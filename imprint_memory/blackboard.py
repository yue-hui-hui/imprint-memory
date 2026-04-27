"""
Blackboard - Short-lived coding handoff area.
For temporary TODOs, handoff notes, and unfinished context.
"""

import json
import secrets
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from .db import _get_db, now_str, now_local
from .memory_manager import daily_log

DEFAULT_TTL_HOURS = 72
MAX_ITEMS = 20


def _gen_id() -> str:
    return f"bb_{secrets.token_hex(4)}"


def _parse_refs(refs: Any) -> List[str]:
    if isinstance(refs, list):
        return refs
    if isinstance(refs, str):
        try:
            parsed = json.loads(refs)
            return parsed if isinstance(parsed, list) else []
        except:
            return [r.strip() for r in refs.split(",") if r.strip()]
    return []


def _row_to_item(row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "type": row["type"] if "type" in row.keys() else "todo",
        "status": row["status"],
        "priority": row["priority"],
        "title": row["title"],
        "body": row["body"],
        "refs": json.loads(row["refs"]) if row["refs"] else [],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "expires_at": row["expires_at"],
    }


def _format_response(scope: str, items: List[Dict], message: str = "") -> Dict[str, Any]:
    open_count = sum(1 for i in items if i["status"] == "open")
    checked_count = sum(1 for i in items if i["status"] == "checked")
    return {
        "scope": scope,
        "updated_at": now_str(),
        "summary": f"{open_count} open, {checked_count} checked",
        "message": message,
        "items": items,
    }


def blackboard_read(scope: str, db=None) -> Dict[str, Any]:
    """Read all non-expired items for a scope."""
    _close = db is None
    if _close:
        db = _get_db()
    now = now_str()
    now_dt = now_local()
    rows = db.execute(
        "SELECT * FROM blackboard WHERE scope = ? AND expires_at > ? ORDER BY priority DESC, created_at DESC",
        (scope, now)
    ).fetchall()
    items = [_row_to_item(r) for r in rows]
    # goal 排在 todo 前面
    items.sort(key=lambda x: (0 if x.get("type") == "goal" else 1, x["id"]))
    if _close:
        db.close()
    result = _format_response(scope, items)

    # Check for items expiring within 24 hours
    expiring = []
    for item in items:
        expires_dt = datetime.strptime(item["expires_at"], "%Y-%m-%d %H:%M:%S")
        # Make both naive for comparison
        now_naive = now_dt.replace(tzinfo=None) if hasattr(now_dt, 'tzinfo') and now_dt.tzinfo else now_dt
        if (expires_dt - now_naive).total_seconds() < 86400:
            expiring.append(item["id"])
    if expiring:
        result["warning"] = f"{len(expiring)} item(s) expiring within 24h"
        result["expiring_ids"] = expiring

    return result


def blackboard_write(
    scope: str,
    title: str,
    body: str = "",
    priority: str = "normal",
    refs: Optional[List[str]] = None,
    ttl_hours: int = DEFAULT_TTL_HOURS,
    item_id: Optional[str] = None,
    item_type: str = "todo",
) -> Dict[str, Any]:
    """Write or update a blackboard item. type='goal' for permanent items."""
    db = _get_db()
    now = now_str()
    # goal type never expires
    if item_type == "goal":
        expires = "9999-12-31 23:59:59"
    else:
        expires = (now_local() + timedelta(hours=ttl_hours)).strftime("%Y-%m-%d %H:%M:%S")
    refs_json = json.dumps(refs or [])

    if item_id:
        db.execute(
            """UPDATE blackboard
               SET title = ?, body = ?, priority = ?, refs = ?, updated_at = ?, expires_at = ?, type = ?
               WHERE id = ? AND scope = ?""",
            (title, body, priority, refs_json, now, expires, item_type, item_id, scope)
        )
        message = f"Updated item {item_id}"
    else:
        item_id = _gen_id()
        current_count = db.execute(
            "SELECT COUNT(*) FROM blackboard WHERE scope = ? AND expires_at > ?",
            (scope, now)
        ).fetchone()[0]

        if current_count >= MAX_ITEMS:
            db.close()
            return {"error": f"Blackboard full ({MAX_ITEMS} items). Erase some items first."}

        db.execute(
            """INSERT INTO blackboard (id, scope, status, priority, title, body, refs, created_at, updated_at, expires_at, type)
               VALUES (?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, ?)""",
            (item_id, scope, priority, title, body, refs_json, now, now, expires, item_type)
        )
        message = f"Created {item_type} {item_id}"

    db.commit()
    result = blackboard_read(scope, db=db)
    result["message"] = message
    result["item_id"] = item_id
    db.close()
    return result


def blackboard_check(scope: str, item_id: str, archive: bool = True) -> Dict[str, Any]:
    """Mark an item as checked (completed) and optionally archive to daily log."""
    db = _get_db()
    now = now_str()

    # Get item content before checking
    row = db.execute(
        "SELECT title, body FROM blackboard WHERE id = ? AND scope = ?",
        (item_id, scope)
    ).fetchone()

    # Archive to daily log if requested
    archived = False
    if archive and row:
        title = row["title"]
        body = row["body"]
        log_text = f"[blackboard] ✓ {title}"
        if body:
            log_text += f": {body[:200]}" if len(body) > 200 else f": {body}"
        daily_log(log_text)
        archived = True

    db.execute(
        "UPDATE blackboard SET status = 'checked', updated_at = ? WHERE id = ? AND scope = ?",
        (now, item_id, scope)
    )
    db.commit()
    result = blackboard_read(scope, db=db)
    result["message"] = f"Checked item {item_id}" + (" (archived)" if archived else "")
    db.close()
    return result


def blackboard_uncheck(scope: str, item_id: str) -> Dict[str, Any]:
    """Uncheck an item (reopen)."""
    db = _get_db()
    now = now_str()
    db.execute(
        "UPDATE blackboard SET status = 'open', updated_at = ? WHERE id = ? AND scope = ?",
        (now, item_id, scope)
    )
    db.commit()
    result = blackboard_read(scope, db=db)
    result["message"] = f"Unchecked item {item_id}"
    db.close()
    return result


def blackboard_erase(scope: str, mode: str = "checked_only") -> Dict[str, Any]:
    """Erase blackboard items.
    mode: 'checked_only' (default) or 'all'
    """
    db = _get_db()
    if mode == "all":
        db.execute("DELETE FROM blackboard WHERE scope = ?", (scope,))
        message = "Erased all items"
    else:
        db.execute("DELETE FROM blackboard WHERE scope = ? AND status = 'checked'", (scope,))
        message = "Erased checked items"
    db.commit()
    result = blackboard_read(scope, db=db)
    result["message"] = message
    db.close()
    return result


DEFAULT_SCOPE = "hui"

def blackboard_action(action: str, scope: str = DEFAULT_SCOPE, payload: Optional[Dict] = None) -> str:
    """Unified entry point for blackboard actions. Default scope is 'hui'."""
    payload = payload or {}

    if action == "read":
        result = blackboard_read(scope)
    elif action == "write":
        result = blackboard_write(
            scope=scope,
            title=payload.get("title", ""),
            body=payload.get("body", ""),
            priority=payload.get("priority", "normal"),
            refs=_parse_refs(payload.get("refs", [])),
            ttl_hours=payload.get("ttl_hours", DEFAULT_TTL_HOURS),
            item_id=payload.get("item_id"),
            item_type=payload.get("type", "todo"),
        )
    elif action == "check":
        item_id = payload.get("item_id", "")
        if not item_id:
            return json.dumps({"error": "item_id required for check action"})
        archive = payload.get("archive", True)
        result = blackboard_check(scope, item_id, archive=archive)
    elif action == "uncheck":
        item_id = payload.get("item_id", "")
        if not item_id:
            return json.dumps({"error": "item_id required for uncheck action"})
        result = blackboard_uncheck(scope, item_id)
    elif action == "erase":
        result = blackboard_erase(scope, mode=payload.get("mode", "checked_only"))
    else:
        return json.dumps({"error": f"Unknown action: {action}"})

    return json.dumps(result, ensure_ascii=False, indent=2)
