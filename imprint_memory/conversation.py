"""
Conversation log — Layer 3 of the memory architecture.
Stores full conversation history from all platforms with FTS5 search.
"""

from .db import _get_db, now_str


def log_message(
    platform: str,
    direction: str,
    content: str,
    speaker: str = "",
    session_id: str = "",
    entrypoint: str = "",
    created_at: str = "",
) -> dict:
    """Write one message to conversation_log."""
    if not content or not content.strip():
        return {"ok": False, "error": "empty content"}

    ts = created_at or now_str()
    db = _get_db()
    try:
        cur = db.execute(
            """INSERT INTO conversation_log
               (platform, direction, speaker, content, session_id, entrypoint, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (platform, direction, speaker, content.strip(), session_id, entrypoint, ts),
        )
        db.commit()
        return {"ok": True, "id": cur.lastrowid}
    finally:
        db.close()


def search_conversations(
    query: str, platform: str = "", limit: int = 20
) -> list[dict]:
    """FTS5 keyword search over conversation history."""
    db = _get_db()
    try:
        if platform:
            rows = db.execute(
                """SELECT c.id, c.platform, c.direction, c.speaker, c.content,
                          c.session_id, c.entrypoint, c.created_at
                   FROM conversation_log_fts f
                   JOIN conversation_log c ON c.id = f.rowid
                   WHERE conversation_log_fts MATCH ? AND c.platform = ?
                   ORDER BY c.id DESC LIMIT ?""",
                (query, platform, limit),
            ).fetchall()
        else:
            rows = db.execute(
                """SELECT c.id, c.platform, c.direction, c.speaker, c.content,
                          c.session_id, c.entrypoint, c.created_at
                   FROM conversation_log_fts f
                   JOIN conversation_log c ON c.id = f.rowid
                   WHERE conversation_log_fts MATCH ?
                   ORDER BY c.id DESC LIMIT ?""",
                (query, limit),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        db.close()


def get_recent(platform: str = "", limit: int = 30) -> list[dict]:
    """Get the most recent N messages, optionally filtered by platform."""
    db = _get_db()
    try:
        if platform:
            rows = db.execute(
                """SELECT id, platform, direction, speaker, content, session_id, entrypoint, created_at
                   FROM conversation_log WHERE platform = ?
                   ORDER BY id DESC LIMIT ?""",
                (platform, limit),
            ).fetchall()
        else:
            rows = db.execute(
                """SELECT id, platform, direction, speaker, content, session_id, entrypoint, created_at
                   FROM conversation_log ORDER BY id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]  # chronological order
    finally:
        db.close()


def format_recent(messages: list[dict], max_content_len: int = 100) -> str:
    """Format recent messages for display."""
    platform_short: dict[str, str] = {}  # users can override; e.g. {"telegram": "tg"}
    lines = []
    for m in messages:
        p = platform_short.get(m["platform"], m["platform"])
        d = "in" if m["direction"] == "in" else "out"
        ts = m["created_at"]
        if len(ts) >= 16:
            ts = ts[5:16]
        content = m["content"]
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."
        lines.append(f"[{ts} {p}/{d}] {content}")
    return "\n".join(lines)


def format_search_results(results: list[dict]) -> str:
    """Format search results for MCP tool output."""
    if not results:
        return "No matching conversation records found"
    lines = []
    for r in results:
        p = r["platform"]
        d = "\u2190" if r["direction"] == "in" else "\u2192"
        ts = r["created_at"]
        content = r["content"]
        if len(content) > 200:
            content = content[:200] + "..."
        lines.append(f"[{ts}] {p}{d} {content}")
    return "\n".join(lines)
