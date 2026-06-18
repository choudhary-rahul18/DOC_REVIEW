import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config import DB_PATH

logger = logging.getLogger(__name__)


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS threads (
            thread_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            title TEXT,
            summary TEXT,
            summary_msg_count INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            metadata TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id);
    """)
    # Migrate existing DBs that predate added columns
    existing_thread_cols = {r[1] for r in conn.execute("PRAGMA table_info(threads)").fetchall()}
    if "title" not in existing_thread_cols:
        conn.execute("ALTER TABLE threads ADD COLUMN title TEXT")
    if "summary" not in existing_thread_cols:
        conn.execute("ALTER TABLE threads ADD COLUMN summary TEXT")
    if "summary_msg_count" not in existing_thread_cols:
        conn.execute("ALTER TABLE threads ADD COLUMN summary_msg_count INTEGER DEFAULT 0")
    existing_msg_cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()}
    if "metadata" not in existing_msg_cols:
        conn.execute("ALTER TABLE messages ADD COLUMN metadata TEXT")
    conn.commit()


def _get_conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    return conn


# ── Thread CRUD ────────────────────────────────────────────────────────────────

def create_thread() -> str:
    thread_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO threads (thread_id, created_at) VALUES (?, ?)",
                (thread_id, now),
            )
    except Exception:
        logger.warning("[memory] create_thread failed for %s", thread_id, exc_info=True)
        raise
    return thread_id


def set_thread_title(thread_id: str, title: str) -> None:
    try:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE threads SET title = ? WHERE thread_id = ?",
                (title, thread_id),
            )
    except Exception:
        logger.warning("[memory] set_thread_title failed for %s", thread_id, exc_info=True)
        raise


def list_threads() -> list[dict]:
    try:
        with _get_conn() as conn:
            rows = conn.execute("""
                SELECT t.thread_id, t.created_at, t.title,
                       COUNT(m.id) as message_count,
                       MAX(m.timestamp) as last_message_at
                FROM threads t
                LEFT JOIN messages m ON t.thread_id = m.thread_id
                GROUP BY t.thread_id
                ORDER BY t.created_at DESC
            """).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        logger.warning("[memory] list_threads failed", exc_info=True)
        raise


def delete_thread(thread_id: str) -> None:
    try:
        conn = _get_conn()
        with conn:
            conn.execute("BEGIN")
            conn.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
        conn.close()
    except Exception:
        logger.warning("[memory] delete_thread failed for %s", thread_id, exc_info=True)
        raise


# ── Message CRUD ───────────────────────────────────────────────────────────────

def add_message(thread_id: str, role: str, content: str, metadata: dict | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    metadata_json = json.dumps(metadata) if metadata is not None else None
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO messages (thread_id, role, content, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
                (thread_id, role, content, now, metadata_json),
            )
    except Exception:
        logger.warning("[memory] add_message failed for thread %s role=%s", thread_id, role, exc_info=True)
        raise


def get_history(thread_id: str, n: int) -> list[dict]:
    """Return last n messages as [{role, content}] in chronological order."""
    try:
        with _get_conn() as conn:
            rows = conn.execute("""
                SELECT role, content FROM (
                    SELECT role, content, timestamp
                    FROM messages
                    WHERE thread_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ) ORDER BY timestamp ASC
            """, (thread_id, n)).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]
    except Exception:
        logger.warning("[memory] get_history failed for %s", thread_id, exc_info=True)
        raise


def get_message_count(thread_id: str) -> int:
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM messages WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        return row["cnt"] if row else 0
    except Exception:
        logger.warning("[memory] get_message_count failed for %s", thread_id, exc_info=True)
        raise


def get_messages_slice(thread_id: str, offset: int, limit: int) -> list[dict]:
    """Return messages[offset : offset+limit] in insertion order."""
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE thread_id = ? ORDER BY id ASC LIMIT ? OFFSET ?",
                (thread_id, limit, offset),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]
    except Exception:
        logger.warning("[memory] get_messages_slice failed for %s", thread_id, exc_info=True)
        raise


def get_thread_summary(thread_id: str) -> dict:
    """Return {summary, summary_msg_count} for the thread."""
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT summary, summary_msg_count FROM threads WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        if row is None:
            return {"summary": None, "summary_msg_count": 0}
        return {"summary": row["summary"], "summary_msg_count": row["summary_msg_count"] or 0}
    except Exception:
        logger.warning("[memory] get_thread_summary failed for %s", thread_id, exc_info=True)
        raise


def set_thread_summary(thread_id: str, summary: str, msg_count: int) -> None:
    try:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE threads SET summary = ?, summary_msg_count = ? WHERE thread_id = ?",
                (summary, msg_count, thread_id),
            )
    except Exception:
        logger.warning("[memory] set_thread_summary failed for %s", thread_id, exc_info=True)
        raise


def get_all_messages(thread_id: str) -> list[dict]:
    try:
        with _get_conn() as conn:
            rows = conn.execute("""
                SELECT role, content, timestamp, metadata
                FROM messages
                WHERE thread_id = ?
                ORDER BY timestamp ASC
            """, (thread_id,)).fetchall()
        result = []
        for r in rows:
            row = dict(r)
            raw_meta = row.get("metadata")
            row["metadata"] = json.loads(raw_meta) if raw_meta else None
            result.append(row)
        return result
    except Exception:
        logger.warning("[memory] get_all_messages failed for %s", thread_id, exc_info=True)
        raise
