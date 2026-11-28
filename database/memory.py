import sqlite3
from pathlib import Path
from app.config import settings

def get_connection() -> sqlite3.Connection:
    db_path = Path(settings.database_url)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database=db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    session_id TEXT NOT NULL, 
                    role TEXT NOT NULL CHECK(role IN ("user", "assistant")), 
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
                """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON messages(session_id, id)")
    conn.commit()
    return conn

def save_turn(session_id: str, user_text: str, assistant_text: str) -> None:
    with get_connection() as conn:
        conn.executemany("""
                    INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)""",
                     [
                         (session_id, "user", user_text),
                         (session_id, "assistant", assistant_text)
                     ],
                    )

def get_history(session_id: str, limit: int = 10) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("""
                    SELECT role, content FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC                
                    LIMIT ?""",
                    (session_id, limit),).fetchall()
                    
    return [{"role": role, "content": content} for role, content in reversed(rows)]

def clear_history(session_id: str) -> None:
    with get_connection() as conn:
        conn.execute("""DELETE FROM messages
                        WHERE session_id = ?""",
                     (session_id,))