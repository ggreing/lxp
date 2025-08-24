"""SQLite‑based performance & conversation analytics."""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd

DB_PATH = "personal_logs.db"

__all__ = ["PersonalLogManager"]


class PersonalLogManager:
    def get_user_sessions(self, user_id: str) -> List[Dict]:
        """Return all finished sessions for a user, with persona type, score, and start time."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """
            SELECT session_id, persona_type, scenario, start_time, performance_score
            FROM session_logs
            WHERE user_id=? AND end_time IS NOT NULL
            ORDER BY start_time DESC
            """, (user_id,)
        ).fetchall()
        conn.close()
        return [
            {
                "id": row[0],
                "persona_type": row[1],
                "scenario": row[2],
                "startedAt": row[3],
                "score": row[4],
            }
            for row in rows
        ]
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    # ---------------------------------------------------------------------
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # users ------------------------------------------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_sessions INTEGER DEFAULT 0,
                total_messages INTEGER DEFAULT 0,
                best_score REAL DEFAULT 0.0
            )
        """)
        # sessions ---------------------------------------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS session_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                session_id TEXT,
                persona_type TEXT,
                scenario TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                message_count INTEGER,
                performance_score REAL,
                feedback TEXT,
                session_duration REAL,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)
        # messages ---------------------------------------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS message_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message_length INTEGER,
                FOREIGN KEY (session_id) REFERENCES session_logs (session_id)
            )
        """)
        # performance tracking --------------------------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS performance_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                date DATE,
                daily_sessions INTEGER,
                avg_daily_score REAL,
                total_messages INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)
        conn.commit()
        conn.close()

    # ---------------------------------------------------------------------
    def create_user(self, user_id: str, username: str | None = None):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR IGNORE INTO users (user_id, username) VALUES (?, ?)",
            (user_id, username or f"User_{user_id[:8]}")
        )
        conn.commit()
        conn.close()

    def log_session_start(self, user_id: str, session_id: str, persona_type: str, scenario: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO session_logs (user_id, session_id, persona_type, scenario, start_time)
            VALUES (?, ?, ?, ?, ?)""",
            (user_id, session_id, persona_type, scenario, datetime.now()),
        )
        conn.commit()
        conn.close()

    def log_message(self, session_id: str, user_id: str, role: str, content: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO message_logs (session_id, user_id, role, content, message_length)
            VALUES (?, ?, ?, ?, ?)""",
            (session_id, user_id, role, content, len(content)),
        )
        conn.commit()
        conn.close()

    def log_session_end(self, session_id: str, performance_score: float, feedback: str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT start_time, user_id FROM session_logs WHERE session_id = ?", (session_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            return
        start_time, user_id = datetime.fromisoformat(row[0]), row[1]
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        c.execute("SELECT COUNT(*) FROM message_logs WHERE session_id = ?", (session_id,))
        msg_count = c.fetchone()[0]
        # update session ---------------------------------------------------
        c.execute(
            """UPDATE session_logs SET end_time=?, message_count=?, performance_score=?, feedback=?,
            session_duration=? WHERE session_id=?""",
            (end_time, msg_count, performance_score, feedback, duration, session_id),
        )
        # update user ------------------------------------------------------
        c.execute(
            """UPDATE users SET total_sessions = total_sessions + 1,
            total_messages = total_messages + ?, best_score = MAX(best_score, ?)
            WHERE user_id = ?""",
            (msg_count, performance_score, user_id),
        )
        # daily perf --------------------------------------------------------
        today = datetime.now().date()
        c.execute("""
            INSERT OR REPLACE INTO performance_tracking (user_id, date, daily_sessions, avg_daily_score, total_messages)
            VALUES (
                ?, ?,
                COALESCE((SELECT daily_sessions FROM performance_tracking WHERE user_id=? AND date=?), 0) + 1,
                COALESCE((SELECT avg_daily_score FROM performance_tracking WHERE user_id=? AND date=?), 0) * 0.5 + ? * 0.5,
                COALESCE((SELECT total_messages FROM performance_tracking WHERE user_id=? AND date=?), 0) + ?
            )""",
            (user_id, today, user_id, today, user_id, today, performance_score, user_id, today, msg_count),
        )
        conn.commit()
        conn.close()

    # ---------------------------------------------------------------------
    def get_user_stats(self, user_id: str) -> Dict:
        conn = sqlite3.connect(self.db_path)
        basic = pd.read_sql_query(
            """SELECT u.total_sessions, u.total_messages, u.best_score,
               COALESCE(AVG(s.performance_score),0) AS avg_performance,
               COALESCE(AVG(s.message_count),0) AS avg_messages_per_session,
               COALESCE(AVG(s.session_duration),0) AS avg_session_duration
            FROM users u LEFT JOIN session_logs s ON u.user_id=s.user_id
            WHERE u.user_id=? AND s.end_time IS NOT NULL""", conn, params=[user_id])
        persona = pd.read_sql_query(
            """SELECT persona_type, COUNT(*) AS sessions,
               AVG(performance_score) AS avg_score, MAX(performance_score) AS best_score
            FROM session_logs WHERE user_id=? AND end_time IS NOT NULL
            GROUP BY persona_type""", conn, params=[user_id])
        recent = pd.read_sql_query(
            """SELECT session_id, persona_type, scenario, start_time, performance_score, message_count
            FROM session_logs WHERE user_id=? AND end_time IS NOT NULL
            ORDER BY start_time DESC LIMIT 10""", conn, params=[user_id])
        weekly = pd.read_sql_query(
            """SELECT date, daily_sessions, avg_daily_score, total_messages
            FROM performance_tracking WHERE user_id=? AND date>=DATE('now','-7 day')""", conn, params=[user_id])
        conn.close()
        return {
            "basic_stats": basic.to_dict("records")[0] if not basic.empty else {},
            "persona_stats": persona.to_dict("records"),
            "recent_sessions": recent.to_dict("records"),
            "weekly_performance": weekly.to_dict("records"),
        }