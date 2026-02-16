# src/core/session/event_store.py
import json
from datetime import datetime
from src.core.storage.db import connect

class EventStore:
    def log(self, session_id: str, event_type: str, payload: dict) -> None:
        conn = connect()
        try:
            conn.execute(
                "INSERT INTO events(session_id, t, type, payload) VALUES (?, ?, ?, ?)",
                (session_id, datetime.now().isoformat(timespec="milliseconds"), event_type, json.dumps(payload))
            )
            conn.commit()
        finally:
            conn.close()
