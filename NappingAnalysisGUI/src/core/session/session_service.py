# src/core/session/session_service.py
import os
import uuid
from datetime import datetime
from src.core.storage.db import connect
from src.core.utils.paths import data_path

class SessionService:
    def start_session(self, protocol_id: str, participant_id: str) -> tuple[str, str]:
        session_id = str(uuid.uuid4())
        started_at = datetime.now().isoformat(timespec="seconds")

        out_dir = data_path("sessions", session_id)
        os.makedirs(out_dir, exist_ok=True)

        conn = connect()
        try:
            conn.execute(
                "INSERT INTO sessions(id, protocol_id, participant_id, started_at, output_dir) VALUES (?, ?, ?, ?, ?)",
                (session_id, protocol_id, participant_id, started_at, out_dir)
            )
            conn.commit()
        finally:
            conn.close()

        return session_id, out_dir

    def end_session(self, session_id: str) -> None:
        conn = connect()
        try:
            conn.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ?",
                (datetime.now().isoformat(timespec="seconds"), session_id)
            )
            conn.commit()
        finally:
            conn.close()
