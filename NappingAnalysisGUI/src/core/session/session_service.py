# src/core/session/session_service.py
import os
import uuid
from datetime import datetime
from src.core.storage.db import connect
from src.core.utils.paths import data_path

def _sanitize_folder_name(s: str) -> str:
    bad = '\\/:*?"<>|'
    for c in bad:
        s = s.replace(c, "_")
    return s.strip().replace(" ", "_")

class SessionService:
    def start_session(self, protocol_id: str, participant_id: str, protocol_name: str) -> tuple[str, str]:
        session_id = str(uuid.uuid4())
        started_at = datetime.now().isoformat(timespec="seconds")

        safe_proto = _sanitize_folder_name(protocol_name or "UNKNOWN_PROTOCOL")
        safe_pid = _sanitize_folder_name(participant_id or "P001")

        # 1) dossier protocole
        proto_dir = data_path("sessions", safe_proto)
        os.makedirs(proto_dir, exist_ok=True)

        # 2) date (YYYY-MM-DD)
        date_str = datetime.now().strftime("%Y-%m-%d")

        # 3) dossier session lisible : PROTO_PID_DATE
        session_dir_name = f"{safe_proto}_{safe_pid}_{date_str}"
        out_dir = os.path.join(proto_dir, session_dir_name)
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



