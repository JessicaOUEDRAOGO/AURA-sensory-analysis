# src/core/session/export_service.py
import json
import os
from datetime import datetime
from src.core.storage.db import connect

class ExportService:
    def export_session_minimal(self, session_id: str) -> None:
        conn = connect()
        try:
            s = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if not s:
                return
            p = conn.execute("SELECT * FROM protocols WHERE id = ?", (s["protocol_id"],)).fetchone()
            out_dir = s["output_dir"]
            os.makedirs(out_dir, exist_ok=True)

            # session.json
            with open(os.path.join(out_dir, "session.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "session_id": s["id"],
                    "protocol_id": s["protocol_id"],
                    "participant_id": s["participant_id"],
                    "started_at": s["started_at"],
                    "ended_at": s["ended_at"],
                    "exported_at": datetime.now().isoformat(timespec="seconds")
                }, f, indent=2, ensure_ascii=False)

            # protocol_snapshot.json
            if p:
                with open(os.path.join(out_dir, "protocol_snapshot.json"), "w", encoding="utf-8") as f:
                    json.dump(dict(p), f, indent=2, ensure_ascii=False)

        finally:
            conn.close()
