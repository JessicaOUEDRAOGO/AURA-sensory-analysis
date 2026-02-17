# src/core/protocol/participants_repository.py
from __future__ import annotations
from typing import List
from src.core.storage.db import connect

class ProtocolParticipantsRepository:
    def list(self, protocol_id: str) -> List[str]:
        conn = connect()
        try:
            rows = conn.execute(
                "SELECT participant_id FROM protocol_participants WHERE protocol_id=? ORDER BY participant_id ASC",
                (protocol_id,)
            ).fetchall()
            return [r["participant_id"] for r in rows]
        finally:
            conn.close()

    def replace_all(self, protocol_id: str, participant_ids: List[str]) -> None:
        conn = connect()
        try:
            conn.execute("DELETE FROM protocol_participants WHERE protocol_id=?", (protocol_id,))
            conn.executemany(
                "INSERT OR IGNORE INTO protocol_participants(protocol_id, participant_id) VALUES(?, ?)",
                [(protocol_id, pid) for pid in participant_ids]
            )
            conn.commit()
        finally:
            conn.close()
