# src/core/protocol/repository.py
import json
from typing import Optional, List
from src.core.storage.db import connect
from src.core.protocol.models import Protocol

class ProtocolRepository:
    def create(self, p: Protocol) -> None:
        conn = connect()
        try:
            conn.execute(
                """
                INSERT INTO protocols(id, name, goal, hypotheses, instruction_type, modules_enabled, data_to_export, locked, created_at, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    p.id, p.name, p.goal, p.hypotheses, p.instruction_type,
                    json.dumps(p.modules_enabled or []),
                    json.dumps(p.data_to_export or []),
                    1 if p.locked else 0,
                    p.created_at, p.version
                )
            )
            conn.commit()
        finally:
            conn.close()

    def get_by_name(self, name: str) -> Optional[Protocol]:
        conn = connect()
        try:
            row = conn.execute("SELECT * FROM protocols WHERE name = ?", (name,)).fetchone()
            return self._row_to_protocol(row) if row else None
        finally:
            conn.close()

    def list(self, search: str = "") -> List[Protocol]:
        conn = connect()
        try:
            if search:
                rows = conn.execute("SELECT * FROM protocols WHERE name LIKE ? ORDER BY created_at DESC", (f"%{search}%",)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM protocols ORDER BY created_at DESC").fetchall()
            return [self._row_to_protocol(r) for r in rows]
        finally:
            conn.close()

    def set_locked(self, protocol_id: str, locked: bool) -> None:
        conn = connect()
        try:
            conn.execute("UPDATE protocols SET locked = ? WHERE id = ?", (1 if locked else 0, protocol_id))
            conn.commit()
        finally:
            conn.close()

    def _row_to_protocol(self, r) -> Protocol:
        import json
        return Protocol(
            id=r["id"],
            name=r["name"],
            goal=r["goal"] or "",
            hypotheses=r["hypotheses"] or "",
            instruction_type=r["instruction_type"],
            modules_enabled=json.loads(r["modules_enabled"] or "[]"),
            data_to_export=json.loads(r["data_to_export"] or "[]"),
            locked=bool(r["locked"]),
            created_at=r["created_at"],
            version=int(r["version"]),
        )

    def update_fields(self, protocol_id: str, goal: str, hypotheses: str, instruction_type: str) -> None:
        conn = connect()
        try:
            conn.execute(
                """
                UPDATE protocols
                SET goal = ?, hypotheses = ?, instruction_type = ?
                WHERE id = ?
                """,
                (goal, hypotheses, instruction_type, protocol_id)
            )
            conn.commit()
        finally:
            conn.close()

