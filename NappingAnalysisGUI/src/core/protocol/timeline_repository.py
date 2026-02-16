# -*- coding: utf-8 -*-
import uuid
from datetime import datetime
from typing import List, Optional
from src.core.storage.db import connect
from src.core.protocol.models import TimelineStep

class TimelineRepository:
    def replace_all(self, protocol_id: str, steps: List[TimelineStep]) -> None:
        conn = connect()
        try:
            conn.execute("DELETE FROM timeline_steps WHERE protocol_id = ?", (protocol_id,))
            for s in steps:
                conn.execute(
                    """
                    INSERT INTO timeline_steps(
                        id, protocol_id, order_index, asset_ref, duration_s, label, repeat, pause, trigger
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        s.id,
                        protocol_id,
                        s.order_index,
                        s.asset_ref,
                        float(s.duration_s),
                        s.label,
                        s.repeat,
                        1 if s.pause else 0 if s.pause is not None else None,
                        s.trigger,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def list(self, protocol_id: str) -> List[TimelineStep]:
        conn = connect()
        try:
            rows = conn.execute(
                "SELECT * FROM timeline_steps WHERE protocol_id = ? ORDER BY order_index ASC",
                (protocol_id,),
            ).fetchall()
            return [self._row_to_step(r) for r in rows]
        finally:
            conn.close()

    def _row_to_step(self, r) -> TimelineStep:
        return TimelineStep(
            id=r["id"],
            protocol_id=r["protocol_id"],
            order_index=int(r["order_index"]),
            asset_ref=r["asset_ref"],
            duration_s=float(r["duration_s"]),
            label=r["label"],
            repeat=r["repeat"],
            pause=bool(r["pause"]) if r["pause"] is not None else None,
            trigger=r["trigger"],
        )

def new_step_id() -> str:
    return str(uuid.uuid4())
