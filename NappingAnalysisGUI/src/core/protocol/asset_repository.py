# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from typing import List
from src.core.storage.db import connect
from src.core.protocol.models import InstructionAsset


class InstructionAssetRepository:
    def add(self, a: InstructionAsset) -> None:
        conn = connect()
        try:
            conn.execute(
                """
                INSERT INTO instruction_assets(id, protocol_id, asset_type, path, meta, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (a.id, a.protocol_id, a.asset_type, a.path, json.dumps(a.meta or {}), a.created_at)
            )
            conn.commit()
        finally:
            conn.close()

    def list_by_protocol(self, protocol_id: str) -> List[InstructionAsset]:
        conn = connect()
        try:
            rows = conn.execute(
                "SELECT * FROM instruction_assets WHERE protocol_id = ? ORDER BY created_at ASC",
                (protocol_id,)
            ).fetchall()

            out: List[InstructionAsset] = []
            for r in rows:
                out.append(
                    InstructionAsset(
                        id=r["id"],
                        protocol_id=r["protocol_id"],
                        asset_type=r["asset_type"],
                        path=r["path"],
                        meta=json.loads(r["meta"] or "{}"),
                        created_at=r["created_at"],
                    )
                )
            return out
        finally:
            conn.close()
            
    def list(self, protocol_id: str) -> List[InstructionAsset]:
        conn = connect()
        try:
            rows = conn.execute(
                "SELECT * FROM instruction_assets WHERE protocol_id = ? ORDER BY created_at ASC",
                (protocol_id,),
            ).fetchall()
            out = []
            for r in rows:
                out.append(
                    InstructionAsset(
                        id=r["id"],
                        protocol_id=r["protocol_id"],
                        asset_type=r["asset_type"],
                        path=r["path"],
                        meta=json.loads(r["meta"] or "{}"),
                        created_at=r["created_at"],
                    )
                )
            return out
        finally:
            conn.close()

    def delete(self, asset_id: str) -> None:
        conn = connect()
        try:
            conn.execute("DELETE FROM instruction_assets WHERE id = ?", (asset_id,))
            conn.commit()
        finally:
            conn.close()
