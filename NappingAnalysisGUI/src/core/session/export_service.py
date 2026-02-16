# -*- coding: utf-8 -*-
import csv
import json
import os
from datetime import datetime
from typing import Any, Dict

from src.core.storage.db import connect


def _sanitize_filename(s: str) -> str:
    if s is None:
        return ""
    bad = '\\/:*?"<>|'
    for c in bad:
        s = s.replace(c, "_")
    return s.strip().replace(" ", "_")


def _safe_json_loads(s: str, default: Any):
    try:
        return json.loads(s) if s else default
    except Exception:
        return default


class ExportService:
    """
    Exporte une session sous 3 fichiers dans s["output_dir"] :

    - <PROTO>_<PARTICIPANT>_session.json
    - <PROTO>_<PARTICIPANT>_protocol.json
    - <PROTO>_<PARTICIPANT>_events.csv

    NOTE:
    - Le dossier out_dir est celui stocké dans la table sessions.output_dir
    - events.payload est stocké en JSON string en DB -> on le parse si possible
    """

    def export_session_minimal(self, session_id: str) -> None:
        conn = connect()
        try:
            s = conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if not s:
                return

            p = conn.execute(
                "SELECT * FROM protocols WHERE id = ?",
                (s["protocol_id"],),
            ).fetchone()

            # Dossier de sortie (déjà défini à la création de session)
            out_dir = s["output_dir"]
            os.makedirs(out_dir, exist_ok=True)

            protocol_name_raw = p["name"] if p else "UNKNOWN_PROTOCOL"
            participant_raw = s["participant_id"] or "P001"

            protocol_name = _sanitize_filename(protocol_name_raw) or "UNKNOWN_PROTOCOL"
            participant_id = _sanitize_filename(participant_raw) or "P001"
            base = f"{protocol_name}_{participant_id}"

            exported_at = datetime.now().isoformat(timespec="seconds")

            # ---------------------------------------------------------
            # 1) SESSION JSON
            # ---------------------------------------------------------
            session_path = os.path.join(out_dir, f"{base}_session.json")
            session_payload: Dict[str, Any] = {
                "session_id": s["id"],
                "protocol_id": s["protocol_id"],
                "protocol_name": protocol_name_raw,
                "participant_id": participant_raw,
                "started_at": s["started_at"],
                "ended_at": s["ended_at"],
                "output_dir": s["output_dir"],
                "exported_at": exported_at,
            }

            with open(session_path, "w", encoding="utf-8") as f:
                json.dump(session_payload, f, indent=2, ensure_ascii=False)

            # ---------------------------------------------------------
            # 2) PROTOCOL SNAPSHOT JSON
            # ---------------------------------------------------------
            if p:
                protocol_path = os.path.join(out_dir, f"{base}_protocol.json")
                with open(protocol_path, "w", encoding="utf-8") as f:
                    # dict(p) car sqlite3.Row
                    json.dump(dict(p), f, indent=2, ensure_ascii=False)

            # ---------------------------------------------------------
            # 3) EVENTS CSV
            # ---------------------------------------------------------
            events = conn.execute(
                "SELECT id, session_id, t, type, payload FROM events WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()

            events_csv_path = os.path.join(out_dir, f"{base}_events.csv")

            # On garde un CSV "plat" + une colonne payload_json (string) + payload_parsed (json si possible)
            # (payload_parsed sera ré-écrit en JSON string pour rester CSV-compatible)
            fieldnames = ["id", "session_id", "t", "type", "payload_json", "payload_parsed"]

            with open(events_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for e in events:
                    payload_json = e["payload"] if e["payload"] is not None else ""
                    payload_obj = _safe_json_loads(payload_json, default={})

                    writer.writerow(
                        {
                            "id": e["id"],
                            "session_id": e["session_id"],
                            "t": e["t"],
                            "type": e["type"],
                            "payload_json": payload_json,
                            "payload_parsed": json.dumps(payload_obj, ensure_ascii=False),
                        }
                    )

        finally:
            conn.close()
