# src/core/session/models.py
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Session:
    id: str
    protocol_id: str
    participant_id: str
    started_at: str
    ended_at: Optional[str]
    output_dir: str
