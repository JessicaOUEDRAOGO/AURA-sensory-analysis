# src/core/protocol/models.py
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass(frozen=True)
class Protocol:
    id: str
    name: str
    goal: str = ""
    hypotheses: str = ""
    instruction_type: str = "image"  # audio|image|video
    modules_enabled: List[str] = None
    data_to_export: List[str] = None
    locked: bool = False
    created_at: str = ""
    version: int = 1

@dataclass(frozen=True)
class InstructionAsset:
    id: str
    protocol_id: str
    asset_type: str
    path: str
    meta: Dict[str, Any]
    created_at: str

@dataclass(frozen=True)
class TimelineStep:
    id: str
    protocol_id: str
    order_index: int
    asset_ref: Optional[str]
    duration_s: float
    label: str
    repeat: Optional[int] = None
    pause: Optional[bool] = None
    trigger: Optional[str] = None
