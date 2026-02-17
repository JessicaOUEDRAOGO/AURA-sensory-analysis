# src/core/protocol/service.py
import uuid
from datetime import datetime
from src.core.protocol.models import Protocol
from src.core.protocol.repository import ProtocolRepository

class ProtocolService:
    def __init__(self, repo: ProtocolRepository):
        self.repo = repo

    def create_new(self, name: str, instruction_type: str = "image") -> Protocol:
        existing = self.repo.get_by_name(name)
        if existing:
            raise ValueError("Nom de protocole déjà utilisé.")

        p = Protocol(
            id=str(uuid.uuid4()),
            name=name.strip(),
            goal="",
            hypotheses="",
            instruction_type=instruction_type,
            modules_enabled=[],
            data_to_export=[],
            locked=False,
            created_at=datetime.now().isoformat(timespec="seconds"),
            version=1
        )
        self.repo.create(p)
        return p

    def open_existing_readonly(self, name: str) -> Protocol:
        p = self.repo.get_by_name(name)
        if not p:
            raise ValueError("Protocole introuvable.")
        # # verrouille en DB (lecture seule)
        # self.repo.set_locked(p.id, True)
        return Protocol(**{**p.__dict__, "locked": True})

    def duplicate(self, source_name: str, new_name: str) -> Protocol:
        src = self.repo.get_by_name(source_name)
        if not src:
            raise ValueError("Protocole source introuvable.")
        if self.repo.get_by_name(new_name):
            raise ValueError("Nouveau nom déjà utilisé.")

        new_p = Protocol(
            id=str(uuid.uuid4()),
            name=new_name.strip(),
            goal=src.goal,
            hypotheses=src.hypotheses,
            instruction_type=src.instruction_type,
            modules_enabled=list(src.modules_enabled or []),
            data_to_export=list(src.data_to_export or []),
            locked=False,
            created_at=datetime.now().isoformat(timespec="seconds"),
            version=src.version + 1
        )
        self.repo.create(new_p)
        # NOTE: on dupliquera assets + timeline ensuite (dans Sprint 2)
        return new_p
