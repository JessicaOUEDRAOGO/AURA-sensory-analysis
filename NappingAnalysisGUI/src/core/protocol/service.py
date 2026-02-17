# src/core/protocol/service.py
import os
import shutil
import uuid
from datetime import datetime
from src.core.protocol.models import Protocol, InstructionAsset, TimelineStep
from src.core.protocol.repository import ProtocolRepository
from src.core.protocol.asset_repository import InstructionAssetRepository
from src.core.protocol.timeline_repository import TimelineRepository, new_step_id
from src.core.protocol.participants_repository import ProtocolParticipantsRepository
from src.core.utils.paths import data_path

def _sanitize_name(s: str) -> str:
    bad = '\\/:*?"<>|'
    for c in bad:
        s = s.replace(c, "_")
    return s.strip().replace(" ", "_")

class ProtocolService:
    def __init__(self, repo: ProtocolRepository):
        self.repo = repo
        self.asset_repo = InstructionAssetRepository()
        self.timeline_repo = TimelineRepository()
        self.part_repo = ProtocolParticipantsRepository()

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

        # 1) créer protocole
        new_p = Protocol(
            id=str(uuid.uuid4()),
            name=new_name.strip(),
            goal=src.goal,
            hypotheses=src.hypotheses,
            instruction_type=src.instruction_type,
            modules_enabled=dict(src.modules_enabled or {}),
            data_to_export=dict(src.data_to_export or {}),
            locked=False,  # ✅ duplication toujours éditable
            created_at=datetime.now().isoformat(timespec="seconds"),
            version=int(src.version) + 1,
        )
        self.repo.create(new_p)

        # 2) copier participants
        try:
            participants = self.part_repo.list(src.id)
            if participants:
                self.part_repo.replace_all(new_p.id, participants)
        except Exception:
            # pas bloquant si vide ou souci mineur
            pass

        # 3) copier assets + fichiers (remap id)
        old_assets = self.asset_repo.list_by_protocol(src.id)
        asset_id_map = {}  # old_asset_id -> new_asset_id

        src_dir = data_path("protocols", _sanitize_name(src.name), "assets")
        dst_dir = data_path("protocols", _sanitize_name(new_p.name), "assets")
        os.makedirs(dst_dir, exist_ok=True)

        for a in old_assets:
            new_asset_id = str(uuid.uuid4())
            asset_id_map[a.id] = new_asset_id

            # chemin destination fichier
            filename = os.path.basename(a.path) if a.path else f"{new_asset_id}"
            dst_path = os.path.join(dst_dir, filename)

            # éviter écrasement
            if os.path.exists(dst_path):
                name, ext = os.path.splitext(filename)
                dst_path = os.path.join(dst_dir, f"{name}_{uuid.uuid4().hex[:6]}{ext}")

            # copie fichier (si présent)
            try:
                if a.path and os.path.exists(a.path):
                    shutil.copy2(a.path, dst_path)
                else:
                    # fallback: si path est cassé, tente dans src_dir
                    alt = os.path.join(src_dir, filename)
                    if os.path.exists(alt):
                        shutil.copy2(alt, dst_path)
            except Exception:
                # si la copie échoue, on garde quand même l’asset en DB
                pass

            new_asset = InstructionAsset(
                id=new_asset_id,
                protocol_id=new_p.id,
                asset_type=a.asset_type,
                path=dst_path,
                meta=dict(a.meta or {}),
                created_at=datetime.now().isoformat(timespec="seconds"),
            )
            self.asset_repo.add(new_asset)

        # 4) copier timeline + remap asset_ref
        old_steps = self.timeline_repo.list(src.id)
        if old_steps:
            new_steps = []
            for s in old_steps:
                new_steps.append(
                    TimelineStep(
                        id=new_step_id(),
                        protocol_id=new_p.id,
                        order_index=s.order_index,
                        asset_ref=asset_id_map.get(s.asset_ref) if s.asset_ref else None,
                        duration_s=s.duration_s,
                        label=s.label,
                        repeat=s.repeat,
                        pause=s.pause,
                        trigger=s.trigger,
                    )
                )
            self.timeline_repo.replace_all(new_p.id, new_steps)

        return new_p
