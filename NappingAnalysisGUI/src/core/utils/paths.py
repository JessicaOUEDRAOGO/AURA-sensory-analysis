# -*- coding: utf-8 -*-
from pathlib import Path


def project_root() -> Path:
    """
    Retourne le dossier racine du projet (NappingAnalysisGUI).
    Fonctionne même si la structure évolue légèrement.
    """
    current = Path(__file__).resolve()

    # Remonte jusqu'à trouver le dossier contenant "src"
    for parent in current.parents:
        if (parent / "src").exists():
            return parent

    raise RuntimeError("Impossible de trouver la racine du projet.")


def gui_path(filename: str) -> str:
    """
    Chemin vers les fichiers Qt (.ui)
    """
    return str(project_root() / "src" / "ui" / "qt_ui" / filename)


def asset_path(*parts: str) -> str:
    """
    Chemin vers assets/
    Exemple: asset_path("textures", "image.png")
    """
    return str(project_root() / "assets" / Path(*parts))


def config_path(*parts: str) -> str:
    """
    Chemin vers config/
    """
    return str(project_root() / "config" / Path(*parts))


def data_path(*parts: str) -> str:
    """
    Chemin vers data/
    Crée automatiquement le dossier s’il n’existe pas.
    """
    data_dir = project_root() / "data"
    data_dir.mkdir(exist_ok=True)
    return str(data_dir / Path(*parts))
