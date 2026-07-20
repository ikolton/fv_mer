from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return payload


def project_path(value: str | Path) -> Path:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    return expanded if expanded.is_absolute() else PROJECT_ROOT / expanded


def fvlm_root() -> Path:
    return project_path(os.environ.get("FVLM_ROOT", PROJECT_ROOT / "fvlm"))

