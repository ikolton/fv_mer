from __future__ import annotations

import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import PROJECT_ROOT

MAE_URL = "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
BIOMED_REPO = "microsoft/BiomedVLP-CXR-BERT-specialized"


def download() -> None:
    root = PROJECT_ROOT / "assets"
    root.mkdir(parents=True, exist_ok=True)
    mae = root / "mae_pretrain_vit_base.pth"
    if not mae.is_file():
        temporary = mae.with_suffix(".tmp")
        urllib.request.urlretrieve(MAE_URL, temporary)
        temporary.replace(mae)
    target = root / "biomed_cxrbert"
    snapshot_download(repo_id=BIOMED_REPO, local_dir=target)
    compatibility_path = PROJECT_ROOT / "BiomedVLP-CXR-BERT-specialized"
    if compatibility_path.is_symlink() and compatibility_path.resolve() != target.resolve():
        compatibility_path.unlink()
    if not compatibility_path.exists():
        compatibility_path.symlink_to(target, target_is_directory=True)
