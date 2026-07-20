from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from .config import PROJECT_ROOT, fvlm_root
from .manifest import checksum


def _prepare_imports() -> None:
    root = fvlm_root()
    if not (root / "lavis").is_dir():
        raise FileNotFoundError(f"fVLM checkout is missing: {root}")
    sys.path.insert(0, str(root))
    os.environ.setdefault("FVLM_MERLIN_ROOT", str(PROJECT_ROOT))
    os.chdir(PROJECT_ROOT)


def _revision(path: Path) -> str:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def run(config_path: Path, options: list[str] | None = None) -> None:
    _prepare_imports()
    import lavis.tasks as tasks
    from lavis.common.config import Config
    from lavis.common.dist_utils import get_rank, init_distributed_mode
    from lavis.common.logger import setup_logger
    from lavis.common.utils import now
    from lavis.common.registry import registry

    for module in ("lavis.common.optims", "lavis.datasets.builders", "lavis.models",
                   "lavis.processors", "lavis.runners", "lavis.tasks"):
        importlib.import_module(module)
    from . import lavis_adapter
    lavis_adapter.register()

    cfg = Config(argparse.Namespace(cfg_path=str(config_path), options=options))
    init_distributed_mode(cfg.run_cfg)
    seed = int(cfg.run_cfg.seed) + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    setup_logger()
    cfg.pretty_print()

    if get_rank() == 0:
        output = Path(cfg.run_cfg.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        train_manifest = Path(cfg.datasets_cfg.abdominal_fvlm_caption.build_info.annotations.train.storage)
        metadata = {
            "project_revision": _revision(PROJECT_ROOT),
            "fvlm_revision": _revision(fvlm_root()),
            "train_manifest": str(train_manifest.resolve()),
            "train_manifest_sha256": checksum(train_manifest),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        }
        (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        from omegaconf import OmegaConf
        OmegaConf.save(cfg.config, output / "resolved_config.yaml", resolve=True)

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    runner_name = cfg.run_cfg.get("runner", "runner_base")
    runner = registry.get_runner_class(runner_name)(
        cfg=cfg, job_id=now(), task=task, model=model, datasets=datasets
    )
    runner.train()
