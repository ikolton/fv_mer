from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from .manifest import load, resolve_paths
from .organs import ORGANS, ORGAN_NAMES, remap_mask


def _check_volume(item: tuple[dict[str, Path], dict[str, Any]]) -> str:
    roots, row = item
    image_path, mask_path = resolve_paths(roots, row)
    image_obj, mask_obj = nib.load(str(image_path)), nib.load(str(mask_path))
    if image_obj.shape != mask_obj.shape or not np.allclose(image_obj.affine, mask_obj.affine, atol=1e-3):
        raise ValueError(f"Image-mask geometry mismatch: {row['dataset']}:{row['study_id']}")
    dense = remap_mask(np.asanyarray(mask_obj.dataobj))
    if not any(np.any(dense == organ.dense_id) for organ in ORGANS):
        raise ValueError(f"No supported organs in mask: {mask_path}")
    return f"{row['dataset']}:{row['study_id']}"


def run(manifest_paths: list[Path], max_per_dataset: int | None = None, workers: int = 4) -> dict:
    if max_per_dataset is not None and max_per_dataset < 1:
        raise ValueError("Volume limit must be positive")
    if workers < 1:
        raise ValueError("Worker count must be positive")
    report = {"manifests": {}, "volumes_checked": 0, "exhaustive": max_per_dataset is None}
    all_keys = set()
    selected_items = []
    for path in manifest_paths:
        roots, rows = load(path)
        keys = {(row["dataset"], row["study_id"]) for row in rows}
        overlap = all_keys & keys
        if overlap:
            raise ValueError(f"Studies occur in multiple manifests: {sorted(overlap)[:5]}")
        all_keys.update(keys)
        by_dataset = Counter(row["dataset"] for row in rows)
        report["manifests"][str(path)] = {"rows": len(rows), "datasets": dict(by_dataset)}
        for row in rows:
            if tuple(row["organ_texts"]) != ORGAN_NAMES or tuple(row["organ_labels"]) != ORGAN_NAMES:
                raise ValueError(f"Organ order mismatch: {row['study_id']}")
        for dataset in sorted(by_dataset):
            dataset_rows = [row for row in rows if row["dataset"] == dataset]
            if max_per_dataset is not None:
                dataset_rows = dataset_rows[:max_per_dataset]
            selected_items.extend((roots, row) for row in dataset_rows)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in executor.map(_check_volume, selected_items):
            report["volumes_checked"] += 1
            if report["volumes_checked"] % 500 == 0:
                print(json.dumps({
                    "volumes_checked": report["volumes_checked"],
                    "total": len(selected_items),
                }), flush=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report
