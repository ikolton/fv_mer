from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from .manifest import load, resolve_paths
from .organs import ORGANS, ORGAN_NAMES, remap_mask


def run(manifest_paths: list[Path], samples_per_dataset: int = 2) -> dict:
    report = {"manifests": {}, "sampled_volumes": 0}
    all_keys = set()
    for path in manifest_paths:
        roots, rows = load(path)
        keys = {(row["dataset"], row["study_id"]) for row in rows}
        overlap = all_keys & keys
        if overlap:
            raise ValueError(f"Studies occur in multiple manifests: {sorted(overlap)[:5]}")
        all_keys.update(keys)
        by_dataset = Counter(row["dataset"] for row in rows)
        report["manifests"][str(path)] = {"rows": len(rows), "datasets": dict(by_dataset)}
        selected = []
        for dataset in sorted(by_dataset):
            dataset_rows = [row for row in rows if row["dataset"] == dataset]
            selected.extend(dataset_rows[:samples_per_dataset])
        for row in selected:
            image_path, mask_path = resolve_paths(roots, row)
            image_obj, mask_obj = nib.load(str(image_path)), nib.load(str(mask_path))
            if image_obj.shape != mask_obj.shape or not np.allclose(image_obj.affine, mask_obj.affine, atol=1e-3):
                raise ValueError(f"Image-mask geometry mismatch: {row['dataset']}:{row['study_id']}")
            dense = remap_mask(np.asanyarray(mask_obj.dataobj))
            if not any(np.any(dense == organ.dense_id) for organ in ORGANS):
                raise ValueError(f"No supported organs in mask: {mask_path}")
            if tuple(row["organ_texts"]) != ORGAN_NAMES or tuple(row["organ_labels"]) != ORGAN_NAMES:
                raise ValueError(f"Organ order mismatch: {row['study_id']}")
            report["sampled_volumes"] += 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return report
