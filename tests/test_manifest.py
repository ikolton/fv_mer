from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fvlm_merlin.manifest import _limited, build, load, resolve_paths
from fvlm_merlin.organs import ORGAN_BY_NAME, ORGAN_NAMES, remap_mask


def _write_dataset(root: Path, split: str, study_id: str) -> None:
    directory = root / split / study_id
    directory.mkdir(parents=True)
    (directory / f"{study_id}_resampled.nii.gz").write_bytes(b"image")
    (directory / f"{study_id}_seg_resampled.nii.gz").write_bytes(b"mask")
    record = {
        "study_id": study_id,
        "cleaned_report": "A report.",
        "findings": {"Liver": "Small liver lesion."},
        "labels": {"Liver": 1},
    }
    (root / split / "combined.json").write_text(json.dumps([record]), encoding="utf-8")


def _config(tmp_path: Path, root: Path) -> Path:
    path = tmp_path / "data.yaml"
    path.write_text(
        f"roots:\n  merlin: {root}\nsources:\n"
        "  - dataset: merlin\n    split: train\n    annotation: train/combined.json\n    image_split: train\n"
        "  - dataset: merlin\n    split: val\n    annotation: val/combined.json\n    image_split: val\n",
        encoding="utf-8",
    )
    return path


def test_manifest_uses_relative_paths_and_canonical_organs(tmp_path: Path) -> None:
    root = tmp_path / "merlin"
    _write_dataset(root, "train", "train-study")
    _write_dataset(root, "val", "val-study")
    paths = build(_config(tmp_path, root), tmp_path / "manifest")
    roots, rows = load(paths["train"])
    row = rows[0]
    assert not Path(row["image"]).is_absolute()
    assert tuple(row["organ_texts"]) == ORGAN_NAMES
    assert row["organ_labels"]["liver"] == 1
    assert resolve_paths(roots, row)[0].is_file()


def test_manifest_fails_on_missing_volume(tmp_path: Path) -> None:
    root = tmp_path / "merlin"
    _write_dataset(root, "train", "train-study")
    _write_dataset(root, "val", "val-study")
    (root / "train/train-study/train-study_resampled.nii.gz").unlink()
    with pytest.raises(FileNotFoundError):
        build(_config(tmp_path, root), tmp_path / "manifest")


def test_mask_remap_preserves_organ_identity() -> None:
    source = np.array([[0, 5, 20], [21, 99, 1]], dtype=np.int16)
    dense = remap_mask(source)
    assert dense[0, 1] == ORGAN_BY_NAME["liver"].dense_id
    assert dense[0, 2] == ORGAN_BY_NAME["colon"].dense_id
    assert dense[1, 0] == ORGAN_BY_NAME["urinary bladder"].dense_id
    assert dense[1, 1] == 0


def test_limited_pooled_manifest_represents_each_dataset() -> None:
    rows = []
    for dataset in ("merlin", "swiss", "turkish"):
        for index in range(4):
            rows.append({
                "dataset": dataset,
                "study_id": f"{dataset}-{index}",
                "organ_labels": {name: int(index == 3) for name in ORGAN_NAMES},
            })

    selected = _limited(rows, limit=6, prefer_abnormal=True)

    assert [row["dataset"] for row in selected] == [
        "merlin", "swiss", "turkish", "merlin", "swiss", "turkish"
    ]
    assert all(sum(row["organ_labels"].values()) for row in selected[:3])
