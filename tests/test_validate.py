from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fvlm_merlin.organs import ORGAN_NAMES
from fvlm_merlin.validate import run


def _manifest(tmp_path: Path, mask_affine: np.ndarray, count: int = 1) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    image = np.zeros((4, 4, 4), dtype=np.float32)
    mask = np.zeros((4, 4, 4), dtype=np.int16)
    mask[1:3, 1:3, 1:3] = 1
    rows = []
    for index in range(count):
        image_name, mask_name = f"image-{index}.nii.gz", f"mask-{index}.nii.gz"
        nib.save(nib.Nifti1Image(image, np.eye(4)), root / image_name)
        nib.save(nib.Nifti1Image(mask, mask_affine), root / mask_name)
        rows.append({
            "dataset": "test",
            "study_id": f"study-{index}",
            "image": image_name,
            "mask": mask_name,
            "organ_texts": {name: "caption" for name in ORGAN_NAMES},
            "organ_labels": {name: 0 for name in ORGAN_NAMES},
        })
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"roots": {"test": str(root)}, "records": rows}), encoding="utf-8")
    return path


def test_validate_reads_nifti_and_checks_supported_organs(tmp_path: Path) -> None:
    report = run([_manifest(tmp_path, np.eye(4))])
    assert report["volumes_checked"] == 1
    assert report["exhaustive"] is True


def test_validate_rejects_image_mask_geometry_mismatch(tmp_path: Path) -> None:
    affine = np.eye(4)
    affine[0, 3] = 5
    with pytest.raises(ValueError, match="geometry mismatch"):
        run([_manifest(tmp_path, affine)])


def test_validate_checks_every_volume_unless_limited(tmp_path: Path) -> None:
    path = _manifest(tmp_path, np.eye(4), count=3)

    complete = run([path], workers=2)
    limited = run([path], max_per_dataset=1)

    assert complete["volumes_checked"] == 3
    assert complete["exhaustive"] is True
    assert limited["volumes_checked"] == 1
    assert limited["exhaustive"] is False
