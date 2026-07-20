from __future__ import annotations

import numpy as np
import torch

import fvlm_merlin.geometry as geometry
from fvlm_merlin.features import _organ_crops
from fvlm_merlin.geometry import crop_bounds, dense_label
from fvlm_merlin.organs import ORGAN_BY_NAME


def test_crop_bounds_keep_fixed_size_at_volume_edges(monkeypatch) -> None:
    roi_size = (4, 6, 8)
    monkeypatch.setattr(geometry, "ROI_SIZE", roi_size)
    shape = tuple(size + 4 for size in roi_size)
    low = crop_bounds(np.zeros(3), shape)
    high = crop_bounds(np.asarray(shape) - 1, shape)

    assert tuple(axis.start for axis in low) == (0, 0, 0)
    assert tuple(axis.stop - axis.start for axis in low) == roi_size
    assert tuple(axis.stop for axis in high) == shape


def test_dense_label_remaps_source_ids_and_keeps_channel() -> None:
    source = torch.tensor([[[1, 5], [21, 99]]], dtype=torch.int16)
    result = dense_label(source)

    assert result.shape == source.shape
    assert result[0, 0, 0] == ORGAN_BY_NAME["spleen"].dense_id
    assert result[0, 0, 1] == ORGAN_BY_NAME["liver"].dense_id
    assert result[0, 1, 0] == ORGAN_BY_NAME["urinary bladder"].dense_id
    assert result[0, 1, 1] == 0


def test_organ_crops_center_present_organs_and_mark_absent_ones(monkeypatch) -> None:
    roi_size = (4, 6, 8)
    monkeypatch.setattr(geometry, "ROI_SIZE", roi_size)
    shape = tuple(size + 2 for size in roi_size)
    image = torch.arange(np.prod(shape), dtype=torch.float32).reshape((1,) + shape)
    mask = torch.zeros((1,) + shape, dtype=torch.int16)
    liver = ORGAN_BY_NAME["liver"]
    center = tuple(size // 2 for size in shape)
    mask[(0,) + center] = liver.dense_id

    crops, masks, present = _organ_crops(image, mask)

    assert crops.shape == (len(present), 1) + roi_size
    assert masks.shape == (len(present),) + roi_size
    assert present.sum().item() == 1
    assert present[liver.dense_id - 1]
    assert (masks[liver.dense_id - 1] == liver.dense_id).sum().item() == 1
