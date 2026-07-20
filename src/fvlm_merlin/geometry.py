from __future__ import annotations

import numpy as np
import torch

from .organs import remap_mask

TARGET_SPACING = (1.5, 1.5, 1.5)
ROI_SIZE = (112, 256, 352)
PATCH_SIZE = (16, 16, 32)


def build_volume_transform():
    from monai.transforms import Compose, EnsureTyped, LoadImaged, SpatialPadd, Spacingd

    return Compose([
        LoadImaged(keys=("image", "label"), image_only=False, ensure_channel_first=True),
        Spacingd(
            keys=("image", "label"),
            pixdim=TARGET_SPACING,
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),
        SpatialPadd(keys=("image", "label"), spatial_size=ROI_SIZE),
        EnsureTyped(keys=("image", "label")),
    ])


def dense_label(value) -> torch.Tensor:
    tensor = value.as_tensor() if hasattr(value, "as_tensor") else value
    dense = remap_mask(tensor[0].detach().cpu().numpy())
    return torch.as_tensor(dense[None], dtype=torch.int16)


def crop_bounds(center: np.ndarray, shape: tuple[int, int, int]) -> tuple[slice, slice, slice]:
    starts = []
    for axis, size in enumerate(ROI_SIZE):
        high = max(shape[axis] - size, 0)
        starts.append(max(0, min(int(round(center[axis] - size / 2)), high)))
    return tuple(slice(start, start + ROI_SIZE[axis]) for axis, start in enumerate(starts))

