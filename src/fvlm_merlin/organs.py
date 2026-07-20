from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Organ:
    name: str
    source_id: int
    dense_id: int


ORGANS = (
    Organ("spleen", 1, 1),
    Organ("kidneys", 2, 2),
    Organ("gallbladder", 4, 3),
    Organ("liver", 5, 4),
    Organ("stomach", 6, 5),
    Organ("pancreas", 7, 6),
    Organ("adrenal glands", 8, 7),
    Organ("small bowel", 18, 8),
    Organ("colon", 20, 9),
    Organ("urinary bladder", 21, 10),
    Organ("prostate", 22, 11),
)
ORGAN_NAMES = tuple(organ.name for organ in ORGANS)
ORGAN_BY_NAME = {organ.name: organ for organ in ORGANS}
NORMAL_TEMPLATE = "{organ} shows no significant abnormalities."


def remap_mask(mask: np.ndarray) -> np.ndarray:
    dense = np.zeros(mask.shape, dtype=np.int16)
    for organ in ORGANS:
        dense[mask == organ.source_id] = organ.dense_id
    return dense


def normal_caption(organ: str) -> str:
    return NORMAL_TEMPLATE.format(organ=organ)

