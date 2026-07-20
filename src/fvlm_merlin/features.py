from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import PROJECT_ROOT, fvlm_root
from .geometry import PATCH_SIZE, ROI_SIZE, build_volume_transform, crop_bounds, dense_label
from .manifest import checksum, load, resolve_paths
from .organs import ORGANS, ORGAN_NAMES

SCHEMA = "fvlm_merlin_features_v1"


def _load_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    from omegaconf import OmegaConf

    sys.path.insert(0, str(fvlm_root()))
    os.chdir(PROJECT_ROOT)
    from .lavis_adapter import AbdominalBlipPretrain

    cfg = OmegaConf.load(config_path).model
    model = AbdominalBlipPretrain.from_config(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    result = model.load_state_dict(state, strict=False)
    allowed_missing = {key for key in result.missing_keys if key.startswith("text_")}
    if set(result.missing_keys) - allowed_missing or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={result.missing_keys[:10]} unexpected={result.unexpected_keys[:10]}"
        )
    return model.eval().to(device)


def _organ_crops(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = image.as_tensor() if hasattr(image, "as_tensor") else image
    dense = mask[0].cpu().numpy()
    crops, masks, present = [], [], []
    for organ in ORGANS:
        coordinates = np.argwhere(dense == organ.dense_id)
        exists = coordinates.size > 0
        center = ((coordinates.min(0) + coordinates.max(0)) / 2 if exists
                  else (np.asarray(dense.shape) - 1) / 2)
        bounds = crop_bounds(center, tuple(int(value) for value in dense.shape))
        crops.append(image[(slice(None),) + bounds])
        masks.append(torch.as_tensor(dense[bounds], dtype=torch.long))
        present.append(exists)
    return torch.stack(crops), torch.stack(masks), torch.tensor(present, dtype=torch.bool)


def _embeddings(model, crops: torch.Tensor, masks: torch.Tensor, device: torch.device) -> torch.Tensor:
    crops = crops.to(device)
    masks = masks.to(device)
    dense_ids = torch.tensor([organ.dense_id for organ in ORGANS], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        _, levels = model.visual_encoder(crops)
    values = []
    for index, dense_id in enumerate(dense_ids):
        pooled = F.max_pool3d(
            (masks[index] == dense_id).float()[None, None], kernel_size=PATCH_SIZE, stride=PATCH_SIZE
        ).flatten().bool()
        if pooled.any():
            keys = torch.cat([level[index, pooled] for level in levels], dim=0)[None]
        else:
            keys = levels[-1][index].mean(0, keepdim=True)[None]
        query = model.query_tokens[index][None, None]
        attended, _ = model.attention(query, keys, keys)
        values.append(F.normalize(model.vision_projs[index](attended[0, 0]).float(), dim=-1))
    return torch.stack(values).cpu()


def export(manifest_path: Path, config_path: Path, checkpoint_path: Path, output_path: Path,
           num_shards: int = 1, shard_index: int = 0) -> None:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid shard selection")
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(config_path, checkpoint_path, device)
    roots, all_rows = load(manifest_path)
    rows = [row for index, row in enumerate(all_rows) if index % num_shards == shard_index]
    transform = build_volume_transform()
    records = {}
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            image_path, mask_path = resolve_paths(roots, row)
            data = transform({"image": str(image_path), "label": str(mask_path)})
            mask = dense_label(data["label"])
            crops, crop_masks, present = _organ_crops(data["image"], mask)
            organ_embeddings = _embeddings(model, crops, crop_masks, device)
            records[f"{row['dataset']}:{row['study_id']}"] = {
                "dataset": row["dataset"],
                "study_id": row["study_id"],
                "organ_embeddings": organ_embeddings,
                "organ_present": present,
                "study_embedding": F.normalize(organ_embeddings.mean(0), dim=-1),
            }
            print(json.dumps({"index": index, "total": len(rows), "study_id": row["study_id"]}), flush=True)
    payload = {
        "schema": SCHEMA,
        "organ_names": ORGAN_NAMES,
        "records": records,
        "metadata": {
            "checkpoint": str(checkpoint_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": checksum(manifest_path),
            "target_spacing": [1.5, 1.5, 1.5],
            "roi_size": list(ROI_SIZE),
            "num_shards": num_shards,
            "shard_index": shard_index,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)


def merge(output_path: Path, shard_paths: list[Path]) -> None:
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    records = {}
    metadata = []
    for path in shard_paths:
        payload = torch.load(path, map_location="cpu")
        if payload.get("schema") != SCHEMA or tuple(payload.get("organ_names", ())) != ORGAN_NAMES:
            raise ValueError(f"Incompatible feature shard: {path}")
        overlap = records.keys() & payload["records"].keys()
        if overlap:
            raise ValueError(f"Duplicate records in feature shards: {sorted(overlap)[:5]}")
        records.update(payload["records"])
        metadata.append(payload["metadata"])
    merged = {"schema": SCHEMA, "organ_names": ORGAN_NAMES, "records": records,
              "metadata": {"merged_shards": metadata}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(merged, temporary)
    temporary.replace(output_path)
