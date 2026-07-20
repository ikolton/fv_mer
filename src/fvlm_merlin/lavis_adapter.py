from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import PROJECT_ROOT
from .geometry import ROI_SIZE, build_volume_transform, dense_label
from .manifest import load, resolve_paths
from .organs import ORGAN_NAMES


def register() -> None:
    """Importing this module registers the dataset and model with LAVIS."""


try:
    from lavis.common.registry import registry
    from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
    from lavis.datasets.datasets.base_dataset import BaseDataset
    from lavis.models.blip_models.blip_pretrain import BlipPretrain
    from lavis.models.blip_models.vit import ViT
    from lavis.models.med import XBertEncoder
    import torch.nn.functional as F
except ImportError as exc:
    raise RuntimeError("fVLM is not importable; run through the project environment") from exc


class AbdominalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, [])
        if len(ann_paths) != 1:
            raise ValueError("Exactly one manifest is required per split")
        self.roots, self.rows = load(Path(ann_paths[0]))
        self.organs = ORGAN_NAMES
        self.volume_transform = build_volume_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path, mask_path = resolve_paths(self.roots, row)
        data = self.volume_transform({"image": str(image_path), "label": str(mask_path)})
        data["label"] = dense_label(data["label"])
        data = self.vis_processor(data)
        image = data["image"].as_tensor() if hasattr(data["image"], "as_tensor") else data["image"]
        label = data["label"].as_tensor() if hasattr(data["label"], "as_tensor") else data["label"]
        texts = dict(row["organ_texts"])
        if self.text_processor is not None:
            texts = self.text_processor(texts)
        return {
            "image": image.float(),
            "seg": label[0].long(),
            "text_input": texts,
            "organ_abnormal_flags": torch.tensor(
                [bool(row["organ_labels"].get(name, 0)) for name in self.organs], dtype=torch.bool
            ),
            "study_id": row["study_id"],
            "dataset": row["dataset"],
        }


@registry.register_builder("abdominal_fvlm_caption")
class AbdominalBuilder(BaseDatasetBuilder):
    train_dataset_cls = AbdominalDataset
    eval_dataset_cls = AbdominalDataset
    DATASET_CONFIG_DICT = {"default": str(PROJECT_ROOT / "configs/model/abdominal_fvlm.yaml")}


@registry.register_model("abdominal_blip_pretrain")
class AbdominalBlipPretrain(BlipPretrain):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "abdominal": str(PROJECT_ROOT / "configs/model/abdominal_fvlm.yaml")
    }

    @classmethod
    def init_tokenizer(cls):
        from transformers import BertTokenizer

        root = Path(os.environ.get("FVLM_BIOMED_ROOT", PROJECT_ROOT / "assets/biomed_cxrbert"))
        if not root.is_dir():
            raise FileNotFoundError(f"CXR-BERT tokenizer is missing: {root}")
        return BertTokenizer.from_pretrained(str(root), local_files_only=True)

    def __init__(self, image_encoder, text_encoder, text_decoder, alpha=0.4, embed_dim=256,
                 tie_enc_dec_weights=False, max_txt_len=175, organ_names=None):
        super().__init__(image_encoder, text_encoder, text_decoder, alpha, embed_dim,
                         tie_enc_dec_weights, max_txt_len)
        self.organs = list(organ_names or ORGAN_NAMES)
        self.vision_projs = nn.ModuleList([nn.Linear(768, embed_dim) for _ in self.organs])
        self.query_tokens = nn.Parameter(torch.zeros(len(self.organs), 768))

    @classmethod
    def from_config(cls, cfg=None):
        mae_path = Path(cfg.get("mae_ckpt_path", PROJECT_ROOT / "assets/mae_pretrain_vit_base.pth"))
        if not mae_path.is_file():
            raise FileNotFoundError(f"MAE checkpoint is missing: {mae_path}")
        encoder = ViT(in_channels=1, img_size=ROI_SIZE, patch_size=(16, 16, 32),
                      num_classes=0, dropout_rate=0.1, qkv_bias=True)
        checkpoint = torch.load(mae_path, map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        encoder.load_state_dict(convert_mae_state_dict(state), strict=False)
        text_encoder = XBertEncoder.from_config(cfg, from_pretrained=True)
        model = cls(
            encoder, text_encoder, None,
            embed_dim=cfg.get("embed_dim", 256), alpha=cfg.get("alpha", 0.5),
            max_txt_len=cfg.get("max_txt_len", 384), organ_names=ORGAN_NAMES,
        )
        model.load_checkpoint_from_config(cfg)
        return model


def convert_mae_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    from collections import OrderedDict

    converted = OrderedDict()
    for key, value in state.items():
        if key.startswith("decoder") or key in {"mask_token", "cls_token"} or key.startswith("patch_embed"):
            continue
        if key.startswith("pos_embed"):
            value = value[0, 1:].reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
            value = F.interpolate(value, size=(16, 11), mode="bilinear", align_corners=False)
            converted["patch_embedding.position_embeddings"] = (
                value.unsqueeze(2).repeat(1, 1, 7, 1, 1).flatten(2).permute(0, 2, 1)
            )
        else:
            converted[key.replace("fc", "linear").replace("proj", "out_proj")] = value
    return converted

