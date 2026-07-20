from pathlib import Path

import pytest
import torch

from fvlm_merlin.features import SCHEMA, merge
from fvlm_merlin.organs import ORGAN_NAMES


def _metadata(shard_index: int, num_shards: int = 2, checkpoint: str = "checkpoint.pth") -> dict:
    return {
        "checkpoint": checkpoint,
        "manifest_sha256": "manifest-hash",
        "num_shards": num_shards,
        "shard_index": shard_index,
    }


def test_feature_merge_rejects_duplicate_records(tmp_path: Path) -> None:
    payload = {
        "schema": SCHEMA,
        "organ_names": ORGAN_NAMES,
        "records": {"merlin:a": {"study_id": "a"}},
        "metadata": _metadata(0, num_shards=1),
    }
    first, second = tmp_path / "first.pt", tmp_path / "second.pt"
    torch.save(payload, first)
    torch.save(payload, second)
    with pytest.raises(ValueError, match="Duplicate"):
        merge(tmp_path / "merged.pt", [first, second])


def test_feature_merge_is_atomic_and_keeps_schema(tmp_path: Path) -> None:
    paths = []
    for index in range(2):
        path = tmp_path / f"{index}.pt"
        torch.save({
            "schema": SCHEMA,
            "organ_names": ORGAN_NAMES,
            "records": {f"merlin:{index}": {"study_id": str(index)}},
            "metadata": _metadata(index),
        }, path)
        paths.append(path)
    output = tmp_path / "merged.pt"
    merge(output, paths)
    result = torch.load(output, map_location="cpu")
    assert result["schema"] == SCHEMA
    assert len(result["records"]) == 2
    assert not (tmp_path / "merged.pt.tmp").exists()


def test_feature_merge_rejects_mixed_checkpoints(tmp_path: Path) -> None:
    paths = []
    for index, checkpoint in enumerate(("first.pth", "second.pth")):
        path = tmp_path / f"{index}.pt"
        torch.save({
            "schema": SCHEMA,
            "organ_names": ORGAN_NAMES,
            "records": {f"merlin:{index}": {}},
            "metadata": _metadata(index, checkpoint=checkpoint),
        }, path)
        paths.append(path)

    with pytest.raises(ValueError, match="different provenance"):
        merge(tmp_path / "merged.pt", paths)


def test_feature_merge_rejects_missing_shards(tmp_path: Path) -> None:
    path = tmp_path / "0.pt"
    torch.save({
        "schema": SCHEMA,
        "organ_names": ORGAN_NAMES,
        "records": {"merlin:0": {}},
        "metadata": _metadata(0),
    }, path)

    with pytest.raises(ValueError, match="Incomplete"):
        merge(tmp_path / "merged.pt", [path])
