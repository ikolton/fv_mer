from pathlib import Path

import pytest
import torch

from fvlm_merlin.features import SCHEMA, merge
from fvlm_merlin.organs import ORGAN_NAMES


def test_feature_merge_rejects_duplicate_records(tmp_path: Path) -> None:
    payload = {
        "schema": SCHEMA,
        "organ_names": ORGAN_NAMES,
        "records": {"merlin:a": {"study_id": "a"}},
        "metadata": {},
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
            "metadata": {"shard_index": index},
        }, path)
        paths.append(path)
    output = tmp_path / "merged.pt"
    merge(output, paths)
    result = torch.load(output, map_location="cpu")
    assert result["schema"] == SCHEMA
    assert len(result["records"]) == 2
    assert not (tmp_path / "merged.pt.tmp").exists()

