from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import load_yaml, project_path
from .organs import ORGAN_NAMES, normal_caption


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _label(value: Any) -> int:
    if isinstance(value, (bool, int, float)):
        return int(bool(value))
    return int(str(value).strip().casefold() in {"1", "true", "yes", "abnormal", "positive"})


def _fold(mapping: Any) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        return {}
    return {str(key).strip().casefold(): value for key, value in mapping.items()}


def _rows(dataset: str, split: str, root: Path, annotation: Path, image_split: str | None,
          on_missing: str = "error") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = _read_json(annotation)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list: {annotation}")
    rows = []
    excluded = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Invalid record {index} in {annotation}")
        study_id = _clean(record.get("study_id"))
        if not study_id:
            raise ValueError(f"Missing study_id at record {index} in {annotation}")
        base = Path(image_split) / study_id if image_split else Path(study_id)
        image = base / f"{study_id}_resampled.nii.gz"
        mask = base / f"{study_id}_seg_resampled.nii.gz"
        missing = [str(root / path) for path in (image, mask) if not (root / path).is_file()]
        if missing:
            if on_missing != "exclude":
                raise FileNotFoundError(f"Missing files for {dataset}/{study_id}: {missing}")
            excluded.append({"dataset": dataset, "split": split, "study_id": study_id, "missing": missing})
            continue
        findings = _fold(record.get("findings"))
        labels = _fold(record.get("labels"))
        rows.append({
            "dataset": dataset,
            "split": split,
            "study_id": study_id,
            "image": image.as_posix(),
            "mask": mask.as_posix(),
            "report": _clean(record.get("cleaned_report")),
            "organ_texts": {name: _clean(findings.get(name)) or normal_caption(name) for name in ORGAN_NAMES},
            "organ_labels": {name: _label(labels.get(name)) for name in ORGAN_NAMES},
        })
    return rows, excluded


def build(config_path: Path, output_dir: Path, limit_per_split: int | None = None,
          prefer_abnormal: bool = False) -> dict[str, Path]:
    cfg = load_yaml(config_path)
    roots = {name: project_path(value) for name, value in cfg["roots"].items()}
    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusions: list[dict[str, Any]] = []
    for source in cfg["sources"]:
        dataset = source["dataset"]
        root = roots[dataset]
        annotation = root / source["annotation"]
        rows, missing = _rows(
            dataset, source["split"], root, annotation, source.get("image_split"),
            source.get("on_missing", "error"),
        )
        splits[source["split"]].extend(rows)
        exclusions.extend(missing)
    seen: set[tuple[str, str]] = set()
    for split, records in splits.items():
        for row in records:
            key = (row["dataset"], row["study_id"])
            if key in seen:
                raise ValueError(f"Duplicate or cross-split study: {key}")
            seen.add(key)
    output_dir.mkdir(parents=True, exist_ok=True)
    root_payload = {name: str(path) for name, path in roots.items()}
    paths = {}
    for split, records in splits.items():
        if limit_per_split is not None:
            records = _limited(records, limit_per_split, prefer_abnormal)
            splits[split] = records
        path = output_dir / f"{split}.json"
        path.write_text(json.dumps({"roots": root_payload, "records": records}, indent=2), encoding="utf-8")
        paths[split] = path
    summary = summarize(splits)
    summary["config"] = str(Path(config_path).resolve())
    summary["excluded_missing_files"] = {
        "count": len(exclusions),
        "examples": exclusions[:20],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    paths["summary"] = summary_path
    return paths


def _limited(records: list[dict[str, Any]], limit: int, prefer_abnormal: bool) -> list[dict[str, Any]]:
    if limit < 0:
        raise ValueError("Split limit must be non-negative")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[row["dataset"]].append(row)
    if prefer_abnormal:
        for rows in groups.values():
            rows.sort(
                key=lambda row: sum(int(value) for value in row["organ_labels"].values()),
                reverse=True,
            )
    selected = []
    offsets = {dataset: 0 for dataset in groups}
    while len(selected) < limit:
        added = False
        for dataset, rows in groups.items():
            offset = offsets[dataset]
            if offset < len(rows) and len(selected) < limit:
                selected.append(rows[offset])
                offsets[dataset] += 1
                added = True
        if not added:
            break
    return selected


def load(path: Path) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    payload = _read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"Invalid manifest: {path}")
    roots = {name: Path(value) for name, value in payload.get("roots", {}).items()}
    return roots, payload["records"]


def resolve_paths(roots: dict[str, Path], row: dict[str, Any]) -> tuple[Path, Path]:
    root = roots[row["dataset"]]
    return root / row["image"], root / row["mask"]


def summarize(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    abnormal: dict[str, dict[str, int]] = {}
    for split, rows in splits.items():
        counts = Counter()
        for row in rows:
            counts.update(name for name, value in row["organ_labels"].items() if value)
        abnormal[split] = {name: counts[name] for name in ORGAN_NAMES}
    return {
        "rows": {split: len(rows) for split, rows in splits.items()},
        "datasets": {split: dict(Counter(row["dataset"] for row in rows)) for split, rows in splits.items()},
        "abnormal": abnormal,
    }


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
