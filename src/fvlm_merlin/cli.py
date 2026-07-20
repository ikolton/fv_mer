from __future__ import annotations

import argparse
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fvlm-merlin",
        description="Train abdominal fVLM encoders and export per-organ features.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-manifest", help="Build train and validation manifests")
    build.add_argument("--config", type=Path, required=True, help="Data preset YAML")
    build.add_argument("--output-dir", type=Path, required=True, help="Manifest output directory")
    build.add_argument("--limit-per-split", type=int, help="Maximum records in each split")
    build.add_argument(
        "--prefer-abnormal", action="store_true", help="Select abnormal studies first when limiting"
    )
    validate = commands.add_parser("validate", help="Check manifests and sampled image-mask pairs")
    validate.add_argument("manifests", type=Path, nargs="+", help="Manifest JSON files")
    validate.add_argument(
        "--samples-per-dataset", type=int, default=2, help="Volumes opened per dataset (default: 2)"
    )
    train = commands.add_parser("train", help="Run fVLM training from a configuration")
    train.add_argument("--config", type=Path, required=True, help="Training YAML")
    train.add_argument("--options", nargs="*", help="LAVIS configuration overrides")
    commands.add_parser("download-assets", help="Download MAE and CXR-BERT initial weights")
    export = commands.add_parser("export-features", help="Export or merge per-organ feature caches")
    export.add_argument("--manifest", type=Path, help="Input manifest JSON")
    export.add_argument("--config", type=Path, help="Model configuration YAML")
    export.add_argument("--checkpoint", type=Path, help="Trained checkpoint")
    export.add_argument("--output", type=Path, required=True, help="Output feature file")
    export.add_argument("--num-shards", type=int, default=1, help="Total extraction shards")
    export.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index")
    export.add_argument("--merge", type=Path, nargs="+", help="Completed feature shards to merge")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build-manifest":
        from .manifest import build
        for name, path in build(
            args.config, args.output_dir, args.limit_per_split, args.prefer_abnormal
        ).items():
            print(f"{name}={path}")
    elif args.command == "validate":
        from .validate import run
        run(args.manifests, args.samples_per_dataset)
    elif args.command == "train":
        from .training import run
        run(args.config, args.options)
    elif args.command == "download-assets":
        from .assets import download
        download()
    elif args.command == "export-features":
        from .features import export, merge
        if args.merge:
            merge(args.output, args.merge)
        elif not all((args.manifest, args.config, args.checkpoint)):
            raise SystemExit("--manifest, --config, and --checkpoint are required unless --merge is used")
        else:
            export(args.manifest, args.config, args.checkpoint, args.output,
                   args.num_shards, args.shard_index)


if __name__ == "__main__":
    main()
