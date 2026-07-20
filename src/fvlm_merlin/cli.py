from __future__ import annotations

import argparse
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fvlm-merlin")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-manifest")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--limit-per-split", type=int)
    build.add_argument("--prefer-abnormal", action="store_true")
    validate = commands.add_parser("validate")
    validate.add_argument("manifests", type=Path, nargs="+")
    validate.add_argument("--samples-per-dataset", type=int, default=2)
    train = commands.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--options", nargs="*")
    assets = commands.add_parser("download-assets")
    export = commands.add_parser("export-features")
    export.add_argument("--manifest", type=Path)
    export.add_argument("--config", type=Path)
    export.add_argument("--checkpoint", type=Path)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--num-shards", type=int, default=1)
    export.add_argument("--shard-index", type=int, default=0)
    export.add_argument("--merge", type=Path, nargs="+")
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
