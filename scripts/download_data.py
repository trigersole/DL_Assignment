from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import gdown

FILE_ID = "1PmfiGy1D1ssa8NFnwH8UDJJYCCj8NkHt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract the assignment dataset")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "dog-breed-identification.zip"
    gdown.download(id=FILE_ID, output=str(archive), quiet=False)
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(f"Downloaded file is not a ZIP archive: {archive}")
    with zipfile.ZipFile(archive) as outer:
        outer.extractall(args.output_dir)
    # Some distributions contain train.zip inside the outer archive.
    for nested in args.output_dir.rglob("*.zip"):
        if nested == archive or not zipfile.is_zipfile(nested):
            continue
        target = nested.parent / nested.stem
        target.mkdir(exist_ok=True)
        with zipfile.ZipFile(nested) as inner:
            inner.extractall(target)
    print(f"Dataset extracted under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
