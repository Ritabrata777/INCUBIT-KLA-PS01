"""
prepare_carinthia.py
One-off data preparation script for the Carinthia SEM defect dataset
(Zenodo DOI: 10.5281/zenodo.10715190, CC BY 4.0, KAI / Infineon).

Splits the raw extracted `carinthia_raw/data/images/` folder into:
  - data/clean            (training set of clean SEM images)
  - data/val_clean         (validation set of clean SEM images)
  - data/holdout_clean     (held-out clean images, reserved for building a
                             demo degraded test set / final qualitative check)

Usage:
    python prepare_carinthia.py --src carinthia_raw/data/images \
        --dst data --val_count 200 --holdout_count 150
"""

import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Split Carinthia SEM images into train/val/holdout")
    parser.add_argument("--src", type=str, default="carinthia_raw/data/images")
    parser.add_argument("--dst", type=str, default="data")
    parser.add_argument("--val_count", type=int, default=200)
    parser.add_argument("--holdout_count", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    src_dir = Path(args.src)
    paths = sorted(src_dir.glob("*.jpg"))
    if len(paths) == 0:
        raise RuntimeError(f"No .jpg images found in {src_dir}")

    random.shuffle(paths)

    holdout_paths = paths[: args.holdout_count]
    val_paths = paths[args.holdout_count: args.holdout_count + args.val_count]
    train_paths = paths[args.holdout_count + args.val_count:]

    dst_dir = Path(args.dst)
    splits = {
        "clean": train_paths,
        "val_clean": val_paths,
        "holdout_clean": holdout_paths,
    }

    for split_name, split_paths in splits.items():
        split_dir = dst_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for p in split_paths:
            shutil.copy2(p, split_dir / p.name)
        print(f"{split_name}: {len(split_paths)} images -> {split_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
