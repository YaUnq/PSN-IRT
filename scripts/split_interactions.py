#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create paper-style model-item interaction splits.

The paper describes a 60/20/20 split over model-item interactions. This script
keeps the original response-matrix shape and masks held-out interactions with
NaN, so the existing Dataset classes can consume the generated CSV files.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="data/combine.csv")
    parser.add_argument("--output_dir", default="data/splits/paper_seed3407")
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main():
    args = parse_args()
    ratios = np.array([args.train_ratio, args.val_ratio, args.test_ratio])
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError("train/val/test ratios must sum to 1.0")

    response_df = pd.read_csv(args.input_csv, header=None)
    values = response_df.to_numpy(dtype=float)
    observed = np.argwhere(~np.isnan(values))

    rng = np.random.default_rng(args.seed)
    rng.shuffle(observed)

    n_total = len(observed)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    splits = {
        "train": observed[:n_train],
        "val": observed[n_train:n_train + n_val],
        "test": observed[n_train + n_val:],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for split_name, indices in splits.items():
        split_values = np.full(values.shape, np.nan)
        split_values[indices[:, 0], indices[:, 1]] = values[indices[:, 0], indices[:, 1]]
        out_path = output_dir / f"{split_name}.csv"
        pd.DataFrame(split_values).to_csv(out_path, header=False, index=False)
        manifest_rows.append({
            "split": split_name,
            "interactions": len(indices),
            "positive_rate": float(values[indices[:, 0], indices[:, 1]].mean()),
            "path": str(out_path),
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
