#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute item-quality diagnostics from fitted PSN-IRT parameters."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def four_pl_probability(theta, a, b, c, d):
    sigmoid = 1.0 / (1.0 + np.exp(-a * (theta - b)))
    return c + (d - c) * sigmoid


def fisher_information(theta, a, b, c, d):
    p = four_pl_probability(theta, a, b, c, d)
    q = 1.0 - p
    core = 1.0 / (1.0 + np.exp(-a * (theta - b)))
    dp = (d - c) * a * core * (1.0 - core)
    return (dp ** 2) / np.clip(p * q, 1e-9, None)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item_parameters", default="results/paper_reproduction/psn_irt_4pl_seed3407/item_parameters.csv")
    parser.add_argument("--student_abilities", default="results/paper_reproduction/psn_irt_4pl_seed3407/student_abilities.csv")
    parser.add_argument("--output_csv", default="results/paper_reproduction/psn_irt_4pl_seed3407/item_quality.csv")
    parser.add_argument("--selection_ratio", type=float, default=0.7)
    return parser.parse_args()


def main():
    args = parse_args()
    item_df = pd.read_csv(args.item_parameters)
    ability_df = pd.read_csv(args.student_abilities)
    if "discriminability" not in item_df.columns and "discrimination" in item_df.columns:
        item_df = item_df.rename(columns={"discrimination": "discriminability"})
    if "item_id" not in item_df.columns:
        item_df.insert(0, "item_id", range(len(item_df)))

    required = {"difficulty", "discriminability", "guessing", "feasibility"}
    missing = required - set(item_df.columns)
    if missing:
        raise ValueError(f"Missing item parameter columns: {sorted(missing)}")

    theta = ability_df["ability"].to_numpy()
    info_rows = []
    for _, row in item_df.iterrows():
        a = row["discriminability"]
        b = row["difficulty"]
        c = row["guessing"]
        d = row["feasibility"]
        infos = fisher_information(theta, a, b, c, d)
        probs = four_pl_probability(theta, a, b, c, d)
        info_rows.append({
            "item_id": int(row["item_id"]),
            "difficulty": b,
            "discriminability": a,
            "guessing": c,
            "feasibility": d,
            "mean_probability": float(np.mean(probs)),
            "mean_fisher_information": float(np.mean(infos)),
            "max_fisher_information": float(np.max(infos)),
        })

    quality_df = pd.DataFrame(info_rows).sort_values(
        "mean_fisher_information", ascending=False
    )
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quality_df.to_csv(output_path, index=False)

    n_select = int(len(quality_df) * args.selection_ratio)
    selected_path = output_path.with_name(f"selected_item_indices_{args.selection_ratio:g}.csv")
    quality_df[["item_id"]].head(n_select).to_csv(selected_path, index=False)
    print(f"Saved {output_path}")
    print(f"Saved {selected_path} ({n_select} items)")


if __name__ == "__main__":
    main()
