#!/usr/bin/env python3
import argparse
import os
import re

import numpy as np


def parse_metrics(path):
    text = open(path, encoding="utf-8", errors="ignore").read()
    metrics = {}
    for k in (1, 5, 10):
        matches = re.findall(
            rf"HR@{k}: ([0-9.]+), NDCG@{k}: ([0-9.]+)", text
        )
        if not matches:
            raise ValueError(f"final HR/NDCG metrics not found in {path}")
        hr, ndcg = matches[-1]
        metrics[f"HR@{k}"] = float(hr)
        metrics[f"NDCG@{k}"] = float(ndcg)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--tag", default="dynamic")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025]
    )
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        path = os.path.join(
            "log", "DisCo", f"{args.task}_{args.tag}_seed{seed}.log"
        )
        metrics = parse_metrics(path)
        rows.append(metrics)
        print(
            f"seed={seed}: HR@10={metrics['HR@10']:.4f}, "
            f"NDCG@10={metrics['NDCG@10']:.4f}"
        )

    print(f"\n{args.task} ({len(rows)} seeds)")
    for name in ("HR@1", "NDCG@1", "HR@5", "NDCG@5", "HR@10", "NDCG@10"):
        values = np.array([row[name] for row in rows])
        print(f"{name}: {values.mean():.4f} +/- {values.std(ddof=1):.4f}")


if __name__ == "__main__":
    main()
