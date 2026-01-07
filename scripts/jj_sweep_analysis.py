"""Sweep-level analysis utilities (knee, BORGT-style scaling)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jjparse.config import setup_logging
from jjparse.metrics import borgt_scaling

LOGGER = logging.getLogger("jjparse")


def _pick_size_column(df: pd.DataFrame, size_cols: list[str] | None) -> str | None:
    if size_cols:
        for col in size_cols:
            if col in df.columns:
                return col
    for col in ["radius", "diameter", "size", "device_size", "turns", "n_turns", "lattice"]:
        if col in df.columns:
            return col
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep-level analysis across files.")
    parser.add_argument("--metrics", default="results/metrics/metrics_long.csv", help="Metrics CSV")
    parser.add_argument("--index", default=None, help="Optional parsed index CSV to merge")
    parser.add_argument("--out", default="results/reports", help="Output folder")
    parser.add_argument("--size-cols", default=None, help="Comma-separated size column candidates")
    parser.add_argument("--metric-col", default="I1", help="Metric column for scaling")
    parser.add_argument("--bootstrap", type=int, default=200, help="Bootstrap iterations")
    args = parser.parse_args()

    setup_logging(args.out)

    metrics_df = pd.read_csv(args.metrics)
    if args.index:
        index_df = pd.read_csv(args.index)
        metrics_df = metrics_df.merge(index_df, on="file_id", how="left", suffixes=("", "_idx"))

    size_cols = args.size_cols.split(",") if args.size_cols else None
    size_col = _pick_size_column(metrics_df, size_cols)
    metric_col = args.metric_col

    if not size_col or metric_col not in metrics_df.columns:
        LOGGER.info("No suitable size/metric columns found for scaling.")
        return

    size_vals = pd.to_numeric(metrics_df[size_col], errors="coerce").to_numpy()
    metric_vals = pd.to_numeric(metrics_df[metric_col], errors="coerce").to_numpy()

    result = borgt_scaling(size_vals, metric_vals, n_boot=args.bootstrap)
    if not result:
        LOGGER.info("Insufficient data for scaling.")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "borgt_scaling.csv"

    row = {"size_col": size_col, "metric_col": metric_col}
    row.update(result)
    pd.DataFrame([row]).to_csv(out_path, index=False)
    LOGGER.info("Wrote BORGT scaling to %s", out_path)


if __name__ == "__main__":
    main()
