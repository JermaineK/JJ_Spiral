"""Phase 4.5 salvage: knee-window vs random-window eta ratio."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jjparse.config import load_config, seed_everything, setup_logging
from jjparse.metrics import compute_voltage_nonreciprocity
from jjparse.version import PIPELINE_VERSION

LOGGER = logging.getLogger("jjparse.phase45_salvage")


def _load_segment(segments_dir: Path, file_id: str, segment_id: int, x_name: str | None) -> pd.DataFrame:
    seg_path = segments_dir / f"{file_id}.parquet"
    if not seg_path.exists():
        return pd.DataFrame()
    seg_all = pd.read_parquet(seg_path)
    seg_df = seg_all[seg_all["segment_id"] == segment_id]
    if x_name and "x_name" in seg_df.columns:
        seg_df = seg_df[seg_df["x_name"] == x_name]
    return seg_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4.5 salvage knee-window eta checks.")
    parser.add_argument("--metrics", default="results/metrics/metrics_long.csv", help="Metrics CSV")
    parser.add_argument("--segments-dir", default="results/parsed/segments", help="Segments parquet folder")
    parser.add_argument("--out", default="results/reports", help="Output folder")
    parser.add_argument("--config", default="config.yaml", help="Config path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed
    if seed is None:
        seed = int(config.get("phase45_salvage", {}).get("random_seed", 123))
    seed_everything(seed)
    setup_logging("results/reports", name="jjparse.phase45_salvage")

    metrics = pd.read_csv(args.metrics, low_memory=False)
    if metrics.empty:
        LOGGER.warning("No metrics to salvage.")
        return

    segments_dir = Path(args.segments_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    window_frac = float(config.get("phase45_salvage", {}).get("knee_window_frac", 0.2))
    grid_n = int(config.get("parity", {}).get("grid_n", 200))
    eps = float(config.get("parity", {}).get("eps", 1e-12))
    min_overlap = float(config.get("parity", {}).get("min_overlap", 0.8))
    den_min = float(config.get("parity", {}).get("den_min", 0.0))

    rng = np.random.default_rng(seed)
    rows = []

    for _, row in metrics.iterrows():
        if not bool(row.get("eta_valid", False)):
            continue
        file_id = str(row.get("file_id"))
        segment_id = int(row.get("segment_id", 0))
        x_name = row.get("x_name") or row.get("axis_primary")
        seg_df = _load_segment(segments_dir, file_id, segment_id, x_name)
        if seg_df.empty:
            continue

        x = pd.to_numeric(seg_df["x"], errors="coerce").to_numpy()
        v = pd.to_numeric(seg_df["V_V"], errors="coerce").to_numpy()
        mask = np.isfinite(x) & np.isfinite(v)
        x = x[mask]
        v = v[mask]
        if x.size < 5:
            continue

        x_min = float(np.nanmin(x))
        x_max = float(np.nanmax(x))
        span = x_max - x_min
        if not np.isfinite(span) or span <= 0:
            continue

        knee_val = row.get("knee_x_value")
        if knee_val is None or not np.isfinite(knee_val):
            knee_norm = row.get("knee_x_norm")
            if knee_norm is not None and np.isfinite(knee_norm):
                knee_val = x_min + float(knee_norm) * span

        if knee_val is None or not np.isfinite(knee_val):
            continue

        half_width = window_frac * span
        if half_width <= 0:
            continue

        knee_low = knee_val - half_width
        knee_high = knee_val + half_width
        knee_mask = (x >= knee_low) & (x <= knee_high)
        if knee_mask.sum() < 5:
            continue

        eta_knee = compute_voltage_nonreciprocity(
            x[knee_mask],
            v[knee_mask],
            grid_n=grid_n,
            eps=eps,
            min_overlap=min_overlap,
            den_min=den_min,
        )
        if not eta_knee.get("eta_valid"):
            continue

        if x_min + half_width >= x_max - half_width:
            continue
        rand_center = float(rng.uniform(x_min + half_width, x_max - half_width))
        rand_mask = (x >= rand_center - half_width) & (x <= rand_center + half_width)
        if rand_mask.sum() < 5:
            continue

        eta_rand = compute_voltage_nonreciprocity(
            x[rand_mask],
            v[rand_mask],
            grid_n=grid_n,
            eps=eps,
            min_overlap=min_overlap,
            den_min=den_min,
        )
        if not eta_rand.get("eta_valid"):
            continue

        eta_knee_val = eta_knee.get("eta_V_L1")
        eta_rand_val = eta_rand.get("eta_V_L1")
        ratio = eta_knee_val / eta_rand_val if eta_rand_val not in (None, 0) else np.nan

        rows.append(
            {
                "file_id": file_id,
                "segment_id": segment_id,
                "x_name": x_name,
                "eta_knee_window": eta_knee_val,
                "eta_random_window": eta_rand_val,
                "eta_knee_random_ratio": ratio,
                "knee_window_frac": window_frac,
                "pipeline_version": PIPELINE_VERSION,
            }
        )

    out_path = out_dir / "phase45_knee_window.csv"
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)

    report_lines = []
    report_lines.append("# Phase 4.5 Salvage Summary")
    report_lines.append("")
    report_lines.append(f"- Pipeline version: {PIPELINE_VERSION}")
    report_lines.append(f"- Rows: {len(out_df)}")
    if not out_df.empty:
        ratio = pd.to_numeric(out_df["eta_knee_random_ratio"], errors="coerce").dropna()
        if not ratio.empty:
            report_lines.append(f"- eta_knee_window / eta_random_window: median={float(ratio.median()):.4g}")
            report_lines.append(f"- fraction > 1: {float((ratio > 1.0).mean()):.4g}")
    report_lines.append("")
    report_lines.append("Notes: ratios use a +/- 20% sweep window around the knee and a random window of equal size.")

    report_path = out_dir / "phase45_salvage.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
