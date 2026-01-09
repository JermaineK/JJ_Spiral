"""Audit metrics and generate defensibility reports."""

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

LOGGER = logging.getLogger("jjparse")


def _distribution(series: pd.Series) -> dict:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return {}
    return {
        "count": int(series.count()),
        "median": float(series.median()),
        "p05": float(series.quantile(0.05)),
        "p95": float(series.quantile(0.95)),
    }


def _load_segment(segments_dir: Path, file_id: str, segment_id: int, x_name: str | None) -> pd.DataFrame:
    seg_path = segments_dir / f"{file_id}.parquet"
    if not seg_path.exists():
        return pd.DataFrame()
    seg_all = pd.read_parquet(seg_path)
    seg_df = seg_all[seg_all["segment_id"] == segment_id]
    if x_name and "x_name" in seg_df.columns:
        seg_df = seg_df[seg_df["x_name"] == x_name]
    return seg_df


def _split_plus_minus(x: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    mask = np.isfinite(x) & np.isfinite(v)
    x = x[mask]
    v = v[mask]
    if x.size < 5:
        return None
    pos_mask = x > 0
    neg_mask = x < 0
    if not pos_mask.any() or not neg_mask.any():
        return None
    x_pos = np.abs(x[pos_mask])
    v_pos = v[pos_mask]
    x_neg = np.abs(x[neg_mask])
    v_neg = v[neg_mask]
    return x_pos, v_pos, x_neg, v_neg


def _random_pair_control(
    segments: list[dict],
    grid_n: int,
    eps: float,
    min_overlap: float,
    den_min: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[dict]] = {}
    for seg in segments:
        key = str(seg.get("x_name") or "unknown")
        groups.setdefault(key, []).append(seg)

    eta_vals = []
    corr_vals = []
    for group in groups.values():
        if len(group) < 2:
            continue
        for seg in group:
            candidates = [other for other in group if other["file_id"] != seg["file_id"]]
            if not candidates:
                continue
            other = rng.choice(candidates)
            x_comb = np.concatenate([seg["x_pos"], -other["x_neg"]])
            v_comb = np.concatenate([seg["v_pos"], other["v_neg"]])
            eta = compute_voltage_nonreciprocity(
                x_comb,
                v_comb,
                grid_n=grid_n,
                eps=eps,
                min_overlap=min_overlap,
                den_min=den_min,
            )
            if eta.get("eta_valid"):
                eta_vals.append(eta.get("eta_V_L1"))
                corr_vals.append(eta.get("corr_pm"))

    eta_series = pd.Series(eta_vals, dtype=float)
    corr_series = pd.Series(corr_vals, dtype=float)
    return {
        "n_pairs": int(len(eta_vals)),
        "eta_median": float(eta_series.median()) if not eta_series.empty else np.nan,
        "eta_p95": float(eta_series.quantile(0.95)) if not eta_series.empty else np.nan,
        "corr_median": float(corr_series.median()) if not corr_series.empty else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit metrics and generate defensibility reports.")
    parser.add_argument("--metrics", default="results/metrics/metrics_long.csv", help="Metrics CSV")
    parser.add_argument("--segments-dir", default="results/parsed/segments", help="Segment parquet folder")
    parser.add_argument("--out", default="results/reports", help="Output folder")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(args.seed)
    setup_logging("results/reports")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metrics, low_memory=False)
    if df.empty:
        LOGGER.warning("No metrics to audit.")
        return

    bidir = df[df.get("is_bidirectional", False)]
    eta_valid = df.get("eta_valid", pd.Series(False, index=df.index)).fillna(False)
    knee_valid = df.get("knee_valid", pd.Series(False, index=df.index)).fillna(False)
    knee_edge = df.get("knee_edge_flag", pd.Series(False, index=df.index)).fillna(False)
    knee_capped = df.get("knee_score_capped", pd.Series(False, index=df.index)).fillna(False)
    knee_cap_val = pd.to_numeric(df.get("knee_score_cap", pd.Series(dtype=float)), errors="coerce").dropna()
    knee_cap = float(knee_cap_val.iloc[0]) if not knee_cap_val.empty else np.nan

    bidir_missing = bidir[~eta_valid.reindex(bidir.index).fillna(False)]
    mismatch_cols = [
        "file_id",
        "segment_id",
        "x_name",
        "eta_reason",
        "f_overlap",
        "eta_denom",
        "x_min_plus",
        "x_max_plus",
        "x_min_minus",
        "x_max_minus",
        "x_grid_min",
        "x_grid_max",
        "n_grid",
    ]
    mismatch = bidir_missing[mismatch_cols].copy() if not bidir_missing.empty else pd.DataFrame(columns=mismatch_cols)
    if not mismatch.empty:
        mismatch["pipeline_version"] = PIPELINE_VERSION
    mismatch.to_csv(out_dir / "bidirectional_mismatch.csv", index=False)

    reason_counts = bidir_missing.get("eta_reason", pd.Series(dtype=str)).fillna("missing").value_counts()
    overlap_stats = _distribution(df.get("f_overlap", pd.Series(dtype=float)))

    knee_stats = _distribution(df.loc[knee_valid, "knee_x_norm"] if "knee_x_norm" in df.columns else pd.Series(dtype=float))
    knee_top = (
        df[knee_valid].dropna(subset=["knee_score_rank"]).sort_values("knee_score_rank", ascending=False).head(50)
    )
    knee_top_stats = _distribution(knee_top.get("knee_x_norm", pd.Series(dtype=float)))

    h_incoh = pd.to_numeric(df.get("H_incoh", pd.Series(dtype=float)), errors="coerce")
    h_raw = pd.to_numeric(df.get("H_incoh_raw", pd.Series(dtype=float)), errors="coerce")
    h_missing = float(h_incoh.isna().mean()) if len(h_incoh) else np.nan
    h_zero_true = float(((h_incoh == 0.0) & (h_raw == 0.0)).mean()) if len(h_incoh) else np.nan

    eta_high = df[df.get("eta_valid", False)].copy()
    high_mismatch = pd.DataFrame()
    if not eta_high.empty:
        eta_high = eta_high.sort_values("eta_V_L1", ascending=False)
        high_mismatch = eta_high[(eta_high["eta_V_L1"] > 0.7) & (eta_high.get("corr_pm", 0.0) > 0.9)].head(10)

    segments_dir = Path(args.segments_dir)
    segments = []
    for _, row in df[df.get("eta_valid", False)].iterrows():
        seg_df = _load_segment(segments_dir, str(row.get("file_id")), int(row.get("segment_id", 0)), row.get("x_name"))
        if seg_df.empty:
            continue
        x = pd.to_numeric(seg_df["x"], errors="coerce").to_numpy()
        v = pd.to_numeric(seg_df["V_V"], errors="coerce").to_numpy()
        split = _split_plus_minus(x, v)
        if split is None:
            continue
        x_pos, v_pos, x_neg, v_neg = split
        segments.append(
            {
                "file_id": str(row.get("file_id")),
                "segment_id": int(row.get("segment_id", 0)),
                "x_name": row.get("x_name"),
                "x_pos": x_pos,
                "v_pos": v_pos,
                "x_neg": x_neg,
                "v_neg": v_neg,
            }
        )

    grid_n = int(config.get("parity", {}).get("grid_n", 200))
    eps = float(config.get("metrics", {}).get("eps", 1e-12))
    min_overlap = float(config.get("parity", {}).get("min_overlap", 0.8))
    den_min = float(config.get("parity", {}).get("den_min", 0.0))
    rand_control = _random_pair_control(segments, grid_n, eps, min_overlap, den_min, args.seed + 10)

    lines = []
    lines.append("# Audit report")
    lines.append("")
    lines.append(f"- Pipeline version: {PIPELINE_VERSION}")
    lines.append("")
    lines.append("## What looked odd")
    lines.append("")
    lines.append("- High eta_V_L1 median and potential pairing inflation.")
    lines.append("- Knee locations clustering late in the sweep.")
    lines.append("- Knee score rank saturation near cap.")
    lines.append("- Bidirectional mismatch counts.")
    lines.append("- Incoherence zeros or missing joins.")
    lines.append("")
    lines.append("## Checks run")
    lines.append("")
    lines.append(f"- Overlap threshold: min_overlap={min_overlap}")
    lines.append(f"- Denominator threshold: den_min={den_min}")
    lines.append("- Alternate eta metrics: eta_V_L2, corr_pm.")
    lines.append("- Knee norm/value tracking with edge flags.")
    lines.append("- Knee score cap and capped flag.")
    lines.append("- Randomized pairing control (cross-file pairing).")
    lines.append("")
    lines.append("## Key outcomes")
    lines.append("")
    lines.append(f"- Bidirectional segments: {len(bidir)}")
    lines.append(f"- eta_valid segments: {int(eta_valid.sum())}")
    lines.append(f"- eta invalid reasons: {', '.join(f'{k}={v}' for k, v in reason_counts.items()) or 'none'}")
    lines.append(f"- knee_edge_flag (valid knees): {knee_edge[knee_valid].mean() * 100:.2f}%")
    lines.append(f"- knee_score_capped: {knee_capped.mean() * 100:.2f}%")
    if np.isfinite(knee_cap):
        lines.append(f"- knee_score_cap (p99): {knee_cap:.6g}")
    if knee_stats:
        lines.append(
            f"- knee_x_norm (valid): median={knee_stats['median']:.3g}, p05={knee_stats['p05']:.3g}, p95={knee_stats['p95']:.3g}"
        )
    if knee_top_stats:
        lines.append(
            f"- knee_x_norm (top knees): median={knee_top_stats['median']:.3g}, p05={knee_top_stats['p05']:.3g}, p95={knee_top_stats['p95']:.3g}"
        )
    if overlap_stats:
        lines.append(
            f"- f_overlap: median={overlap_stats['median']:.3g}, p05={overlap_stats['p05']:.3g}, p95={overlap_stats['p95']:.3g}"
        )
    lines.append(f"- H_incoh missing: {h_missing * 100:.2f}%")
    lines.append(f"- H_incoh == 0 (raw==0): {h_zero_true * 100:.2f}%")
    lines.append(
        f"- Randomized pairing control: n={rand_control['n_pairs']}, eta_median={rand_control['eta_median']:.3g}, eta_p95={rand_control['eta_p95']:.3g}, corr_median={rand_control['corr_median']:.3g}"
    )
    lines.append("")
    lines.append("## High-eta cross-checks")
    lines.append("")
    if high_mismatch.empty:
        lines.append("- No high-eta + high-corr conflicts detected.")
    else:
        for _, row in high_mismatch.iterrows():
            lines.append(
                f"- {row.get('file_id')} seg={row.get('segment_id')} eta_L1={row.get('eta_V_L1'):.3g} corr_pm={row.get('corr_pm'):.3g}"
            )
    lines.append("")
    lines.append("## What changed")
    lines.append("")
    lines.append("- eta_valid gating now enforces overlap and denominator thresholds.")
    lines.append("- Knee position stored as norm/value with edge flags; cap rate exposed.")
    lines.append("- Added randomized pairing control and mismatch CSV.")
    lines.append("")
    lines.append("## Remaining limitations")
    lines.append("")
    lines.append("- Overlap threshold may exclude narrow sweeps; review low_overlap cases.")
    lines.append("- Randomized pairing control uses same x_name but not device metadata.")
    lines.append("")
    lines.append("## Audit tables")
    lines.append("")
    lines.append("- Bidirectional mismatch list: results/reports/bidirectional_mismatch.csv")
    lines.append("")

    (out_dir / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    one = []
    one.append("# Diagnostics (one-page)")
    one.append("")
    one.append(f"- Pipeline version: {PIPELINE_VERSION}")
    one.append(f"- eta_valid segments: {int(eta_valid.sum())} / {len(df)}")
    one.append(f"- knee_edge_flag (valid knees): {knee_edge[knee_valid].mean() * 100:.2f}%")
    one.append(f"- knee_score_capped: {knee_capped.mean() * 100:.2f}%")
    if np.isfinite(knee_cap):
        one.append(f"- knee_score_cap (p99): {knee_cap:.6g}")
    one.append(f"- H_incoh missing: {h_missing * 100:.2f}%")
    one.append(f"- H_incoh == 0 (raw==0): {h_zero_true * 100:.2f}%")
    one.append(f"- Random pairing control median eta: {rand_control['eta_median']:.3g}")
    one.append(f"- Random pairing control median corr_pm: {rand_control['corr_median']:.3g}")
    one.append(f"- Bidirectional mismatch rows: {len(mismatch)}")
    one.append("")
    one.append("See audit_report.md for details.")
    (out_dir / "diagnostic_onepage.md").write_text("\n".join(one) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
