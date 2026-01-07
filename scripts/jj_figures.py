"""Generate Phase 3 figures and exemplar plots."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jjparse.config import load_config, seed_everything, setup_logging

LOGGER = logging.getLogger("jjparse")

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
BLACK = "#000000"
GRAY = "#666666"
RED = "#d62728"


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (6.5, 4.5),
            "figure.dpi": 300,
            "font.family": "sans-serif",
            "lines.linewidth": 1.5,
        }
    )


def _save_fig(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), dpi=300)
    fig.savefig(base_path.with_suffix(".pdf"))
    plt.close(fig)


def _downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    idx = np.linspace(0, x.size - 1, max_points).astype(int)
    return x[idx], y[idx]


def _select_top_unique(df: pd.DataFrame, column: str, n: int) -> list[pd.Series]:
    if column not in df.columns:
        return []
    rows = []
    seen = set()
    for _, row in df.dropna(subset=[column]).sort_values(column, ascending=False).iterrows():
        file_id = row.get("file_id")
        if file_id in seen:
            continue
        seen.add(file_id)
        rows.append(row)
        if len(rows) >= n:
            break
    return rows


def _select_random_unique(df: pd.DataFrame, n: int, rng: np.random.Generator, exclude: set[str]) -> list[pd.Series]:
    if df.empty:
        return []
    file_ids = [fid for fid in df["file_id"].dropna().unique() if fid not in exclude]
    if not file_ids:
        return []
    pick = rng.choice(file_ids, size=min(n, len(file_ids)), replace=False)
    rows = []
    for fid in pick:
        row = df[df["file_id"] == fid].iloc[0]
        rows.append(row)
    return rows


def _add_selection(selection: dict, row: pd.Series, reason: str) -> None:
    key = (str(row.get("file_id")), int(row.get("segment_id", 0)))
    entry = selection.setdefault(key, {"row": row, "reasons": set()})
    entry["reasons"].add(reason)


def _load_segment(segments_dir: Path, file_id: str, segment_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    seg_path = segments_dir / f"{file_id}.parquet"
    if not seg_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    seg_all = pd.read_parquet(seg_path)
    seg_df = seg_all[seg_all["segment_id"] == segment_id]
    return seg_df, seg_all


def _plot_exemplar(
    row: pd.Series,
    segments_dir: Path,
    out_dir: Path,
    max_points: int,
) -> None:
    file_id = str(row.get("file_id"))
    segment_id = int(row.get("segment_id", 0))
    seg_df, seg_all = _load_segment(segments_dir, file_id, segment_id)
    if seg_df.empty:
        LOGGER.warning("Missing segment data for %s seg %s", file_id, segment_id)
        return

    x = pd.to_numeric(seg_df["x"], errors="coerce").to_numpy()
    v = pd.to_numeric(seg_df["V_V"], errors="coerce").to_numpy()
    mask = np.isfinite(x) & np.isfinite(v)
    x = x[mask]
    v = v[mask]
    if x.size == 0:
        return

    fig, ax = plt.subplots()
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)

    plotted = False
    if not seg_all.empty and "segment_dir" in seg_all.columns:
        up = seg_all[seg_all["segment_dir"] == "up"]
        down = seg_all[seg_all["segment_dir"] == "down"]
        if not up.empty and not down.empty:
            x_up = pd.to_numeric(up["x"], errors="coerce").to_numpy()
            v_up = pd.to_numeric(up["V_V"], errors="coerce").to_numpy()
            x_down = pd.to_numeric(down["x"], errors="coerce").to_numpy()
            v_down = pd.to_numeric(down["V_V"], errors="coerce").to_numpy()
            mask_up = np.isfinite(x_up) & np.isfinite(v_up)
            mask_down = np.isfinite(x_down) & np.isfinite(v_down)
            x_up, v_up = x_up[mask_up], v_up[mask_up]
            x_down, v_down = x_down[mask_down], v_down[mask_down]
            x_up, v_up = _downsample_xy(x_up, v_up, max_points)
            x_down, v_down = _downsample_xy(x_down, v_down, max_points)
            ax.plot(x_up, v_up, color=BLUE, label="forward", alpha=0.8)
            ax.plot(x_down, v_down, color=ORANGE, label="reverse", alpha=0.8)
            plotted = True

    if not plotted and np.any(x < 0) and np.any(x > 0):
        pos_mask = x > 0
        neg_mask = x < 0
        x_pos, v_pos = _downsample_xy(x[pos_mask], v[pos_mask], max_points)
        x_neg, v_neg = _downsample_xy(x[neg_mask], v[neg_mask], max_points)
        ax.plot(x_pos, v_pos, color=BLUE, label="forward", alpha=0.8)
        ax.plot(x_neg, v_neg, color=ORANGE, label="reverse", alpha=0.8)
        plotted = True

    if not plotted:
        x_plot, v_plot = _downsample_xy(x, v, max_points)
        ax.plot(x_plot, v_plot, color=BLACK, label="trace", alpha=0.8)

    knee_x = row.get("knee_x")
    if knee_x is not None and np.isfinite(knee_x):
        ax.axvline(float(knee_x), color=RED, linestyle="--", linewidth=1.5, label="knee", alpha=0.9)

    x_name = str(row.get("x_name", "x"))
    ax.set_xlabel(x_name)
    ax.set_ylabel("V (V)")
    ax.set_title(f"{file_id} seg{segment_id}")

    annotations = []
    for label, key in [
        ("eta_V_L1", "eta_V_L1"),
        ("eta_norm", "eta_norm"),
        ("H_incoh", "H_incoh"),
        ("parity_bit", "parity_bit"),
        ("knee_x", "knee_x"),
    ]:
        value = row.get(key)
        if value is not None and np.isfinite(value):
            if key == "parity_bit":
                annotations.append(f"{label}={int(value)}")
            else:
                annotations.append(f"{label}={value:.4g}")

    if annotations:
        ax.text(
            0.02,
            0.98,
            "\n".join(annotations),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )

    ax.legend(loc="best", fontsize=8)
    out_path = out_dir / f"{file_id}__seg{segment_id}__trace_knee_parity"
    _save_fig(fig, out_path)


def _plot_eta_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    if "eta_V_L1" not in df.columns or "eta_norm" not in df.columns:
        return
    raw = pd.to_numeric(df["eta_V_L1"], errors="coerce").dropna()
    norm = pd.to_numeric(df["eta_norm"], errors="coerce").dropna()
    if raw.empty or norm.empty:
        return

    combined = pd.concat([raw, norm], ignore_index=True)
    bins = np.histogram_bin_edges(combined, bins=30)

    fig, ax = plt.subplots()
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.hist(raw, bins=bins, color=BLUE, alpha=0.6, label="eta_V_L1")
    ax.hist(norm, bins=bins, color=ORANGE, alpha=0.6, label="eta_norm")

    for series, color, label in [
        (raw, BLUE, "raw"),
        (norm, ORANGE, "norm"),
    ]:
        median = float(np.median(series))
        p95 = float(np.percentile(series, 95))
        ax.axvline(median, color=color, linestyle="-", linewidth=1.2, label=f"{label} median")
        ax.axvline(p95, color=color, linestyle="--", linewidth=1.2, label=f"{label} p95")

    ax.set_xlabel("eta")
    ax.set_ylabel("count")
    ax.set_title("Odd-channel distribution")
    ax.legend(fontsize=8)
    _save_fig(fig, out_dir / "dist_eta_raw_vs_norm")


def _plot_eta_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    if not {"eta_V_L1", "eta_norm", "H_incoh"}.issubset(df.columns):
        return
    data = df[["eta_V_L1", "eta_norm", "H_incoh"]].copy()
    data = data.apply(pd.to_numeric, errors="coerce").dropna()
    if data.empty:
        return

    fig, ax = plt.subplots()
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    sc = ax.scatter(
        data["eta_V_L1"],
        data["eta_norm"],
        c=data["H_incoh"],
        cmap="Greys",
        s=12,
        alpha=0.8,
    )
    ax.set_xlabel("eta_V_L1")
    ax.set_ylabel("eta_norm")
    ax.set_title("Eta vs normalized eta (colored by incoherence)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("H_incoh")
    _save_fig(fig, out_dir / "eta_vs_eta_norm_colored_by_incoherence")


def _plot_parity_behavior(summary_path: Path, out_dir: Path) -> None:
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    if summary.empty:
        return
    row = summary.iloc[0]
    frac_flip = float(row.get("fraction_flip", np.nan))
    frac_lock = float(row.get("fraction_lock_after", np.nan))
    if not np.isfinite(frac_flip) or not np.isfinite(frac_lock):
        return

    fig, ax = plt.subplots()
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4, axis="y")
    labels = ["parity flip near knee", "parity lock after knee"]
    values = [frac_flip, frac_lock]
    ax.bar(labels, values, color=[GRAY, BLACK], alpha=0.8)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("fraction")
    ax.set_title("Parity behavior relative to knee")
    _save_fig(fig, out_dir / "parity_behavior_relative_to_knee")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 3 figures and exemplars.")
    parser.add_argument("--metrics", default="results/metrics/metrics_long.csv", help="Metrics CSV")
    parser.add_argument("--segments-dir", default="results/parsed/segments", help="Segment parquet folder")
    parser.add_argument("--out", default="results/figures", help="Output folder")
    parser.add_argument("--summary", default="results/reports/knee_parity_summary.csv", help="Knee parity summary CSV")
    parser.add_argument("--top-n", type=int, default=5, help="Top N by eta_norm and knee_score_rank")
    parser.add_argument("--random-n", type=int, default=3, help="Random controls per bidirectional/unidirectional")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(args.seed)
    setup_logging("results/reports")

    _set_style()

    metrics_path = Path(args.metrics)
    segments_dir = Path(args.segments_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_path, low_memory=False)
    if df.empty:
        LOGGER.warning("No metrics in %s", metrics_path)
        return

    rng = np.random.default_rng(args.seed)
    selection: dict[tuple[str, int], dict] = {}

    for row in _select_top_unique(df, "eta_norm", args.top_n):
        _add_selection(selection, row, "top_eta_norm")
    for row in _select_top_unique(df, "knee_score_rank", args.top_n):
        _add_selection(selection, row, "top_knee_score")

    bidir_mask = df.get("is_bidirectional", pd.Series(False, index=df.index)).fillna(False)
    bidir = df[bidir_mask]
    unidir = df[~bidir_mask]

    used_files = {key[0] for key in selection}
    for row in _select_random_unique(bidir, args.random_n, rng, used_files):
        _add_selection(selection, row, "random_bidirectional")
    used_files = {key[0] for key in selection}
    for row in _select_random_unique(unidir, args.random_n, rng, used_files):
        _add_selection(selection, row, "random_unidirectional")

    max_points = int(config.get("qc", {}).get("max_points", 5000))
    index_rows = []
    for (file_id, segment_id), entry in selection.items():
        row = entry["row"]
        reasons = sorted(entry["reasons"])
        index_rows.append(
            {
                "file_id": file_id,
                "segment_id": segment_id,
                "reasons": ";".join(reasons),
                "x_name": row.get("x_name"),
                "eta_V_L1": row.get("eta_V_L1"),
                "eta_norm": row.get("eta_norm"),
                "H_incoh": row.get("H_incoh"),
                "parity_bit": row.get("parity_bit"),
                "knee_x": row.get("knee_x"),
                "knee_score_rank": row.get("knee_score_rank"),
            }
        )
        _plot_exemplar(row, segments_dir, out_dir, max_points)

    if index_rows:
        pd.DataFrame(index_rows).to_csv(out_dir / "exemplar_index.csv", index=False)

    _plot_eta_distributions(df, out_dir)
    _plot_eta_scatter(df, out_dir)
    _plot_parity_behavior(Path(args.summary), out_dir)


if __name__ == "__main__":
    main()
