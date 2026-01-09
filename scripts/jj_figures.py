"""Generate Phase 4 figures and exemplar plots."""

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
from jjparse.version import PIPELINE_VERSION

LOGGER = logging.getLogger("jjparse")
QC_LOGGER = logging.getLogger("jjparse.plot_qc")

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


def _setup_qc_logger(log_path: Path) -> None:
    QC_LOGGER.setLevel(logging.INFO)
    QC_LOGGER.handlers.clear()
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    QC_LOGGER.addHandler(handler)


def _save_fig(fig: plt.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
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
    if df.empty or n <= 0:
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
    key = (str(row.get("file_id")), int(row.get("segment_id", 0)), str(row.get("x_name", "")))
    entry = selection.setdefault(key, {"row": row, "reasons": set()})
    entry["reasons"].add(reason)


def _build_segment_ranges(metrics_df: pd.DataFrame, segments_dir: Path) -> dict[tuple[str, int], dict]:
    ranges: dict[tuple[str, int], dict] = {}
    for file_id, group in metrics_df.groupby("file_id"):
        seg_path = segments_dir / f"{file_id}.parquet"
        if not seg_path.exists():
            QC_LOGGER.warning("Missing segment parquet for %s", file_id)
            continue
        try:
            seg_all = pd.read_parquet(seg_path, columns=["segment_id", "x", "x_name"])
        except (ValueError, KeyError):
            seg_all = pd.read_parquet(seg_path)
        if seg_all.empty:
            continue
        for seg_id, seg_df in seg_all.groupby("segment_id"):
            x = pd.to_numeric(seg_df["x"], errors="coerce").to_numpy()
            x = x[np.isfinite(x)]
            if x.size == 0:
                continue
            x_min = float(np.nanmin(x))
            x_max = float(np.nanmax(x))
            x_name = None
            if "x_name" in seg_df.columns:
                first = seg_df["x_name"].dropna()
                if not first.empty:
                    x_name = str(first.iloc[0])
            ranges[(str(file_id), int(seg_id))] = {"x_min": x_min, "x_max": x_max, "x_name": x_name}
    return ranges


def _compute_knee_fields(row: pd.Series, range_info: dict | None, config: dict) -> dict:
    knee_value = row.get("knee_x_value")
    knee_norm = row.get("knee_x_norm")

    if range_info is None:
        return {"knee_x_value": knee_value, "knee_x_norm": knee_norm, "knee_valid": False, "knee_edge_flag": False}

    x_min = range_info["x_min"]
    x_max = range_info["x_max"]
    span = x_max - x_min
    if not np.isfinite(span) or span <= 0:
        return {"knee_x_value": knee_value, "knee_x_norm": knee_norm, "knee_valid": False, "knee_edge_flag": False}

    if knee_value is None or not np.isfinite(knee_value):
        raw_knee = row.get("knee_x")
        if raw_knee is not None and np.isfinite(raw_knee):
            knee_value = float(raw_knee)

    if knee_norm is None or not np.isfinite(knee_norm):
        if knee_value is not None and np.isfinite(knee_value):
            knee_norm = float((knee_value - x_min) / span)

    if (knee_value is None or not np.isfinite(knee_value)) and knee_norm is not None and np.isfinite(knee_norm):
        knee_value = float(x_min + knee_norm * span)

    knee_valid = bool(knee_norm is not None and np.isfinite(knee_norm) and 0.0 <= knee_norm <= 1.0)
    edge_frac = float(config.get("knee", {}).get("conf_edge_frac", 0.05))
    knee_edge_flag = bool(knee_valid and (knee_norm <= edge_frac or knee_norm >= 1.0 - edge_frac))

    return {
        "knee_x_value": knee_value if knee_value is not None and np.isfinite(knee_value) else np.nan,
        "knee_x_norm": knee_norm if knee_norm is not None and np.isfinite(knee_norm) else np.nan,
        "knee_valid": knee_valid,
        "knee_edge_flag": knee_edge_flag,
    }


def _attach_knee_fields(metrics_df: pd.DataFrame, ranges: dict, config: dict) -> pd.DataFrame:
    knee_value_list = []
    knee_norm_list = []
    knee_valid_list = []
    knee_edge_list = []
    for _, row in metrics_df.iterrows():
        key = (str(row.get("file_id")), int(row.get("segment_id", 0)))
        range_info = ranges.get(key)
        row_x = row.get("x_name") or row.get("axis_primary")
        if range_info and row_x and range_info.get("x_name") and str(row_x) != str(range_info["x_name"]):
            QC_LOGGER.warning(
                "x_name mismatch for %s seg %s: metrics=%s segments=%s",
                row.get("file_id"),
                row.get("segment_id"),
                row_x,
                range_info.get("x_name"),
            )
            range_info = None
        fields = _compute_knee_fields(row, range_info, config)
        knee_value_list.append(fields["knee_x_value"])
        knee_norm_list.append(fields["knee_x_norm"])
        knee_valid_list.append(fields["knee_valid"])
        knee_edge_list.append(fields["knee_edge_flag"])
    metrics_df = metrics_df.copy()
    metrics_df["knee_x_value"] = knee_value_list
    metrics_df["knee_x_norm"] = knee_norm_list
    metrics_df["knee_valid"] = knee_valid_list
    metrics_df["knee_edge_flag"] = knee_edge_list
    return metrics_df


def _build_exemplar_index(
    metrics_df: pd.DataFrame,
    rng: np.random.Generator,
    top_n: int,
    random_n: int,
) -> pd.DataFrame:
    selection: dict[tuple[str, int, str], dict] = {}

    bidir_mask = metrics_df.get("is_bidirectional", pd.Series(False, index=metrics_df.index)).fillna(False)
    bidir_df = metrics_df[bidir_mask]
    unidir_df = metrics_df[~bidir_mask]

    eta_candidates = bidir_df.copy()
    if "eta_valid" in eta_candidates.columns:
        eta_candidates = eta_candidates[eta_candidates["eta_valid"].fillna(False)]
    eta_candidates = eta_candidates[eta_candidates["H_incoh"].notna()]
    for row in _select_top_unique(eta_candidates.dropna(subset=["eta_norm"]), "eta_norm", top_n):
        _add_selection(selection, row, "top_eta_norm")

    knee_candidates = metrics_df.copy()
    if "knee_valid" in knee_candidates.columns:
        knee_candidates = knee_candidates[knee_candidates["knee_valid"].fillna(False)]
    for row in _select_top_unique(knee_candidates.dropna(subset=["knee_score_rank"]), "knee_score_rank", top_n):
        _add_selection(selection, row, "top_knee_score")

    used_files = {key[0] for key in selection}
    for row in _select_random_unique(bidir_df, random_n, rng, used_files):
        _add_selection(selection, row, "random_bidirectional")

    used_files = {key[0] for key in selection}
    for row in _select_random_unique(unidir_df, random_n, rng, used_files):
        _add_selection(selection, row, "random_unidirectional")

    index_rows = []
    for (_, _, _), entry in selection.items():
        row = entry["row"]
        reasons = sorted(entry["reasons"])
        index_rows.append(
            {
                "file_id": row.get("file_id"),
                "segment_id": row.get("segment_id"),
                "x_name": row.get("x_name"),
                "bidirectional": bool(row.get("is_bidirectional")),
                "reasons": ";".join(reasons),
                "pipeline_version": PIPELINE_VERSION,
                "eta_V_L1": row.get("eta_V_L1"),
                "eta_norm": row.get("eta_norm"),
                "eta_V_signed": row.get("eta_V_signed"),
                "eta_V_L2": row.get("eta_V_L2"),
                "corr_pm": row.get("corr_pm"),
                "eta_valid": row.get("eta_valid"),
                "eta_reason": row.get("eta_reason"),
                "f_overlap": row.get("f_overlap"),
                "eta_denom": row.get("eta_denom"),
                "parity_bit": row.get("parity_bit"),
                "H_incoh": row.get("H_incoh"),
                "H_incoh_raw": row.get("H_incoh_raw"),
                "knee_x_value": row.get("knee_x_value"),
                "knee_x_norm": row.get("knee_x_norm"),
                "knee_valid": row.get("knee_valid"),
                "knee_edge_flag": row.get("knee_edge_flag"),
                "knee_score_capped": row.get("knee_score_capped"),
                "knee_score_rank": row.get("knee_score_rank"),
            }
        )
    return pd.DataFrame(index_rows)


def _load_segments(
    segments_dir: Path,
    file_id: str,
    segment_id: int,
    x_name: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    seg_path = segments_dir / f"{file_id}.parquet"
    if not seg_path.exists():
        return pd.DataFrame(), pd.DataFrame(), False
    seg_all = pd.read_parquet(seg_path)
    seg_df = seg_all[seg_all["segment_id"] == segment_id]
    matched = True
    if x_name and "x_name" in seg_df.columns:
        seg_named = seg_df[seg_df["x_name"] == x_name]
        if seg_named.empty:
            matched = False
        else:
            seg_df = seg_named
    return seg_df, seg_all, matched


def _mask_missing_metrics(row: pd.Series) -> pd.Series:
    row = row.copy()
    for key in [
        "eta_V_L1",
        "eta_norm",
        "eta_V_signed",
        "parity_bit",
        "H_incoh",
        "H_incoh_raw",
        "knee_x_value",
        "knee_x_norm",
        "knee_valid",
        "knee_edge_flag",
    ]:
        row[key] = np.nan
    row["knee_valid"] = False
    return row


def _plot_trace_on_ax(
    ax: plt.Axes,
    row: pd.Series,
    seg_df: pd.DataFrame,
    seg_all: pd.DataFrame,
    max_points: int,
    config: dict,
) -> dict:
    annotations = []
    bidirectional = bool(row.get("bidirectional") or row.get("is_bidirectional"))
    plotted_pairs = False

    if seg_df.empty:
        QC_LOGGER.warning("Missing segment data for %s seg %s", row.get("file_id"), row.get("segment_id"))
        return {"plotted_pairs": False, "annotations": annotations}

    x = pd.to_numeric(seg_df["x"], errors="coerce").to_numpy()
    v = pd.to_numeric(seg_df["V_V"], errors="coerce").to_numpy()
    mask = np.isfinite(x) & np.isfinite(v)
    x = x[mask]
    v = v[mask]
    if x.size == 0:
        return {"plotted_pairs": False, "annotations": annotations}

    if bidirectional and not seg_all.empty and "segment_dir" in seg_all.columns:
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
            order_up = np.argsort(x_up)
            order_down = np.argsort(x_down)
            x_up, v_up = x_up[order_up], v_up[order_up]
            x_down, v_down = x_down[order_down], v_down[order_down]
            x_up, v_up = _downsample_xy(x_up, v_up, max_points)
            x_down, v_down = _downsample_xy(x_down, v_down, max_points)
            ax.plot(x_up, v_up, color=BLUE, label="forward", alpha=0.8)
            ax.plot(x_down, v_down, color=ORANGE, label="reverse", alpha=0.8)
            plotted_pairs = True

    if bidirectional and not plotted_pairs:
        pos_mask = x > 0
        neg_mask = x < 0
        if pos_mask.any() and neg_mask.any():
            x_pos, v_pos = _downsample_xy(x[pos_mask], v[pos_mask], max_points)
            x_neg, v_neg = _downsample_xy(x[neg_mask], v[neg_mask], max_points)
            order_pos = np.argsort(x_pos)
            order_neg = np.argsort(x_neg)
            ax.plot(x_pos[order_pos], v_pos[order_pos], color=BLUE, label="forward", alpha=0.8)
            ax.plot(x_neg[order_neg], v_neg[order_neg], color=ORANGE, label="reverse", alpha=0.8)
            plotted_pairs = True

    if not plotted_pairs:
        x_plot, v_plot = _downsample_xy(x, v, max_points)
        order = np.argsort(x_plot)
        ax.plot(x_plot[order], v_plot[order], color=GRAY if bidirectional else BLACK, label="trace", alpha=0.8)
        if bidirectional:
            QC_LOGGER.warning("Bidirectional file plotted as single trace for %s seg %s", row.get("file_id"), row.get("segment_id"))

    knee_valid = bool(row.get("knee_valid"))
    knee_value = row.get("knee_x_value")
    if knee_valid and knee_value is not None and np.isfinite(knee_value):
        ax.axvline(float(knee_value), color=RED, linestyle="--", linewidth=1.5, alpha=0.9)
    elif knee_value is not None and np.isfinite(knee_value):
        annotations.append("knee invalid")

    x_name = str(row.get("x_name") or row.get("axis_primary") or "x")
    ax.set_xlabel(x_name)
    ax.set_ylabel("V (V)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)

    if bidirectional:
        eta_valid = row.get("eta_valid")
        eta_valid_str = "NA" if eta_valid is None or not np.isfinite(eta_valid) else str(bool(eta_valid))
        annotations.append(f"eta_valid={eta_valid_str}")
        for label, key in [
            ("eta_V_L1", "eta_V_L1"),
            ("eta_norm", "eta_norm"),
            ("eta_V_signed", "eta_V_signed"),
            ("parity_bit", "parity_bit"),
        ]:
            value = row.get(key)
            if value is None or not np.isfinite(value):
                annotations.append(f"{label}=NA")
                QC_LOGGER.warning("Missing %s for %s seg %s", key, row.get("file_id"), row.get("segment_id"))
            else:
                if key == "parity_bit":
                    annotations.append(f"{label}={int(value)}")
                else:
                    annotations.append(f"{label}={value:.4g}")
        f_overlap = row.get("f_overlap")
        if f_overlap is not None and np.isfinite(f_overlap):
            annotations.append(f"f_overlap={float(f_overlap):.3f}")
        else:
            annotations.append("f_overlap=NA")

    h_incoh = row.get("H_incoh")
    h_raw = row.get("H_incoh_raw")
    if h_incoh is None or not np.isfinite(h_incoh):
        annotations.append("H_incoh=NA")
        QC_LOGGER.warning("Missing H_incoh for %s seg %s", row.get("file_id"), row.get("segment_id"))
    elif abs(float(h_incoh)) < 1e-12:
        if h_raw is not None and np.isfinite(h_raw):
            if abs(float(h_raw)) < 1e-12:
                annotations.append("H_incoh=0 (raw=0)")
            else:
                annotations.append(f"H_incoh≈0 (raw={float(h_raw):.4g})")
                QC_LOGGER.warning(
                    "H_incoh clipped to zero with nonzero raw for %s seg %s",
                    row.get("file_id"),
                    row.get("segment_id"),
                )
        else:
            annotations.append("H_incoh=NA")
            QC_LOGGER.warning("H_incoh is zero without raw confirmation for %s seg %s", row.get("file_id"), row.get("segment_id"))
    else:
        annotations.append(f"H_incoh={float(h_incoh):.4g}")

    if knee_valid and knee_value is not None and np.isfinite(knee_value):
        knee_norm = row.get("knee_x_norm")
        if knee_norm is not None and np.isfinite(knee_norm):
            annotations.append(f"knee_x={knee_value:.4g} (norm={knee_norm:.3f})")
        else:
            annotations.append(f"knee_x={knee_value:.4g}")
    if row.get("knee_valid") is not None:
        annotations.append(f"knee_valid={bool(row.get('knee_valid'))}")
    if row.get("knee_edge_flag") is not None:
        annotations.append(f"knee_edge={bool(row.get('knee_edge_flag'))}")
    if row.get("knee_score_capped") is not None:
        annotations.append(f"knee_capped={bool(row.get('knee_score_capped'))}")

    if annotations:
        ax.text(
            0.02,
            0.98,
            "\n".join(annotations),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    if plotted_pairs:
        ax.legend(loc="best", fontsize=8)

    return {"plotted_pairs": plotted_pairs, "annotations": annotations}


def _plot_exemplar_trace(
    row: pd.Series,
    segments_dir: Path,
    out_path: Path,
    max_points: int,
    config: dict,
) -> None:
    file_id = str(row.get("file_id"))
    segment_id = int(row.get("segment_id", 0))
    x_name = row.get("x_name")
    seg_df, seg_all, matched = _load_segments(segments_dir, file_id, segment_id, x_name)
    if seg_df.empty:
        return
    if not matched:
        QC_LOGGER.warning("x_name mismatch for %s seg %s; annotations set to NA", file_id, segment_id)
        row = _mask_missing_metrics(row)

    fig, ax = plt.subplots()
    _plot_trace_on_ax(ax, row, seg_df, seg_all, max_points, config)
    ax.set_title(f"{file_id} seg{segment_id}")
    _save_fig(fig, out_path)


def _plot_eta_distributions(df: pd.DataFrame, out_path: Path, ax: plt.Axes | None = None) -> None:
    if "eta_V_L1" not in df.columns or "eta_norm" not in df.columns:
        return
    raw = pd.to_numeric(df["eta_V_L1"], errors="coerce").dropna()
    norm = pd.to_numeric(df["eta_norm"], errors="coerce").dropna()
    if raw.empty or norm.empty:
        return

    combined = pd.concat([raw, norm], ignore_index=True)
    bins = np.histogram_bin_edges(combined, bins=30)

    fig = None
    if ax is None:
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

    if fig is not None:
        _save_fig(fig, out_path)


def _plot_eta_scatter(df: pd.DataFrame, out_path: Path, ax: plt.Axes | None = None) -> None:
    if not {"eta_V_L1", "eta_norm", "H_incoh"}.issubset(df.columns):
        return
    data = df[["eta_V_L1", "eta_norm", "H_incoh"]].copy()
    data = data.apply(pd.to_numeric, errors="coerce").dropna()
    if data.empty:
        return

    fig = None
    if ax is None:
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
    ax.set_title("Eta vs normalized eta")
    if fig is not None:
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("H_incoh")
        _save_fig(fig, out_path)


def _plot_parity_behavior(summary_path: Path, out_path: Path, ax: plt.Axes | None = None) -> None:
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

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4, axis="y")
    labels = ["parity flip near knee", "parity lock after knee"]
    values = [frac_flip, frac_lock]
    ax.bar(labels, values, color=[GRAY, BLACK], alpha=0.8)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("fraction")
    ax.set_title("Parity behavior relative to knee")
    if fig is not None:
        _save_fig(fig, out_path)


def _plot_overview_2x2(
    metrics_df: pd.DataFrame,
    summary_path: Path,
    exemplar_row: pd.Series,
    segments_dir: Path,
    out_path: Path,
    config: dict,
    max_points: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.5), dpi=300)
    _plot_eta_distributions(metrics_df, out_path, ax=axes[0, 0])
    _plot_eta_scatter(metrics_df, out_path, ax=axes[0, 1])
    _plot_parity_behavior(summary_path, out_path, ax=axes[1, 0])

    file_id = str(exemplar_row.get("file_id"))
    segment_id = int(exemplar_row.get("segment_id", 0))
    x_name = exemplar_row.get("x_name")
    seg_df, seg_all, matched = _load_segments(segments_dir, file_id, segment_id, x_name)
    if not seg_df.empty:
        if not matched:
            exemplar_row = _mask_missing_metrics(exemplar_row)
        _plot_trace_on_ax(axes[1, 1], exemplar_row, seg_df, seg_all, max_points, config)
        axes[1, 1].set_title(f"{file_id} seg{segment_id}")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _check_qc(metrics_df: pd.DataFrame, exemplar_df: pd.DataFrame) -> None:
    knee_valid = metrics_df.get("knee_valid", pd.Series(False, index=metrics_df.index)).fillna(False)
    knee_norm = pd.to_numeric(metrics_df["knee_x_norm"], errors="coerce")
    valid_mask = knee_valid & knee_norm.notna()
    if valid_mask.any():
        in_range = ((knee_norm[valid_mask] >= 0.0) & (knee_norm[valid_mask] <= 1.0)).mean()
        if in_range < 0.99:
            QC_LOGGER.warning("knee_x_norm within [0,1] only %.2f%% of valid knees", 100.0 * in_range)

    if not exemplar_df.empty and "H_incoh" in exemplar_df.columns:
        h_vals = pd.to_numeric(exemplar_df["H_incoh"], errors="coerce")
        h_raw = pd.to_numeric(exemplar_df.get("H_incoh_raw"), errors="coerce")
        annotated = h_vals.notna()
        if annotated.any():
            true_zero = (h_vals == 0.0) & (h_raw.fillna(np.nan) == 0.0)
            frac_zero = float(true_zero.sum() / annotated.sum())
            if frac_zero > 0.01:
                QC_LOGGER.warning("H_incoh == 0 (raw==0) for %.2f%% of exemplars", 100.0 * frac_zero)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 4 figures and exemplars.")
    parser.add_argument("--metrics", default="results/metrics/metrics_long.csv", help="Metrics CSV")
    parser.add_argument("--segments-dir", default="results/parsed/segments", help="Segment parquet folder")
    parser.add_argument("--out", default="results/figures/final", help="Output folder for figures")
    parser.add_argument("--summary", default="results/reports/knee_parity_summary.csv", help="Knee parity summary CSV")
    parser.add_argument("--index", default="results/figures/exemplar_index.csv", help="Exemplar index CSV")
    parser.add_argument("--top-n", type=int, default=6, help="Top N by eta_norm and knee_score_rank")
    parser.add_argument("--random-n", type=int, default=6, help="Random controls per bidirectional/unidirectional")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(args.seed)
    setup_logging("results/reports")

    _set_style()
    _setup_qc_logger(Path("results/reports/plot_qc_warnings.log"))
    QC_LOGGER.info("Pipeline version: %s", PIPELINE_VERSION)

    metrics_path = Path(args.metrics)
    segments_dir = Path(args.segments_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_csv(metrics_path, low_memory=False)
    if metrics_df.empty:
        LOGGER.warning("No metrics in %s", metrics_path)
        return

    ranges = _build_segment_ranges(metrics_df, segments_dir)
    metrics_df = _attach_knee_fields(metrics_df, ranges, config)

    rng = np.random.default_rng(args.seed)
    exemplar_df = _build_exemplar_index(metrics_df, rng, args.top_n, args.random_n)
    exemplar_path = Path(args.index)
    exemplar_path.parent.mkdir(parents=True, exist_ok=True)
    exemplar_df.to_csv(exemplar_path, index=False)

    max_points = int(config.get("qc", {}).get("max_points", 5000))
    exemplars_dir = out_dir / "exemplars"
    for _, row in exemplar_df.iterrows():
        file_id = str(row.get("file_id"))
        segment_id = int(row.get("segment_id", 0))
        out_path = exemplars_dir / f"{file_id}__seg{segment_id}__trace_knee_parity"
        _plot_exemplar_trace(row, segments_dir, out_path, max_points, config)

    _plot_eta_distributions(metrics_df, out_dir / "dist_eta_raw_vs_norm")
    _plot_eta_scatter(metrics_df, out_dir / "eta_vs_eta_norm_colored_by_incoherence")
    _plot_parity_behavior(Path(args.summary), out_dir / "parity_behavior_relative_to_knee")

    best = metrics_df.copy()
    best = best[
        best.get("is_bidirectional", False)
        & best.get("knee_valid", False)
        & pd.to_numeric(best.get("H_incoh"), errors="coerce").notna()
    ]
    if not best.empty:
        best = best.sort_values("eta_norm", ascending=False)
        best_row = best.iloc[0]
        _plot_overview_2x2(
            metrics_df,
            Path(args.summary),
            best_row,
            segments_dir,
            out_dir / "figure_overview_2x2.pdf",
            config,
            max_points,
        )

    _check_qc(metrics_df, exemplar_df)


if __name__ == "__main__":
    main()
