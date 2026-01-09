"""Extract metrics from JJ datasets."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jjparse.config import load_config, seed_everything, setup_logging
from jjparse.io import make_file_id, read_dat
from jjparse.metrics import METRICS, axis_properties, compute_voltage_nonreciprocity, run_metric, split_segments
from jjparse.preprocess import apply_unit_conversions, infer_phase_radians
from jjparse.schema import build_signal_pack, coerce_types, detect_capabilities, infer_primary_axis, standardize_columns
from jjparse.version import PIPELINE_VERSION

LOGGER = logging.getLogger("jjparse")


def _iter_files(in_dir: Path, glob_pattern: str) -> list[Path]:
    files = sorted(in_dir.glob(glob_pattern))
    return [path for path in files if path.is_file()]


def _has_direction_meta(meta: dict) -> bool:
    for key, value in meta.items():
        text = f"{key} {value}".lower()
        if "forward" in text or "reverse" in text or "up" in text or "down" in text:
            return True
    return False


def _first_non_null(series: pd.Series) -> float | str | None:
    series = series.dropna()
    if series.empty:
        return None
    return series.iloc[0]


def _load_pairing_map(csv_path: str | None) -> tuple[list[dict], dict[str, str]]:
    if not csv_path:
        return [], {}
    df = pd.read_csv(csv_path)
    pairs = []
    file_to_pair = {}
    if {"pair_id", "file_left", "file_right"}.issubset(df.columns):
        for _, row in df.iterrows():
            pair_id = str(row["pair_id"])
            left = str(row["file_left"])
            right = str(row["file_right"])
            pairs.append({"pair_id": pair_id, "left": left, "right": right})
            file_to_pair[left] = pair_id
            file_to_pair[right] = pair_id
    elif {"left", "right"}.issubset(df.columns):
        for idx, row in df.iterrows():
            pair_id = str(idx)
            left = str(row["left"])
            right = str(row["right"])
            pairs.append({"pair_id": pair_id, "left": left, "right": right})
            file_to_pair[left] = pair_id
            file_to_pair[right] = pair_id
    return pairs, file_to_pair


def _pair_metrics(metrics_df: pd.DataFrame, pairs: list[dict]) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    summary_cols = [
        "eta_V_signed",
        "eta_V_L1",
        "knee_x",
        "cp_main_x",
        "V_std",
        "H_incoh",
        "H_incoh_raw",
    ]
    available = [col for col in summary_cols if col in metrics_df.columns]
    file_summary = metrics_df.groupby("file_id")[available].agg(_first_non_null)

    rows = []
    for pair in pairs:
        left = pair["left"]
        right = pair["right"]
        if left not in file_summary.index or right not in file_summary.index:
            continue
        left_row = file_summary.loc[left]
        right_row = file_summary.loc[right]

        row = {
            "pair_id": pair["pair_id"],
            "left_file_id": left,
            "right_file_id": right,
        }

        if "eta_V_signed" in file_summary.columns:
            lval = left_row.get("eta_V_signed")
            rval = right_row.get("eta_V_signed")
            if pd.notna(lval) and pd.notna(rval):
                row["eta_V_signed_flips"] = np.sign(lval) == -np.sign(rval)
        for col in ["knee_x", "cp_main_x"]:
            if col in file_summary.columns:
                lval = left_row.get(col)
                rval = right_row.get(col)
                if pd.notna(lval) and pd.notna(rval):
                    row[f"{col}_shift"] = float(rval - lval)
        for col in ["V_std", "H_incoh", "H_incoh_raw"]:
            if col in file_summary.columns:
                lval = left_row.get(col)
                rval = right_row.get(col)
                if pd.notna(lval) and pd.notna(rval):
                    denom = abs(lval) + abs(rval)
                    row[f"{col}_rel_diff"] = abs(lval - rval) / denom if denom else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def _knee_confident(knee_x: float | None, slope_pre: float | None, slope_post: float | None, x: np.ndarray, config: dict) -> bool:
    if knee_x is None or not np.isfinite(knee_x):
        return False
    if slope_pre is None or slope_post is None:
        return False
    if not np.isfinite(slope_pre) or not np.isfinite(slope_post):
        return False

    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return False
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return False

    edge_frac = float(config.get("knee", {}).get("conf_edge_frac", 0.05))
    edge = edge_frac * (x_max - x_min)
    if knee_x <= x_min + edge or knee_x >= x_max - edge:
        return False

    eps = float(config.get("metrics", {}).get("eps", 1e-12))
    abs_pre = abs(float(slope_pre))
    abs_post = abs(float(slope_post))
    ratio = max(abs_pre, abs_post) / (min(abs_pre, abs_post) + eps)
    factor = float(config.get("knee", {}).get("conf_slope_factor", 3.0))
    return bool(ratio >= factor)


def _parity_bit(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(1 if value > 0 else 0)


def _parity_stability(series: pd.Series) -> float | None:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return None
    counts = series.value_counts()
    return float(counts.max() / counts.sum())


def _normalize_incoherence(series: pd.Series, config: dict) -> pd.Series:
    raw = pd.to_numeric(series, errors="coerce")
    raw = raw.dropna()
    if raw.empty:
        return pd.Series(index=series.index, dtype=float)

    norm_cfg = config.get("incoherence", {}).get("normalize", {})
    method = str(norm_cfg.get("method", "percentile")).lower()
    if method == "minmax":
        low = float(raw.min())
        high = float(raw.max())
    else:
        p_low = float(norm_cfg.get("p_low", 1.0))
        p_high = float(norm_cfg.get("p_high", 99.0))
        low = float(np.nanpercentile(raw, p_low))
        high = float(np.nanpercentile(raw, p_high))

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return pd.Series(index=series.index, data=np.nan, dtype=float)

    normalized = (pd.to_numeric(series, errors="coerce") - low) / (high - low)
    return normalized.clip(lower=0.0, upper=1.0)


def _write_knee_parity_summary(metrics_long: pd.DataFrame, out_dir: Path) -> None:
    if metrics_long.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = metrics_long[
        metrics_long.get("is_bidirectional", False)
        & metrics_long.get("knee_x", pd.Series(dtype=float)).notna()
        & metrics_long.get("parity_pre", pd.Series(dtype=float)).notna()
        & metrics_long.get("parity_post", pd.Series(dtype=float)).notna()
    ]
    total = int(len(subset))
    if total == 0:
        summary = pd.DataFrame(
            [
                {
                    "n_total": 0,
                    "n_flip": 0,
                    "fraction_flip": np.nan,
                    "n_lock_after": 0,
                    "fraction_lock_after": np.nan,
                    "pipeline_version": PIPELINE_VERSION,
                }
            ]
        )
    else:
        flips = int((subset["parity_pre"] != subset["parity_post"]).sum())
        locks = int((subset["parity_post"] == subset.get("parity_bit")).sum())
        summary = pd.DataFrame(
            [
                {
                    "n_total": total,
                    "n_flip": flips,
                    "fraction_flip": float(flips / total),
                    "n_lock_after": locks,
                    "fraction_lock_after": float(locks / total),
                    "pipeline_version": PIPELINE_VERSION,
                }
            ]
        )
    out_path = out_dir / "knee_parity_summary.csv"
    summary.to_csv(out_path, index=False)


def _should_skip(metric, available: set[str], skip_counts: dict[str, int], skip_missing: dict[str, set[str]]) -> bool:
    missing = metric.requires - available
    if not missing:
        return False
    skip_counts[metric.name] = skip_counts.get(metric.name, 0) + 1
    skip_missing.setdefault(metric.name, set()).update(missing)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract metrics from JJ datasets.")
    parser.add_argument("--in", dest="in_dir", default="data/raw", help="Input data folder")
    parser.add_argument("--out", dest="out_dir", default="results/metrics", help="Output folder")
    parser.add_argument("--glob", default="**/*.dat", help="Glob pattern for data files")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--pairing-csv", default=None, help="CSV with pair_id, file_left, file_right")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(args.seed)
    setup_logging("results/reports")

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    segments_dir = Path("results/parsed/segments")
    segments_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_files(in_dir, args.glob)
    if args.max_files:
        files = files[: args.max_files]

    rows = []
    skip_counts: dict[str, int] = {metric.name: 0 for metric in METRICS}
    skip_missing: dict[str, set[str]] = {metric.name: set() for metric in METRICS}

    pairing_csv = args.pairing_csv or config.get("pairing", {}).get("csv")
    pairs, file_to_pair = _load_pairing_map(pairing_csv)

    for path in tqdm(files, desc="Metrics"):
        df_raw, meta = read_dat(path)
        if df_raw.empty:
            LOGGER.warning("No data parsed for %s", path)
            continue
        file_id = make_file_id(path, in_dir)
        meta["file_id"] = file_id
        run_id = str(meta.get("run_id") or file_id)

        df_std = standardize_columns(df_raw, meta)
        df_std = coerce_types(df_std)
        df_std = apply_unit_conversions(df_std, meta, config)
        df_std = infer_phase_radians(df_std, meta, config)

        capabilities = detect_capabilities(df_std)
        pack = build_signal_pack(df_std, meta)

        x_name, x_values = infer_primary_axis(df_std, meta)
        axis_info = axis_properties(x_values)
        x_finite = x_values[np.isfinite(x_values)]
        has_direction_meta = _has_direction_meta(meta)
        is_bidirectional = bool(x_finite.size and np.nanmin(x_finite) < 0 and np.nanmax(x_finite) > 0)
        if has_direction_meta:
            is_bidirectional = True

        is_cyclic = axis_info.get("is_cyclic", False)
        is_time_series = bool(x_name == "t_s" and axis_info.get("is_monotonic", False) and not is_cyclic)

        data = pd.DataFrame({"x": x_values})
        if "V_V" in df_std.columns:
            data["V_V"] = pd.to_numeric(df_std["V_V"], errors="coerce").to_numpy()
        else:
            data["V_V"] = np.nan

        extra_cols = ["I_A", "B_T", "phi_rad", "gate_V", "gate1_V", "gate2_V", "t_s"]
        for col in extra_cols:
            if col in df_std.columns:
                data[col] = pd.to_numeric(df_std[col], errors="coerce").to_numpy()

        if capabilities.get("cap_has_V"):
            mask = np.isfinite(data["x"]) & np.isfinite(data["V_V"])
        else:
            mask = np.isfinite(data["x"])

        data_clean = data.loc[mask].reset_index(drop=True)
        x_clean = data_clean["x"].to_numpy()
        v_clean = data_clean["V_V"].to_numpy()

        if capabilities.get("cap_has_V"):
            segments = split_segments(x_clean, v_clean, min_points=config.get("segments", {}).get("min_points", 5))
        else:
            segments = []

        if not segments:
            segments = [
                {
                    "segment_id": 0,
                    "x": x_clean,
                    "V": v_clean,
                    "direction": "unknown",
                    "indices": np.arange(x_clean.size),
                }
            ]

        n_segments = len(segments) if segments else 1

        segment_frames = []
        for seg in segments:
            idx = seg.get("indices", np.arange(data_clean.shape[0]))
            seg_df = data_clean.iloc[idx].copy()
            seg_df["file_id"] = file_id
            seg_df["run_id"] = run_id
            seg_df["segment_id"] = seg["segment_id"]
            seg_df["segment_dir"] = seg["direction"]
            seg_df["x_name"] = x_name
            segment_frames.append(seg_df)

        if segment_frames:
            segment_out = pd.concat(segment_frames, ignore_index=True)
            segment_out.to_parquet(segments_dir / f"{file_id}.parquet", index=False)

        file_signals = {"x": x_values}
        if pack.has("V"):
            file_signals["V"] = pack.signals["V"].to_numpy()
        if pack.has("I"):
            file_signals["I"] = pack.signals["I"].to_numpy()
        if pack.has("phi"):
            file_signals["phi"] = pack.signals["phi"].to_numpy()
        if pack.has("B"):
            file_signals["B"] = pack.signals["B"].to_numpy()
        if pack.has("gate"):
            file_signals["gate"] = pack.signals["gate"].to_numpy()
        if pack.has("t"):
            file_signals["t"] = pack.signals["t"].to_numpy()

        file_ctx = {
            "signals": file_signals,
            "segments": segments,
            "config": config,
            "is_time_series": is_time_series,
        }

        file_metrics = {}
        available_file = set(file_signals.keys())
        for metric in METRICS:
            if metric.level != "file":
                continue
            if _should_skip(metric, available_file, skip_counts, skip_missing):
                continue
            result = run_metric(metric, available_file, file_ctx)
            file_metrics.update(result)

        eps = float(config.get("metrics", {}).get("eps", 1e-12))
        grid_n = int(config.get("parity", {}).get("grid_n", 200))

        for seg in segments:
            seg_x = seg["x"]
            seg_v = seg["V"]
            seg_signals = {"x": seg_x, "V": seg_v}
            seg_ctx = {
                "signals": seg_signals,
                "segments": segments,
                "config": config,
                "is_time_series": is_time_series,
            }
            available_seg = set(seg_signals.keys())

            seg_metrics = {}
            for metric in METRICS:
                if metric.level != "segment":
                    continue
                if _should_skip(metric, available_seg, skip_counts, skip_missing):
                    continue
                result = run_metric(metric, available_seg, seg_ctx)
                seg_metrics.update(result)

            row = {
                "file_id": file_id,
                "run_id": run_id,
                "segment_id": seg["segment_id"],
                "segment_dir": seg["direction"],
                "x_name": x_name,
                "axis_primary": x_name,
                "n_points": int(seg["x"].size),
                "is_bidirectional": bool(is_bidirectional),
                "is_cyclic": bool(is_cyclic),
                "n_segments": int(n_segments),
                "is_time_series": bool(is_time_series),
                "has_direction_meta": bool(has_direction_meta),
                "source_file": meta.get("source_file"),
                "pipeline_version": PIPELINE_VERSION,
            }
            row.update(capabilities)
            row.update(file_metrics)
            row.update(seg_metrics)

            x_min = float(np.nanmin(seg_x)) if np.isfinite(seg_x).any() else np.nan
            x_max = float(np.nanmax(seg_x)) if np.isfinite(seg_x).any() else np.nan
            span = x_max - x_min if np.isfinite(x_max) and np.isfinite(x_min) else np.nan
            row["x_min"] = x_min
            row["x_max"] = x_max
            knee_val = row.get("knee_x")
            if knee_val is not None and np.isfinite(knee_val) and np.isfinite(span) and span > 0:
                row["knee_x_value"] = float(knee_val)
                row["knee_x_norm"] = float((knee_val - x_min) / span)
            else:
                row["knee_x_value"] = np.nan
                row["knee_x_norm"] = np.nan

            knee_norm = row.get("knee_x_norm")
            if knee_norm is not None and np.isfinite(knee_norm) and 0.0 <= knee_norm <= 1.0:
                row["knee_valid"] = True
                edge_frac = float(config.get("knee", {}).get("conf_edge_frac", 0.05))
                row["knee_edge_flag"] = bool(knee_norm <= edge_frac or knee_norm >= 1.0 - edge_frac)
            else:
                row["knee_valid"] = False
                row["knee_edge_flag"] = False

            row["knee_confident"] = _knee_confident(
                row.get("knee_x"),
                row.get("knee_slope_pre"),
                row.get("knee_slope_post"),
                seg_x,
                config,
            )

            eta_valid = bool(row.get("eta_valid")) if row.get("eta_valid") is not None else False
            row["eta_valid"] = eta_valid if row.get("is_bidirectional") else False

            if "eta_V_signed" in row and row.get("eta_valid"):
                parity_bit = _parity_bit(row.get("eta_V_signed"))
                if parity_bit is not None:
                    row["parity_bit"] = parity_bit

            if row.get("is_bidirectional") and row.get("eta_valid") and row.get("knee_valid"):
                knee_x = float(row["knee_x_value"]) if np.isfinite(row.get("knee_x_value", np.nan)) else np.nan
                if np.isfinite(knee_x):
                    x_abs = np.abs(seg_x)
                    knee_thresh = abs(knee_x)
                    pre_mask = x_abs < knee_thresh
                    post_mask = x_abs > knee_thresh
                    pre = compute_voltage_nonreciprocity(
                        seg_x[pre_mask],
                        seg_v[pre_mask],
                        grid_n=grid_n,
                        eps=eps,
                        min_overlap=float(config.get("parity", {}).get("min_overlap", 0.8)),
                        den_min=float(config.get("parity", {}).get("den_min", 0.0)),
                    )
                    post = compute_voltage_nonreciprocity(
                        seg_x[post_mask],
                        seg_v[post_mask],
                        grid_n=grid_n,
                        eps=eps,
                        min_overlap=float(config.get("parity", {}).get("min_overlap", 0.8)),
                        den_min=float(config.get("parity", {}).get("den_min", 0.0)),
                    )
                    if pre.get("eta_valid"):
                        parity_pre = _parity_bit(pre.get("eta_V_signed"))
                        if parity_pre is not None:
                            row["parity_pre"] = parity_pre
                    if post.get("eta_valid"):
                        parity_post = _parity_bit(post.get("eta_V_signed"))
                        if parity_post is not None:
                            row["parity_post"] = parity_post
                    if row.get("parity_pre") is not None and row.get("parity_post") is not None:
                        row["parity_flip_at_knee"] = bool(row.get("parity_pre") != row.get("parity_post"))

            if file_to_pair:
                row["pair_id"] = file_to_pair.get(file_id)

            rows.append(row)

    metrics_long = pd.DataFrame(rows)
    if metrics_long.empty:
        metrics_long = pd.DataFrame(columns=["file_id"])
    else:
        if "H_incoh_raw" in metrics_long.columns:
            metrics_long["H_incoh"] = _normalize_incoherence(metrics_long["H_incoh_raw"], config)

        eps = float(config.get("metrics", {}).get("eps", 1e-12))
        if "eta_valid" in metrics_long.columns:
            metrics_long["eta_valid"] = metrics_long["eta_valid"].fillna(False)
        else:
            metrics_long["eta_valid"] = False

        if "eta_V_L1" in metrics_long.columns and "H_incoh" in metrics_long.columns:
            eta_mask = metrics_long["eta_valid"] & metrics_long["H_incoh"].notna()
            metrics_long.loc[eta_mask, "eta_norm"] = metrics_long.loc[eta_mask, "eta_V_L1"] / (
                eps + metrics_long.loc[eta_mask, "H_incoh"]
            )
        if "K_max" in metrics_long.columns and "H_incoh" in metrics_long.columns:
            metrics_long["knee_score_norm"] = metrics_long["K_max"] / (eps + metrics_long["H_incoh"])

        if "knee_score_norm" in metrics_long.columns:
            rank_pct = float(config.get("knee", {}).get("rank_cap_percentile", 99.0))
            ks = pd.to_numeric(metrics_long["knee_score_norm"], errors="coerce").dropna()
            if not ks.empty:
                kcap = float(np.nanpercentile(ks, rank_pct))
                metrics_long["knee_score_cap"] = kcap
                metrics_long["knee_score_capped"] = (
                    pd.to_numeric(metrics_long["knee_score_norm"], errors="coerce") >= kcap
                )
                metrics_long["knee_score_rank"] = np.log1p(
                    np.minimum(pd.to_numeric(metrics_long["knee_score_norm"], errors="coerce"), kcap)
                )

        if "parity_bit" in metrics_long.columns:
            stability = metrics_long[metrics_long["eta_valid"]].groupby("file_id")["parity_bit"].apply(_parity_stability)
            metrics_long["parity_stability"] = metrics_long["file_id"].map(stability)

        _write_knee_parity_summary(metrics_long, Path("results/reports"))

        for name, count in sorted(skip_counts.items()):
            if count <= 0:
                continue
            missing = ",".join(sorted(skip_missing.get(name, set()))) or "unknown"
            LOGGER.info("Metric %s skipped for %d rows (missing: %s)", name, count, missing)

    long_path = out_dir / "metrics_long.csv"
    metrics_long.to_csv(long_path, index=False)

    wide_path = out_dir / "metrics_wide.csv"
    metrics_long.to_csv(wide_path, index=False)

    if pairs:
        pair_df = _pair_metrics(metrics_long, pairs)
        pair_df.to_csv(out_dir / "pairs.csv", index=False)

    LOGGER.info("Wrote metrics to %s", long_path)


if __name__ == "__main__":
    main()
