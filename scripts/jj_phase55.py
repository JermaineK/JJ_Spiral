"""Phase 5.5 knee-conditioned coherence tests."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from jjparse.config import load_config, seed_everything, setup_logging
from jjparse.metrics import spectral_flatness
from jjparse.version import PIPELINE_VERSION

import jj_phase5 as phase5

LOGGER = logging.getLogger("jjparse.phase55")


@dataclass
class SegmentData:
    file_id: str
    segment_id: int
    condition_group: str
    condition_value: float | None
    dominant_direction: str
    i_abs: np.ndarray
    v_pos: np.ndarray
    v_neg: np.ndarray
    v_even: np.ndarray
    v_odd: np.ndarray
    window_centers: np.ndarray
    window_indices: list[np.ndarray]
    odd_energy: np.ndarray
    knee_value: float | None
    knee_valid: bool


def _dominant_direction(i_vals: np.ndarray, threshold: float) -> tuple[str, float, float]:
    diffs = np.diff(i_vals)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return "mixed", 0.0, 0.0
    pos_frac = float(np.mean(diffs > 0))
    neg_frac = float(np.mean(diffs < 0))
    if pos_frac >= threshold:
        return "increasing", pos_frac, neg_frac
    if neg_frac >= threshold:
        return "decreasing", pos_frac, neg_frac
    return "mixed", pos_frac, neg_frac


def _window_indices(log_i: np.ndarray, width: float, step: float, min_points: int) -> tuple[np.ndarray, list[np.ndarray]]:
    if log_i.size < min_points:
        return np.array([]), []
    start = float(np.nanmin(log_i))
    stop = float(np.nanmax(log_i) - width)
    if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
        return np.array([]), []
    centers: list[float] = []
    indices: list[np.ndarray] = []
    current = start
    while current <= stop + 1e-12:
        end = current + width
        idx = np.where((log_i >= current) & (log_i <= end))[0]
        if idx.size >= min_points:
            centers.append(0.5 * (current + end))
            indices.append(idx)
        current += step
    return np.asarray(centers, dtype=float), indices


def _window_metrics(
    v_even: np.ndarray,
    v_odd: np.ndarray,
    v_pos: np.ndarray,
    v_neg: np.ndarray,
    indices: list[np.ndarray],
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    odd_energy = []
    flatness = []
    corr_pm = []
    for idx in indices:
        v_odd_w = v_odd[idx]
        v_even_w = v_even[idx]
        denom = float(np.linalg.norm(v_even_w) + np.linalg.norm(v_odd_w) + eps)
        odd_energy.append(float(np.linalg.norm(v_odd_w) / denom) if denom > 0 else np.nan)

        flat = spectral_flatness(v_odd_w - np.nanmean(v_odd_w))
        flatness.append(float(flat) if flat is not None else np.nan)

        corr_val = np.nan
        if np.std(v_pos[idx]) > 0 and np.std(v_neg[idx]) > 0:
            corr_val = float(np.corrcoef(v_pos[idx], v_neg[idx])[0, 1])
        corr_pm.append(corr_val)

    return (
        np.asarray(odd_energy, dtype=float),
        np.asarray(flatness, dtype=float),
        np.asarray(corr_pm, dtype=float),
    )


def _coherence_gradients(
    centers: np.ndarray,
    odd_energy: np.ndarray,
    log_knee: float | None,
    min_windows: int,
) -> tuple[float | None, float | None, float | None, int, int]:
    mask = np.isfinite(centers) & np.isfinite(odd_energy)
    centers = centers[mask]
    odd_energy = odd_energy[mask]
    if log_knee is None or not np.isfinite(log_knee):
        return None, None, None, 0, 0
    if centers.size < min_windows:
        return None, None, None, 0, 0

    aligned = centers - log_knee
    pre_mask = aligned < 0
    post_mask = aligned > 0

    def _slope(x: np.ndarray, y: np.ndarray) -> float | None:
        if x.size < 2:
            return None
        try:
            coef = np.polyfit(x, y, 1)
        except (ValueError, np.linalg.LinAlgError):
            return None
        return float(coef[0])

    pre = _slope(centers[pre_mask], odd_energy[pre_mask]) if pre_mask.sum() >= 2 else None
    post = _slope(centers[post_mask], odd_energy[post_mask]) if post_mask.sum() >= 2 else None
    delta = None
    if pre is not None and post is not None:
        delta = float(post - pre)

    return pre, post, delta, int(pre_mask.sum()), int(post_mask.sum())


def _odd_ratio_post(i_abs: np.ndarray, v_even: np.ndarray, v_odd: np.ndarray, knee: float, eps: float) -> float | None:
    mask = i_abs > knee
    if not mask.any():
        return None
    denom = float(np.mean(np.abs(v_even[mask])) + eps)
    if denom <= 0:
        return None
    return float(np.mean(np.abs(v_odd[mask])) / denom)


def _knee_alignment_gain(
    i_abs: np.ndarray,
    v_even: np.ndarray,
    v_odd: np.ndarray,
    centers: np.ndarray,
    odd_energy: np.ndarray,
    knee: float | None,
    config: dict,
    rng: np.random.Generator,
    null_mode: bool = False,
) -> tuple[float | None, float | None, float | None, float | None]:
    if knee is None or not np.isfinite(knee):
        return None, None, None, None
    eps = float(config.get("phase55", {}).get("eps", 1e-12))
    log_knee = math.log(knee) if knee > 0 else None
    if log_knee is None:
        return None, None, None, None

    real_post = _odd_ratio_post(i_abs, v_even, v_odd, knee, eps)
    _, real_grad_post, _, _, _ = _coherence_gradients(
        centers, odd_energy, log_knee, min_windows=2
    )

    jitter_frac = float(config.get("phase55", {}).get("knee_jitter_frac", 0.2))
    jitter_n = int(config.get("phase55", {}).get("knee_jitter_n", 100))
    if null_mode:
        jitter_n = int(config.get("phase55", {}).get("knee_jitter_n_null", 25))

    i_min = float(np.nanmin(i_abs))
    i_max = float(np.nanmax(i_abs))
    span = i_max - i_min
    if not np.isfinite(span) or span <= 0:
        return None, None, real_post, real_grad_post

    jitter_post = []
    jitter_grad = []
    for _ in range(jitter_n):
        jitter = (rng.random() * 2.0 - 1.0) * jitter_frac * span
        knee_j = knee + jitter
        knee_j = float(np.clip(knee_j, i_min + 1e-12, i_max - 1e-12))
        post_val = _odd_ratio_post(i_abs, v_even, v_odd, knee_j, eps)
        _, grad_post, _, _, _ = _coherence_gradients(
            centers, odd_energy, math.log(knee_j), min_windows=2
        )
        if post_val is not None and np.isfinite(post_val):
            jitter_post.append(float(post_val))
        if grad_post is not None and np.isfinite(grad_post):
            jitter_grad.append(float(grad_post))

    gain_post = None
    gain_grad = None
    if real_post is not None and jitter_post:
        med = float(np.nanmedian(jitter_post))
        if med != 0:
            gain_post = float(real_post / med)
    if real_grad_post is not None and jitter_grad:
        med = float(np.nanmedian(jitter_grad))
        if med != 0:
            gain_grad = float(real_grad_post / med)

    return gain_post, gain_grad, real_post, real_grad_post


def _local_shuffle_odd(log_i: np.ndarray, v_odd: np.ndarray, bins: int, rng: np.random.Generator) -> np.ndarray:
    if bins <= 1 or log_i.size == 0:
        return v_odd.copy()
    edges = np.linspace(float(np.nanmin(log_i)), float(np.nanmax(log_i)), bins + 1)
    shuffled = v_odd.copy()
    for idx in range(bins):
        mask = (log_i >= edges[idx]) & (log_i < edges[idx + 1])
        if mask.sum() < 2:
            continue
        subset = shuffled[mask]
        rng.shuffle(subset)
        shuffled[mask] = subset
    return shuffled


def _directional_score(df: pd.DataFrame, metric: str) -> float | None:
    subset = df[["dominant_direction", metric]].dropna()
    if subset.empty:
        return None
    medians = subset.groupby("dominant_direction")[metric].median()
    if len(medians) < 2:
        return None
    return float(medians.max() - medians.min())


def _directional_permutation(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    n_iter: int,
    rng: np.random.Generator,
) -> tuple[float | None, float | None, float | None]:
    subset = df[[group_col, "dominant_direction", metric]].copy()
    subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
    subset = subset.dropna()
    if subset.empty:
        return None, None, None

    obs = _directional_score(subset, metric)
    if obs is None:
        return None, None, None

    scores = []
    for _ in range(n_iter):
        shuffled = []
        for _, grp in subset.groupby(group_col):
            labels = grp["dominant_direction"].to_numpy()
            rng.shuffle(labels)
            tmp = grp.copy()
            tmp["dominant_direction"] = labels
            shuffled.append(tmp)
        perm_df = pd.concat(shuffled, ignore_index=True)
        score = _directional_score(perm_df, metric)
        if score is not None:
            scores.append(score)
    if not scores:
        return obs, None, None
    scores_arr = np.asarray(scores, dtype=float)
    p = float((np.sum(scores_arr >= obs) + 1) / (scores_arr.size + 1))
    return obs, p, float(np.nanmedian(scores_arr))


def _summary_stats(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    mean = float(arr.mean())
    std = float(arr.std(ddof=1) if arr.size > 1 else 0.0)
    return mean, std


def _z_score(obs: float, mean: float | None, std: float | None) -> float | None:
    if mean is None or std is None or std == 0.0 or not np.isfinite(std):
        return None
    return float((obs - mean) / std)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5.5 knee-conditioned coherence tests.")
    parser.add_argument("--in", dest="in_dir", default="Data_for_analysis/D4.2", help="Input folder")
    parser.add_argument("--glob", default="**/*", help="Glob for input files")
    parser.add_argument("--out", default="results/phase55", help="Output folder")
    parser.add_argument("--config", default="config.yaml", help="Config path")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(args.seed)
    setup_logging("results/reports", name="jjparse.phase55")

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = phase5._iter_files(in_dir, args.glob)
    if args.max_files:
        files = files[: args.max_files]
    LOGGER.info("Phase55 reading %d files from %s", len(files), in_dir)

    grid_n = int(config.get("phase5", {}).get("grid_n", 401))
    min_overlap = float(config.get("phase5", {}).get("min_overlap", 0.8))
    min_points = int(config.get("phase5", {}).get("min_points", 10))

    win_width = float(config.get("phase55", {}).get("window_log_width", 0.15))
    win_step = float(config.get("phase55", {}).get("window_log_step", win_width / 2.0))
    win_min_points = int(config.get("phase55", {}).get("window_min_points", 6))
    direction_thresh = float(config.get("phase55", {}).get("direction_threshold", 0.6))
    eps = float(config.get("phase55", {}).get("eps", 1e-12))
    local_bins = int(config.get("phase55", {}).get("local_shuffle_bins", 20))

    rows = []
    segments: list[SegmentData] = []
    rng = np.random.default_rng(int(args.seed))

    for path in files:
        i_vals, v_vals, _ = phase5._parse_iv_file(path)
        if i_vals.size < 3 or v_vals.size < 3:
            continue

        file_id = phase5._make_file_id(path, in_dir)
        condition_group, condition_value = phase5._infer_condition(path, in_dir)

        segments_meta = phase5._split_segments(i_vals, v_vals, min_points)
        for seg in segments_meta:
            idx = seg["indices"]
            seg_i = i_vals[idx]
            seg_v = v_vals[idx]
            mask = np.isfinite(seg_i) & np.isfinite(seg_v)
            seg_i = seg_i[mask]
            seg_v = seg_v[mask]
            if seg_i.size < min_points:
                continue

            direction, pos_frac, neg_frac = _dominant_direction(seg_i, direction_thresh)

            interp = phase5._interp_even_odd(seg_i, seg_v, grid_n, min_overlap)
            parity_valid = bool(interp.get("parity_valid", False))

            row = {
                "file_id": file_id,
                "segment_id": int(seg["segment_id"]),
                "segment_dir": seg.get("direction", "unknown"),
                "condition_group": condition_group,
                "condition_value": condition_value,
                "pipeline_version": PIPELINE_VERSION,
                "I_min": float(np.nanmin(seg_i)),
                "I_max": float(np.nanmax(seg_i)),
                "N_points": int(seg_i.size),
                "dominant_direction": direction,
                "direction_pos_frac": pos_frac,
                "direction_neg_frac": neg_frac,
                "parity_valid": parity_valid,
                "parity_reason": interp.get("parity_reason"),
                "f_overlap": interp.get("f_overlap"),
            }

            if not parity_valid:
                rows.append(row)
                continue

            i_abs = interp["i_abs_grid"]
            v_pos = interp["v_pos_grid"]
            v_neg = interp["v_neg_grid"]
            mask_pos = i_abs > 0
            i_abs = i_abs[mask_pos]
            v_pos = v_pos[mask_pos]
            v_neg = v_neg[mask_pos]
            if i_abs.size < win_min_points:
                rows.append(row)
                continue

            v_even = 0.5 * (v_pos + v_neg)
            v_odd = 0.5 * (v_pos - v_neg)

            log_i = np.log(i_abs)
            centers, indices = _window_indices(log_i, win_width, win_step, win_min_points)
            odd_energy, flatness, corr_pm = _window_metrics(v_even, v_odd, v_pos, v_neg, indices, eps)

            knee = phase5._knee_loglog(i_abs, 0.5 * (np.abs(v_pos) + np.abs(v_neg)), config)
            knee_value = knee.get("I_knee") if knee else None
            knee_valid = bool(knee.get("knee_valid", False)) if knee else False
            log_knee = math.log(knee_value) if knee_value and knee_value > 0 else None

            grad_pre, grad_post, grad_delta, n_pre, n_post = _coherence_gradients(
                centers, odd_energy, log_knee, min_windows=2
            )

            pre_mask = centers < log_knee if log_knee is not None else np.array([], dtype=bool)
            post_mask = centers > log_knee if log_knee is not None else np.array([], dtype=bool)

            odd_energy_pre_med = float(np.nanmedian(odd_energy[pre_mask])) if pre_mask.any() else np.nan
            odd_energy_post_med = float(np.nanmedian(odd_energy[post_mask])) if post_mask.any() else np.nan
            flat_pre_med = float(np.nanmedian(flatness[pre_mask])) if pre_mask.any() else np.nan
            flat_post_med = float(np.nanmedian(flatness[post_mask])) if post_mask.any() else np.nan
            corr_pre_med = float(np.nanmedian(corr_pm[pre_mask])) if pre_mask.any() else np.nan
            corr_post_med = float(np.nanmedian(corr_pm[post_mask])) if post_mask.any() else np.nan

            odd_ratio_pre = np.nan
            odd_ratio_post = np.nan
            if knee_value is not None and np.isfinite(knee_value):
                pre_mask_i = i_abs < knee_value
                post_mask_i = i_abs > knee_value
                if pre_mask_i.any():
                    odd_ratio_pre = float(
                        np.mean(np.abs(v_odd[pre_mask_i])) / (np.mean(np.abs(v_even[pre_mask_i])) + eps)
                    )
                if post_mask_i.any():
                    odd_ratio_post = float(
                        np.mean(np.abs(v_odd[post_mask_i])) / (np.mean(np.abs(v_even[post_mask_i])) + eps)
                    )

            gain_post, gain_grad, real_post, real_grad = _knee_alignment_gain(
                i_abs,
                v_even,
                v_odd,
                centers,
                odd_energy,
                knee_value,
                config,
                rng,
                null_mode=False,
            )

            row.update(
                {
                    "odd_energy_L2": float(np.linalg.norm(v_odd) / (np.linalg.norm(v_even) + np.linalg.norm(v_odd) + eps)),
                    "odd_ratio_pre": odd_ratio_pre,
                    "odd_ratio_post": odd_ratio_post,
                    "I_knee": knee_value,
                    "I_knee_norm": knee.get("I_knee_norm") if knee else np.nan,
                    "knee_score": knee.get("knee_score") if knee else np.nan,
                    "knee_valid": knee_valid,
                    "knee_edge_flag": knee.get("knee_edge_flag") if knee else False,
                    "coherence_gradient_pre": grad_pre,
                    "coherence_gradient_post": grad_post,
                    "delta_coherence_gradient": grad_delta,
                    "odd_energy_pre_median": odd_energy_pre_med,
                    "odd_energy_post_median": odd_energy_post_med,
                    "odd_flatness_pre_median": flat_pre_med,
                    "odd_flatness_post_median": flat_post_med,
                    "corr_pm_pre_median": corr_pre_med,
                    "corr_pm_post_median": corr_post_med,
                    "window_count_pre": n_pre,
                    "window_count_post": n_post,
                    "knee_alignment_gain_odd_ratio_post": gain_post,
                    "knee_alignment_gain_coherence_post": gain_grad,
                }
            )

            rows.append(row)

            segments.append(
                SegmentData(
                    file_id=file_id,
                    segment_id=int(seg["segment_id"]),
                    condition_group=condition_group,
                    condition_value=condition_value,
                    dominant_direction=direction,
                    i_abs=i_abs,
                    v_pos=v_pos,
                    v_neg=v_neg,
                    v_even=v_even,
                    v_odd=v_odd,
                    window_centers=centers,
                    window_indices=indices,
                    odd_energy=odd_energy,
                    knee_value=knee_value,
                    knee_valid=knee_valid,
                )
            )

    metrics = pd.DataFrame(rows)
    metrics_path = out_dir / "phase55_metrics_long.csv"
    metrics.to_csv(metrics_path, index=False)

    # Directional dependence
    permute_n = int(config.get("phase55", {}).get("permute_n", 500))
    rng_perm = np.random.default_rng(int(config.get("phase55", {}).get("nulls", {}).get("seed", 123)) + 7)
    dir_rows = []
    for metric in ["I_knee_norm", "odd_ratio_post", "coherence_gradient_post"]:
        obs, pval, null_med = _directional_permutation(metrics, metric, "condition_group", permute_n, rng_perm)
        note = ""
        if obs is None:
            note = "insufficient class coverage"
        dir_rows.append(
            {
                "metric": metric,
                "directional_dependence_score": obs,
                "perm_p": pval,
                "perm_median": null_med,
                "note": note,
                "pipeline_version": PIPELINE_VERSION,
            }
        )
    dir_df = pd.DataFrame(dir_rows)
    dir_df.to_csv(out_dir / "phase55_directional_dependence.csv", index=False)

    # Summary metrics
    summary_rows = []
    for metric in [
        "delta_coherence_gradient",
        "knee_alignment_gain_odd_ratio_post",
        "knee_alignment_gain_coherence_post",
    ]:
        vals = pd.to_numeric(metrics.get(metric), errors="coerce").dropna()
        summary_rows.append(
            {
                "metric": metric,
                "median": float(np.nanmedian(vals)) if not vals.empty else np.nan,
                "n": int(len(vals)),
                "pipeline_version": PIPELINE_VERSION,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "phase55_metrics_summary.csv", index=False)

    # Nulls
    null_cfg = config.get("phase55", {}).get("nulls", {})
    phase_n = int(null_cfg.get("phase_n", 100))
    pair_n = int(null_cfg.get("pair_n", 100))
    local_n = int(null_cfg.get("local_shuffle_n", 100))
    null_seed = int(null_cfg.get("seed", 123))
    rng_null = np.random.default_rng(null_seed)

    sample_max = int(config.get("phase55", {}).get("null_sample_segments", 0))
    segments_for_null = segments
    if sample_max > 0 and len(segments) > sample_max:
        idx = rng_null.choice(len(segments), size=sample_max, replace=False)
        segments_for_null = [segments[i] for i in idx]

    # Pair-breaking candidates by i_max
    i_max = np.array([seg.i_abs.max() for seg in segments_for_null])
    if i_max.size == 0:
        LOGGER.warning("No parity-valid segments for nulls.")
        return
    bins = pd.qcut(i_max, q=min(10, len(i_max)), duplicates="drop")
    bin_map: dict[int, list[int]] = {}
    for idx, bin_id in enumerate(bins.codes):
        bin_map.setdefault(int(bin_id), []).append(idx)

    def _null_iteration(model: str) -> tuple[list[float], list[float], list[float]]:
        delta_vals = []
        gain_vals = []
        gain_grad_vals = []
        for idx, seg in enumerate(segments_for_null):
            v_pos = seg.v_pos
            v_neg = seg.v_neg
            if model == "phase_scramble":
                full_i, full_v = phase5._build_full_grid(seg.i_abs, v_pos, v_neg)
                scrambled = phase5._phase_scramble(full_v, rng_null)
                half = seg.i_abs.size
                neg_len = full_i.size - half
                v_neg_full = scrambled[:neg_len][::-1]
                v_pos = scrambled[-half:]
                if v_neg_full.size < half:
                    v_neg = np.concatenate([v_neg_full, [v_pos[0]]])
                else:
                    v_neg = v_neg_full[:half]
            elif model == "pair_breaking":
                bin_id = bins.codes[idx]
                candidates = [j for j in bin_map.get(int(bin_id), []) if j != idx]
                if not candidates:
                    continue
                other = segments_for_null[int(rng_null.choice(candidates))]
                i_max_common = min(seg.i_abs.max(), other.i_abs.max())
                if i_max_common <= 0:
                    continue
                i_abs = np.linspace(seg.i_abs.min(), i_max_common, seg.i_abs.size)
                v_pos = np.interp(i_abs, seg.i_abs, seg.v_pos)
                v_neg = np.interp(i_abs, other.i_abs, other.v_neg)
            elif model == "local_shuffle":
                log_i = np.log(seg.i_abs)
                v_odd = _local_shuffle_odd(log_i, seg.v_odd, local_bins, rng_null)
                v_even = seg.v_even
                v_pos = v_even + v_odd
                v_neg = v_even - v_odd

            v_even = 0.5 * (v_pos + v_neg)
            v_odd = 0.5 * (v_pos - v_neg)
            odd_energy, _, _ = _window_metrics(
                v_even, v_odd, v_pos, v_neg, seg.window_indices, eps
            )
            log_knee = math.log(seg.knee_value) if seg.knee_value and seg.knee_value > 0 else None
            _, _, delta, _, _ = _coherence_gradients(seg.window_centers, odd_energy, log_knee, min_windows=2)
            if delta is not None and np.isfinite(delta):
                delta_vals.append(float(delta))
            gain_post, gain_grad, _, _ = _knee_alignment_gain(
                seg.i_abs,
                v_even,
                v_odd,
                seg.window_centers,
                odd_energy,
                seg.knee_value,
                config,
                rng_null,
                null_mode=True,
            )
            if gain_post is not None and np.isfinite(gain_post):
                gain_vals.append(float(gain_post))
            if gain_grad is not None and np.isfinite(gain_grad):
                gain_grad_vals.append(float(gain_grad))
        return delta_vals, gain_vals, gain_grad_vals

    null_rows = []
    for model, n_iter in [
        ("phase_scramble", phase_n),
        ("pair_breaking", pair_n),
        ("local_shuffle", local_n),
    ]:
        delta_medians = []
        gain_medians = []
        gain_grad_medians = []
        for _ in range(n_iter):
            delta_vals, gain_vals, gain_grad_vals = _null_iteration(model)
            if delta_vals:
                delta_medians.append(float(np.nanmedian(delta_vals)))
            if gain_vals:
                gain_medians.append(float(np.nanmedian(gain_vals)))
            if gain_grad_vals:
                gain_grad_medians.append(float(np.nanmedian(gain_grad_vals)))

        obs_delta = float(np.nanmedian(pd.to_numeric(metrics.get("delta_coherence_gradient"), errors="coerce")))
        obs_gain = float(np.nanmedian(pd.to_numeric(metrics.get("knee_alignment_gain_odd_ratio_post"), errors="coerce")))
        obs_gain_grad = float(
            np.nanmedian(pd.to_numeric(metrics.get("knee_alignment_gain_coherence_post"), errors="coerce"))
        )

        mean, std = _summary_stats(delta_medians)
        null_rows.append(
            {
                "statistic": "deltaG",
                "null_model": model,
                "observed": obs_delta,
                "null_mean": mean,
                "null_std": std,
                "z_score": _z_score(obs_delta, mean, std),
                "n": int(len(delta_medians)),
                "pipeline_version": PIPELINE_VERSION,
            }
        )

        mean, std = _summary_stats(gain_medians)
        null_rows.append(
            {
                "statistic": "knee_alignment_gain",
                "null_model": model,
                "observed": obs_gain,
                "null_mean": mean,
                "null_std": std,
                "z_score": _z_score(obs_gain, mean, std),
                "n": int(len(gain_medians)),
                "pipeline_version": PIPELINE_VERSION,
            }
        )

        mean, std = _summary_stats(gain_grad_medians)
        null_rows.append(
            {
                "statistic": "knee_alignment_gain_coherence",
                "null_model": model,
                "observed": obs_gain_grad,
                "null_mean": mean,
                "null_std": std,
                "z_score": _z_score(obs_gain_grad, mean, std),
                "n": int(len(gain_grad_medians)),
                "pipeline_version": PIPELINE_VERSION,
            }
        )

    null_df = pd.DataFrame(null_rows)
    null_df.to_csv(out_dir / "phase55_null_summary.csv", index=False)

    # Report
    report_lines = []
    report_lines.append("# Phase 5.5 Summary")
    report_lines.append("")
    report_lines.append(f"- Pipeline version: {PIPELINE_VERSION}")
    report_lines.append(f"- Input folder: {in_dir}")
    report_lines.append(f"- Total segments: {len(metrics)}")
    report_lines.append(f"- Parity-valid segments: {int(metrics['parity_valid'].sum())}")
    report_lines.append("")

    report_lines.append("## Primary observables")
    report_lines.append("")
    for metric in ["delta_coherence_gradient", "knee_alignment_gain_odd_ratio_post", "knee_alignment_gain_coherence_post"]:
        vals = pd.to_numeric(metrics.get(metric), errors="coerce").dropna()
        if vals.empty:
            report_lines.append(f"- {metric}: n=0")
        else:
            report_lines.append(f"- {metric}: n={len(vals)}, median={float(np.nanmedian(vals)):.6g}")
    report_lines.append("")

    report_lines.append("## Directional sensitivity")
    report_lines.append("")
    for _, row in dir_df.iterrows():
        if pd.isna(row["directional_dependence_score"]):
            report_lines.append(f"- {row['metric']}: insufficient class coverage")
        else:
            report_lines.append(
                f"- {row['metric']}: score={row['directional_dependence_score']:.4g}, perm_p={row['perm_p']:.3g}"
            )
    report_lines.append("")

    report_lines.append("## Null comparisons")
    report_lines.append("")
    if null_df.empty:
        report_lines.append("- No null summaries available.")
    else:
        for _, row in null_df.iterrows():
            report_lines.append(
                f"- {row['statistic']} vs {row['null_model']}: observed={row['observed']:.4g}, "
                f"null_mean={row['null_mean']:.4g}, null_std={row['null_std']:.4g}, z={row['z_score']:.3g}"
            )
    report_lines.append("")

    report_lines.append("## Summary table")
    report_lines.append("")
    report_lines.append("Metric\tReal\tPhase-scramble\tPair-break")
    for stat in ["deltaG", "knee_alignment_gain", "knee_alignment_gain_coherence"]:
        real_val = null_df[null_df["statistic"] == stat]["observed"].iloc[0] if not null_df.empty else np.nan
        phase_val = null_df[(null_df["statistic"] == stat) & (null_df["null_model"] == "phase_scramble")][
            "null_mean"
        ]
        pair_val = null_df[(null_df["statistic"] == stat) & (null_df["null_model"] == "pair_breaking")][
            "null_mean"
        ]
        phase_out = float(phase_val.iloc[0]) if not phase_val.empty else np.nan
        pair_out = float(pair_val.iloc[0]) if not pair_val.empty else np.nan
        report_lines.append(f"{stat}\t{real_val:.4g}\t{phase_out:.4g}\t{pair_out:.4g}")

    report_lines.append("")
    report_lines.append("## Interpretation")
    report_lines.append("")
    report_lines.append(
        "- Focus is on knee-conditioned coherence change and alignment gain; raw odd_ratio is not treated as diagnostic."
    )
    report_lines.append(
        "- Nulls include phase scramble, pair-breaking, and local shuffle to separate organized structure from envelope effects."
    )

    report_path = Path("results/reports/phase55_summary.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
