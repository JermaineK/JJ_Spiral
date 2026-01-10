"""Phase 7 structured significance analysis."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
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
import jj_phase6 as phase6
import jj_phase55 as phase55

LOGGER = logging.getLogger("jjparse.phase7")

DEVICE_RE = re.compile(r"(?:^|[_-])(BS\d+_[A-Z0-9]+S\d+)", re.IGNORECASE)
FAMILY_RE = re.compile(r"(?i)(fig\s*\d+|iv[_-]?data|squid[_-]?data|fraunhofer|cpr|iv|sweep)")
TEMP_RE = re.compile(r"(?i)(?:temp|t)[_\-]*([+-]?\d+(?:p\d+)?(?:\.\d+)?(?:e[-+]?\d+)?)(m?K)")
TEMP_FALLBACK_RE = re.compile(r"(?i)([+-]?\d+(?:p\d+)?(?:\.\d+)?(?:e[-+]?\d+)?)(m?K)")
COIL_RE = re.compile(
    r"(?i)(?:i[_-]?coil|coil|bcoil|ibias)[_\-]*([+-]?\d+(?:p\d+)?(?:\.\d+)?(?:e[-+]?\d+)?)(mA|uA|A)"
)
BRACKET_RE = re.compile(r"\[([+-]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\]")
VGAIN_RE = re.compile(r"(?i)v[_-]?gain[_-]*([+-]?\d+(?:p\d+)?(?:\.\d+)?(?:e[-+]?\d+)?)")
IGAIN_RE = re.compile(r"(?i)i[_-]?gain[_-]*([+-]?\d+(?:p\d+)?(?:\.\d+)?(?:e[-+]?\d+)?)")
FRQ_RE = re.compile(r"(?i)(?:frq|freq|frequency)[_\-]*([+-]?\d+(?:p\d+)?(?:\.\d+)?(?:e[-+]?\d+)?)([kmg]?)")
AVG_RE = re.compile(r"(?i)avg[_-]*([+-]?\d+(?:p\d+)?(?:\.\d+)?(?:e[-+]?\d+)?)")
VPP_RE = re.compile(r"(?i)vpp[_-]*([+-]?\d+(?:p\d+)?(?:\.\d+)?(?:e[-+]?\d+)?)")
RUN_RE = re.compile(r"\b\d{6,14}\b")


@dataclass
class BranchData:
    file_id: str
    branch_id: int
    device_id: str | None
    experiment_family: str | None
    temp_K: float | None
    coil_mA: float | None
    b_proxy: float | None
    direction: str
    i_vals: np.ndarray
    v_vals: np.ndarray
    i_abs_grid: np.ndarray
    v_pos_grid: np.ndarray
    v_neg_grid: np.ndarray
    knee: float | None
    odd_ratio_post: float | None
    odd_energy_ratio: float | None
    alignment_score: float | None


def _parse_number_token(token: str) -> float | None:
    cleaned = token.strip()
    if "p" in cleaned and "e" not in cleaned.lower():
        cleaned = cleaned.replace("p", ".", 1)
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_experiment_family(path: Path) -> str | None:
    parts = [part for part in path.parts if part]
    for part in parts:
        match = FAMILY_RE.search(part)
        if match:
            return match.group(1).replace(" ", "")
    return None


def _parse_temperature(text: str) -> float | None:
    match = TEMP_RE.search(text)
    if match:
        value = _parse_number_token(match.group(1))
        unit = match.group(2).lower()
        if value is None:
            return None
        return value * (1e-3 if unit == "mk" else 1.0)
    match = TEMP_FALLBACK_RE.search(text)
    if match:
        value = _parse_number_token(match.group(1))
        unit = match.group(2).lower()
        if value is None:
            return None
        return value * (1e-3 if unit == "mk" else 1.0)
    return None


def _parse_coil(text: str) -> float | None:
    match = COIL_RE.search(text)
    if match:
        value = _parse_number_token(match.group(1))
        unit = match.group(2).lower()
        if value is None:
            return None
        if unit == "a":
            return value * 1e3
        if unit == "ua":
            return value * 1e-3
        return value
    if "coil" in text.lower():
        bracket = BRACKET_RE.search(text)
        if bracket:
            value = _parse_number_token(bracket.group(1))
            return value if value is not None else None
    return None


def _parse_gain(pattern: re.Pattern, text: str) -> float | None:
    match = pattern.search(text)
    if not match:
        return None
    return _parse_number_token(match.group(1))


def _parse_freq(text: str) -> float | None:
    match = FRQ_RE.search(text)
    if not match:
        return None
    value = _parse_number_token(match.group(1))
    if value is None:
        return None
    scale = {"k": 1e3, "m": 1e6, "g": 1e9}.get(match.group(2).lower(), 1.0)
    return value * scale

def _extract_metadata(path: Path) -> dict:
    text = path.as_posix()
    meta: dict[str, object] = {}

    device = DEVICE_RE.search(text)
    if device:
        meta["device_id"] = device.group(1)

    family = _extract_experiment_family(path)
    if family:
        meta["experiment_family"] = family

    temp = _parse_temperature(text)
    if temp is not None:
        meta["T_K"] = temp

    coil = _parse_coil(text)
    if coil is not None:
        meta["I_coil_mA"] = coil
    else:
        bracket = BRACKET_RE.search(text)
        if bracket:
            value = _parse_number_token(bracket.group(1))
            if value is not None:
                meta["B_proxy"] = value

    vgain = _parse_gain(VGAIN_RE, text)
    if vgain is not None:
        meta["Vgain"] = vgain

    igain = _parse_gain(IGAIN_RE, text)
    if igain is not None:
        meta["Igain"] = igain

    frq = _parse_freq(text)
    if frq is not None:
        meta["FRQ_Hz"] = frq

    avg = _parse_gain(AVG_RE, text)
    if avg is not None:
        meta["AVG"] = avg

    vpp = _parse_gain(VPP_RE, text)
    if vpp is not None:
        meta["Vpp"] = vpp

    run_match = RUN_RE.search(path.name)
    if run_match:
        meta["run_id"] = run_match.group(0)

    return meta


def _normalize_pattern(name: str) -> str:
    text = re.sub(r"\d", "#", name)
    text = re.sub(r"#+", "#", text)
    text = re.sub(r"#([a-zA-Z])", r"#\1", text)
    return text


def _spacing_outlier_flag(spacing_cv: float | None, threshold: float) -> bool:
    if spacing_cv is None or not np.isfinite(spacing_cv):
        return False
    return spacing_cv > threshold


def _trimmed_median(values: pd.Series, trim_frac: float) -> float | None:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if vals.size == 0:
        return None
    vals = np.sort(vals)
    trim = int(math.floor(trim_frac * vals.size))
    if trim > 0:
        vals = vals[trim:-trim]
    if vals.size == 0:
        return None
    return float(np.nanmedian(vals))


def _alignment_components(
    i_abs: np.ndarray,
    v_pos: np.ndarray,
    v_neg: np.ndarray,
    v_even: np.ndarray,
    v_odd: np.ndarray,
    knee: float | None,
    config: dict,
) -> dict:
    eps = float(config.get("phase5", {}).get("eps", 1e-12))
    align_cfg = config.get("phase7", {}).get("alignment", {})
    width = float(align_cfg.get("window_log_width", 0.15))
    step = float(align_cfg.get("window_log_step", 0.075))
    min_points = int(align_cfg.get("window_min_points", 6))

    loc_ratio = None
    log_knee = None
    if knee is not None and np.isfinite(knee) and knee > 0:
        log_knee = math.log(knee)

    log_i = np.log(i_abs[i_abs > 0])
    centers, indices = phase55._window_indices(log_i, width, step, min_points)
    odd_energy, flatness, corr_pm = phase55._window_metrics(v_even, v_odd, v_pos, v_neg, indices, eps)

    if log_knee is not None and centers.size:
        pre = odd_energy[centers < log_knee]
        post = odd_energy[centers > log_knee]
        if pre.size and post.size:
            loc_ratio = float(np.nanmedian(post) / (np.nanmedian(pre) + eps))

    return {
        "odd_energy_ratio": loc_ratio,
        "odd_energy_post": float(np.nanmedian(odd_energy[centers > log_knee]))
        if log_knee is not None and (centers > log_knee).any()
        else None,
        "odd_energy_pre": float(np.nanmedian(odd_energy[centers < log_knee]))
        if log_knee is not None and (centers < log_knee).any()
        else None,
        "odd_flatness": float(np.nanmedian(flatness)) if flatness.size else None,
        "corr_pm": float(np.nanmedian(corr_pm)) if corr_pm.size else None,
    }


def _coherence_stability(
    i_vals: np.ndarray,
    v_vals: np.ndarray,
    base_grid_n: int,
    min_overlap: float,
    config: dict,
) -> float | None:
    factors = [0.7, 1.0, 1.3]
    ratios = []
    for factor in factors:
        grid_n = max(50, int(round(base_grid_n * factor)))
        interp = phase6._interp_even_odd_method(i_vals, v_vals, grid_n, min_overlap, "linear")
        if not interp.get("parity_valid"):
            continue
        v_pos = interp["v_pos_grid"]
        v_neg = interp["v_neg_grid"]
        v_even = 0.5 * (v_pos + v_neg)
        v_odd = 0.5 * (v_pos - v_neg)
        knee = phase5._knee_loglog(interp["i_abs_grid"], 0.5 * (np.abs(v_pos) + np.abs(v_neg)), config)
        parity = phase5._parity_pre_post(interp["i_abs_grid"], v_even, v_odd, knee.get("I_knee"), config)
        ratio = parity.get("odd_ratio_post")
        if ratio is not None and np.isfinite(ratio):
            ratios.append(float(ratio))
    if len(ratios) < 2:
        return None
    mean = float(np.mean(ratios))
    if mean == 0:
        return None
    cv = float(np.std(ratios) / mean)
    return float(1.0 / (1.0 + cv))


def _normalize_series(values: pd.Series, config: dict) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    norm_cfg = config.get("phase7", {}).get("alignment", {}).get("normalize", {})
    p_low = float(norm_cfg.get("p_low", 5.0))
    p_high = float(norm_cfg.get("p_high", 95.0))
    if vals.dropna().empty:
        return vals
    low = float(np.nanpercentile(vals, p_low))
    high = float(np.nanpercentile(vals, p_high))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return vals
    return ((vals - low) / (high - low)).clip(0.0, 1.0)


def _compute_flatness_r2(df: pd.DataFrame, group_field: str, metric: str) -> float | None:
    subset = df[[group_field, metric]].copy()
    subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
    subset = subset.dropna()
    if subset.empty:
        return None
    overall_mean = float(subset[metric].mean())
    total_var = float(subset[metric].var(ddof=0))
    if total_var <= 0:
        return None
    grouped = subset.groupby(group_field)
    between = 0.0
    total_n = len(subset)
    for _, group in grouped:
        mean = float(group[metric].mean())
        between += len(group) * (mean - overall_mean) ** 2
    between /= max(total_n, 1)
    return float(between / total_var)


def _direction_null(df: pd.DataFrame, n_iter: int, rng: np.random.Generator) -> dict:
    grouped = df.groupby("file_id")
    obs_diffs = []
    file_groups = []
    for _, group in grouped:
        up = pd.to_numeric(group.loc[group["direction"] == "up", "odd_ratio_post"], errors="coerce")
        down = pd.to_numeric(group.loc[group["direction"] == "down", "odd_ratio_post"], errors="coerce")
        if up.dropna().empty or down.dropna().empty:
            continue
        obs_diffs.append(float(np.nanmedian(up) - np.nanmedian(down)))
        file_groups.append(group)
    if not obs_diffs:
        return {
            "status": "insufficient_data",
            "observed": None,
            "null_mean": None,
            "null_std": None,
            "z_score": None,
            "p_empirical": None,
            "n": 0,
        }
    obs_stat = float(np.nanmedian(np.abs(obs_diffs)))

    null_stats = []
    for _ in range(n_iter):
        diffs = []
        for group in file_groups:
            shuffled = group.copy()
            shuffled["direction"] = rng.permutation(shuffled["direction"].values)
            up = pd.to_numeric(shuffled.loc[shuffled["direction"] == "up", "odd_ratio_post"], errors="coerce")
            down = pd.to_numeric(shuffled.loc[shuffled["direction"] == "down", "odd_ratio_post"], errors="coerce")
            if up.dropna().empty or down.dropna().empty:
                continue
            diffs.append(float(np.nanmedian(up) - np.nanmedian(down)))
        if diffs:
            null_stats.append(float(np.nanmedian(np.abs(diffs))))
    null_stats = np.asarray(null_stats, dtype=float)
    if null_stats.size == 0:
        return {
            "status": "null_failed",
            "observed": obs_stat,
            "null_mean": None,
            "null_std": None,
            "z_score": None,
            "p_empirical": None,
            "n": 0,
        }
    mean = float(np.nanmean(null_stats))
    std = float(np.nanstd(null_stats, ddof=1)) if null_stats.size > 1 else 0.0
    z = float((obs_stat - mean) / std) if std > 0 else None
    p = float((np.sum(null_stats >= obs_stat) + 1) / (null_stats.size + 1))
    return {
        "observed": obs_stat,
        "null_mean": mean,
        "null_std": std,
        "z_score": z,
        "p_empirical": p,
        "n": int(null_stats.size),
    }

def _pairing_null(
    df: pd.DataFrame,
    branch_data: list[BranchData],
    n_iter: int,
    sample_n: int,
    rng: np.random.Generator,
    config: dict,
) -> dict:
    parity_df = df[df["parity_valid"]].copy()
    obs = pd.to_numeric(parity_df.get("odd_ratio_post"), errors="coerce").dropna()
    if obs.empty:
        return {}
    obs_stat = float(np.nanmedian(obs))

    groups: dict[str, list[BranchData]] = {}
    for item in branch_data:
        key_parts = [item.device_id or "unknown", item.experiment_family or "unknown"]
        if item.temp_K is not None:
            key_parts.append(f"temp{item.temp_K:.4g}")
        key = "|".join(key_parts)
        groups.setdefault(key, []).append(item)

    eligible = [items for items in groups.values() if len(items) >= 2]
    if not eligible:
        return {}

    null_stats = []
    grid_n = int(config.get("phase7", {}).get("grid_n", 401))

    for _ in range(n_iter):
        ratios = []
        for _ in range(sample_n):
            group = eligible[int(rng.integers(0, len(eligible)))]
            a, b = rng.choice(group, size=2, replace=False)
            i_max = min(a.i_abs_grid[-1], b.i_abs_grid[-1])
            if not np.isfinite(i_max) or i_max <= 0:
                continue
            i_abs = np.linspace(0.0, i_max, grid_n)
            v_pos = np.interp(i_abs, a.i_abs_grid, a.v_pos_grid)
            v_neg = np.interp(i_abs, b.i_abs_grid, b.v_neg_grid)
            v_even = 0.5 * (v_pos + v_neg)
            v_odd = 0.5 * (v_pos - v_neg)
            knee = phase5._knee_loglog(i_abs, 0.5 * (np.abs(v_pos) + np.abs(v_neg)), config)
            parity = phase5._parity_pre_post(i_abs, v_even, v_odd, knee.get("I_knee"), config)
            ratio = parity.get("odd_ratio_post")
            if ratio is not None and np.isfinite(ratio):
                ratios.append(float(ratio))
        if ratios:
            null_stats.append(float(np.nanmedian(ratios)))

    null_stats = np.asarray(null_stats, dtype=float)
    if null_stats.size == 0:
        return {}
    mean = float(np.nanmean(null_stats))
    std = float(np.nanstd(null_stats, ddof=1)) if null_stats.size > 1 else 0.0
    z = float((obs_stat - mean) / std) if std > 0 else None
    p = float((np.sum(null_stats >= obs_stat) + 1) / (null_stats.size + 1))
    return {
        "status": "ok",
        "observed": obs_stat,
        "null_mean": mean,
        "null_std": std,
        "z_score": z,
        "p_empirical": p,
        "n": int(null_stats.size),
    }


def _apply_warp(
    i_vals: np.ndarray,
    v_vals: np.ndarray,
    warp: str,
    config: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    i_vals = np.asarray(i_vals, dtype=float)
    v_vals = np.asarray(v_vals, dtype=float)
    span = float(np.nanmax(i_vals) - np.nanmin(i_vals))
    if not np.isfinite(span) or span == 0:
        return i_vals, v_vals

    nuisance_cfg = config.get("phase7", {}).get("nuisance", {})
    if warp == "piecewise_linear":
        slope_bound = float(nuisance_cfg.get("piecewise_slope", 0.15))
        knot = rng.uniform(0.35, 0.65) * span
        slope1 = 1.0 + rng.uniform(-slope_bound, slope_bound)
        slope2 = 1.0 + rng.uniform(-slope_bound, slope_bound)
        abs_i = np.abs(i_vals)
        warped = np.where(abs_i <= knot, slope1 * abs_i, slope1 * knot + slope2 * (abs_i - knot))
        i_prime = np.sign(i_vals) * warped
        return i_prime, v_vals
    if warp == "offset":
        offset = float(nuisance_cfg.get("offset_frac", 0.02)) * span
        shift = rng.uniform(-offset, offset)
        return i_vals + shift, v_vals
    if warp == "loop_squash":
        c = float(nuisance_cfg.get("loop_squash", 0.1))
        i_max = float(np.nanmax(np.abs(i_vals)))
        if i_max <= 0:
            return i_vals, v_vals
        warp_term = c * (np.abs(i_vals) / i_max) ** 2 * i_max
        return i_vals + np.sign(i_vals) * warp_term, v_vals
    if warp == "v_drift":
        v_offset = float(nuisance_cfg.get("v_offset_frac", 0.05)) * float(np.nanmax(np.abs(v_vals)) or 0.0)
        slope = rng.uniform(-v_offset, v_offset) / span if span > 0 else 0.0
        return i_vals, v_vals + rng.uniform(-v_offset, v_offset) + slope * i_vals
    return i_vals, v_vals

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 structured significance analysis.")
    parser.add_argument("--in", dest="in_dir", default="Data_for_analysis/D4.2", help="Input folder")
    parser.add_argument("--glob", default="**/*", help="Glob for input files")
    parser.add_argument("--out", default="results/phase7", help="Output folder")
    parser.add_argument("--config", default="config.yaml", help="Config path")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--skip-nulls", action="store_true", help="Skip null model evaluations")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(args.seed)
    setup_logging("results/reports", name="jjparse.phase7")

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cond_dir = out_dir / "condition_tables"
    cond_dir.mkdir(parents=True, exist_ok=True)
    null_dir = out_dir / "nulls"
    null_dir.mkdir(parents=True, exist_ok=True)

    phase7_cfg = config.get("phase7", {})
    grid_n = int(phase7_cfg.get("grid_n", config.get("phase5", {}).get("grid_n", 401)))
    min_overlap = float(phase7_cfg.get("min_overlap", config.get("phase5", {}).get("min_overlap", 0.8)))

    branch_cfg = phase7_cfg.get("branch", {})
    branch_window = int(branch_cfg.get("smooth_window", 11))
    branch_poly = int(branch_cfg.get("smooth_polyorder", 3))
    branch_thresh = float(branch_cfg.get("diff_threshold", 0.1))
    branch_min_points = int(branch_cfg.get("min_points", 12))

    spacing_thresh = float(phase7_cfg.get("spacing_cv_warn", config.get("phase5", {}).get("spacing_cv_warn", 0.2)))

    bins_cfg = phase7_cfg.get("bins", {})
    temp_bins = int(bins_cfg.get("temp_bins", 4))
    coil_bins = int(bins_cfg.get("coil_bins", 4))
    min_bin_n = int(bins_cfg.get("min_bin_n", 8))
    trim_frac = float(bins_cfg.get("trim_frac", 0.1))

    null_cfg = phase7_cfg.get("nulls", {})
    null_seed = int(null_cfg.get("seed", 123))
    direction_n = int(null_cfg.get("direction_n", 200))
    pairing_n = int(null_cfg.get("pairing_n", 200))
    pairing_samples = int(null_cfg.get("pairing_samples", 200))
    nuisance_samples = int(null_cfg.get("nuisance_samples", 200))

    rng = np.random.default_rng(args.seed)
    rng_null = np.random.default_rng(null_seed)

    metadata_rows = []
    branch_rows = []
    branches_rows = []
    branch_data: list[BranchData] = []
    unparsed_patterns: dict[str, int] = {}

    files = phase5._iter_files(in_dir, args.glob)
    if args.max_files:
        files = files[: args.max_files]
    LOGGER.info("Phase7 reading %d files from %s", len(files), in_dir)

    for path in files:
        i_vals, v_vals, _ = phase5._parse_iv_file(path)
        if i_vals.size < 3 or v_vals.size < 3:
            continue
        file_id = phase5._make_file_id(path, in_dir)
        meta = _extract_metadata(path)
        spacing_cv = phase5._spacing_cv(i_vals)
        spacing_flag = _spacing_outlier_flag(spacing_cv, spacing_thresh)

        missing_key = (
            not meta.get("device_id")
            or (meta.get("T_K") is None and meta.get("I_coil_mA") is None and meta.get("B_proxy") is None)
        )
        if missing_key:
            pattern = _normalize_pattern(path.name)
            unparsed_patterns[pattern] = unparsed_patterns.get(pattern, 0) + 1

        metadata_rows.append(
            {
                "file_id": file_id,
                "n_points": int(i_vals.size),
                "I_min": float(np.nanmin(i_vals)),
                "I_max": float(np.nanmax(i_vals)),
                "I_range": float(np.nanmax(i_vals) - np.nanmin(i_vals)),
                "spacing_cv": spacing_cv,
                "spacing_outlier_flag": spacing_flag,
                "x_name": "I_A",
                "pipeline_version": PIPELINE_VERSION,
                **meta,
            }
        )

        branches = phase6._split_branches(i_vals, branch_min_points, branch_window, branch_poly, branch_thresh)
        for branch in branches:
            idx = branch["indices"]
            seg_i = i_vals[idx]
            seg_v = v_vals[idx]
            mask = np.isfinite(seg_i) & np.isfinite(seg_v)
            seg_i = seg_i[mask]
            seg_v = seg_v[mask]
            if seg_i.size < branch_min_points:
                continue

            branch_id = int(branch["branch_id"])
            direction = branch.get("direction", "unknown")
            monotonicity = phase6._branch_monotonicity(seg_i)

            interp = phase6._interp_even_odd_method(seg_i, seg_v, grid_n, min_overlap, "linear")
            parity_valid = bool(interp.get("parity_valid", False))
            f_overlap = interp.get("f_overlap") if parity_valid else None

            branches_rows.append(
                {
                    "file_id": file_id,
                    "branch_id": branch_id,
                    "direction": direction,
                    "n_points": int(seg_i.size),
                    "monotonicity": monotonicity,
                    "overlap_fraction": f_overlap,
                }
            )

            row = {
                "file_id": file_id,
                "branch_id": branch_id,
                "direction": direction,
                "n_points": int(seg_i.size),
                "monotonicity": monotonicity,
                "parity_valid": parity_valid,
                "overlap_fraction": f_overlap,
                "pipeline_version": PIPELINE_VERSION,
                "x_name": "I_A",
            }
            row.update(meta)

            if parity_valid:
                i_abs = interp["i_abs_grid"]
                v_pos = interp["v_pos_grid"]
                v_neg = interp["v_neg_grid"]
                v_even = 0.5 * (v_pos + v_neg)
                v_odd = 0.5 * (v_pos - v_neg)
                v_mag = 0.5 * (np.abs(v_pos) + np.abs(v_neg))

                odd_ratio, odd_energy = phase5._odd_ratios(v_even, v_odd, float(config.get("phase5", {}).get("eps", 1e-12)))
                knee = phase5._knee_loglog(i_abs, v_mag, config)
                parity = phase5._parity_pre_post(i_abs, v_even, v_odd, knee.get("I_knee"), config)
                phi0 = phase5._phi0_proxy(i_abs, v_even, v_odd, float(config.get("phase5", {}).get("eps", 1e-12)))

                align = _alignment_components(i_abs, v_pos, v_neg, v_even, v_odd, knee.get("I_knee"), config)
                stability = _coherence_stability(seg_i, seg_v, grid_n, min_overlap, config)

                row.update(
                    {
                        "odd_ratio_L1": odd_ratio,
                        "odd_energy_L2": odd_energy,
                        "odd_ratio_post": parity.get("odd_ratio_post"),
                        "odd_ratio_pre": parity.get("odd_ratio_pre"),
                        "parity_lock_after_knee": parity.get("parity_lock_flag"),
                        "phi0_proxy": phi0.get("phi0_proxy"),
                        "phi0_confidence": phi0.get("phi0_confidence"),
                        "I_knee": knee.get("I_knee"),
                        "I_knee_norm": knee.get("I_knee_norm"),
                        "knee_score": knee.get("knee_score"),
                        "knee_valid": knee.get("knee_valid"),
                        "knee_edge_flag": knee.get("knee_edge_flag"),
                        "odd_energy_ratio": align.get("odd_energy_ratio"),
                        "odd_energy_post": align.get("odd_energy_post"),
                        "odd_energy_pre": align.get("odd_energy_pre"),
                        "odd_flatness": align.get("odd_flatness"),
                        "corr_pm": align.get("corr_pm"),
                        "coherence_stability": stability,
                        "noise_flatness": spectral_flatness(v_odd) if v_odd.size > 0 else None,
                    }
                )

                branch_data.append(
                    BranchData(
                        file_id=file_id,
                        branch_id=branch_id,
                        device_id=meta.get("device_id"),
                        experiment_family=meta.get("experiment_family"),
                        temp_K=meta.get("T_K"),
                        coil_mA=meta.get("I_coil_mA"),
                        b_proxy=meta.get("B_proxy"),
                        direction=direction,
                        i_vals=seg_i,
                        v_vals=seg_v,
                        i_abs_grid=i_abs,
                        v_pos_grid=v_pos,
                        v_neg_grid=v_neg,
                        knee=knee.get("I_knee"),
                        odd_ratio_post=parity.get("odd_ratio_post"),
                        odd_energy_ratio=align.get("odd_energy_ratio"),
                        alignment_score=None,
                    )
                )

            branch_rows.append(row)

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(out_dir / "metadata.csv", index=False)

    branches_df = pd.DataFrame(branches_rows)
    branches_df.to_csv(out_dir / "branches.csv", index=False)

    branch_df = pd.DataFrame(branch_rows)
    if branch_df.empty:
        LOGGER.warning("No branch metrics generated.")
        return

    branch_df["odd_energy_ratio_log"] = np.log1p(pd.to_numeric(branch_df.get("odd_energy_ratio"), errors="coerce"))
    branch_df["odd_energy_ratio_norm"] = _normalize_series(branch_df["odd_energy_ratio_log"], config)
    branch_df["knee_score_norm"] = _normalize_series(branch_df["knee_score"], config)
    branch_df["alignment_score"] = (
        branch_df[["odd_energy_ratio_norm", "knee_score_norm", "coherence_stability"]]
        .astype(float)
        .mean(axis=1, skipna=True)
    )

    branch_df.to_csv(out_dir / "branch_metrics.csv", index=False)

    coverage_rows = []
    for field in [
        "device_id",
        "experiment_family",
        "T_K",
        "I_coil_mA",
        "B_proxy",
        "Vgain",
        "Igain",
        "FRQ_Hz",
        "AVG",
        "Vpp",
        "run_id",
    ]:
        if field in metadata_df.columns:
            coverage = float(metadata_df[field].notna().mean())
        else:
            coverage = 0.0
        coverage_rows.append({"field": field, "coverage": coverage})
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(out_dir / "metadata_coverage.csv", index=False)

    if unparsed_patterns:
        patterns_df = pd.DataFrame(
            sorted(unparsed_patterns.items(), key=lambda item: item[1], reverse=True)[:20],
            columns=["pattern", "count"],
        )
        patterns_df.to_csv(out_dir / "unparsed_patterns.csv", index=False)

    if "T_K" in branch_df.columns:
        temp_bins_series = pd.qcut(branch_df["T_K"], q=temp_bins, duplicates="drop")
        branch_df["temp_bin"] = temp_bins_series.astype(str)
    if "I_coil_mA" in branch_df.columns or "B_proxy" in branch_df.columns:
        coil_source = branch_df["I_coil_mA"] if "I_coil_mA" in branch_df.columns else None
        if coil_source is None or coil_source.isna().all():
            coil_source = branch_df.get("B_proxy")
        if coil_source is not None:
            coil_bins_series = pd.qcut(coil_source, q=coil_bins, duplicates="drop")
            branch_df["coil_bin"] = coil_bins_series.astype(str)

    metrics = ["odd_ratio_post", "phi0_proxy", "I_knee_norm", "alignment_score", "parity_lock_after_knee"]
    group_fields = [
        "device_id",
        "experiment_family",
        "temp_bin",
        "coil_bin",
        "x_name",
    ]

    top_bins_rows = []
    flatness_rows = []
    for field in group_fields:
        if field not in branch_df.columns:
            continue
        group_df = branch_df[branch_df[field].notna()].copy()
        if group_df.empty:
            continue
        summaries = []
        for value, group in group_df.groupby(field):
            if len(group) < min_bin_n:
                continue
            row = {"group_field": field, "bin": str(value), "n": int(len(group))}
            for metric in metrics:
                series = pd.to_numeric(group.get(metric), errors="coerce")
                if series.dropna().empty:
                    continue
                if metric == "phi0_proxy":
                    series = np.abs(series)
                row[f"{metric}_median"] = float(np.nanmedian(series))
                trimmed = _trimmed_median(series, trim_frac)
                if trimmed is not None:
                    row[f"{metric}_trimmed"] = trimmed
            summaries.append(row)
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(cond_dir / f"{field}.csv", index=False)

        if not summary_df.empty:
            if "odd_ratio_post_median" in summary_df.columns:
                top = summary_df.sort_values("odd_ratio_post_median", ascending=False).head(5)
                for _, row in top.iterrows():
                    top_bins_rows.append(
                        {
                            "group_field": field,
                            "metric": "odd_ratio_post",
                            "bin": row["bin"],
                            "n": row["n"],
                            "median": row["odd_ratio_post_median"],
                        }
                    )
            if "alignment_score_median" in summary_df.columns:
                top = summary_df.sort_values("alignment_score_median", ascending=False).head(5)
                for _, row in top.iterrows():
                    top_bins_rows.append(
                        {
                            "group_field": field,
                            "metric": "alignment_score",
                            "bin": row["bin"],
                            "n": row["n"],
                            "median": row["alignment_score_median"],
                        }
                    )

        for metric in metrics:
            if metric not in branch_df.columns:
                continue
            r2 = _compute_flatness_r2(branch_df, field, metric)
            if r2 is not None:
                flatness_rows.append({"group_field": field, "metric": metric, "r2": r2})

    if top_bins_rows:
        top_bins_df = pd.DataFrame(top_bins_rows)
        top_bins_df.to_csv(out_dir / "top_bins.csv", index=False)

    if flatness_rows:
        flatness_df = pd.DataFrame(flatness_rows)
        flatness_df.to_csv(out_dir / "flatness_report.csv", index=False)

    if not args.skip_nulls:
        direction_null = _direction_null(branch_df, direction_n, rng_null)
        if direction_null:
            pd.DataFrame([direction_null]).to_csv(null_dir / "branch_direction_null.csv", index=False)

        pairing_null = _pairing_null(branch_df, branch_data, pairing_n, pairing_samples, rng_null, config)
        if pairing_null:
            pd.DataFrame([pairing_null]).to_csv(null_dir / "pairing_null.csv", index=False)

        nuisance_rows = []
        if branch_data:
            sample = rng_null.choice(branch_data, size=min(nuisance_samples, len(branch_data)), replace=False)
            for warp in ["piecewise_linear", "offset", "loop_squash", "v_drift"]:
                deltas = []
                align_deltas = []
                for item in sample:
                    i_warp, v_warp = _apply_warp(item.i_vals, item.v_vals, warp, config, rng_null)
                    interp = phase6._interp_even_odd_method(i_warp, v_warp, grid_n, min_overlap, "linear")
                    if not interp.get("parity_valid"):
                        continue
                    v_pos = interp["v_pos_grid"]
                    v_neg = interp["v_neg_grid"]
                    v_even = 0.5 * (v_pos + v_neg)
                    v_odd = 0.5 * (v_pos - v_neg)
                    v_mag = 0.5 * (np.abs(v_pos) + np.abs(v_neg))
                    knee = phase5._knee_loglog(interp["i_abs_grid"], v_mag, config)
                    parity = phase5._parity_pre_post(interp["i_abs_grid"], v_even, v_odd, knee.get("I_knee"), config)
                    ratio = parity.get("odd_ratio_post")
                    if ratio is not None and item.odd_ratio_post is not None:
                        deltas.append(float(ratio - item.odd_ratio_post))

                    align = _alignment_components(
                        interp["i_abs_grid"],
                        v_pos,
                        v_neg,
                        v_even,
                        v_odd,
                        knee.get("I_knee"),
                        config,
                    )
                    loc_ratio = align.get("odd_energy_ratio")
                    if loc_ratio is not None and item.odd_energy_ratio is not None:
                        align_deltas.append(float(loc_ratio - item.odd_energy_ratio))

                if deltas:
                    nuisance_rows.append(
                        {
                            "warp": warp,
                            "metric": "odd_ratio_post",
                            "delta_median": float(np.nanmedian(deltas)),
                            "delta_iqr": float(np.nanpercentile(deltas, 75) - np.nanpercentile(deltas, 25)),
                            "n": int(len(deltas)),
                        }
                    )
                if align_deltas:
                    nuisance_rows.append(
                        {
                            "warp": warp,
                            "metric": "odd_energy_ratio",
                            "delta_median": float(np.nanmedian(align_deltas)),
                            "delta_iqr": float(np.nanpercentile(align_deltas, 75) - np.nanpercentile(align_deltas, 25)),
                            "n": int(len(align_deltas)),
                        }
                    )
        if nuisance_rows:
            pd.DataFrame(nuisance_rows).to_csv(null_dir / "nuisance_matrix.csv", index=False)

    nuisance_rows = []
    for file_id, group in branch_df.groupby("file_id"):
        file_meta = metadata_df.loc[metadata_df["file_id"] == file_id].iloc[0] if not metadata_df.empty else {}
        odd_post = pd.to_numeric(group.get("odd_ratio_post"), errors="coerce").dropna()
        knee_norm = pd.to_numeric(group.get("I_knee_norm"), errors="coerce").dropna()
        noise = pd.to_numeric(group.get("noise_flatness"), errors="coerce").dropna()
        alignment = pd.to_numeric(group.get("alignment_score"), errors="coerce").dropna()

        hysteresis = None
        up = group[group["direction"] == "up"]
        down = group[group["direction"] == "down"]
        if not up.empty and not down.empty:
            up_val = pd.to_numeric(up.get("odd_ratio_post"), errors="coerce").dropna()
            down_val = pd.to_numeric(down.get("odd_ratio_post"), errors="coerce").dropna()
            if not up_val.empty and not down_val.empty:
                hysteresis = float(np.nanmedian(np.abs(up_val - down_val)))

        nuisance_rows.append(
            {
                "file_id": file_id,
                "odd_ratio_post_median": float(np.nanmedian(odd_post)) if not odd_post.empty else None,
                "alignment_score_median": float(np.nanmedian(alignment)) if not alignment.empty else None,
                "knee_norm_span": float(np.nanmax(knee_norm) - np.nanmin(knee_norm)) if knee_norm.size else None,
                "noise_flatness_median": float(np.nanmedian(noise)) if not noise.empty else None,
                "hysteresis_proxy": hysteresis,
                "spacing_cv": file_meta.get("spacing_cv") if isinstance(file_meta, dict) else file_meta.get("spacing_cv"),
                "spacing_outlier_flag": file_meta.get("spacing_outlier_flag")
                if isinstance(file_meta, dict)
                else file_meta.get("spacing_outlier_flag"),
                "pipeline_version": PIPELINE_VERSION,
            }
        )

    nuisance_df = pd.DataFrame(nuisance_rows)
    nuisance_df.to_csv(out_dir / "nuisance_fingerprints.csv", index=False)

    report_lines = [
        "# Phase 7 Summary",
        "",
        f"- Pipeline version: {PIPELINE_VERSION}",
        f"- Files processed: {metadata_df['file_id'].nunique() if not metadata_df.empty else 0}",
        f"- Branches: {len(branch_df)}",
        "",
        "## Metadata coverage",
    ]
    for _, row in coverage_df.iterrows():
        report_lines.append(f"- {row['field']}: {row['coverage']:.3f}")

    if unparsed_patterns:
        report_lines.append("")
        report_lines.append("## Top unparsed patterns")
        patterns = sorted(unparsed_patterns.items(), key=lambda item: item[1], reverse=True)[:10]
        for pattern, count in patterns:
            report_lines.append(f"- {pattern}: {count}")

    report_lines.append("")
    report_lines.append("## Outputs")
    report_lines.append(f"- metadata.csv: {out_dir / 'metadata.csv'}")
    report_lines.append(f"- branches.csv: {out_dir / 'branches.csv'}")
    report_lines.append(f"- branch_metrics.csv: {out_dir / 'branch_metrics.csv'}")
    report_lines.append(f"- condition_tables/: {cond_dir}")
    report_lines.append(f"- nulls/: {null_dir}")

    report_path = Path("results/reports/phase7_summary.md")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "metadata": str(out_dir / "metadata.csv"),
        "branches": str(out_dir / "branches.csv"),
        "branch_metrics": str(out_dir / "branch_metrics.csv"),
        "condition_tables": str(cond_dir),
        "nulls": str(null_dir),
        "nuisance_fingerprints": str(out_dir / "nuisance_fingerprints.csv"),
    }
    (out_dir / "phase7_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
