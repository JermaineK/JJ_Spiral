
"""Phase 6 attribution and systematics kill-switches."""

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
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from jjparse.config import load_config, seed_everything, setup_logging
from jjparse.version import PIPELINE_VERSION

import jj_phase5 as phase5

LOGGER = logging.getLogger("jjparse.phase6")

TEMP_RE = re.compile(r"(?i)(?:temp|t)[_\\-]*([0-9]+(?:p[0-9]+)?(?:e[-+]?\\d+)?)(m?K)")
COIL_RE = re.compile(r"(?i)(?:i[_-]?coil|coil)[+_-]*([0-9]+(?:p[0-9]+)?(?:e[-+]?\\d+)?)([munp]?A|mA|uA)")
FREQ_RE = re.compile(r"(?i)(?:frq|freq|frequency)[_\\-]*([0-9]+(?:p[0-9]+)?(?:e[-+]?\\d+)?)([kmg]?)(?:hz)?")
VPP_RE = re.compile(r"(?i)vpp[_\\-]*([0-9]+(?:p[0-9]+)?(?:e[-+]?\\d+)?)")
SAMPLE_RE = re.compile(r"\b([A-Z]\\d+S\\d+)\b")
DATE_RE = re.compile(r"\b(\\d{6}|\\d{8})\b")


@dataclass
class BranchCache:
    file_id: str
    branch_id: int
    condition_group: str
    condition_value: float | None
    i_pos: np.ndarray
    v_pos: np.ndarray
    i_neg: np.ndarray
    v_neg: np.ndarray
    i_abs_grid: np.ndarray
    v_pos_grid: np.ndarray
    v_neg_grid: np.ndarray
    i_max_common: float
    i_knee: float | None
    knee_valid: bool


def _parse_number_token(token: str) -> float | None:
    cleaned = token.strip()
    if "p" in cleaned and "e" not in cleaned.lower():
        cleaned = cleaned.replace("p", ".", 1)
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_metadata(path: Path) -> dict:
    text = path.as_posix()
    meta: dict[str, object] = {}

    temp = TEMP_RE.search(text)
    if temp:
        value = _parse_number_token(temp.group(1))
        unit = temp.group(2).lower()
        if value is not None:
            meta["meta_temp_K"] = value * (1e-3 if unit == "mk" else 1.0)

    coil = COIL_RE.search(text)
    if coil:
        value = _parse_number_token(coil.group(1))
        unit = coil.group(2).lower()
        if value is not None:
            scale = 1.0
            if unit.startswith("m"):
                scale = 1e-3
            elif unit.startswith("u"):
                scale = 1e-6
            elif unit.startswith("n"):
                scale = 1e-9
            elif unit.startswith("p"):
                scale = 1e-12
            meta["meta_coil_A"] = value * scale

    freq = FREQ_RE.search(text)
    if freq:
        value = _parse_number_token(freq.group(1))
        if value is not None:
            scale = {"k": 1e3, "m": 1e6, "g": 1e9}.get(freq.group(2).lower(), 1.0)
            meta["meta_freq_Hz"] = value * scale

    vpp = VPP_RE.search(text)
    if vpp:
        value = _parse_number_token(vpp.group(1))
        if value is not None:
            meta["meta_vpp"] = value

    sample = SAMPLE_RE.search(text)
    if sample:
        meta["meta_sample_id"] = sample.group(1)

    date = DATE_RE.search(text)
    if date:
        meta["meta_date"] = date.group(1)

    return meta


def _smooth_series(values: np.ndarray, window: int, poly: int) -> np.ndarray:
    if window < 3 or values.size < window:
        return values
    if window % 2 == 0:
        window += 1
    poly = min(poly, window - 1)
    return savgol_filter(values, window_length=window, polyorder=poly, mode="interp")


def _split_branches(
    i_vals: np.ndarray,
    min_points: int,
    window: int,
    poly: int,
    diff_threshold: float,
) -> list[dict]:
    if i_vals.size < min_points:
        return [{"branch_id": 0, "indices": np.arange(i_vals.size), "direction": "unknown"}]

    smooth = _smooth_series(i_vals, window, poly)
    diffs = np.diff(smooth)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return [{"branch_id": 0, "indices": np.arange(i_vals.size), "direction": "unknown"}]

    scale = float(np.nanmedian(np.abs(diffs))) if np.isfinite(diffs).any() else 0.0
    thresh = diff_threshold * scale if scale > 0 else 0.0
    diffs_full = np.diff(smooth)
    signs = np.sign(diffs_full)
    if thresh > 0:
        signs[np.abs(diffs_full) < thresh] = 0.0
    signs = phase5._fill_zero_signs(signs)
    if signs.size == 0 or np.all(signs == 0):
        return [{"branch_id": 0, "indices": np.arange(i_vals.size), "direction": "unknown"}]

    boundaries = [0]
    for idx in range(1, len(signs)):
        if signs[idx] != signs[idx - 1]:
            boundaries.append(idx + 1)
    boundaries.append(len(i_vals))

    branches = []
    branch_id = 0
    for start, end in zip(boundaries, boundaries[1:]):
        if end - start < min_points:
            continue
        seg_i = i_vals[start:end]
        direction = "unknown"
        if np.nanmean(np.diff(seg_i)) > 0:
            direction = "up"
        elif np.nanmean(np.diff(seg_i)) < 0:
            direction = "down"
        branches.append({"branch_id": branch_id, "indices": np.arange(start, end), "direction": direction})
        branch_id += 1

    if not branches:
        branches = [{"branch_id": 0, "indices": np.arange(i_vals.size), "direction": "unknown"}]
    return branches


def _branch_monotonicity(i_vals: np.ndarray) -> float:
    if i_vals.size < 3:
        return 0.0
    diffs = np.diff(i_vals)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 0.0
    pos = float(np.mean(diffs > 0))
    neg = float(np.mean(diffs < 0))
    return max(pos, neg)


def _interp_series(x: np.ndarray, y: np.ndarray, grid: np.ndarray, method: str) -> np.ndarray:
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    if x_sorted.size == 0:
        return np.full_like(grid, np.nan, dtype=float)
    if np.unique(x_sorted).size < x_sorted.size:
        df = pd.DataFrame({"x": x_sorted, "y": y_sorted}).groupby("x", as_index=False).mean()
        x_sorted = df["x"].to_numpy()
        y_sorted = df["y"].to_numpy()
    if method == "pchip":
        interp = PchipInterpolator(x_sorted, y_sorted, extrapolate=False)
        return interp(grid)
    return np.interp(grid, x_sorted, y_sorted)


def _interp_even_odd_method(
    i_vals: np.ndarray,
    v_vals: np.ndarray,
    grid_n: int,
    min_overlap: float,
    method: str,
) -> dict:
    pos_mask = i_vals > 0
    neg_mask = i_vals < 0
    if not pos_mask.any() or not neg_mask.any():
        return {"parity_valid": False, "parity_reason": "missing_sign"}

    i_pos = i_vals[pos_mask]
    v_pos = v_vals[pos_mask]
    i_neg = np.abs(i_vals[neg_mask])
    v_neg = v_vals[neg_mask]

    i_min_pos = float(np.nanmin(i_pos))
    i_max_pos = float(np.nanmax(i_pos))
    i_min_neg = float(np.nanmin(i_neg))
    i_max_neg = float(np.nanmax(i_neg))
    union_min = min(i_min_pos, i_min_neg)
    union_max = max(i_max_pos, i_max_neg)
    overlap_min = max(i_min_pos, i_min_neg)
    overlap_max = min(i_max_pos, i_max_neg)
    union_range = union_max - union_min
    overlap_range = overlap_max - overlap_min
    if union_range <= 0 or overlap_range <= 0:
        return {"parity_valid": False, "parity_reason": "low_overlap", "f_overlap": 0.0}
    f_overlap = float(overlap_range / union_range)
    if f_overlap < min_overlap:
        return {"parity_valid": False, "parity_reason": "low_overlap", "f_overlap": f_overlap}

    i_max_common = overlap_max
    i_abs_grid = np.linspace(0.0, i_max_common, grid_n)
    v_pos_grid = _interp_series(i_pos, v_pos, i_abs_grid, method)
    v_neg_grid = _interp_series(i_neg, v_neg, i_abs_grid, method)

    if not np.isfinite(v_pos_grid).any() or not np.isfinite(v_neg_grid).any():
        return {"parity_valid": False, "parity_reason": "interp_failed", "f_overlap": f_overlap}

    return {
        "parity_valid": True,
        "parity_reason": "ok",
        "f_overlap": f_overlap,
        "i_abs_grid": i_abs_grid,
        "v_pos_grid": v_pos_grid,
        "v_neg_grid": v_neg_grid,
        "i_pos": i_pos,
        "v_pos": v_pos,
        "i_neg": i_neg,
        "v_neg": v_neg,
        "i_max_common": i_max_common,
    }


def _branch_metrics(
    seg_i: np.ndarray,
    seg_v: np.ndarray,
    grid_n: int,
    min_overlap: float,
    method: str,
    config: dict,
) -> tuple[dict, BranchCache | None]:
    interp = _interp_even_odd_method(seg_i, seg_v, grid_n, min_overlap, method)
    parity_valid = bool(interp.get("parity_valid", False))
    eps = float(config.get("phase5", {}).get("eps", 1e-12))
    sign_eps = float(config.get("phase5", {}).get("sign_eps", 1e-15))

    odd_ratio = np.nan
    odd_energy = np.nan
    odd_mode = np.nan
    odd_cons = np.nan
    odd_entropy = np.nan
    odd_sign_p = np.nan
    knee = {}
    phi0 = {}
    parity_knee = {}
    cache = None

    if parity_valid:
        i_abs = interp["i_abs_grid"]
        v_pos_grid = interp["v_pos_grid"]
        v_neg_grid = interp["v_neg_grid"]
        v_even = 0.5 * (v_pos_grid + v_neg_grid)
        v_odd = 0.5 * (v_pos_grid - v_neg_grid)
        v_mag = 0.5 * (np.abs(v_pos_grid) + np.abs(v_neg_grid))

        odd_ratio, odd_energy = phase5._odd_ratios(v_even, v_odd, eps)
        odd_mode, odd_cons, odd_entropy, odd_sign_p = phase5._odd_sign_stats(v_odd, sign_eps)
        knee = phase5._knee_loglog(i_abs, v_mag, config)
        phi0 = phase5._phi0_proxy(i_abs, v_even, v_odd, eps)
        parity_knee = phase5._parity_pre_post(i_abs, v_even, v_odd, knee.get("I_knee"), config)

        cache = BranchCache(
            file_id="",
            branch_id=0,
            condition_group="",
            condition_value=None,
            i_pos=interp["i_pos"],
            v_pos=interp["v_pos"],
            i_neg=interp["i_neg"],
            v_neg=interp["v_neg"],
            i_abs_grid=i_abs,
            v_pos_grid=v_pos_grid,
            v_neg_grid=v_neg_grid,
            i_max_common=float(interp["i_max_common"]),
            i_knee=knee.get("I_knee"),
            knee_valid=bool(knee.get("knee_valid", False)),
        )

    row = {
        "parity_valid": parity_valid,
        "parity_reason": interp.get("parity_reason"),
        "f_overlap": interp.get("f_overlap"),
        "I_max_common": interp.get("i_max_common"),
        "odd_ratio_L1": odd_ratio,
        "odd_energy_L2": odd_energy,
        "odd_sign_mode": odd_mode,
        "odd_sign_consistency": odd_cons,
        "odd_sign_entropy": odd_entropy,
        "odd_sign_pos_fraction": odd_sign_p,
    }
    row.update(knee)
    row.update(phi0)
    row.update(parity_knee)
    return row, cache


def _iqr(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25))


def _warp_i(i_vals: np.ndarray, model: str, a: float, b: float) -> np.ndarray:
    if model == "odd_kink":
        return (1.0 + a) * i_vals + b * np.sign(i_vals)
    return i_vals + a * (i_vals**3)


def _fit_warp_params(
    i_abs: np.ndarray,
    v_even: np.ndarray,
    model: str,
    config: dict,
    i_knee: float | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    mask = np.isfinite(i_abs) & np.isfinite(v_even) & (i_abs > 0) & (v_even > 0)
    if i_knee is not None and np.isfinite(i_knee):
        mask = mask & (i_abs < i_knee)
    if mask.sum() < 6:
        return None, None, None, None

    i_abs = i_abs[mask]
    v_even = v_even[mask]
    log_v = np.log(v_even)

    warp_cfg = config.get("phase6", {}).get("nuisance_warp", {})
    a_min = float(warp_cfg.get("a_min", -0.2))
    a_max = float(warp_cfg.get("a_max", 0.2))
    b_min = float(warp_cfg.get("b_min", 0.0))
    b_max_frac = float(warp_cfg.get("b_max_frac", 0.2))
    b_max = float(b_max_frac * np.nanmax(i_abs)) if np.isfinite(np.nanmax(i_abs)) else 0.0
    if model == "odd_kink" and b_max <= b_min:
        return None, None, None, None

    def residual(params: np.ndarray) -> np.ndarray:
        a = float(params[0])
        b = float(params[1]) if model == "odd_kink" else 0.0
        i_prime = _warp_i(i_abs, model, a, b)
        if np.any(i_prime <= 0) or not np.isfinite(i_prime).all():
            return np.full_like(log_v, 1e6)
        log_i = np.log(i_prime)
        A = np.vstack([log_i, np.ones_like(log_i)]).T
        coeff, _, _, _ = np.linalg.lstsq(A, log_v, rcond=None)
        pred = A @ coeff
        return log_v - pred

    if model == "odd_kink":
        x0 = np.array([0.0, 0.0], dtype=float)
        bounds = ([a_min, b_min], [a_max, b_max])
    else:
        x0 = np.array([0.0], dtype=float)
        bounds = ([a_min], [a_max])
    result = least_squares(residual, x0=x0, bounds=bounds)
    res = residual(result.x)
    rss = float(np.sum(res**2))
    tss = float(np.sum((log_v - np.nanmean(log_v)) ** 2))
    r2 = float(1.0 - rss / tss) if tss > 0 else np.nan
    a = float(result.x[0])
    b = float(result.x[1]) if model == "odd_kink" and result.x.size > 1 else 0.0
    return a, b, rss, r2


def _odd_metrics_after_warp(
    i_vals: np.ndarray,
    v_vals: np.ndarray,
    model: str,
    a: float,
    b: float,
    grid_n: int,
    min_overlap: float,
    method: str,
    config: dict,
) -> dict:
    i_warp = _warp_i(i_vals, model, a, b)
    interp = _interp_even_odd_method(i_warp, v_vals, grid_n, min_overlap, method)
    if not interp.get("parity_valid"):
        return {"parity_valid": False}
    i_abs = interp["i_abs_grid"]
    v_pos = interp["v_pos_grid"]
    v_neg = interp["v_neg_grid"]
    v_even = 0.5 * (v_pos + v_neg)
    v_odd = 0.5 * (v_pos - v_neg)
    v_mag = 0.5 * (np.abs(v_pos) + np.abs(v_neg))
    knee = phase5._knee_loglog(i_abs, v_mag, config)
    parity = phase5._parity_pre_post(i_abs, v_even, v_odd, knee.get("I_knee"), config)
    return {
        "parity_valid": True,
        "odd_ratio_post": parity.get("odd_ratio_post"),
        "odd_ratio_pre": parity.get("odd_ratio_pre"),
    }


def _collect_null_stats(stats_list: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    odd = []
    lock = []
    phi0 = []
    for item in stats_list:
        if item.get("odd_ratio_post") is not None:
            odd.append(item["odd_ratio_post"])
        if item.get("parity_lock_flag") is not None:
            lock.append(float(item["parity_lock_flag"]))
        if item.get("phi0_proxy") is not None:
            phi0.append(abs(float(item["phi0_proxy"])))
    return np.asarray(odd), np.asarray(lock), np.asarray(phi0)


def _summarize_null(obs_odd: float, obs_lock: float, obs_phi0: float, null_stats: list[list[dict]]) -> dict:
    t_odd = []
    t_lock = []
    t_phi0 = []
    for stats in null_stats:
        odd, lock, phi0 = _collect_null_stats(stats)
        if odd.size:
            t_odd.append(float(np.nanmedian(odd)))
        if lock.size:
            t_lock.append(float(np.nanmean(lock)))
        if phi0.size and np.isfinite(phi0).any():
            t_phi0.append(float(np.nanmedian(phi0)))
    t_odd = np.asarray(t_odd)
    t_lock = np.asarray(t_lock)
    t_phi0 = np.asarray(t_phi0)
    mean_odd, std_odd = phase5._compute_stats(t_odd)
    mean_lock, std_lock = phase5._compute_stats(t_lock)
    mean_phi0, std_phi0 = phase5._compute_stats(t_phi0)
    return {
        "odd_ratio_post_mean": mean_odd,
        "odd_ratio_post_std": std_odd,
        "odd_ratio_post_z": phase5._z_score(obs_odd, mean_odd, std_odd),
        "parity_lock_mean": mean_lock,
        "parity_lock_std": std_lock,
        "parity_lock_z": phase5._z_score(obs_lock, mean_lock, std_lock),
        "phi0_abs_mean": mean_phi0,
        "phi0_abs_std": std_phi0,
        "phi0_abs_z": phase5._z_score(obs_phi0, mean_phi0, std_phi0),
        "n_null": int(t_odd.size),
    }


def _bin_numeric(series: pd.Series, n_bins: int) -> pd.Series | None:
    series = pd.to_numeric(series, errors="coerce")
    series = series.dropna()
    if series.size < max(5, n_bins * 2):
        return None
    try:
        return pd.qcut(series, q=n_bins, duplicates="drop")
    except ValueError:
        return None


def _stratify_metrics(df: pd.DataFrame, field: str, min_group: int) -> pd.DataFrame:
    rows = []
    for value, group in df.groupby(field):
        if len(group) < min_group:
            continue
        rows.append(
            {
                "strat_field": field,
                "bin_label": str(value),
                "n_segments": int(len(group)),
                "odd_ratio_post_median": float(np.nanmedian(group.get("odd_ratio_post"))),
                "parity_lock_fraction": float(np.nanmean(group.get("parity_lock_flag"))),
                "phi0_proxy_abs_median": float(np.nanmedian(np.abs(group.get("phi0_proxy")))),
                "I_knee_norm_median": float(np.nanmedian(group.get("I_knee_norm"))),
            }
        )
    return pd.DataFrame(rows)


def _plot_stratified_bar(df: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if df.empty:
        return
    fig, ax = plt.subplots()
    ax.bar(df["bin_label"], df["odd_ratio_post_median"], color="#666666", alpha=0.8)
    ax.set_ylabel("odd_ratio_post median")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4, axis="y")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 attribution and systematics checks.")
    parser.add_argument("--in", dest="in_dir", default="Data_for_analysis/D4.2", help="Input folder")
    parser.add_argument("--glob", default="**/*", help="Glob for input files")
    parser.add_argument("--out", default="results/phase6", help="Output folder")
    parser.add_argument("--config", default="config.yaml", help="Config path")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--skip-sensitivity", action="store_true", help="Skip sensitivity grid")
    parser.add_argument("--skip-nuisance", action="store_true", help="Skip nuisance warp tests")
    parser.add_argument("--skip-stratify", action="store_true", help="Skip metadata stratification")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(args.seed)
    setup_logging("results/reports", name="jjparse.phase6")

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    branch_cfg = config.get("phase6", {}).get("branch", {})
    branch_window = int(branch_cfg.get("smooth_window", 11))
    branch_poly = int(branch_cfg.get("smooth_polyorder", 3))
    branch_thresh = float(branch_cfg.get("diff_threshold", 0.1))
    branch_min_points = int(branch_cfg.get("min_points", 12))

    base_grid_n = int(config.get("phase5", {}).get("grid_n", 401))
    base_overlap = float(config.get("phase5", {}).get("min_overlap", 0.8))
    base_method = "linear"

    sens_cfg = config.get("phase6", {}).get("sensitivity", {})
    sens_min_overlap = list(sens_cfg.get("min_overlap", [0.2, 0.35, 0.5, 0.65, 0.8]))
    sens_methods = list(sens_cfg.get("grid_method", ["linear", "pchip"]))
    sens_densities = list(sens_cfg.get("grid_density", ["low", "medium", "high"]))
    sens_grid_base = int(sens_cfg.get("grid_n_base", base_grid_n))
    robust_pct = float(sens_cfg.get("robust_pct", 0.1))
    sens_null_cfg = sens_cfg.get("nulls", {})

    def _grid_n_for_density(density: str) -> int:
        factor = {"low": 0.5, "medium": 1.0, "high": 2.0}.get(density, 1.0)
        return max(50, int(round(sens_grid_base * factor)))

    settings = []
    for overlap in sens_min_overlap:
        for method in sens_methods:
            for density in sens_densities:
                settings.append(
                    {
                        "min_overlap": float(overlap),
                        "grid_method": str(method),
                        "grid_density": str(density),
                        "grid_n": _grid_n_for_density(str(density)),
                    }
                )

    sens_track: dict[str, dict] = {}
    if not args.skip_sensitivity:
        for setting in settings:
            key = f"{setting['min_overlap']}_{setting['grid_method']}_{setting['grid_density']}"
            sens_track[key] = {
                "setting": setting,
                "odd_ratio_post": [],
                "parity_lock": [],
                "phi0_abs": [],
                "cache": [],
                "seen": 0,
            }

    branch_rows = []
    branch_qc = []
    branch_groups: dict[str, list[dict]] = {}
    warp_fit_rows = []
    warp_robust_rows = []

    files = phase5._iter_files(in_dir, args.glob)
    if args.max_files:
        files = files[: args.max_files]
    LOGGER.info("Phase6 reading %d files from %s", len(files), in_dir)

    for path in files:
        i_vals, v_vals, _ = phase5._parse_iv_file(path)
        if i_vals.size < 3 or v_vals.size < 3:
            continue
        file_id = phase5._make_file_id(path, in_dir)
        condition_group, condition_value = phase5._infer_condition(path, in_dir)
        meta = _extract_metadata(path)

        branches = _split_branches(i_vals, branch_min_points, branch_window, branch_poly, branch_thresh)
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
            branch_dir = branch.get("direction", "unknown")
            branch_mono = _branch_monotonicity(seg_i)
            branch_qc.append(
                {
                    "file_id": file_id,
                    "branch_id": branch_id,
                    "branch_dir": branch_dir,
                    "n_points": int(seg_i.size),
                    "monotonicity": branch_mono,
                }
            )

            metrics, cache = _branch_metrics(seg_i, seg_v, base_grid_n, base_overlap, base_method, config)
            if cache is not None:
                cache.file_id = file_id
                cache.branch_id = branch_id
                cache.condition_group = condition_group
                cache.condition_value = condition_value

            row = {
                "file_id": file_id,
                "branch_id": branch_id,
                "branch_dir": branch_dir,
                "condition_group": condition_group,
                "condition_value": condition_value,
                "pipeline_version": PIPELINE_VERSION,
                "I_min": float(np.nanmin(seg_i)),
                "I_max": float(np.nanmax(seg_i)),
                "N_points": int(seg_i.size),
            }
            row.update(meta)
            row.update(metrics)
            branch_rows.append(row)

            if metrics.get("parity_valid"):
                branch_groups.setdefault(file_id, []).append(row)

            if not args.skip_nuisance and metrics.get("parity_valid"):
                i_abs = cache.i_abs_grid if cache else None
                if i_abs is not None:
                    v_even = 0.5 * (cache.v_pos_grid + cache.v_neg_grid)
                    warp_cfg = config.get("phase6", {}).get("nuisance_warp", {})
                    for model in warp_cfg.get("models", ["odd_cubic", "odd_kink"]):
                        a, b, rss, r2 = _fit_warp_params(i_abs, v_even, model, config, cache.i_knee)
                        if a is None:
                            continue
                        fit_row = {
                            "file_id": file_id,
                            "branch_id": branch_id,
                            "model": model,
                            "a": a,
                            "b": b,
                            "fit_rss": rss,
                            "fit_r2": r2,
                            "pipeline_version": PIPELINE_VERSION,
                        }
                        warp_fit_rows.append(fit_row)

                        warped = _odd_metrics_after_warp(
                            seg_i,
                            seg_v,
                            model,
                            a,
                            b,
                            base_grid_n,
                            base_overlap,
                            base_method,
                            config,
                        )
                        odd_before = row.get("odd_ratio_post")
                        odd_after = warped.get("odd_ratio_post")
                        reduction = np.nan
                        if odd_before not in (None, np.nan) and odd_after not in (None, np.nan):
                            if np.isfinite(odd_before) and odd_before != 0:
                                reduction = float((odd_before - odd_after) / odd_before)
                        warp_robust_rows.append(
                            {
                                "file_id": file_id,
                                "branch_id": branch_id,
                                "model": model,
                                "odd_ratio_post_before": odd_before,
                                "odd_ratio_post_after": odd_after,
                                "odd_ratio_post_reduction": reduction,
                                "warp_magnitude": abs(a) + abs(b),
                                "pipeline_version": PIPELINE_VERSION,
                            }
                        )

            if not args.skip_sensitivity:
                for key, state in sens_track.items():
                    setting = state["setting"]
                    metrics_s, cache_s = _branch_metrics(
                        seg_i,
                        seg_v,
                        setting["grid_n"],
                        setting["min_overlap"],
                        setting["grid_method"],
                        config,
                    )
                    if metrics_s.get("odd_ratio_post") is not None:
                        state["odd_ratio_post"].append(metrics_s.get("odd_ratio_post"))
                    if metrics_s.get("parity_lock_flag") is not None:
                        state["parity_lock"].append(float(metrics_s.get("parity_lock_flag")))
                    if metrics_s.get("phi0_proxy") is not None:
                        state["phi0_abs"].append(abs(float(metrics_s.get("phi0_proxy"))))

                    if cache_s is None:
                        continue
                    sample_max = int(sens_null_cfg.get("sample_segments", 200))
                    if sample_max <= 0:
                        continue
                    state["seen"] += 1
                    if len(state["cache"]) < sample_max:
                        cache_s.file_id = file_id
                        cache_s.branch_id = branch_id
                        cache_s.condition_group = condition_group
                        cache_s.condition_value = condition_value
                        state["cache"].append(cache_s)
                    else:
                        rng = np.random.default_rng(int(sens_null_cfg.get("seed", 123)) + 17)
                        idx = int(rng.integers(0, state["seen"]))
                        if idx < sample_max:
                            cache_s.file_id = file_id
                            cache_s.branch_id = branch_id
                            cache_s.condition_group = condition_group
                            cache_s.condition_value = condition_value
                            state["cache"][idx] = cache_s

    branch_df = pd.DataFrame(branch_rows)
    if branch_df.empty:
        LOGGER.warning("No branch metrics generated.")
        return
    branch_df.to_csv(out_dir / "phase6_branch_metrics.csv", index=False)

    qc_df = pd.DataFrame(branch_qc)
    qc_df.to_csv(Path("results/reports") / "branch_split_qc.csv", index=False)

    consistency_rows = []
    for file_id, rows in branch_groups.items():
        df = pd.DataFrame(rows)
        odd_post = pd.to_numeric(df.get("odd_ratio_post"), errors="coerce").dropna()
        knee_norm = pd.to_numeric(df.get("I_knee_norm"), errors="coerce").dropna()
        lock_vals = pd.to_numeric(df.get("parity_lock_flag"), errors="coerce").dropna()
        odd_median = float(np.nanmedian(odd_post)) if not odd_post.empty else np.nan
        odd_iqr = _iqr(odd_post.to_numpy()) if not odd_post.empty else np.nan
        odd_span = float(np.nanmax(odd_post) - np.nanmin(odd_post)) if odd_post.size else np.nan
        knee_median = float(np.nanmedian(knee_norm)) if not knee_norm.empty else np.nan
        knee_span = float(np.nanmax(knee_norm) - np.nanmin(knee_norm)) if knee_norm.size else np.nan
        lock_fraction = float(np.nanmean(lock_vals)) if not lock_vals.empty else np.nan
        lock_consistency = np.nan
        if np.isfinite(lock_fraction):
            lock_consistency = float(max(lock_fraction, 1.0 - lock_fraction))
        consistency_rows.append(
            {
                "file_id": file_id,
                "n_branches": int(len(df)),
                "n_parity_valid": int(df["parity_valid"].sum()),
                "odd_ratio_post_branchsafe": odd_median,
                "odd_ratio_post_dispersion": odd_iqr,
                "delta_odd_ratio_post": odd_span,
                "I_knee_norm_branchsafe": knee_median,
                "delta_I_knee_norm": knee_span,
                "parity_lock_fraction": lock_fraction,
                "parity_lock_consistency": lock_consistency,
                "pipeline_version": PIPELINE_VERSION,
            }
        )

    consistency_df = pd.DataFrame(consistency_rows)
    consistency_df.to_csv(Path("results/reports") / "branch_consistency.csv", index=False)

    if not args.skip_nuisance and warp_fit_rows:
        warp_fit_df = pd.DataFrame(warp_fit_rows)
        warp_fit_df.to_csv(Path("results/reports") / "nuisance_warp_fit.csv", index=False)
    if not args.skip_nuisance and warp_robust_rows:
        warp_robust_df = pd.DataFrame(warp_robust_rows)
        warp_robust_df.to_csv(Path("results/reports") / "nuisance_robustness.csv", index=False)

    if not args.skip_sensitivity:
        sens_rows = []
        for key, state in sens_track.items():
            setting = state["setting"]
            odd_vals = np.asarray(state["odd_ratio_post"], dtype=float)
            lock_vals = np.asarray(state["parity_lock"], dtype=float)
            phi_vals = np.asarray(state["phi0_abs"], dtype=float)
            obs_odd = float(np.nanmedian(odd_vals)) if odd_vals.size else np.nan
            obs_lock = float(np.nanmean(lock_vals)) if lock_vals.size else np.nan
            obs_phi0 = float(np.nanmedian(phi_vals)) if phi_vals.size and np.isfinite(phi_vals).any() else np.nan

            cache = state["cache"]
            null_stats = {}
            if cache:
                null_seed = int(sens_null_cfg.get("seed", 123))
                n_sign = int(sens_null_cfg.get("sign_n", 50))
                n_phase = int(sens_null_cfg.get("phase_n", 50))
                n_pair = int(sens_null_cfg.get("pair_n", 50))
                pair_tol = float(sens_null_cfg.get("pair_match_tol", 0.2))
                rng_null = np.random.default_rng(null_seed)

                def _compute_from_arrays(i_abs, v_pos, v_neg) -> dict:
                    v_even = 0.5 * (v_pos + v_neg)
                    v_odd = 0.5 * (v_pos - v_neg)
                    v_mag = 0.5 * (np.abs(v_pos) + np.abs(v_neg))
                    knee = phase5._knee_loglog(i_abs, v_mag, config)
                    parity = phase5._parity_pre_post(i_abs, v_even, v_odd, knee.get("I_knee"), config)
                    phi0 = phase5._phi0_proxy(i_abs, v_even, v_odd, float(config.get("phase5", {}).get("eps", 1e-12)))
                    return {
                        "odd_ratio_post": parity.get("odd_ratio_post"),
                        "parity_lock_flag": parity.get("parity_lock_flag"),
                        "phi0_proxy": phi0.get("phi0_proxy"),
                    }

                sign_null = []
                for _ in range(n_sign):
                    stats = []
                    for seg in cache:
                        sign = rng_null.choice([-1.0, 1.0])
                        stats.append(_compute_from_arrays(seg.i_abs_grid, seg.v_pos_grid * sign, seg.v_neg_grid * sign))
                    sign_null.append(stats)

                phase_null = []
                for _ in range(n_phase):
                    stats = []
                    for seg in cache:
                        full_i, full_v = phase5._build_full_grid(seg.i_abs_grid, seg.v_pos_grid, seg.v_neg_grid)
                        scrambled = phase5._phase_scramble(full_v, rng_null)
                        half = seg.i_abs_grid.size
                        neg_len = full_i.size - half
                        v_neg_full = scrambled[:neg_len][::-1]
                        v_pos = scrambled[-half:]
                        if v_neg_full.size < half:
                            v_neg = np.concatenate([v_neg_full, [v_pos[0]]])
                        else:
                            v_neg = v_neg_full[:half]
                        stats.append(_compute_from_arrays(seg.i_abs_grid, v_pos, v_neg))
                    phase_null.append(stats)

                bins = pd.qcut(
                    [seg.i_max_common for seg in cache],
                    q=min(10, len(cache)),
                    duplicates="drop",
                )
                bin_map: dict[int, list[int]] = {}
                for idx, bin_id in enumerate(bins.codes):
                    bin_map.setdefault(int(bin_id), []).append(idx)

                pair_null = []
                for _ in range(n_pair):
                    stats = []
                    for idx, seg in enumerate(cache):
                        bin_id = bins.codes[idx]
                        candidates = []
                        for j in bin_map.get(int(bin_id), []):
                            if j == idx:
                                continue
                            other = cache[j]
                            if seg.i_max_common <= 0:
                                continue
                            rel = abs(other.i_max_common - seg.i_max_common) / seg.i_max_common
                            if rel <= pair_tol:
                                candidates.append(j)
                        if not candidates:
                            continue
                        other = cache[int(rng_null.choice(candidates))]
                        i_max_common = min(seg.i_max_common, other.i_max_common)
                        if i_max_common <= 0:
                            continue
                        i_abs = np.linspace(0.0, i_max_common, setting["grid_n"])
                        v_pos = np.interp(i_abs, np.sort(seg.i_pos), seg.v_pos[np.argsort(seg.i_pos)])
                        v_neg = np.interp(i_abs, np.sort(other.i_neg), other.v_neg[np.argsort(other.i_neg)])
                        stats.append(_compute_from_arrays(i_abs, v_pos, v_neg))
                    pair_null.append(stats)

                null_stats["sign"] = _summarize_null(obs_odd, obs_lock, obs_phi0, sign_null)
                null_stats["phase"] = _summarize_null(obs_odd, obs_lock, obs_phi0, phase_null)
                null_stats["pair"] = _summarize_null(obs_odd, obs_lock, obs_phi0, pair_null)

            row = {
                "min_overlap": setting["min_overlap"],
                "grid_method": setting["grid_method"],
                "grid_density": setting["grid_density"],
                "grid_n": setting["grid_n"],
                "n_segments": int(len(odd_vals)),
                "odd_ratio_post_median": obs_odd,
                "parity_lock_fraction": obs_lock,
                "phi0_abs_median": obs_phi0,
                "pipeline_version": PIPELINE_VERSION,
            }
            for label, stats in null_stats.items():
                row[f"{label}_odd_ratio_post_z"] = stats.get("odd_ratio_post_z")
                row[f"{label}_parity_lock_z"] = stats.get("parity_lock_z")
                row[f"{label}_phi0_abs_z"] = stats.get("phi0_abs_z")
                row[f"{label}_null_n"] = stats.get("n_null")

            sens_rows.append(row)

        sens_df = pd.DataFrame(sens_rows)
        sens_path = Path("results/reports") / "sensitivity_grid.csv"
        sens_df.to_csv(sens_path, index=False)

        if not sens_df.empty:
            base_mask = (
                (sens_df["min_overlap"] == base_overlap)
                & (sens_df["grid_method"] == base_method)
                & (sens_df["grid_density"] == "medium")
            )
            baseline = sens_df[base_mask].iloc[0] if base_mask.any() else sens_df.iloc[0]
            for col in ["odd_ratio_post_median", "parity_lock_fraction", "phi0_abs_median"]:
                base_val = baseline[col]
                if base_val and np.isfinite(base_val):
                    sens_df[f"{col}_pct_change"] = (sens_df[col] - base_val) / base_val
                else:
                    sens_df[f"{col}_pct_change"] = np.nan
            sens_df["robust_within_pct"] = (
                sens_df[["odd_ratio_post_median_pct_change", "parity_lock_fraction_pct_change", "phi0_abs_median_pct_change"]]
                .abs()
                .max(axis=1)
                <= robust_pct
            )
            sens_df.to_csv(sens_path, index=False)
            robust_path = Path("results/reports") / "sensitivity_robust_summary.csv"
            robust_df = sens_df[["min_overlap", "grid_method", "grid_density", "robust_within_pct"]]
            robust_df.to_csv(robust_path, index=False)

    if not args.skip_stratify:
        strat_cfg = config.get("phase6", {}).get("stratify", {})
        min_group = int(strat_cfg.get("min_group", 8))
        strat_rows = []

        if "meta_temp_K" in branch_df.columns:
            temp_bins = int(strat_cfg.get("temp_bins", 4))
            bins = _bin_numeric(branch_df["meta_temp_K"], temp_bins)
            if bins is not None:
                temp_df = branch_df.loc[bins.index].copy()
                temp_df["temp_bin"] = bins.astype(str).values
                temp_metrics = _stratify_metrics(temp_df, "temp_bin", min_group)
                strat_rows.append(temp_metrics)
                _plot_stratified_bar(temp_metrics, Path("results/plots/stratified_temp_bins"), "Odd ratio by temperature bin")

        if "meta_coil_A" in branch_df.columns:
            coil_bins = int(strat_cfg.get("coil_bins", 4))
            bins = _bin_numeric(branch_df["meta_coil_A"], coil_bins)
            if bins is not None:
                coil_df = branch_df.loc[bins.index].copy()
                coil_df["coil_bin"] = bins.astype(str).values
                coil_metrics = _stratify_metrics(coil_df, "coil_bin", min_group)
                strat_rows.append(coil_metrics)
                _plot_stratified_bar(coil_metrics, Path("results/plots/stratified_coil_bins"), "Odd ratio by coil bin")

        if "meta_sample_id" in branch_df.columns:
            sample_metrics = _stratify_metrics(branch_df, "meta_sample_id", min_group)
            if not sample_metrics.empty:
                strat_rows.append(sample_metrics)
                _plot_stratified_bar(sample_metrics, Path("results/plots/stratified_samples"), "Odd ratio by sample")

        if "meta_freq_Hz" in branch_df.columns:
            freq_bins = int(strat_cfg.get("coil_bins", 4))
            bins = _bin_numeric(branch_df["meta_freq_Hz"], freq_bins)
            if bins is not None:
                freq_df = branch_df.loc[bins.index].copy()
                freq_df["freq_bin"] = bins.astype(str).values
                freq_metrics = _stratify_metrics(freq_df, "freq_bin", min_group)
                strat_rows.append(freq_metrics)
                _plot_stratified_bar(freq_metrics, Path("results/plots/stratified_freq_bins"), "Odd ratio by freq bin")

        if "meta_date" in branch_df.columns:
            date_metrics = _stratify_metrics(branch_df, "meta_date", min_group)
            if not date_metrics.empty:
                strat_rows.append(date_metrics)
                _plot_stratified_bar(date_metrics, Path("results/plots/stratified_dates"), "Odd ratio by run date")

        if strat_rows:
            strat_df = pd.concat(strat_rows, ignore_index=True)
        else:
            strat_df = pd.DataFrame(
                columns=[
                    "strat_field",
                    "bin_label",
                    "n_segments",
                    "odd_ratio_post_median",
                    "parity_lock_fraction",
                    "phi0_proxy_abs_median",
                    "I_knee_norm_median",
                ]
            )
        strat_df.to_csv(Path("results/reports") / "stratified_metrics.csv", index=False)

    report = {
        "pipeline_version": PIPELINE_VERSION,
        "branch_metrics": str(out_dir / "phase6_branch_metrics.csv"),
        "branch_split_qc": "results/reports/branch_split_qc.csv",
        "branch_consistency": "results/reports/branch_consistency.csv",
        "sensitivity_grid": "results/reports/sensitivity_grid.csv" if not args.skip_sensitivity else None,
        "nuisance_warp_fit": "results/reports/nuisance_warp_fit.csv" if not args.skip_nuisance else None,
        "nuisance_robustness": "results/reports/nuisance_robustness.csv" if not args.skip_nuisance else None,
        "stratified_metrics": "results/reports/stratified_metrics.csv" if not args.skip_stratify else None,
    }
    report_path = Path("results/reports/phase6_summary.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
