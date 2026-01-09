"""Metric extraction for JJ datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

LOGGER = logging.getLogger("jjparse")


def basic_stats(df: pd.DataFrame) -> dict:
    stats = {"n_points": int(len(df))}
    if "I_A" in df.columns:
        series = pd.to_numeric(df["I_A"], errors="coerce")
        stats["I_range"] = float(series.max(skipna=True) - series.min(skipna=True))
    if "V_V" in df.columns:
        series = pd.to_numeric(df["V_V"], errors="coerce")
        stats["V_range"] = float(series.max(skipna=True) - series.min(skipna=True))
    return stats


def detect_sweep_type(df: pd.DataFrame) -> str | None:
    candidates = [
        ("gate1_V", "gate1"),
        ("gate2_V", "gate2"),
        ("gate_V", "gate"),
        ("B_T", "B"),
        ("T_K", "T"),
        ("freq_Hz", "freq"),
    ]
    for col, label in candidates:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.nunique(dropna=True) > 1:
                return label
    return None


def _interp_on_abs_grid(sweep: np.ndarray, resp: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(sweep)
    sweep_sorted = sweep[order]
    resp_sorted = resp[order]
    return np.interp(grid, sweep_sorted, resp_sorted, left=np.nan, right=np.nan)


def compute_nonreciprocity(
    df: pd.DataFrame,
    sweep_col: str,
    resp_col: str,
    grid_n: int = 200,
    eps: float = 1e-12,
) -> float | None:
    sweep = pd.to_numeric(df[sweep_col], errors="coerce").to_numpy()
    resp = pd.to_numeric(df[resp_col], errors="coerce").to_numpy()
    mask = ~np.isnan(sweep) & ~np.isnan(resp)
    sweep = sweep[mask]
    resp = resp[mask]
    if sweep.size < 5:
        return None

    pos_mask = sweep > 0
    neg_mask = sweep < 0
    if not pos_mask.any() or not neg_mask.any():
        return None

    sweep_pos = np.abs(sweep[pos_mask])
    sweep_neg = np.abs(sweep[neg_mask])
    resp_pos = resp[pos_mask]
    resp_neg = resp[neg_mask]

    max_common = min(sweep_pos.max(), sweep_neg.max())
    if not np.isfinite(max_common) or max_common <= 0:
        return None

    grid = np.linspace(0, max_common, grid_n)
    interp_pos = _interp_on_abs_grid(sweep_pos, resp_pos, grid)
    interp_neg = _interp_on_abs_grid(sweep_neg, resp_neg, grid)

    valid = ~np.isnan(interp_pos) & ~np.isnan(interp_neg)
    if not valid.any():
        return None

    diff = np.abs(interp_pos[valid] - interp_neg[valid])
    denom = np.sum(np.abs(interp_pos[valid])) + np.sum(np.abs(interp_neg[valid])) + eps
    return float(np.sum(diff) / denom)


def compute_voltage_nonreciprocity(
    x: np.ndarray,
    v: np.ndarray,
    grid_n: int = 200,
    eps: float = 1e-12,
    min_overlap: float = 0.8,
    den_min: float | None = None,
) -> dict:
    mask = np.isfinite(x) & np.isfinite(v)
    x = x[mask]
    v = v[mask]
    if x.size < 5:
        return {"eta_valid": False, "eta_reason": "insufficient_points"}

    pos_mask = x > 0
    neg_mask = x < 0
    if not pos_mask.any() or not neg_mask.any():
        return {"eta_valid": False, "eta_reason": "missing_pair"}

    x_pos = np.abs(x[pos_mask])
    x_neg = np.abs(x[neg_mask])
    v_pos = v[pos_mask]
    v_neg = v[neg_mask]

    x_min_plus = float(np.nanmin(x_pos))
    x_max_plus = float(np.nanmax(x_pos))
    x_min_minus = float(np.nanmin(x_neg))
    x_max_minus = float(np.nanmax(x_neg))
    union_min = min(x_min_plus, x_min_minus)
    union_max = max(x_max_plus, x_max_minus)
    overlap_min = max(x_min_plus, x_min_minus)
    overlap_max = min(x_max_plus, x_max_minus)
    union_range = union_max - union_min
    overlap_range = overlap_max - overlap_min
    if not np.isfinite(union_range) or union_range <= 0 or not np.isfinite(overlap_range) or overlap_range <= 0:
        return {
            "eta_valid": False,
            "eta_reason": "low_overlap",
            "x_min_plus": x_min_plus,
            "x_max_plus": x_max_plus,
            "x_min_minus": x_min_minus,
            "x_max_minus": x_max_minus,
        }

    f_overlap = overlap_range / union_range
    if f_overlap < min_overlap:
        return {
            "eta_valid": False,
            "eta_reason": "low_overlap",
            "x_min_plus": x_min_plus,
            "x_max_plus": x_max_plus,
            "x_min_minus": x_min_minus,
            "x_max_minus": x_max_minus,
            "f_overlap": float(f_overlap),
        }

    grid = np.linspace(overlap_min, overlap_max, grid_n)
    v_plus = _interp_on_abs_grid(x_pos, v_pos, grid)
    v_minus = _interp_on_abs_grid(x_neg, v_neg, grid)

    valid = np.isfinite(v_plus) & np.isfinite(v_minus)
    if not valid.any():
        return {
            "eta_valid": False,
            "eta_reason": "no_overlap",
            "x_min_plus": x_min_plus,
            "x_max_plus": x_max_plus,
            "x_min_minus": x_min_minus,
            "x_max_minus": x_max_minus,
            "f_overlap": float(f_overlap),
        }

    denom = np.mean(np.abs(v_plus[valid]) + np.abs(v_minus[valid])) + eps
    if den_min is None:
        den_min = 0.0
    if denom < den_min:
        return {
            "eta_valid": False,
            "eta_reason": "tiny_denominator",
            "x_min_plus": x_min_plus,
            "x_max_plus": x_max_plus,
            "x_min_minus": x_min_minus,
            "x_max_minus": x_max_minus,
            "x_grid_min": float(overlap_min),
            "x_grid_max": float(overlap_max),
            "n_grid": int(grid_n),
            "eta_denom": float(denom),
            "f_overlap": float(f_overlap),
        }
    eta_l1 = float(np.mean(np.abs(v_plus[valid] - v_minus[valid])) / denom)
    eta_signed = float(np.mean(v_plus[valid] + v_minus[valid]) / denom)
    diff = v_plus[valid] - v_minus[valid]
    l2_num = float(np.sqrt(np.mean(diff**2)))
    l2_denom = float(np.sqrt(np.mean(v_plus[valid] ** 2)) + np.sqrt(np.mean(v_minus[valid] ** 2)) + eps)
    eta_l2 = float(l2_num / l2_denom) if l2_denom > 0 else np.nan
    corr_pm = np.nan
    if valid.sum() >= 2:
        v_plus_valid = v_plus[valid]
        v_minus_valid = v_minus[valid]
        if np.std(v_plus_valid) > 0 and np.std(v_minus_valid) > 0:
            corr_pm = float(np.corrcoef(v_plus_valid, v_minus_valid)[0, 1])

    return {
        "eta_valid": True,
        "eta_reason": "ok",
        "eta_V_L1": eta_l1,
        "eta_V_L2": eta_l2,
        "eta_V_signed": eta_signed,
        "corr_pm": corr_pm,
        "x_min_plus": x_min_plus,
        "x_max_plus": x_max_plus,
        "x_min_minus": x_min_minus,
        "x_max_minus": x_max_minus,
        "x_grid_min": float(overlap_min),
        "x_grid_max": float(overlap_max),
        "n_grid": int(grid_n),
        "eta_denom": float(denom),
        "f_overlap": float(f_overlap),
    }


def _cpr_model(params: np.ndarray, phi: np.ndarray) -> np.ndarray:
    i1, i2, phi0, offset = params
    return i1 * np.sin(phi + phi0) + i2 * np.sin(2 * (phi + phi0)) + offset


def fit_cpr(phi: np.ndarray, current: np.ndarray, phi0_bounds: Iterable[float]) -> dict:
    phi = np.unwrap(phi)
    current = current.astype(float)

    i1_guess = 0.5 * (np.nanmax(current) - np.nanmin(current))
    i2_guess = 0.1 * i1_guess
    phi0_guess = 0.0
    offset_guess = float(np.nanmean(current))

    lower = [-np.inf, -np.inf, float(phi0_bounds[0]), -np.inf]
    upper = [np.inf, np.inf, float(phi0_bounds[1]), np.inf]

    def residuals(params: np.ndarray) -> np.ndarray:
        return _cpr_model(params, phi) - current

    result = least_squares(residuals, x0=[i1_guess, i2_guess, phi0_guess, offset_guess], bounds=(lower, upper))
    fitted = _cpr_model(result.x, phi)

    ss_res = float(np.nansum((current - fitted) ** 2))
    ss_tot = float(np.nansum((current - np.nanmean(current)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(ss_res / len(current))) if len(current) else np.nan

    return {
        "phi0_rad": float(result.x[2]),
        "I1": float(result.x[0]),
        "I2": float(result.x[1]),
        "offset_C": float(result.x[3]),
        "fit_r2": r2,
        "fit_rmse": rmse,
    }


def extract_cpr_metrics(df: pd.DataFrame, phi0_bounds: Iterable[float]) -> dict:
    if "phi_rad" not in df.columns or "I_A" not in df.columns:
        return {}
    phi = pd.to_numeric(df["phi_rad"], errors="coerce").to_numpy()
    current = pd.to_numeric(df["I_A"], errors="coerce").to_numpy()
    mask = ~np.isnan(phi) & ~np.isnan(current)
    if mask.sum() < 10:
        return {}
    return fit_cpr(phi[mask], current[mask], phi0_bounds)


def phase_shift_proxy(df: pd.DataFrame) -> dict:
    if "phi_rad" in df.columns:
        return {}
    if "B_T" in df.columns and "I_A" in df.columns:
        field = pd.to_numeric(df["B_T"], errors="coerce")
        current = pd.to_numeric(df["I_A"], errors="coerce")
        if field.dropna().empty or current.dropna().empty:
            return {}
        idx = current.abs().idxmax()
        if pd.isna(idx):
            return {}
        proxy = float(field.loc[idx])
        return {"phase_shift_proxy_B_T": proxy}
    return {}


def _two_segment_knee(x: np.ndarray, y: np.ndarray, min_segment: int) -> dict:
    n = len(x)
    best = None
    for split in range(min_segment, n - min_segment + 1):
        x1, y1 = x[:split], y[:split]
        x2, y2 = x[split:], y[split:]
        coef1 = np.polyfit(x1, y1, 1)
        coef2 = np.polyfit(x2, y2, 1)
        y1_fit = np.polyval(coef1, x1)
        y2_fit = np.polyval(coef2, x2)
        sse = np.sum((y1 - y1_fit) ** 2) + np.sum((y2 - y2_fit) ** 2)
        if best is None or sse < best["sse"]:
            best = {
                "split": split,
                "sse": float(sse),
                "slope_pre": float(coef1[0]),
                "slope_post": float(coef2[0]),
            }
    if best is None:
        return {}
    return best


def detect_knee(x: np.ndarray, y: np.ndarray, min_segment: int = 3) -> dict:
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    if len(x_sorted) < min_segment * 2:
        return {}

    single_coef = np.polyfit(x_sorted, y_sorted, 1)
    single_fit = np.polyval(single_coef, x_sorted)
    sse_single = float(np.sum((y_sorted - single_fit) ** 2))

    best = _two_segment_knee(x_sorted, y_sorted, min_segment)
    if not best:
        return {}

    split_idx = best["split"]
    knee_x = float(x_sorted[split_idx - 1])
    rss_ratio = float(best["sse"] / sse_single) if sse_single > 0 else np.nan
    return {
        "knee_x": knee_x,
        "knee_slope_pre": best["slope_pre"],
        "knee_slope_post": best["slope_post"],
        "knee_rss_ratio": rss_ratio,
    }


def borgt_scaling(x: np.ndarray, y: np.ndarray, n_boot: int = 200) -> dict:
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return {}

    logx = np.log(x)
    logy = np.log(y)
    slope, intercept = np.polyfit(logx, logy, 1)

    if n_boot <= 0:
        return {"slope": float(slope), "intercept": float(intercept)}

    rng = np.random.default_rng(123)
    slopes = []
    for _ in range(n_boot):
        idx = rng.choice(len(logx), size=len(logx), replace=True)
        boot_slope, _ = np.polyfit(logx[idx], logy[idx], 1)
        slopes.append(boot_slope)

    slopes = np.array(slopes)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_ci_low": float(np.percentile(slopes, 2.5)),
        "slope_ci_high": float(np.percentile(slopes, 97.5)),
    }


def _finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _fill_zero_signs(signs: np.ndarray) -> np.ndarray:
    if signs.size == 0:
        return signs
    for idx in range(1, len(signs)):
        if signs[idx] == 0:
            signs[idx] = signs[idx - 1]
    if signs[0] == 0:
        for idx, val in enumerate(signs):
            if val != 0:
                signs[:idx] = val
                break
    for idx in range(len(signs) - 2, -1, -1):
        if signs[idx] == 0:
            signs[idx] = signs[idx + 1]
    return signs


def split_segments(x: np.ndarray, v: np.ndarray, min_points: int = 5) -> list[dict]:
    x, v = _finite_xy(x, v)
    if x.size == 0:
        return []
    if x.size < min_points:
        return [
            {
                "segment_id": 0,
                "x": x,
                "V": v,
                "direction": "unknown",
                "indices": np.arange(x.size),
            }
        ]

    dx = np.diff(x)
    signs = np.sign(dx)
    signs = _fill_zero_signs(signs)

    if signs.size == 0 or np.all(signs == 0):
        return [
            {
                "segment_id": 0,
                "x": x,
                "V": v,
                "direction": "unknown",
                "indices": np.arange(x.size),
            }
        ]

    boundaries = [0]
    for idx in range(1, len(signs)):
        if signs[idx] == 0 or signs[idx - 1] == 0:
            continue
        if signs[idx] != signs[idx - 1]:
            boundaries.append(idx + 1)
    boundaries.append(len(x))

    segments = []
    seg_id = 0
    for start, end in zip(boundaries, boundaries[1:]):
        if end - start < 2:
            continue
        seg_x = x[start:end]
        seg_v = v[start:end]
        if seg_x.size < min_points and len(boundaries) > 2:
            continue
        dx_seg = np.diff(seg_x)
        direction = "unknown"
        if np.nanmean(dx_seg) > 0:
            direction = "up"
        elif np.nanmean(dx_seg) < 0:
            direction = "down"
        segments.append(
            {
                "segment_id": seg_id,
                "x": seg_x,
                "V": seg_v,
                "direction": direction,
                "indices": np.arange(start, end),
            }
        )
        seg_id += 1

    if not segments:
        segments = [
            {
                "segment_id": 0,
                "x": x,
                "V": v,
                "direction": "unknown",
                "indices": np.arange(x.size),
            }
        ]

    return segments


def axis_properties(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if x.size < 3:
        return {"is_monotonic": False, "is_cyclic": False, "n_segments": 1}
    dx = np.diff(x)
    dx = dx[np.isfinite(dx)]
    if dx.size == 0:
        return {"is_monotonic": False, "is_cyclic": False, "n_segments": 1}
    signs = np.sign(dx)
    signs = signs[signs != 0]
    if signs.size == 0:
        return {"is_monotonic": False, "is_cyclic": False, "n_segments": 1}
    monotonic = np.all(signs > 0) or np.all(signs < 0)
    sign_changes = np.sum(np.diff(np.sign(signs)) != 0)
    return {
        "is_monotonic": bool(monotonic),
        "is_cyclic": bool(sign_changes >= 1),
        "n_segments": int(sign_changes + 1),
    }


def _smooth_series(values: np.ndarray, method: str, window: int, polyorder: int) -> np.ndarray:
    if values.size < 5:
        return values
    if method == "savgol":
        win = min(window, values.size if values.size % 2 == 1 else values.size - 1)
        if win < 3:
            return values
        if win % 2 == 0:
            win -= 1
        poly = min(polyorder, win - 1)
        return savgol_filter(values, window_length=win, polyorder=poly, mode="interp")
    if method == "median":
        win = min(window, values.size)
        if win < 3:
            return values
        series = pd.Series(values).rolling(window=win, center=True, min_periods=1).median()
        return series.to_numpy()
    return values


def compute_dv_dx_stats(x: np.ndarray, v: np.ndarray, config: dict) -> dict:
    x, v = _finite_xy(x, v)
    if x.size < 3:
        return {}
    dv_dx = np.gradient(v, x)
    abs_dv_dx = np.abs(dv_dx)

    spike_thresh = config.get("metrics", {}).get("dv_dx_spike_thresh")
    if spike_thresh is None:
        mult = config.get("metrics", {}).get("dv_dx_spike_multiplier", 5.0)
        median = float(np.nanmedian(abs_dv_dx))
        spike_thresh = median * mult

    spike_count = int(np.sum(abs_dv_dx > spike_thresh)) if np.isfinite(spike_thresh) else 0
    x_range = float(np.nanmax(x) - np.nanmin(x)) if np.isfinite(x).any() else np.nan
    spike_rate = float(spike_count / (x_range + 1e-12)) if np.isfinite(x_range) else np.nan

    return {
        "dv_dx_mean_abs": float(np.nanmean(abs_dv_dx)),
        "dv_dx_median_abs": float(np.nanmedian(abs_dv_dx)),
        "dv_dx_p95_abs": float(np.nanpercentile(abs_dv_dx, 95)),
        "dv_dx_spike_rate": spike_rate,
        "dv_dx_spike_thresh": float(spike_thresh),
    }


def compute_knee_score(x: np.ndarray, v: np.ndarray, config: dict) -> dict:
    x, v = _finite_xy(x, v)
    if x.size < 5:
        return {}
    dv_dx = np.gradient(v, x)
    s = np.abs(dv_dx)

    smoothing = config.get("smoothing", {})
    s = _smooth_series(
        s,
        method=smoothing.get("method", "savgol"),
        window=int(smoothing.get("window", 11)),
        polyorder=int(smoothing.get("polyorder", 3)),
    )

    s1 = np.gradient(s, x)
    s2 = np.gradient(s1, x)
    eps = float(config.get("metrics", {}).get("eps", 1e-12))
    kappa = s1 / (eps + np.abs(s2))
    idx = int(np.nanargmax(np.abs(kappa))) if np.isfinite(kappa).any() else 0

    v_abs = np.abs(v)
    vmax_idx = int(np.nanargmax(v_abs)) if np.isfinite(v_abs).any() else 0

    return {
        "K_max": float(np.nanmax(np.abs(kappa))) if np.isfinite(kappa).any() else np.nan,
        "K_at_max_x": float(x[idx]) if idx < x.size else np.nan,
        "knee_before_peak": bool(x[idx] <= x[vmax_idx]) if x.size and vmax_idx < x.size else False,
    }


def compute_changepoints(x: np.ndarray, v: np.ndarray, config: dict) -> dict:
    if not config.get("knee", {}).get("use_ruptures", True):
        return {}
    try:
        import ruptures as rpt
    except ImportError:
        return {}

    x, v = _finite_xy(x, v)
    if x.size < 6:
        return {}

    order = np.argsort(x)
    x_sorted = x[order]
    v_sorted = v[order]

    model = config.get("knee", {}).get("ruptures_model", "rbf")
    penalty = config.get("knee", {}).get("ruptures_penalty", 5.0)

    algo = rpt.Pelt(model=model).fit(v_sorted)
    cp_idx = algo.predict(pen=penalty)
    cp_idx = [idx for idx in cp_idx if idx < len(x_sorted)]
    if not cp_idx:
        return {}
    cp_x = [float(x_sorted[idx - 1]) for idx in cp_idx if idx > 0]
    if not cp_x:
        return {}

    return {
        "cp_x_list": ";".join(f"{val:.6g}" for val in cp_x),
        "cp_main_x": cp_x[0],
    }


def spectral_flatness(values: np.ndarray) -> float | None:
    values = np.asarray(values)
    if values.size < 8:
        return None
    values = values - np.nanmean(values)
    spectrum = np.abs(np.fft.rfft(values)) ** 2
    if spectrum.size == 0:
        return None
    eps = 1e-12
    geometric = np.exp(np.mean(np.log(spectrum + eps)))
    arithmetic = np.mean(spectrum)
    if arithmetic <= 0:
        return None
    return float(geometric / (arithmetic + eps))


def permutation_entropy(values: np.ndarray, order: int = 3, delay: int = 1) -> float | None:
    values = np.asarray(values)
    n = values.size
    if n < order * delay + 1:
        return None
    patterns = {}
    for idx in range(n - delay * (order - 1)):
        window = values[idx : idx + delay * order : delay]
        if np.any(~np.isfinite(window)):
            continue
        ranks = tuple(np.argsort(window))
        patterns[ranks] = patterns.get(ranks, 0) + 1
    if not patterns:
        return None
    counts = np.array(list(patterns.values()), dtype=float)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(math.factorial(order))
    return float(entropy / max_entropy) if max_entropy > 0 else None


def _detrend_linear(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, v = _finite_xy(x, v)
    if v.size < 3:
        return v
    x_span = float(np.nanmax(x) - np.nanmin(x))
    if not np.isfinite(x_span) or x_span == 0.0:
        return v - np.nanmean(v)
    try:
        coef = np.polyfit(x, v, 1)
        trend = np.polyval(coef, x)
        return v - trend
    except (ValueError, np.linalg.LinAlgError):
        return v - np.nanmean(v)


def rolling_variance_ratio(values: np.ndarray, window: int, eps: float) -> float | None:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    global_std = float(np.nanstd(values))
    if not np.isfinite(global_std):
        return None
    window = max(2, min(int(window), values.size))
    rolling = pd.Series(values).rolling(window=window, center=True, min_periods=2).std()
    median_std = float(np.nanmedian(rolling.to_numpy()))
    if not np.isfinite(median_std):
        return None
    ratio = median_std / (global_std + eps)
    return float(np.clip(ratio, 0.0, 1.0))


def compute_incoherence(x: np.ndarray, v: np.ndarray, config: dict) -> dict:
    x = np.asarray(x)
    v = np.asarray(v)
    x, v = _finite_xy(x, v)
    if v.size < 4:
        return {}

    incoh_cfg = config.get("incoherence", {})
    min_points = int(incoh_cfg.get("min_points", 16))
    window = int(incoh_cfg.get("rolling_window", 11))
    eps = float(config.get("metrics", {}).get("eps", 1e-12))

    h_var = rolling_variance_ratio(v, window=window, eps=eps)
    h_flat = None
    if v.size >= min_points:
        detrended = _detrend_linear(x, v)
        h_flat = spectral_flatness(detrended)

    if h_flat is not None:
        h_raw = h_flat
        method = "spectral_flatness"
    elif h_var is not None:
        h_raw = h_var
        method = "rolling_variance"
    else:
        return {}

    out = {
        "H_incoh_raw": float(h_raw),
        "H_method": method,
    }
    if h_flat is not None:
        out["H_flat_raw"] = float(h_flat)
    if h_var is not None:
        out["H_var_raw"] = float(h_var)
    return out


@dataclass
class Metric:
    name: str
    requires: set[str]
    optional: set[str] = field(default_factory=set)
    level: str = "segment"
    func: Callable[[dict], dict] = lambda context: {}
    requires_pairing: bool = False


def run_metric(metric: Metric, available: set[str], context: dict) -> dict:
    missing = metric.requires - available
    if missing:
        LOGGER.debug("Skipping metric %s (missing %s)", metric.name, ",".join(sorted(missing)))
        return {}
    try:
        return metric.func(context)
    except Exception as exc:
        LOGGER.warning("Metric %s failed: %s", metric.name, exc)
        return {}


def metric_v_stats(context: dict) -> dict:
    v = np.asarray(context["signals"]["V"])
    if v.size == 0:
        return {}
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    return {
        "V_mean": float(np.mean(v)),
        "V_std": float(np.std(v)),
        "V_p95": float(np.percentile(v, 95)),
        "V_range": float(np.max(v) - np.min(v)),
    }


def metric_dv_dx(context: dict) -> dict:
    x = context["signals"]["x"]
    v = context["signals"]["V"]
    return compute_dv_dx_stats(x, v, context.get("config", {}))


def metric_knee_two_segment(context: dict) -> dict:
    x = context["signals"]["x"]
    v = context["signals"]["V"]
    min_segment = int(context.get("config", {}).get("knee", {}).get("min_segment", 3))
    return detect_knee(x, v, min_segment=min_segment)


def metric_knee_changepoints(context: dict) -> dict:
    x = context["signals"]["x"]
    v = context["signals"]["V"]
    return compute_changepoints(x, v, context.get("config", {}))


def metric_knee_score(context: dict) -> dict:
    x = context["signals"]["x"]
    v = context["signals"]["V"]
    return compute_knee_score(x, v, context.get("config", {}))


def metric_incoherence(context: dict) -> dict:
    x = context["signals"]["x"]
    v = context["signals"]["V"]
    return compute_incoherence(x, v, context.get("config", {}))


def metric_eta_voltage(context: dict) -> dict:
    x = context["signals"]["x"]
    v = context["signals"]["V"]
    grid_n = int(context.get("config", {}).get("parity", {}).get("grid_n", 200))
    eps = float(context.get("config", {}).get("parity", {}).get("eps", 1e-12))
    min_overlap = float(context.get("config", {}).get("parity", {}).get("min_overlap", 0.8))
    den_min = context.get("config", {}).get("parity", {}).get("den_min", 0.0)
    return compute_voltage_nonreciprocity(x, v, grid_n=grid_n, eps=eps, min_overlap=min_overlap, den_min=den_min)


def metric_hysteresis(context: dict) -> dict:
    segments = context.get("segments", [])
    grid_n = int(context.get("config", {}).get("resample", {}).get("grid_n", 200))
    eps = float(context.get("config", {}).get("metrics", {}).get("eps", 1e-12))

    up_segments = [seg for seg in segments if seg.get("direction") == "up"]
    down_segments = [seg for seg in segments if seg.get("direction") == "down"]
    if not up_segments or not down_segments:
        return {}

    up = max(up_segments, key=lambda seg: seg.get("x").size)
    down = max(down_segments, key=lambda seg: seg.get("x").size)

    x_up = up["x"]
    v_up = up["V"]
    x_down = down["x"]
    v_down = down["V"]

    x_min = max(np.nanmin(x_up), np.nanmin(x_down))
    x_max = min(np.nanmax(x_up), np.nanmax(x_down))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return {}

    grid = np.linspace(x_min, x_max, grid_n)

    v_up_i = _interp_on_abs_grid(x_up, v_up, grid)
    v_down_i = _interp_on_abs_grid(x_down, v_down, grid)
    valid = np.isfinite(v_up_i) & np.isfinite(v_down_i)
    if not valid.any():
        return {}

    area = float(np.trapz(np.abs(v_up_i[valid] - v_down_i[valid]), grid[valid]))
    denom = float(np.trapz(np.abs(v_up_i[valid]) + np.abs(v_down_i[valid]), grid[valid])) + eps

    return {
        "H_area": area,
        "H_norm": float(area / denom),
    }


def metric_phi0_fit(context: dict) -> dict:
    phi = context["signals"]["phi"]
    current = context["signals"]["I"]
    phi0_bounds = context.get("config", {}).get("fit", {}).get("phi0_bounds", [-3.14, 3.14])
    mask = np.isfinite(phi) & np.isfinite(current)
    if mask.sum() < 10:
        return {}
    return fit_cpr(phi[mask], current[mask], phi0_bounds)


METRICS = [
    Metric(name="v_stats", requires={"V"}, level="segment", func=metric_v_stats),
    Metric(name="dv_dx", requires={"V", "x"}, level="segment", func=metric_dv_dx),
    Metric(name="knee_two_segment", requires={"V", "x"}, level="segment", func=metric_knee_two_segment),
    Metric(name="knee_changepoints", requires={"V", "x"}, level="segment", func=metric_knee_changepoints),
    Metric(name="knee_score", requires={"V", "x"}, level="segment", func=metric_knee_score),
    Metric(name="incoherence", requires={"V", "x"}, level="segment", func=metric_incoherence),
    Metric(name="eta_voltage", requires={"V", "x"}, level="segment", func=metric_eta_voltage),
    Metric(name="hysteresis", requires={"V", "x"}, level="file", func=metric_hysteresis),
    Metric(name="phi0_fit", requires={"I", "phi"}, level="file", func=metric_phi0_fit),
]
