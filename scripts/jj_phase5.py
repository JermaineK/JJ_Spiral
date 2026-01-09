"""Phase 5 CPR-grade Josephson analysis for IV sweeps."""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jjparse.config import load_config, seed_everything, setup_logging
from jjparse.version import PIPELINE_VERSION

LOGGER = logging.getLogger("jjparse.phase5")

FLOAT_RE = re.compile(r"[-+]?((\d+(\.\d*)?)|(\.\d+))([eE][-+]?\d+)?")
BRACKET_RE = re.compile(r"\[([+-]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\]")


@dataclass
class SegmentCache:
    file_id: str
    segment_id: int
    condition_group: str
    condition_value: float | None
    i_pos: np.ndarray
    v_pos: np.ndarray
    i_neg: np.ndarray
    v_neg: np.ndarray
    i_max_common: float
    i_abs_grid: np.ndarray
    v_pos_grid: np.ndarray
    v_neg_grid: np.ndarray
    i_knee: float | None
    knee_valid: bool


def _iter_files(in_dir: Path, glob_pattern: str) -> list[Path]:
    files = sorted(in_dir.glob(glob_pattern))
    skip_suffix = {".pdf", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    out = []
    for path in files:
        if not path.is_file():
            continue
        if path.suffix.lower() in skip_suffix:
            continue
        out.append(path)
    return out


def _parse_iv_file(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        LOGGER.warning("Failed to read %s: %s", path, exc)
        return np.array([]), np.array([]), []

    lines = text.splitlines()
    header = []
    i_vals = []
    v_vals = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        numbers = FLOAT_RE.findall(stripped)
        if len(numbers) < 2:
            header.append(stripped[:200])
            continue
        tokens = FLOAT_RE.finditer(stripped)
        vals = []
        for match in tokens:
            if len(vals) >= 2:
                break
            vals.append(float(match.group(0)))
        if len(vals) < 2:
            continue
        i_vals.append(vals[0])
        v_vals.append(vals[1])

    return np.asarray(i_vals, dtype=float), np.asarray(v_vals, dtype=float), header


def _sweep_monotonicity(i_vals: np.ndarray) -> tuple[float, str, int]:
    if i_vals.size < 3:
        return 0.0, "unknown", 1
    diffs = np.diff(i_vals)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 0.0, "unknown", 1
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if signs.size == 0:
        return 0.0, "unknown", 1
    pos_frac = float(np.mean(signs > 0))
    neg_frac = float(np.mean(signs < 0))
    direction = "up" if pos_frac >= neg_frac else "down"
    monotonicity = max(pos_frac, neg_frac)
    sign_changes = int(np.sum(np.diff(np.sign(signs)) != 0))
    return monotonicity, direction, sign_changes + 1


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


def _split_segments(i_vals: np.ndarray, v_vals: np.ndarray, min_points: int) -> list[dict]:
    if i_vals.size < min_points:
        return [
            {
                "segment_id": 0,
                "indices": np.arange(i_vals.size),
                "direction": "unknown",
            }
        ]
    diffs = np.diff(i_vals)
    signs = np.sign(diffs)
    signs = _fill_zero_signs(signs)
    if signs.size == 0 or np.all(signs == 0):
        return [
            {
                "segment_id": 0,
                "indices": np.arange(i_vals.size),
                "direction": "unknown",
            }
        ]
    boundaries = [0]
    for idx in range(1, len(signs)):
        if signs[idx] != signs[idx - 1]:
            boundaries.append(idx + 1)
    boundaries.append(len(i_vals))

    segments = []
    seg_id = 0
    for start, end in zip(boundaries, boundaries[1:]):
        if end - start < min_points:
            continue
        seg_i = i_vals[start:end]
        direction = "unknown"
        if np.nanmean(np.diff(seg_i)) > 0:
            direction = "up"
        elif np.nanmean(np.diff(seg_i)) < 0:
            direction = "down"
        segments.append({"segment_id": seg_id, "indices": np.arange(start, end), "direction": direction})
        seg_id += 1
    if not segments:
        segments = [
            {
                "segment_id": 0,
                "indices": np.arange(i_vals.size),
                "direction": "unknown",
            }
        ]
    return segments


def _spacing_cv(i_vals: np.ndarray) -> float | None:
    diffs = np.diff(i_vals)
    diffs = np.abs(diffs[np.isfinite(diffs)])
    if diffs.size == 0:
        return None
    mean = float(np.mean(diffs))
    if mean <= 0:
        return None
    return float(np.std(diffs) / mean)


def _infer_condition(path: Path, root: Path) -> tuple[str, float | None]:
    try:
        rel = path.relative_to(root)
        parts = list(rel.parts)
    except ValueError:
        parts = list(path.parts)
    condition_group = "unknown"
    if "SQUID_data" in parts:
        idx = parts.index("SQUID_data")
        if idx + 1 < len(parts):
            condition_group = parts[idx + 1]
    elif len(parts) >= 2:
        condition_group = parts[-2]

    match = BRACKET_RE.search(path.name)
    cond_val = None
    if match:
        try:
            cond_val = float(match.group(1))
        except ValueError:
            cond_val = None
    return condition_group, cond_val


def _make_file_id(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
        name = rel.as_posix()
    except ValueError:
        name = path.as_posix()
    suffix = path.suffix.lower()
    if suffix in {".txt", ".dat"}:
        name = name[: -len(suffix)]
    return re.sub(r"[\\/]+", "__", name)


def _interp_even_odd(
    i_vals: np.ndarray,
    v_vals: np.ndarray,
    grid_n: int,
    min_overlap: float,
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
        return {
            "parity_valid": False,
            "parity_reason": "low_overlap",
            "f_overlap": 0.0,
        }
    f_overlap = float(overlap_range / union_range)
    if f_overlap < min_overlap:
        return {
            "parity_valid": False,
            "parity_reason": "low_overlap",
            "f_overlap": f_overlap,
        }

    i_max_common = overlap_max
    i_abs_grid = np.linspace(0.0, i_max_common, grid_n)

    order_pos = np.argsort(i_pos)
    order_neg = np.argsort(i_neg)
    v_pos_grid = np.interp(i_abs_grid, i_pos[order_pos], v_pos[order_pos])
    v_neg_grid = np.interp(i_abs_grid, i_neg[order_neg], v_neg[order_neg])

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
        "i_min_pos": i_min_pos,
        "i_max_pos": i_max_pos,
        "i_min_neg": i_min_neg,
        "i_max_neg": i_max_neg,
    }

def _odd_sign_stats(values: np.ndarray, sign_eps: float) -> tuple[float | None, float | None, float | None, float | None]:
    mask = np.abs(values) > sign_eps
    if not mask.any():
        return None, None, None, None
    signs = np.sign(values[mask])
    pos = np.sum(signs > 0)
    neg = np.sum(signs < 0)
    total = pos + neg
    if total == 0:
        return None, None, None, None
    p = pos / total
    mode = 1.0 if p >= 0.5 else -1.0
    consistency = max(p, 1.0 - p)
    entropy = 0.0
    for frac in (p, 1.0 - p):
        if frac > 0:
            entropy -= frac * math.log(frac)
    entropy /= math.log(2.0)
    return float(mode), float(consistency), float(entropy), float(p)


def _knee_loglog(i_abs: np.ndarray, v_mag: np.ndarray, config: dict) -> dict:
    mask = (i_abs > 0) & (v_mag > 0)
    if mask.sum() < 5:
        return {}
    log_i = np.log(i_abs[mask])
    log_v = np.log(v_mag[mask])

    window = int(config.get("phase5", {}).get("knee_savgol_window", 0))
    poly = int(config.get("phase5", {}).get("knee_savgol_polyorder", 3))
    if window >= 5 and window < log_v.size:
        if window % 2 == 0:
            window += 1
        poly = min(poly, window - 1)
        log_v = savgol_filter(log_v, window_length=window, polyorder=poly, mode="interp")

    alpha = np.gradient(log_v, log_i)
    curvature = np.gradient(alpha, log_i)
    if not np.isfinite(curvature).any():
        return {}
    idx = int(np.nanargmax(curvature))
    knee_score = float(curvature[idx])
    i_knee = float(i_abs[mask][idx])
    i_min = float(i_abs[mask][0])
    i_max = float(i_abs[mask][-1])
    norm = float((i_knee - i_min) / (i_max - i_min)) if i_max > i_min else np.nan

    edge_frac = float(config.get("phase5", {}).get("knee_edge_frac", 0.05))
    knee_valid = bool(np.isfinite(norm) and 0.0 <= norm <= 1.0 and knee_score > 0)
    knee_edge = bool(knee_valid and (norm <= edge_frac or norm >= 1.0 - edge_frac))

    return {
        "I_knee": i_knee,
        "I_knee_norm": norm,
        "knee_score": knee_score,
        "knee_valid": knee_valid,
        "knee_edge_flag": knee_edge,
    }


def _phi0_proxy(i_abs: np.ndarray, v_even: np.ndarray, v_odd: np.ndarray, eps: float) -> dict:
    if i_abs.size < 3:
        return {}
    integ_even = float(np.trapezoid(v_even, i_abs))
    integ_odd = float(np.trapezoid(v_odd, i_abs))
    denom = abs(integ_even) + eps
    phi0 = float(math.atan2(integ_odd, integ_even))
    phi0_sign = 1.0 if phi0 > 0 else (-1.0 if phi0 < 0 else 0.0)
    confidence = float(min(1.0, abs(integ_odd) / denom))
    return {
        "phi0_proxy": phi0,
        "phi0_sign": phi0_sign,
        "phi0_confidence": confidence,
        "phi0_method": "odd_integral",
    }


def _odd_ratios(v_even: np.ndarray, v_odd: np.ndarray, eps: float) -> tuple[float, float]:
    denom = float(np.mean(np.abs(v_even)) + eps)
    odd_ratio = float(np.mean(np.abs(v_odd)) / denom)
    v_combined = np.concatenate([v_even + v_odd, v_even - v_odd])
    odd_energy = float(np.linalg.norm(v_odd) / (np.linalg.norm(v_combined) + eps))
    return odd_ratio, odd_energy


def _parity_pre_post(
    i_abs: np.ndarray,
    v_even: np.ndarray,
    v_odd: np.ndarray,
    i_knee: float | None,
    config: dict,
) -> dict:
    if i_knee is None or not np.isfinite(i_knee):
        return {}
    pre_mask = i_abs < i_knee
    post_mask = i_abs > i_knee
    if not pre_mask.any() or not post_mask.any():
        return {}

    eps = float(config.get("phase5", {}).get("eps", 1e-12))
    odd_pre = float(np.mean(np.abs(v_odd[pre_mask])) / (np.mean(np.abs(v_even[pre_mask])) + eps))
    odd_post = float(np.mean(np.abs(v_odd[post_mask])) / (np.mean(np.abs(v_even[post_mask])) + eps))

    sign_eps = float(config.get("phase5", {}).get("sign_eps", 1e-15))
    mode_pre, cons_pre, ent_pre, _ = _odd_sign_stats(v_odd[pre_mask], sign_eps)
    mode_post, cons_post, ent_post, _ = _odd_sign_stats(v_odd[post_mask], sign_eps)

    stable_thresh = float(config.get("phase5", {}).get("parity_stable_thresh", 0.8))
    unstable_thresh = float(config.get("phase5", {}).get("parity_unstable_thresh", 0.6))
    parity_lock = bool(cons_post is not None and cons_pre is not None and cons_post >= stable_thresh and cons_pre <= unstable_thresh)
    parity_flip = bool(mode_pre is not None and mode_post is not None and mode_pre != mode_post)

    return {
        "odd_ratio_pre": odd_pre,
        "odd_ratio_post": odd_post,
        "odd_sign_mode_pre": mode_pre,
        "odd_sign_consistency_pre": cons_pre,
        "odd_sign_entropy_pre": ent_pre,
        "odd_sign_mode_post": mode_post,
        "odd_sign_consistency_post": cons_post,
        "odd_sign_entropy_post": ent_post,
        "parity_lock_flag": parity_lock,
        "parity_flip_at_knee": parity_flip,
    }


def _build_full_grid(i_abs: np.ndarray, v_pos: np.ndarray, v_neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    neg_i = -i_abs[::-1]
    neg_v = v_neg[::-1]
    if i_abs.size > 1 and i_abs[0] == 0:
        neg_i = neg_i[:-1]
        neg_v = neg_v[:-1]
    full_i = np.concatenate([neg_i, i_abs])
    full_v = np.concatenate([neg_v, v_pos])
    return full_i, full_v


def _phase_scramble(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if values.size < 4:
        return values.copy()
    mean = float(np.nanmean(values))
    centered = values - mean
    spectrum = np.fft.rfft(centered)
    mag = np.abs(spectrum)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.size)
    phases[0] = 0.0
    if spectrum.size > 1 and values.size % 2 == 0:
        phases[-1] = 0.0
    scrambled = mag * np.exp(1j * phases)
    scrambled[0] = spectrum[0]
    if spectrum.size > 1 and values.size % 2 == 0:
        scrambled[-1] = spectrum[-1]
    out = np.fft.irfft(scrambled, n=values.size)
    return out + mean


def _perm_test_median(a: np.ndarray, b: np.ndarray, n_iter: int, rng: np.random.Generator) -> float:
    if a.size == 0 or b.size == 0:
        return np.nan
    observed = float(np.median(a) - np.median(b))
    combined = np.concatenate([a, b])
    count = 0
    for _ in range(n_iter):
        rng.shuffle(combined)
        aa = combined[: a.size]
        bb = combined[a.size :]
        stat = float(np.median(aa) - np.median(bb))
        if abs(stat) >= abs(observed):
            count += 1
    return float((count + 1) / (n_iter + 1))


def _compute_stats(values: np.ndarray) -> tuple[float | None, float | None]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None
    mean = float(values.mean())
    std = float(values.std(ddof=1) if values.size > 1 else 0.0)
    return mean, std


def _z_score(obs: float, mean: float | None, std: float | None) -> float | None:
    if mean is None or std is None or std == 0.0 or not np.isfinite(std):
        return None
    return float((obs - mean) / std)


def _select_exemplar(df: pd.DataFrame, condition: str) -> pd.Series | None:
    subset = df[(df["condition_group"] == condition) & df["parity_valid"].fillna(False)]
    subset = subset.dropna(subset=["odd_ratio_post"])
    if subset.empty:
        return None
    return subset.sort_values("odd_ratio_post", ascending=False).iloc[0]


def _save_plot(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))


def _set_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (6.5, 4.5),
            "figure.dpi": 300,
            "font.family": "sans-serif",
            "lines.linewidth": 1.5,
        }
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 CPR-grade analysis for IV data.")
    parser.add_argument("--in", dest="in_dir", default="Data_for_analysis/D4.2", help="Input folder")
    parser.add_argument("--glob", default="**/*", help="Glob for input files")
    parser.add_argument("--out", default="results/phase5", help="Output folder")
    parser.add_argument("--config", default="config.yaml", help="Config path")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(args.seed)
    setup_logging("results/reports", name="jjparse.phase5")

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_files(in_dir, args.glob)
    if args.max_files:
        files = files[: args.max_files]
    LOGGER.info("Phase5 reading %d files from %s", len(files), in_dir)

    rows = []
    cache: list[SegmentCache] = []
    non_monotonic = 0
    non_monotonic_examples: list[str] = []
    spacing_outliers = 0
    spacing_examples: list[str] = []

    min_points = int(config.get("phase5", {}).get("min_points", 10))
    grid_n = int(config.get("phase5", {}).get("grid_n", 401))
    min_overlap = float(config.get("phase5", {}).get("min_overlap", 0.8))
    eps = float(config.get("phase5", {}).get("eps", 1e-12))
    sign_eps = float(config.get("phase5", {}).get("sign_eps", 1e-15))
    monotonic_thresh = float(config.get("phase5", {}).get("monotonicity_threshold", 0.95))
    spacing_warn = float(config.get("phase5", {}).get("spacing_cv_warn", 0.2))

    for path in files:
        i_vals, v_vals, _ = _parse_iv_file(path)
        if i_vals.size < 3 or v_vals.size < 3:
            LOGGER.warning("Skipping %s (insufficient data)", path)
            continue
        file_id = _make_file_id(path, in_dir)
        condition_group, condition_value = _infer_condition(path, in_dir)

        i_min = float(np.nanmin(i_vals))
        i_max = float(np.nanmax(i_vals))
        has_both_signs = bool(i_min < 0 and i_max > 0)
        monotonicity, sweep_dir, n_segments = _sweep_monotonicity(i_vals)
        spacing_cv = _spacing_cv(i_vals)
        if spacing_cv is not None and spacing_cv > spacing_warn:
            spacing_outliers += 1
            if len(spacing_examples) < 5:
                spacing_examples.append(file_id)
        if monotonicity < monotonic_thresh:
            non_monotonic += 1
            if len(non_monotonic_examples) < 5:
                non_monotonic_examples.append(file_id)

        segments = _split_segments(i_vals, v_vals, min_points)
        for seg in segments:
            idx = seg["indices"]
            seg_i = i_vals[idx]
            seg_v = v_vals[idx]
            mask = np.isfinite(seg_i) & np.isfinite(seg_v)
            seg_i = seg_i[mask]
            seg_v = seg_v[mask]
            if seg_i.size < min_points:
                continue

            interp = _interp_even_odd(seg_i, seg_v, grid_n, min_overlap)
            parity_valid = bool(interp.get("parity_valid", False))

            odd_ratio = np.nan
            odd_energy = np.nan
            odd_mode = np.nan
            odd_cons = np.nan
            odd_entropy = np.nan
            odd_sign_p = np.nan
            knee = {}
            phi0 = {}
            parity_knee = {}

            if parity_valid:
                i_abs = interp["i_abs_grid"]
                v_pos_grid = interp["v_pos_grid"]
                v_neg_grid = interp["v_neg_grid"]
                v_even = 0.5 * (v_pos_grid + v_neg_grid)
                v_odd = 0.5 * (v_pos_grid - v_neg_grid)
                v_mag = 0.5 * (np.abs(v_pos_grid) + np.abs(v_neg_grid))

                odd_ratio, odd_energy = _odd_ratios(v_even, v_odd, eps)
                odd_mode, odd_cons, odd_entropy, odd_sign_p = _odd_sign_stats(v_odd, sign_eps)
                knee = _knee_loglog(i_abs, v_mag, config)
                phi0 = _phi0_proxy(i_abs, v_even, v_odd, eps)
                parity_knee = _parity_pre_post(i_abs, v_even, v_odd, knee.get("I_knee"), config)

                cache.append(
                    SegmentCache(
                        file_id=file_id,
                        segment_id=int(seg["segment_id"]),
                        condition_group=condition_group,
                        condition_value=condition_value,
                        i_pos=interp["i_pos"],
                        v_pos=interp["v_pos"],
                        i_neg=interp["i_neg"],
                        v_neg=interp["v_neg"],
                        i_max_common=float(interp["i_max_common"]),
                        i_abs_grid=i_abs,
                        v_pos_grid=v_pos_grid,
                        v_neg_grid=v_neg_grid,
                        i_knee=knee.get("I_knee"),
                        knee_valid=bool(knee.get("knee_valid", False)),
                    )
                )

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
                "has_both_signs": has_both_signs,
                "sweep_monotonicity": monotonicity,
                "sweep_direction": sweep_dir,
                "n_segments": n_segments,
                "spacing_cv": spacing_cv,
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
            rows.append(row)

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        LOGGER.warning("No metrics generated.")
        return

    LOGGER.info("Non-monotonic sweeps: %d (examples=%s)", non_monotonic, ",".join(non_monotonic_examples))
    LOGGER.info("Spacing outliers: %d (examples=%s)", spacing_outliers, ",".join(spacing_examples))

    metrics_path = out_dir / "phase5_metrics_long.csv"
    metrics.to_csv(metrics_path, index=False)
    LOGGER.info("Wrote %s", metrics_path)

    summary_rows = []
    for cond, group in metrics.groupby("condition_group"):
        summary_rows.append(
            {
                "condition_group": cond,
                "n_segments": int(len(group)),
                "odd_ratio_L1_median": float(np.nanmedian(group["odd_ratio_L1"])),
                "odd_ratio_post_median": float(np.nanmedian(group.get("odd_ratio_post"))),
                "phi0_proxy_median": float(np.nanmedian(np.abs(group.get("phi0_proxy")))),
                "knee_score_median": float(np.nanmedian(group.get("knee_score"))),
                "parity_lock_fraction": float(np.nanmean(group.get("parity_lock_flag"))),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df["pipeline_version"] = PIPELINE_VERSION
    summary_path = out_dir / "phase5_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    comparisons = []
    rng = np.random.default_rng(int(config.get("phase5", {}).get("nulls", {}).get("seed", 123)))
    cond_zero_thresh = float(config.get("phase5", {}).get("cond_zero_thresh", 1e-12))
    perm_n = 500

    for cond, group in metrics.groupby("condition_group"):
        if "condition_value" not in group.columns:
            continue
        zero_mask = group["condition_value"].abs() <= cond_zero_thresh
        nonzero_mask = group["condition_value"].abs() > cond_zero_thresh
        if zero_mask.sum() < 5 or nonzero_mask.sum() < 5:
            continue
        for metric in ["odd_ratio_post", "phi0_proxy", "I_knee_norm"]:
            vals_zero = pd.to_numeric(group.loc[zero_mask, metric], errors="coerce").dropna().to_numpy()
            vals_nonzero = pd.to_numeric(group.loc[nonzero_mask, metric], errors="coerce").dropna().to_numpy()
            if vals_zero.size < 5 or vals_nonzero.size < 5:
                continue
            ks = ks_2samp(vals_zero, vals_nonzero)
            perm_p = _perm_test_median(vals_zero, vals_nonzero, perm_n, rng)
            comparisons.append(
                {
                    "condition_group": cond,
                    "metric": metric,
                    "n_zero": int(vals_zero.size),
                    "n_nonzero": int(vals_nonzero.size),
                    "ks_stat": float(ks.statistic),
                    "ks_p": float(ks.pvalue),
                    "perm_p": float(perm_p),
                }
            )
    comparisons_df = pd.DataFrame(comparisons)
    if comparisons_df.empty:
        comparisons_df = pd.DataFrame(
            columns=[
                "condition_group",
                "metric",
                "n_zero",
                "n_nonzero",
                "ks_stat",
                "ks_p",
                "perm_p",
                "pipeline_version",
            ]
        )
    else:
        comparisons_df["pipeline_version"] = PIPELINE_VERSION
    comparisons_path = out_dir / "phase5_condition_tests.csv"
    comparisons_df.to_csv(comparisons_path, index=False)

    null_cfg = config.get("phase5", {}).get("nulls", {})
    n_sign = max(100, int(null_cfg.get("sign_n", 100)))
    n_phase = max(100, int(null_cfg.get("phase_n", 100)))
    n_pair = max(100, int(null_cfg.get("pair_n", 100)))
    pair_tol = float(null_cfg.get("pair_match_tol", 0.2))

    def _stats_from_metrics(df: pd.DataFrame) -> tuple[float | None, float | None, float | None]:
        odd_post = pd.to_numeric(df.get("odd_ratio_post"), errors="coerce").dropna()
        lock = df.get("parity_lock_flag")
        lock = lock[lock.notna()] if lock is not None else pd.Series(dtype=float)
        phi0 = pd.to_numeric(df.get("phi0_proxy"), errors="coerce").dropna()
        t_odd = float(np.nanmedian(odd_post)) if not odd_post.empty else np.nan
        t_lock = float(np.nanmean(lock)) if not lock.empty else np.nan
        t_phi0 = float(np.nanmedian(np.abs(phi0))) if not phi0.empty else np.nan
        return t_odd, t_lock, t_phi0

    obs_t_odd, obs_t_lock, obs_t_phi0 = _stats_from_metrics(metrics)

    null_rows = []
    rng_null = np.random.default_rng(int(null_cfg.get("seed", 123)))

    def _compute_from_arrays(i_abs, v_pos, v_neg) -> dict:
        v_even = 0.5 * (v_pos + v_neg)
        v_odd = 0.5 * (v_pos - v_neg)
        v_mag = 0.5 * (np.abs(v_pos) + np.abs(v_neg))
        knee = _knee_loglog(i_abs, v_mag, config)
        parity_knee = _parity_pre_post(i_abs, v_even, v_odd, knee.get("I_knee"), config)
        phi0 = _phi0_proxy(i_abs, v_even, v_odd, eps)
        return {
            "odd_ratio_post": parity_knee.get("odd_ratio_post"),
            "parity_lock_flag": parity_knee.get("parity_lock_flag"),
            "phi0_proxy": phi0.get("phi0_proxy"),
        }

    def _collect_null(stats_list: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            full_i, full_v = _build_full_grid(seg.i_abs_grid, seg.v_pos_grid, seg.v_neg_grid)
            scrambled = _phase_scramble(full_v, rng_null)
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
            i_abs = np.linspace(0.0, i_max_common, grid_n)
            v_pos = np.interp(i_abs, np.sort(seg.i_pos), seg.v_pos[np.argsort(seg.i_pos)])
            v_neg = np.interp(i_abs, np.sort(other.i_neg), other.v_neg[np.argsort(other.i_neg)])
            stats.append(_compute_from_arrays(i_abs, v_pos, v_neg))
        pair_null.append(stats)

    def _summarize_null(label: str, null_stats: list[list[dict]]) -> list[dict]:
        t_odd = []
        t_lock = []
        t_phi0 = []
        for stats in null_stats:
            odd, lock, phi0 = _collect_null(stats)
            if odd.size:
                t_odd.append(float(np.nanmedian(odd)))
            if lock.size:
                t_lock.append(float(np.nanmean(lock)))
            if phi0.size:
                t_phi0.append(float(np.nanmedian(phi0)))
        t_odd = np.asarray(t_odd)
        t_lock = np.asarray(t_lock)
        t_phi0 = np.asarray(t_phi0)
        rows = []
        for name, obs, values in [
            ("odd_ratio_post", obs_t_odd, t_odd),
            ("parity_lock_fraction", obs_t_lock, t_lock),
            ("phi0_proxy_abs", obs_t_phi0, t_phi0),
        ]:
            mean, std = _compute_stats(values)
            rows.append(
                {
                    "statistic": name,
                    "null_model": label,
                    "observed": obs,
                    "null_mean": mean,
                    "null_std": std,
                    "z_score": _z_score(obs, mean, std),
                    "n": int(values.size),
                }
            )
        return rows

    null_rows.extend(_summarize_null("sign_randomization", sign_null))
    null_rows.extend(_summarize_null("phase_scramble", phase_null))
    null_rows.extend(_summarize_null("pair_breaking", pair_null))

    null_df = pd.DataFrame(null_rows)
    null_df["pipeline_version"] = PIPELINE_VERSION
    null_path = out_dir / "phase5_null_summary.csv"
    null_df.to_csv(null_path, index=False)

    import matplotlib.pyplot as plt

    _set_style()
    top_conditions = (
        metrics["condition_group"].value_counts().head(6).index.tolist()
        if "condition_group" in metrics.columns
        else []
    )

    for cond in top_conditions:
        exemplar = _select_exemplar(metrics, cond)
        if exemplar is None:
            continue
        seg = next((c for c in cache if c.file_id == exemplar["file_id"] and c.segment_id == exemplar["segment_id"]), None)
        if seg is None:
            continue
        v_even = 0.5 * (seg.v_pos_grid + seg.v_neg_grid)
        v_odd = 0.5 * (seg.v_pos_grid - seg.v_neg_grid)

        fig, ax = plt.subplots()
        ax.plot(seg.i_abs_grid, seg.v_pos_grid, color="#1f77b4", label="V(+I)", alpha=0.8)
        ax.plot(seg.i_abs_grid, seg.v_neg_grid, color="#ff7f0e", label="V(-I)", alpha=0.8)
        ax.plot(seg.i_abs_grid, v_even, color="#000000", linestyle="--", label="V_even", alpha=0.8)
        ax.plot(seg.i_abs_grid, v_odd, color="#666666", linestyle="-", label="V_odd", alpha=0.8)
        if seg.knee_valid and seg.i_knee is not None:
            ax.axvline(seg.i_knee, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.9)
        ax.set_xlabel("|I| (A)")
        ax.set_ylabel("V (V)")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
        ax.set_title(f"Even/odd overlay: {cond}")
        ax.legend(fontsize=8)
        _save_plot(fig, plots_dir / f"example_even_odd__{cond}")
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(seg.i_abs_grid, v_odd, color="#666666", label="V_odd", alpha=0.8)
        if seg.knee_valid and seg.i_knee is not None:
            ax.axvline(seg.i_knee, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.9)
        ax.set_xlabel("|I| (A)")
        ax.set_ylabel("V_odd (V)")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
        ax.set_title(f"V_odd with knee: {cond}")
        ax.legend(fontsize=8)
        _save_plot(fig, plots_dir / f"odd_with_knee__{cond}")
        plt.close(fig)

    fig, ax = plt.subplots()
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    for idx, cond in enumerate(top_conditions):
        subset = metrics[metrics["condition_group"] == cond]
        ax.scatter(
            subset["odd_ratio_pre"],
            subset["odd_ratio_post"],
            s=12,
            alpha=0.8,
            label=cond,
            color="#1f77b4" if idx == 0 else "#ff7f0e" if idx == 1 else "#666666",
        )
    ax.set_xlabel("odd_ratio_pre")
    ax.set_ylabel("odd_ratio_post")
    ax.set_title("Odd ratio pre vs post knee")
    ax.legend(fontsize=8, loc="best")
    _save_plot(fig, plots_dir / "odd_ratio_pre_vs_post")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    for idx, cond in enumerate(top_conditions):
        subset = metrics[metrics["condition_group"] == cond]
        vals = pd.to_numeric(subset["phi0_proxy"], errors="coerce").dropna()
        if vals.empty:
            continue
        color = "#1f77b4" if idx == 0 else "#ff7f0e" if idx == 1 else "#666666"
        ax.hist(vals, bins=30, alpha=0.5, color=color, label=cond)
    ax.set_xlabel("phi0_proxy (rad)")
    ax.set_ylabel("count")
    ax.set_title("Phi0 proxy by condition")
    ax.legend(fontsize=8, loc="best")
    _save_plot(fig, plots_dir / "phi0_proxy_hist_by_condition")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4, axis="y")
    lock_rates = []
    labels = []
    for cond in top_conditions:
        subset = metrics[metrics["condition_group"] == cond]
        lock_rates.append(float(np.nanmean(subset.get("parity_lock_flag"))))
        labels.append(cond)
    ax.bar(labels, lock_rates, color="#666666", alpha=0.8)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("fraction")
    ax.set_title("Parity lock after knee")
    _save_plot(fig, plots_dir / "parity_lock_by_condition")
    plt.close(fig)

    lines = []
    lines.append("# Phase 5 Summary")
    lines.append("")
    lines.append(f"- Pipeline version: {PIPELINE_VERSION}")
    lines.append(f"- Input folder: {in_dir}")
    file_count = int(metrics["file_id"].nunique())
    parity_count = int(metrics["parity_valid"].sum())
    lines.append(f"- Total files: {file_count}")
    lines.append(f"- Total segments: {len(metrics)}")
    lines.append(f"- Parity-valid segments: {parity_count}")
    lines.append("")
    lines.append("## Core checks")
    lines.append("")
    files_with_both = int(metrics.groupby("file_id")["has_both_signs"].max().sum())
    lines.append(f"- Files with both signs: {files_with_both}")
    lines.append(f"- Median sweep monotonicity (file-order): {float(np.nanmedian(metrics['sweep_monotonicity'])):.3g}")
    lines.append(f"- Median overlap fraction: {float(np.nanmedian(metrics['f_overlap'])):.3g}")
    lines.append(f"- Non-monotonic sweeps: {non_monotonic}")
    if non_monotonic_examples:
        lines.append(f"- Non-monotonic examples: {', '.join(non_monotonic_examples)}")
    lines.append(f"- Spacing outliers: {spacing_outliers}")
    if spacing_examples:
        lines.append(f"- Spacing outlier examples: {', '.join(spacing_examples)}")
    lines.append("")
    lines.append("## Metrics (medians)")
    lines.append("")
    for key in ["odd_ratio_L1", "odd_energy_L2", "odd_ratio_pre", "odd_ratio_post", "phi0_proxy", "I_knee_norm"]:
        vals = pd.to_numeric(metrics.get(key), errors="coerce").dropna()
        if vals.empty:
            lines.append(f"- {key}: n=0")
        else:
            lines.append(f"- {key}: n={len(vals)}, median={float(np.nanmedian(vals)):.6g}")
    lines.append("")
    lines.append("## Condition tests (zero vs nonzero)")
    lines.append("")
    if comparisons_df.empty:
        lines.append("- No condition comparisons available.")
    else:
        for _, row in comparisons_df.iterrows():
            lines.append(
                f"- {row['condition_group']} {row['metric']}: ks_p={row['ks_p']:.3g}, perm_p={row['perm_p']:.3g}"
            )
    lines.append("")
    lines.append("## Null models (algorithmic significance)")
    lines.append("")
    def _fmt(val: float | None) -> str:
        if val is None or not np.isfinite(val):
            return "NA"
        return f"{val:.4g}"

    for _, row in null_df.iterrows():
        lines.append(
            f"- {row['null_model']} {row['statistic']}: observed={_fmt(row['observed'])}, "
            f"null_mean={_fmt(row['null_mean'])}, null_std={_fmt(row['null_std'])}, z={_fmt(row['z_score'])}"
        )
    lines.append("")
    lines.append("## What was tested")
    lines.append("")
    lines.append("- Even/odd decomposition of V(I) on symmetric current grids.")
    lines.append("- Log-log knee detection on |V| vs |I| with pre/post parity metrics.")
    lines.append("- Phi0 proxy from odd-integral and parity-lock tests.")
    lines.append("- Nulls: sign randomization, phase scrambling, pair-breaking.")
    lines.append("")
    lines.append("## What survived nulls")
    lines.append("")
    def _z(stat: str, model: str) -> float | None:
        match = null_df[(null_df["statistic"] == stat) & (null_df["null_model"] == model)]
        if match.empty:
            return None
        val = match["z_score"].iloc[0]
        return float(val) if np.isfinite(val) else None

    z_phase_odd = _z("odd_ratio_post", "phase_scramble")
    z_pair_odd = _z("odd_ratio_post", "pair_breaking")
    if z_phase_odd is not None and abs(z_phase_odd) >= 2:
        lines.append(f"- odd_ratio_post deviates from phase-scramble null (z={z_phase_odd:.3g}).")
    if z_pair_odd is not None and abs(z_pair_odd) >= 2:
        lines.append(f"- odd_ratio_post deviates from pair-breaking null (z={z_pair_odd:.3g}).")
    if not lines[-1].startswith("- odd_ratio_post"):
        lines.append("- No metric exceeded |z|>=2 across all nulls.")
    lines.append("")
    lines.append("## What failed or is bounded")
    lines.append("")
    z_lock_phase = _z("parity_lock_fraction", "phase_scramble")
    if obs_t_lock == 0 or (z_lock_phase is not None and z_lock_phase < 0):
        lines.append("- parity_lock_fraction is at or below null expectation (no post-knee locking).")
    z_phi0_phase = _z("phi0_proxy_abs", "phase_scramble")
    if z_phi0_phase is not None and z_phi0_phase < 0:
        lines.append(f"- phi0_proxy_abs is below phase-scramble null (z={z_phi0_phase:.3g}); treat as upper bound.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- Effects are reported as algorithmic significance (z-scores vs nulls), not as physical discovery claims."
    )
    lines.append("- Metrics that fail parity-valid checks are excluded from parity and knee analyses.")

    report_path = Path("results/reports/phase5_summary.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
