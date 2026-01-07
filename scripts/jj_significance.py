"""Estimate algorithmic significance using null models."""

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
from jjparse.metrics import compute_incoherence, compute_knee_score, compute_voltage_nonreciprocity, detect_knee

LOGGER = logging.getLogger("jjparse")


def _normalize_bounds(series: pd.Series, config: dict) -> tuple[float, float] | None:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return None
    norm_cfg = config.get("incoherence", {}).get("normalize", {})
    method = str(norm_cfg.get("method", "percentile")).lower()
    if method == "minmax":
        low = float(series.min())
        high = float(series.max())
    else:
        p_low = float(norm_cfg.get("p_low", 1.0))
        p_high = float(norm_cfg.get("p_high", 99.0))
        low = float(np.nanpercentile(series, p_low))
        high = float(np.nanpercentile(series, p_high))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None
    return low, high


def _normalize_value(raw: float | None, bounds: tuple[float, float] | None) -> float | None:
    if raw is None or not np.isfinite(raw) or not bounds:
        return None
    low, high = bounds
    value = (raw - low) / (high - low)
    return float(np.clip(value, 0.0, 1.0))


def _phase_scramble(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
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


def _load_segments(
    metrics_df: pd.DataFrame,
    segments_dir: Path,
    only_bidirectional: bool = False,
) -> list[dict]:
    cache: dict[str, pd.DataFrame] = {}
    segments: list[dict] = []
    for _, row in metrics_df.iterrows():
        if only_bidirectional and not bool(row.get("is_bidirectional", False)):
            continue
        file_id = str(row.get("file_id"))
        segment_id = int(row.get("segment_id", 0))
        if file_id not in cache:
            seg_path = segments_dir / f"{file_id}.parquet"
            if not seg_path.exists():
                continue
            cache[file_id] = pd.read_parquet(seg_path)
        seg_df = cache[file_id]
        seg = seg_df[seg_df["segment_id"] == segment_id]
        if seg.empty:
            continue
        x = pd.to_numeric(seg["x"], errors="coerce").to_numpy()
        v = pd.to_numeric(seg["V_V"], errors="coerce").to_numpy()
        mask = np.isfinite(x) & np.isfinite(v)
        x = x[mask]
        v = v[mask]
        if x.size < 5:
            continue
        segments.append(
            {
                "file_id": file_id,
                "segment_id": segment_id,
                "x": x,
                "v": v,
                "H_incoh": row.get("H_incoh"),
                "knee_x": row.get("knee_x"),
                "is_bidirectional": bool(row.get("is_bidirectional", False)),
            }
        )
    return segments


def _direction_randomization_null(
    segments: list[dict],
    n_iter: int,
    grid_n: int,
    eps: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    file_ids = sorted({seg["file_id"] for seg in segments})
    file_index = {fid: idx for idx, fid in enumerate(file_ids)}
    counts0 = np.zeros(len(file_ids), dtype=int)
    counts1 = np.zeros(len(file_ids), dtype=int)

    eta_medians = []
    lock_fracs = []

    for it in range(n_iter):
        eta_vals = []
        lock_num = 0
        lock_den = 0
        for seg in segments:
            x = seg["x"]
            v = seg["v"]
            h_incoh = seg.get("H_incoh")
            if h_incoh is None or not np.isfinite(h_incoh):
                continue
            signs = rng.choice([-1.0, 1.0], size=x.size)
            x_rand = x * signs
            eta = compute_voltage_nonreciprocity(x_rand, v, grid_n=grid_n, eps=eps)
            if "eta_V_L1" in eta:
                eta_vals.append(float(eta["eta_V_L1"] / (eps + h_incoh)))
            parity_bit = None
            if "eta_V_signed" in eta:
                parity_sign = 1.0 if eta["eta_V_signed"] > 0 else -1.0
                parity_sign *= rng.choice([-1.0, 1.0])
                parity_bit = 1 if parity_sign > 0 else 0
                idx = file_index[seg["file_id"]]
                if parity_bit == 1:
                    counts1[idx] += 1
                else:
                    counts0[idx] += 1

            knee_x = seg.get("knee_x")
            if knee_x is None or not np.isfinite(knee_x):
                continue
            if "eta_V_signed" not in eta or parity_bit is None:
                continue
            knee_thresh = abs(float(knee_x))
            x_abs = np.abs(x_rand)
            pre_mask = x_abs < knee_thresh
            post_mask = x_abs > knee_thresh
            if not pre_mask.any() or not post_mask.any():
                continue
            post = compute_voltage_nonreciprocity(x_rand[post_mask], v[post_mask], grid_n=grid_n, eps=eps)
            if "eta_V_signed" not in post:
                continue
            parity_post_sign = 1.0 if post["eta_V_signed"] > 0 else -1.0
            parity_post_sign *= rng.choice([-1.0, 1.0])
            parity_post = 1 if parity_post_sign > 0 else 0
            lock_num += int(parity_post == parity_bit)
            lock_den += 1

        if eta_vals:
            eta_medians.append(float(np.median(eta_vals)))
        if lock_den:
            lock_fracs.append(float(lock_num / lock_den))

    counts_sum = counts0 + counts1
    valid = counts_sum > 0

    return np.array(eta_medians), counts0[valid], counts1[valid], np.array(lock_fracs)


def _phase_scramble_null(
    segments: list[dict],
    n_iter: int,
    grid_n: int,
    eps: float,
    rng: np.random.Generator,
    config: dict,
    bounds: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bidir_files = sorted({seg["file_id"] for seg in segments if seg.get("is_bidirectional")})
    file_index = {fid: idx for idx, fid in enumerate(bidir_files)}
    counts0 = np.zeros(len(bidir_files), dtype=int)
    counts1 = np.zeros(len(bidir_files), dtype=int)

    eta_medians = []
    lock_fracs = []

    min_segment = int(config.get("knee", {}).get("min_segment", 3))

    for it in range(n_iter):
        eta_vals = []
        lock_num = 0
        lock_den = 0
        for seg in segments:
            x = seg["x"]
            v = seg["v"]
            v_scrambled = _phase_scramble(v, rng)

            _ = compute_knee_score(x, v_scrambled, config)
            knee = detect_knee(x, v_scrambled, min_segment=min_segment)
            knee_x = knee.get("knee_x") if knee else None

            incoh = compute_incoherence(x, v_scrambled, config)
            h_raw = incoh.get("H_incoh_raw") if incoh else None
            h_incoh = _normalize_value(h_raw, bounds)

            if not seg.get("is_bidirectional"):
                continue
            eta = compute_voltage_nonreciprocity(x, v_scrambled, grid_n=grid_n, eps=eps)
            if h_incoh is not None and "eta_V_L1" in eta:
                eta_vals.append(float(eta["eta_V_L1"] / (eps + h_incoh)))
            parity_bit = None
            if "eta_V_signed" in eta:
                parity_sign = 1.0 if eta["eta_V_signed"] > 0 else -1.0
                parity_sign *= rng.choice([-1.0, 1.0])
                parity_bit = 1 if parity_sign > 0 else 0
                idx = file_index[seg["file_id"]]
                if parity_bit == 1:
                    counts1[idx] += 1
                else:
                    counts0[idx] += 1

            if knee_x is None or not np.isfinite(knee_x) or "eta_V_signed" not in eta or parity_bit is None:
                continue
            knee_thresh = abs(float(knee_x))
            x_abs = np.abs(x)
            pre_mask = x_abs < knee_thresh
            post_mask = x_abs > knee_thresh
            if not pre_mask.any() or not post_mask.any():
                continue
            post = compute_voltage_nonreciprocity(x[post_mask], v_scrambled[post_mask], grid_n=grid_n, eps=eps)
            if "eta_V_signed" not in post:
                continue
            parity_post_sign = 1.0 if post["eta_V_signed"] > 0 else -1.0
            parity_post_sign *= rng.choice([-1.0, 1.0])
            parity_post = 1 if parity_post_sign > 0 else 0
            lock_num += int(parity_post == parity_bit)
            lock_den += 1

        if eta_vals:
            eta_medians.append(float(np.median(eta_vals)))
        if lock_den:
            lock_fracs.append(float(lock_num / lock_den))

    counts_sum = counts0 + counts1
    valid = counts_sum > 0

    return np.array(eta_medians), counts0[valid], counts1[valid], np.array(lock_fracs)


def _summary_stats(values: np.ndarray) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None
    return float(values.mean()), float(values.std(ddof=1) if values.size > 1 else 0.0)


def _z_score(obs: float, mean: float | None, std: float | None) -> float | None:
    if mean is None or std is None or std == 0.0 or not np.isfinite(std):
        return None
    return float((obs - mean) / std)


def _bootstrap_parity_stability(
    counts0: np.ndarray,
    counts1: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    counts0 = np.asarray(counts0, dtype=float)
    counts1 = np.asarray(counts1, dtype=float)
    total = counts0 + counts1
    valid = total > 0
    if not np.any(valid) or n_boot <= 0:
        return np.array([])
    total = total[valid]
    p = counts1[valid] / total
    medians = []
    for _ in range(n_boot):
        draws = rng.binomial(total.astype(int), p)
        stability = np.maximum(draws, total - draws) / total
        medians.append(float(np.median(stability)))
    return np.array(medians)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute significance under null models.")
    parser.add_argument("--metrics", default="results/metrics/metrics_long.csv", help="Metrics CSV")
    parser.add_argument("--segments-dir", default="results/parsed/segments", help="Segment parquet folder")
    parser.add_argument("--out", default="results/reports", help="Output folder")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    args = parser.parse_args()

    config = load_config(args.config)
    base_seed = args.seed or config.get("null_models", {}).get("seed", 123)
    seed_everything(base_seed)
    setup_logging("results/reports")

    metrics_path = Path(args.metrics)
    segments_dir = Path(args.segments_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_path, low_memory=False)
    if df.empty:
        LOGGER.warning("No metrics in %s", metrics_path)
        return

    bidir_mask = df.get("is_bidirectional", pd.Series(False, index=df.index)).fillna(False)
    bidir_df = df[bidir_mask & df["eta_norm"].notna()]

    obs_eta = float(np.median(pd.to_numeric(bidir_df["eta_norm"], errors="coerce").dropna()))

    file_summary = df.groupby("file_id").agg(lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan)
    obs_parity = pd.to_numeric(file_summary.get("parity_stability", pd.Series(dtype=float)), errors="coerce").dropna()
    obs_b = float(np.median(obs_parity)) if not obs_parity.empty else np.nan

    knee_parity_path = out_dir / "knee_parity_summary.csv"
    obs_lock = np.nan
    if knee_parity_path.exists():
        kp = pd.read_csv(knee_parity_path)
        if not kp.empty and "fraction_lock_after" in kp.columns:
            obs_lock = float(kp["fraction_lock_after"].iloc[0])

    bounds = _normalize_bounds(df.get("H_incoh_raw", pd.Series(dtype=float)), config)
    if bounds is None:
        LOGGER.warning("Unable to compute H_incoh normalization bounds; null eta_norm may be skipped.")

    direction_n = int(config.get("null_models", {}).get("direction_n", 200))
    phase_n = int(config.get("null_models", {}).get("phase_n", 50))
    phase_max_segments = int(config.get("null_models", {}).get("phase_max_segments", 0))
    parity_boot = int(config.get("null_models", {}).get("parity_boot", 200))
    phase_all = bool(config.get("null_models", {}).get("phase_all_segments", True))
    phase_full_pass = bool(config.get("null_models", {}).get("phase_full_pass", False))

    grid_n = int(config.get("parity", {}).get("grid_n", 200))
    eps = float(config.get("metrics", {}).get("eps", 1e-12))
    rng = np.random.default_rng(base_seed)
    rng_full = np.random.default_rng(base_seed + 1)
    rng_boot = np.random.default_rng(base_seed + 2)
    rng_sample = np.random.default_rng(base_seed + 3)

    segments_bidir = _load_segments(bidir_df, segments_dir, only_bidirectional=True)
    if not segments_bidir:
        LOGGER.warning("No bidirectional segments available for null models.")
        return
    before = len(segments_bidir)
    segments_bidir = [seg for seg in segments_bidir if np.isfinite(seg.get("H_incoh", np.nan))]
    dropped = before - len(segments_bidir)
    if dropped:
        LOGGER.info("Dropped %d bidirectional segments without H_incoh for direction null.", dropped)

    LOGGER.info("Running direction-randomized null with N=%d", direction_n)
    eta_null_a, counts0_a, counts1_a, lock_null_a = _direction_randomization_null(
        segments_bidir,
        n_iter=direction_n,
        grid_n=grid_n,
        eps=eps,
        rng=rng,
    )
    tb_null_a = _bootstrap_parity_stability(counts0_a, counts1_a, parity_boot, rng_boot)

    if phase_all:
        segments_all = _load_segments(df, segments_dir, only_bidirectional=False)
    else:
        segments_all = segments_bidir

    if phase_max_segments > 0 and len(segments_all) > phase_max_segments:
        idx = rng_sample.choice(len(segments_all), size=phase_max_segments, replace=False)
        segments_all = [segments_all[i] for i in idx]
        LOGGER.info("Phase-scramble using %d sampled segments.", len(segments_all))

    if phase_full_pass and not phase_all:
        all_segments = _load_segments(df, segments_dir, only_bidirectional=False)
        total = 0
        min_segment = int(config.get("knee", {}).get("min_segment", 3))
        for seg in all_segments:
            v_scrambled = _phase_scramble(seg["v"], rng_full)
            _ = compute_knee_score(seg["x"], v_scrambled, config)
            _ = detect_knee(seg["x"], v_scrambled, min_segment=min_segment)
            total += 1
        LOGGER.info("Phase-scramble knee metrics recomputed for %d segments (single pass).", total)

    LOGGER.info("Running phase-scrambled null with N=%d", phase_n)
    eta_null_b, counts0_b, counts1_b, lock_null_b = _phase_scramble_null(
        segments_all,
        n_iter=phase_n,
        grid_n=grid_n,
        eps=eps,
        rng=rng,
        config=config,
        bounds=bounds,
    )
    tb_null_b = _bootstrap_parity_stability(counts0_b, counts1_b, parity_boot, rng_boot)

    rows = []
    for label, eta_null, tb_null, lock_null in [
        ("direction_randomization", eta_null_a, tb_null_a, lock_null_a),
        ("phase_scramble", eta_null_b, tb_null_b, lock_null_b),
    ]:
        eta_mean, eta_std = _summary_stats(eta_null)
        tb_mean, tb_std = _summary_stats(tb_null)
        lock_mean, lock_std = _summary_stats(lock_null)

        rows.append(
            {
                "statistic": "T_eta",
                "null_model": label,
                "observed": obs_eta,
                "null_mean": eta_mean,
                "null_std": eta_std,
                "z_score": _z_score(obs_eta, eta_mean, eta_std),
                "n": int(len(eta_null)),
            }
        )
        rows.append(
            {
                "statistic": "T_b",
                "null_model": label,
                "observed": obs_b,
                "null_mean": tb_mean,
                "null_std": tb_std,
                "z_score": _z_score(obs_b, tb_mean, tb_std),
                "n": int(len(tb_null)),
            }
        )
        rows.append(
            {
                "statistic": "T_lock",
                "null_model": label,
                "observed": obs_lock,
                "null_mean": lock_mean,
                "null_std": lock_std,
                "z_score": _z_score(obs_lock, lock_mean, lock_std),
                "n": int(len(lock_null)),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "significance_summary.csv", index=False)

    def _find_z(stat: str, model: str) -> float | None:
        match = summary_df[(summary_df["statistic"] == stat) & (summary_df["null_model"] == model)]
        if match.empty:
            return None
        return match["z_score"].iloc[0]

    lines = []
    lines.append("# Significance summary")
    lines.append("")
    lines.append("## Null models")
    lines.append("")
    lines.append(
        f"- Direction randomization: per-point sign flips (50/50) on bidirectional traces; N={direction_n}."
    )
    lines.append(
        f"- Phase scramble: randomize FFT phases per segment (amplitude preserved); N={phase_n}, phase_all_segments={phase_all}, phase_full_pass={phase_full_pass}, phase_max_segments={phase_max_segments}."
    )
    lines.append(
        "- Parity-bit null handling: eta_V_signed is randomly sign-flipped (50/50) per evaluation to emulate direction-label ambiguity."
    )
    lines.append(
        f"- Parity stability null uses a binomial bootstrap over per-file parity counts; bootstrap_n={parity_boot}."
    )
    lines.append("")
    lines.append("## Observed statistics")
    lines.append("")
    lines.append(f"- T_eta (median eta_norm): {obs_eta:.6g}")
    lines.append(f"- T_b (median parity_stability): {obs_b:.6g}")
    lines.append(f"- T_lock (fraction parity locks after knee): {obs_lock:.6g}")
    lines.append("")
    lines.append("## Z-scores")
    lines.append("")
    for model in ["direction_randomization", "phase_scramble"]:
        z_eta = _find_z("T_eta", model)
        z_b = _find_z("T_b", model)
        z_lock = _find_z("T_lock", model)
        lines.append(f"- {model}: z_eta={z_eta:.4g}, z_b={z_b:.4g}, z_lock={z_lock:.4g}")
        if all(val is not None for val in [z_eta, z_b, z_lock]):
            z_combined = float(np.sqrt(z_eta**2 + z_b**2 + z_lock**2))
            lines.append(f"- {model}: z_combined={z_combined:.4g} (assumes approximate independence)")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    name_map = {
        "direction_randomization": "direction-randomized",
        "phase_scramble": "phase-scrambled",
    }
    for model in ["direction_randomization", "phase_scramble"]:
        z_eta = _find_z("T_eta", model)
        z_lock = _find_z("T_lock", model)
        if z_eta is None or z_lock is None:
            continue
        z_level = min(abs(z_eta), abs(z_lock))
        label = name_map.get(model, model)
        lines.append(
            f"- Under {label} null models, observed odd-channel strength and post-knee parity locking deviate from null expectations at the z~{z_level:.3g} level."
        )
    lines.append(
        "- Sigma here is algorithmic significance, not a particle-physics discovery claim; null definitions, N, and dependence notes are reported above."
    )

    (out_dir / "significance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
