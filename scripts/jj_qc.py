"""Generate QC plots for JJ .dat files."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jjparse.config import load_config, seed_everything, setup_logging
from jjparse.io import make_file_id, read_dat
from jjparse.metrics import compute_changepoints, compute_knee_score, compute_voltage_nonreciprocity, detect_knee, split_segments
from jjparse.plots import plot_cpr, plot_fraunhofer, plot_histograms, plot_iv, plot_missing, plot_voltage_trace
from jjparse.preprocess import apply_unit_conversions, infer_phase_radians
from jjparse.schema import coerce_types, detect_capabilities, infer_primary_axis, standardize_columns

LOGGER = logging.getLogger("jjparse")


def _iter_files(in_dir: Path, glob_pattern: str) -> list[Path]:
    files = sorted(in_dir.glob(glob_pattern))
    return [path for path in files if path.is_file()]


def _parse_cp_list(cp_list: str | None) -> list[float]:
    if not cp_list:
        return []
    out = []
    for item in str(cp_list).split(";"):
        try:
            out.append(float(item))
        except ValueError:
            continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QC plots for JJ .dat files.")
    parser.add_argument("--in", dest="in_dir", default="data/raw", help="Input data folder")
    parser.add_argument("--out", dest="out_dir", default="results/qc_plots", help="Output folder")
    parser.add_argument("--glob", default="**/*.dat", help="Glob pattern for data files")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(args.seed)
    setup_logging("results/reports")

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_files(in_dir, args.glob)
    if args.max_files:
        files = files[: args.max_files]

    for path in tqdm(files, desc="QC"):
        df_raw, meta = read_dat(path)
        if df_raw.empty:
            LOGGER.warning("No data parsed for %s", path)
            continue

        file_id = make_file_id(path, in_dir)
        df_std = standardize_columns(df_raw, meta)
        df_std = coerce_types(df_std)
        df_std = apply_unit_conversions(df_std, meta, config)
        df_std = infer_phase_radians(df_std, meta, config)

        plots = {
            "iv": plot_iv,
            "cpr": plot_cpr,
            "fraunhofer": plot_fraunhofer,
            "hist": plot_histograms,
            "missing": plot_missing,
        }

        for suffix, func in plots.items():
            out_path = out_dir / f"{file_id}__{suffix}.png"
            if out_path.exists() and not args.overwrite:
                continue
            if suffix in ["iv", "cpr"]:
                ok = func(df_std, str(out_path), max_points=config.get("qc", {}).get("max_points", 5000))
            elif suffix == "hist":
                ok = func(df_std, str(out_path), bins=config.get("qc", {}).get("hist_bins", 50))
            else:
                ok = func(df_std, str(out_path))
            if not ok and out_path.exists():
                out_path.unlink(missing_ok=True)

        capabilities = detect_capabilities(df_std)
        if not capabilities.get("cap_has_V"):
            continue

        x_name, x_values = infer_primary_axis(df_std, meta)
        data = df_std.copy()
        data = data.assign(x=x_values)
        data = data[["x", "V_V"]].copy()
        data = data[np.isfinite(data["x"]) & np.isfinite(data["V_V"])].reset_index(drop=True)
        if data.empty:
            continue

        x = data["x"].to_numpy()
        v = data["V_V"].to_numpy()

        segments = split_segments(x, v, min_points=config.get("segments", {}).get("min_points", 5))
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

        for seg in segments:
            seg_id = seg["segment_id"]
            seg_x = seg["x"]
            seg_v = seg["V"]
            knee = detect_knee(seg_x, seg_v, min_segment=config.get("knee", {}).get("min_segment", 3))
            cp = compute_changepoints(seg_x, seg_v, config)
            score = compute_knee_score(seg_x, seg_v, config)

            knee_x = knee.get("knee_x") if knee else None
            cp_list = _parse_cp_list(cp.get("cp_x_list") if cp else None)
            annotations = []
            if score and "K_max" in score:
                annotations.append(f"K_max={score['K_max']:.3g}")

            out_path = out_dir / f"{file_id}__V_vs_x__seg{seg_id}.png"
            if out_path.exists() and not args.overwrite:
                continue
            plot_voltage_trace(
                seg_x,
                seg_v,
                str(out_path),
                x_label=x_name,
                title=f"V vs {x_name} (seg {seg_id}, {seg['direction']})",
                knee_x=knee_x,
                cp_x=cp_list,
                max_points=config.get("qc", {}).get("max_points", 5000),
                annotations=annotations,
            )

        up_segments = [seg for seg in segments if seg.get("direction") == "up"]
        down_segments = [seg for seg in segments if seg.get("direction") == "down"]
        if up_segments and down_segments:
            up = max(up_segments, key=lambda seg: seg["x"].size)
            down = max(down_segments, key=lambda seg: seg["x"].size)
            out_path = out_dir / f"{file_id}__V_vs_x__hyst.png"
            if not out_path.exists() or args.overwrite:
                plot_voltage_trace(
                    up["x"],
                    up["V"],
                    str(out_path),
                    x_label=x_name,
                    title="Hysteresis overlay",
                    overlays=[(down["x"], down["V"], "down")],
                    max_points=config.get("qc", {}).get("max_points", 5000),
                )

        x_finite = x[np.isfinite(x)]
        if x_finite.size and np.nanmin(x_finite) < 0 and np.nanmax(x_finite) > 0:
            pos_mask = x > 0
            neg_mask = x < 0
            eta = compute_voltage_nonreciprocity(
                x,
                v,
                grid_n=config.get("parity", {}).get("grid_n", 200),
                eps=config.get("parity", {}).get("eps", 1e-12),
            )
            annotations = []
            if "eta_V_L1" in eta:
                annotations.append(f"eta_V_L1={eta['eta_V_L1']:.3g}")
            if "eta_V_signed" in eta:
                annotations.append(f"eta_V_signed={eta['eta_V_signed']:.3g}")

            x_abs = np.abs(x)
            out_path = out_dir / f"{file_id}__V_vs_x__pm.png"
            if not out_path.exists() or args.overwrite:
                plot_voltage_trace(
                    x_abs[pos_mask],
                    v[pos_mask],
                    str(out_path),
                    x_label=f"|{x_name}|",
                    title="Plus/Minus overlay",
                    overlays=[(x_abs[neg_mask], v[neg_mask], "minus")],
                    max_points=config.get("qc", {}).get("max_points", 5000),
                    annotations=annotations,
                )


if __name__ == "__main__":
    main()
