"""Generate validation plots for selected segments."""

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
from jjparse.plots import plot_voltage_trace

LOGGER = logging.getLogger("jjparse")


def _pick_one_per_file(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for _, group in df.groupby("file_id"):
        if group.shape[0] == 1:
            rows.append(group.iloc[0])
        else:
            idx = rng.integers(0, group.shape[0])
            rows.append(group.iloc[int(idx)])
    return pd.DataFrame(rows)


def _select_top(df: pd.DataFrame, column: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame()
    subset = df.dropna(subset=[column]).sort_values(column, ascending=False).head(n)
    return _pick_one_per_file(subset, rng)


def _select_random(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if df.empty:
        return df
    file_ids = df["file_id"].dropna().unique()
    if file_ids.size == 0:
        return pd.DataFrame()
    if file_ids.size <= n:
        subset = df[df["file_id"].isin(file_ids)]
    else:
        picked = rng.choice(file_ids, size=n, replace=False)
        subset = df[df["file_id"].isin(picked)]
    return _pick_one_per_file(subset, rng)


def _add_selection(selection: dict, row: pd.Series, reason: str) -> None:
    key = (str(row.get("file_id")), int(row.get("segment_id", 0)))
    entry = selection.setdefault(key, {"row": row, "reasons": set()})
    entry["reasons"].add(reason)


def _format_value(value: float | None, fmt: str = "{:.4g}") -> str | None:
    if value is None or not np.isfinite(value):
        return None
    return fmt.format(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate validation plots for selected JJ segments.")
    parser.add_argument("--metrics", default="results/metrics/metrics_long.csv", help="Metrics CSV")
    parser.add_argument("--segments-dir", default="results/parsed/segments", help="Segment parquet folder")
    parser.add_argument("--out", default="results/validation_plots", help="Output folder")
    parser.add_argument("--top-n", type=int, default=10, help="Top N to select by eta/knee")
    parser.add_argument("--random-n", type=int, default=10, help="Random N bidirectional/unidirectional")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing plots")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(args.seed)
    setup_logging("results/reports")

    metrics_path = Path(args.metrics)
    segments_dir = Path(args.segments_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_path, low_memory=False)
    if df.empty:
        LOGGER.warning("No metrics to plot in %s", metrics_path)
        return

    rng = np.random.default_rng(args.seed)
    selection: dict[tuple[str, int], dict] = {}

    for _, row in _select_top(df, "eta_norm", args.top_n, rng).iterrows():
        _add_selection(selection, row, "top_eta")
    for _, row in _select_top(df, "knee_score_rank", args.top_n, rng).iterrows():
        _add_selection(selection, row, "top_knee")

    bidir_mask = df.get("is_bidirectional", pd.Series(False, index=df.index)).fillna(False)
    bidir = df[bidir_mask]
    unidir = df[~bidir_mask]
    for _, row in _select_random(bidir, args.random_n, rng).iterrows():
        _add_selection(selection, row, "random_bidirectional")
    for _, row in _select_random(unidir, args.random_n, rng).iterrows():
        _add_selection(selection, row, "random_unidirectional")

    index_rows = []
    for (file_id, seg_id), entry in selection.items():
        reasons = sorted(entry["reasons"])
        row = entry["row"]

        seg_path = segments_dir / f"{file_id}.parquet"
        if not seg_path.exists():
            LOGGER.warning("Missing segments for %s", file_id)
            continue
        seg_df = pd.read_parquet(seg_path)
        seg_df = seg_df[seg_df["segment_id"] == seg_id]
        if seg_df.empty:
            LOGGER.warning("Missing segment %s for %s", seg_id, file_id)
            continue

        x = pd.to_numeric(seg_df["x"], errors="coerce").to_numpy()
        v = pd.to_numeric(seg_df["V_V"], errors="coerce").to_numpy()

        x_name = str(row.get("x_name", "x"))
        x_label = x_name
        overlays = []
        knee_x = row.get("knee_x")
        is_bidir = bool(row.get("is_bidirectional", False))

        if is_bidir:
            seg_all = pd.read_parquet(seg_path)
            if "segment_dir" in seg_all.columns and seg_all["segment_id"].nunique() > 1:
                up = seg_all[seg_all["segment_dir"] == "up"]
                down = seg_all[seg_all["segment_dir"] == "down"]
                if not up.empty and not down.empty:
                    up = up.sort_values("x")
                    down = down.sort_values("x")
                    x = pd.to_numeric(up["x"], errors="coerce").to_numpy()
                    v = pd.to_numeric(up["V_V"], errors="coerce").to_numpy()
                    overlays.append(
                        (
                            pd.to_numeric(down["x"], errors="coerce").to_numpy(),
                            pd.to_numeric(down["V_V"], errors="coerce").to_numpy(),
                            "down",
                        )
                    )
                elif np.any(x < 0) and np.any(x > 0):
                    x_abs = np.abs(x)
                    pos_mask = x > 0
                    neg_mask = x < 0
                    v_full = v
                    x = x_abs[pos_mask]
                    v = v_full[pos_mask]
                    overlays.append((x_abs[neg_mask], v_full[neg_mask], "minus"))
                    x_label = f"|{x_name}|"
                    if knee_x is not None and np.isfinite(knee_x):
                        knee_x = abs(float(knee_x))
            elif np.any(x < 0) and np.any(x > 0):
                x_abs = np.abs(x)
                pos_mask = x > 0
                neg_mask = x < 0
                v_full = v
                x = x_abs[pos_mask]
                v = v_full[pos_mask]
                overlays.append((x_abs[neg_mask], v_full[neg_mask], "minus"))
                x_label = f"|{x_name}|"
                if knee_x is not None and np.isfinite(knee_x):
                    knee_x = abs(float(knee_x))

        annotations = []
        for label, key in [
            ("eta_V_L1", "eta_V_L1"),
            ("eta_norm", "eta_norm"),
            ("eta_V_signed", "eta_V_signed"),
            ("knee_x", "knee_x"),
            ("knee_score_rank", "knee_score_rank"),
            ("H_incoh", "H_incoh"),
        ]:
            value = _format_value(row.get(key))
            if value is not None:
                annotations.append(f"{label}={value}")
        if row.get("H_method"):
            annotations.append(f"H_method={row.get('H_method')}")
        if row.get("parity_bit") is not None and np.isfinite(row.get("parity_bit")):
            annotations.append(f"parity_bit={int(row.get('parity_bit'))}")

        out_path = out_dir / f"{file_id}__seg{seg_id}__validation.png"
        if out_path.exists() and not args.overwrite:
            continue
        plot_voltage_trace(
            x,
            v,
            str(out_path),
            x_label=x_label,
            title=f"{file_id} seg{seg_id} ({';'.join(reasons)})",
            knee_x=knee_x if knee_x is None else float(knee_x),
            overlays=overlays,
            max_points=config.get("qc", {}).get("max_points", 5000),
            annotations=annotations,
        )

        index_rows.append(
            {
                "file_id": file_id,
                "segment_id": seg_id,
                "reasons": ";".join(reasons),
                "x_name": x_name,
                "is_bidirectional": bool(row.get("is_bidirectional", False)),
                "eta_V_L1": row.get("eta_V_L1"),
                "eta_norm": row.get("eta_norm"),
                "knee_x": row.get("knee_x"),
                "knee_score_rank": row.get("knee_score_rank"),
                "H_incoh": row.get("H_incoh"),
                "H_method": row.get("H_method"),
                "parity_bit": row.get("parity_bit"),
                "parity_stability": row.get("parity_stability"),
                "knee_confident": row.get("knee_confident"),
            }
        )

    if index_rows:
        index_path = out_dir / "index.csv"
        pd.DataFrame(index_rows).to_csv(index_path, index=False)
        LOGGER.info("Wrote validation index to %s", index_path)


if __name__ == "__main__":
    main()
