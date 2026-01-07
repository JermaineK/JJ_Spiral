"""Create a summary report from metrics output."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _first_non_null(series: pd.Series):
    series = series.dropna()
    if series.empty:
        return None
    return series.iloc[0]


def _distribution(series: pd.Series) -> dict:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return {}
    return {
        "count": int(series.count()),
        "median": float(series.median()),
        "p95": float(series.quantile(0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate summary report from metrics_long.csv")
    parser.add_argument("--metrics", default="results/metrics/metrics_long.csv", help="Metrics CSV")
    parser.add_argument("--out", default="results/reports/summary.md", help="Output markdown")
    parser.add_argument("--top-n", type=int, default=20, help="Top N candidates to list")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_path, low_memory=False)
    if df.empty:
        out_path.write_text("# Summary\n\nNo metrics available.\n", encoding="utf-8")
        return

    file_summary = df.groupby("file_id").agg(_first_non_null)

    total_files = len(file_summary)
    voltage_only = file_summary[
        (file_summary.get("cap_has_V", False))
        & (~file_summary.get("cap_has_I", False))
        & (~file_summary.get("cap_has_phi", False))
    ]
    bidirectional = file_summary[file_summary.get("is_bidirectional", False)]
    cyclic = file_summary[file_summary.get("is_cyclic", False)]

    axis_counts = file_summary.get("axis_primary", pd.Series(dtype=str)).value_counts(dropna=True)

    metrics_dist = {
        "knee_x": _distribution(df.get("knee_x", pd.Series(dtype=float))),
        "H_incoh_raw": _distribution(df.get("H_incoh_raw", pd.Series(dtype=float))),
        "H_incoh": _distribution(df.get("H_incoh", pd.Series(dtype=float))),
        "eta_V_L1": _distribution(df.get("eta_V_L1", pd.Series(dtype=float))),
        "eta_norm": _distribution(df.get("eta_norm", pd.Series(dtype=float))),
        "K_max": _distribution(df.get("K_max", pd.Series(dtype=float))),
        "knee_score_norm": _distribution(df.get("knee_score_norm", pd.Series(dtype=float))),
        "knee_score_rank": _distribution(df.get("knee_score_rank", pd.Series(dtype=float))),
    }

    top_eta = df.dropna(subset=["eta_norm"]).sort_values("eta_norm", ascending=False).head(args.top_n)
    top_knee = df.dropna(subset=["knee_score_rank"]).sort_values("knee_score_rank", ascending=False).head(args.top_n)

    h_method_counts = df.get("H_method", pd.Series(dtype=str)).value_counts(dropna=True)
    parity_stability = file_summary.get("parity_stability", pd.Series(dtype=float))
    parity_stats = _distribution(parity_stability)

    eta_compare = df.dropna(subset=["eta_V_L1", "eta_norm", "H_incoh"])
    suppressed = pd.DataFrame()
    promoted = pd.DataFrame()
    if not eta_compare.empty:
        eta_compare = eta_compare.copy()
        eta_compare["rank_raw"] = eta_compare["eta_V_L1"].rank(ascending=False, method="min")
        eta_compare["rank_norm"] = eta_compare["eta_norm"].rank(ascending=False, method="min")
        eta_compare["rank_delta"] = eta_compare["rank_norm"] - eta_compare["rank_raw"]
        suppressed = eta_compare[eta_compare["H_incoh"] >= 0.8].sort_values("rank_delta", ascending=False).head(5)
        promoted = eta_compare[eta_compare["H_incoh"] <= 0.2].sort_values("rank_delta", ascending=True).head(5)

    knee_parity_path = Path("results/reports/knee_parity_summary.csv")
    knee_parity_summary = None
    if knee_parity_path.exists():
        knee_parity_summary = pd.read_csv(knee_parity_path)
        if knee_parity_summary.empty:
            knee_parity_summary = None

    lines = []
    lines.append("# Summary")
    lines.append("")
    lines.append("## Dataset capabilities")
    lines.append("")
    lines.append(f"- Total files: {total_files}")
    lines.append(f"- Voltage-only files: {len(voltage_only)}")
    lines.append(f"- Bidirectional files: {len(bidirectional)}")
    lines.append(f"- Cyclic files: {len(cyclic)}")
    lines.append("")

    lines.append("## Axis usage")
    lines.append("")
    if axis_counts.empty:
        lines.append("- No axis counts available.")
    else:
        for name, count in axis_counts.head(10).items():
            lines.append(f"- {name}: {count}")
    lines.append("")

    lines.append("## Incoherence methods")
    lines.append("")
    if h_method_counts.empty:
        lines.append("- No incoherence methods recorded.")
    else:
        for name, count in h_method_counts.items():
            lines.append(f"- {name}: {count}")
    lines.append("")

    lines.append("## Metric distributions")
    lines.append("")
    for key, stats in metrics_dist.items():
        if not stats:
            lines.append(f"- {key}: n=0")
        else:
            lines.append(f"- {key}: n={stats['count']}, median={stats['median']:.6g}, p95={stats['p95']:.6g}")
    lines.append("")

    lines.append("## Normalization effects")
    lines.append("")
    if suppressed.empty:
        lines.append("- Suppressed (high incoherence): None")
    else:
        lines.append("- Suppressed (high incoherence):")
        for _, row in suppressed.iterrows():
            lines.append(
                f"- {row.get('file_id')} seg={row.get('segment_id')} eta={row.get('eta_V_L1'):.4g} eta_norm={row.get('eta_norm'):.4g} H_incoh={row.get('H_incoh'):.3g}"
            )
    if promoted.empty:
        lines.append("- Promoted (low incoherence): None")
    else:
        lines.append("- Promoted (low incoherence):")
        for _, row in promoted.iterrows():
            lines.append(
                f"- {row.get('file_id')} seg={row.get('segment_id')} eta={row.get('eta_V_L1'):.4g} eta_norm={row.get('eta_norm'):.4g} H_incoh={row.get('H_incoh'):.3g}"
            )
    lines.append("")

    lines.append("## Parity stability")
    lines.append("")
    if not parity_stats:
        lines.append("- parity_stability: n=0")
    else:
        lines.append(
            f"- parity_stability: n={parity_stats['count']}, median={parity_stats['median']:.6g}, p95={parity_stats['p95']:.6g}"
        )
    if knee_parity_summary is None:
        lines.append("- Knee parity summary: unavailable")
    else:
        row = knee_parity_summary.iloc[0]
        lines.append(
            f"- Knee parity flips: n={int(row['n_flip'])} / {int(row['n_total'])} (fraction={row['fraction_flip']:.4g})"
        )
        lines.append(
            f"- Parity locks after knee: n={int(row['n_lock_after'])} / {int(row['n_total'])} (fraction={row['fraction_lock_after']:.4g})"
        )
    lines.append("")

    lines.append("## Top odd-channel candidates (eta_norm)")
    lines.append("")
    if top_eta.empty:
        lines.append("- None")
    else:
        for _, row in top_eta.iterrows():
            lines.append(
                f"- {row.get('file_id')} seg={row.get('segment_id')} eta_norm={row.get('eta_norm'):.6g} x={row.get('x_name')}"
            )
    lines.append("")

    lines.append("## Top knee evidence (knee_score_rank)")
    lines.append("")
    if top_knee.empty:
        lines.append("- None")
    else:
        for _, row in top_knee.iterrows():
            lines.append(
                f"- {row.get('file_id')} seg={row.get('segment_id')} knee_score_rank={row.get('knee_score_rank'):.6g} x={row.get('x_name')}"
            )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- Incoherence normalization rescales eta and knee scores, promoting low-incoherence traces while keeping raw values intact."
    )
    lines.append("- Parity stability and knee-parity stats quantify whether odd-channel signals persist across segments.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
