"""Plotting utilities for JJ datasets."""

from __future__ import annotations

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    idx = np.linspace(0, len(df) - 1, num=max_points).astype(int)
    return df.iloc[idx]


def plot_voltage_trace(
    x: np.ndarray,
    v: np.ndarray,
    out_path: str,
    x_label: str,
    title: str | None = None,
    knee_x: float | None = None,
    cp_x: list[float] | None = None,
    overlays: list[tuple[np.ndarray, np.ndarray, str]] | None = None,
    max_points: int = 5000,
    annotations: list[str] | None = None,
) -> bool:
    if x.size == 0 or v.size == 0:
        return False
    mask = np.isfinite(x) & np.isfinite(v)
    x = x[mask]
    v = v[mask]
    if x.size == 0:
        return False

    if x.size > max_points:
        idx = np.linspace(0, x.size - 1, num=max_points).astype(int)
        x = x[idx]
        v = v[idx]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(x, v, ".", markersize=2, label="segment")

    if overlays:
        for ox, ov, label in overlays:
            mask = np.isfinite(ox) & np.isfinite(ov)
            ax.plot(ox[mask], ov[mask], "-", linewidth=1.0, label=label)

    if knee_x is not None and np.isfinite(knee_x):
        ax.axvline(knee_x, color="red", linestyle="--", linewidth=1.0, label="knee")
    if cp_x:
        for val in cp_x:
            if np.isfinite(val):
                ax.axvline(val, color="orange", linestyle=":", linewidth=1.0)

    ax.set_xlabel(x_label)
    ax.set_ylabel("V (V)")
    if title:
        ax.set_title(title)

    if annotations:
        text = "\n".join(annotations)
        ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left", fontsize=7)

    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_iv(df: pd.DataFrame, out_path: str, max_points: int = 5000) -> bool:
    if "I_A" not in df.columns or "V_V" not in df.columns:
        return False
    data = df[["I_A", "V_V"]].dropna()
    if data.empty:
        return False
    data = _downsample(data, max_points)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(data["I_A"], data["V_V"], ".", markersize=2)
    ax.set_xlabel("I (A)")
    ax.set_ylabel("V (V)")
    ax.set_title("I-V")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_cpr(df: pd.DataFrame, out_path: str, max_points: int = 5000) -> bool:
    if "phi_rad" not in df.columns or "I_A" not in df.columns:
        return False
    data = df[["phi_rad", "I_A"]].dropna()
    if data.empty:
        return False
    data = _downsample(data, max_points)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(data["phi_rad"], data["I_A"], ".", markersize=2)
    ax.set_xlabel("phi (rad)")
    ax.set_ylabel("I (A)")
    ax.set_title("CPR")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_fraunhofer(df: pd.DataFrame, out_path: str) -> bool:
    if "B_T" not in df.columns or "I_A" not in df.columns:
        return False
    data = df[["B_T", "I_A"]].dropna()
    if data.empty:
        return False

    field = data["B_T"].to_numpy()
    current = data["I_A"].to_numpy()
    rounded = np.round(field, 8)
    grouped = pd.DataFrame({"B_T": rounded, "I_A": current})
    ic = grouped.groupby("B_T")["I_A"].apply(lambda x: np.max(np.abs(x))).reset_index()

    if ic.empty:
        return False

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(ic["B_T"], ic["I_A"], "-o", markersize=2)
    ax.set_xlabel("B (T)")
    ax.set_ylabel("Ic (A)")
    ax.set_title("Fraunhofer-like")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_histograms(df: pd.DataFrame, out_path: str, bins: int = 50) -> bool:
    cols = [
        "I_A",
        "V_V",
        "B_T",
        "phi_rad",
        "gate_V",
        "gate1_V",
        "gate2_V",
        "T_K",
        "freq_Hz",
    ]
    cols = [col for col in cols if col in df.columns]
    if not cols:
        return False

    ncols = 2
    nrows = math.ceil(len(cols) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 3 * nrows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.hist(series, bins=bins, color="gray", alpha=0.8)
        ax.set_title(col)

    for ax in axes[len(cols):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_missing(df: pd.DataFrame, out_path: str) -> bool:
    counts = df.isna().sum()
    if counts.empty:
        return False

    fig, ax = plt.subplots(figsize=(6, 3))
    counts.plot(kind="bar", ax=ax, color="gray")
    ax.set_ylabel("NaN count")
    ax.set_title("Missing values")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True
