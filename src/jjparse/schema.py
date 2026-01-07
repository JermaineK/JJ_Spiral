"""Schema normalization for JJ datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

import numpy as np
import pandas as pd

STANDARD_COLS: list[tuple[str, list[str]]] = [
    ("t_s", ["time", "t_s", "seconds", "sec"]),
    ("I_A", ["ibias", "i_bias", "idc", "i_dc", "current", "isquid", "ich"]),
    ("V_V", ["vdc", "vsquid", "voltage", "vbias", "v_bias"]),
    ("B_T", ["bfield", "b_t", "field", "mag", "flux", "bt"]),
    ("phi_rad", ["phi", "phase"]),
    ("gate1_V", ["vg1", "gate1", "vgjju", "vgjj1"]),
    ("gate2_V", ["vg2", "gate2", "vgjjl", "vgjj2"]),
    ("gate_V", ["vg", "gate"]),
    ("T_K", ["temp", "temperature", "t_k", "tk"]),
    ("freq_Hz", ["freq", "frequency", "rf", "hz"]),
]

NUMERIC_STD_COLS = {
    "t_s",
    "I_A",
    "V_V",
    "B_T",
    "phi_rad",
    "gate_V",
    "gate1_V",
    "gate2_V",
    "T_K",
    "freq_Hz",
}

AXIS_PRIORITY = [
    "I_A",
    "B_T",
    "gate_V",
    "gate1_V",
    "gate2_V",
    "phi_rad",
    "T_K",
    "freq_Hz",
    "t_s",
]


@dataclass
class SignalPack:
    signals: dict[str, pd.Series] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    axis_candidates: list[str] = field(default_factory=list)

    def has(self, name: str) -> bool:
        if name not in self.signals:
            return False
        series = self.signals[name]
        return series is not None and not series.dropna().empty


def _normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _pattern_score(col: str, patterns: Iterable[str]) -> int | None:
    norm = _normalize_col(col)
    for idx, pat in enumerate(patterns):
        if pat in norm:
            return idx
    return None


def standardize_columns(df: pd.DataFrame, meta: dict | None = None) -> pd.DataFrame:
    meta = meta or {}
    std_df = df.copy()
    std_df.attrs.update(df.attrs)
    assigned_raw: set[str] = set()

    for std_name, patterns in STANDARD_COLS:
        if std_name in std_df.columns:
            continue
        best_col = None
        best_score = None
        for col in std_df.columns:
            if col in assigned_raw:
                continue
            score = _pattern_score(col, patterns)
            if score is None:
                continue
            if best_score is None or score < best_score:
                best_score = score
                best_col = col
        if best_col is not None:
            std_df[std_name] = std_df[best_col]
            std_df.attrs[f"source_{std_name}"] = best_col
            assigned_raw.add(best_col)

    if meta.get("sample_id") and "sample_id" not in std_df.columns:
        std_df["sample_id"] = meta.get("sample_id")
    if meta.get("run_id") and "run_id" not in std_df.columns:
        std_df["run_id"] = meta.get("run_id")
    if meta.get("file_id") and "file_id" not in std_df.columns:
        std_df["file_id"] = meta.get("file_id")

    for col in NUMERIC_STD_COLS:
        if col not in std_df.columns:
            std_df[col] = pd.NA

    return std_df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_STD_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _axis_candidate_from_meta(df: pd.DataFrame, meta: dict) -> str | None:
    step_names = []
    for key, value in meta.items():
        if key.startswith("step") and key.endswith("_name"):
            step_names.append(str(value))
    if not step_names:
        return None
    for name in step_names:
        if name in df.columns:
            return name
        norm = _normalize_col(name)
        for col in df.columns:
            if _normalize_col(col) == norm:
                return col
    return None


def detect_capabilities(df: pd.DataFrame) -> dict[str, bool]:
    def _has_col(col: str) -> bool:
        return col in df.columns and not pd.to_numeric(df[col], errors="coerce").dropna().empty

    has_gate = any(_has_col(col) for col in ["gate_V", "gate1_V", "gate2_V"])
    return {
        "cap_has_V": _has_col("V_V"),
        "cap_has_I": _has_col("I_A"),
        "cap_has_phi": _has_col("phi_rad"),
        "cap_has_B": _has_col("B_T"),
        "cap_has_gate": has_gate,
        "cap_has_time": _has_col("t_s"),
    }


def build_signal_pack(df: pd.DataFrame, meta: dict | None = None) -> SignalPack:
    meta = meta or {}
    signals: dict[str, pd.Series] = {}
    units: dict[str, str] = {}

    if "V_V" in df.columns:
        signals["V"] = pd.to_numeric(df["V_V"], errors="coerce")
        units["V"] = "V"
    if "I_A" in df.columns:
        signals["I"] = pd.to_numeric(df["I_A"], errors="coerce")
        units["I"] = "A"
    if "phi_rad" in df.columns:
        signals["phi"] = pd.to_numeric(df["phi_rad"], errors="coerce")
        units["phi"] = "rad"
    if "B_T" in df.columns:
        signals["B"] = pd.to_numeric(df["B_T"], errors="coerce")
        units["B"] = "T"

    gate_col = None
    for candidate in ["gate_V", "gate1_V", "gate2_V"]:
        if candidate in df.columns and not pd.to_numeric(df[candidate], errors="coerce").dropna().empty:
            gate_col = candidate
            break
    if gate_col:
        signals["gate"] = pd.to_numeric(df[gate_col], errors="coerce")
        units["gate"] = "V"

    if "t_s" in df.columns:
        signals["t"] = pd.to_numeric(df["t_s"], errors="coerce")
        units["t"] = "s"

    axis_candidates = []
    for col in AXIS_PRIORITY:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.nunique(dropna=True) > 1:
                axis_candidates.append(col)

    meta_axis = _axis_candidate_from_meta(df, meta)
    if meta_axis and meta_axis not in axis_candidates:
        axis_candidates.insert(0, meta_axis)

    return SignalPack(signals=signals, units=units, axis_candidates=axis_candidates)


def _monotonic_score(values: np.ndarray) -> float:
    diffs = np.diff(values)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 0.0
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if signs.size == 0:
        return 0.0
    frac_pos = np.mean(signs > 0)
    frac_neg = np.mean(signs < 0)
    return float(max(frac_pos, frac_neg))


def infer_primary_axis(df: pd.DataFrame, meta: dict | None = None) -> tuple[str, np.ndarray]:
    meta = meta or {}
    meas_cols = set()
    if "meas" in meta:
        meas_cols = {_normalize_col(item.strip()) for item in str(meta["meas"]).split(",") if item.strip()}

    for col in AXIS_PRIORITY:
        if col in df.columns and _normalize_col(col) not in meas_cols:
            values = pd.to_numeric(df[col], errors="coerce").to_numpy()
            if np.isfinite(values).sum() > 1 and pd.Series(values).nunique(dropna=True) > 1:
                return col, values

    meta_axis = _axis_candidate_from_meta(df, meta)
    if meta_axis and meta_axis in df.columns:
        values = pd.to_numeric(df[meta_axis], errors="coerce").to_numpy()
        if np.isfinite(values).sum() > 1 and pd.Series(values).nunique(dropna=True) > 1:
            return meta_axis, values

    best_col = None
    best_score = 0.0
    for col in df.columns:
        if _normalize_col(col) in meas_cols:
            continue
        values = pd.to_numeric(df[col], errors="coerce").to_numpy()
        if np.isfinite(values).sum() < 3:
            continue
        score = _monotonic_score(values[np.isfinite(values)])
        if score > best_score:
            best_score = score
            best_col = col
    if best_col:
        return best_col, pd.to_numeric(df[best_col], errors="coerce").to_numpy()

    return "index", np.arange(len(df), dtype=float)
