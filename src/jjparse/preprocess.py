"""Preprocessing helpers: unit conversion and cleaning."""

from __future__ import annotations

import logging
import re
from typing import Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("jjparse")


def _scale_from_name(name: str, unit_map: dict[str, float]) -> float | None:
    lower = name.lower()
    for unit, scale in unit_map.items():
        if re.search(rf"\b{re.escape(unit)}\b", lower):
            return scale
    return None


def _parse_value_with_unit(text: str) -> Tuple[float | None, str | None]:
    match = re.search(r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([a-zA-Z]+)?", text)
    if not match:
        return None, None
    value = float(match.group(1))
    unit = match.group(2).lower() if match.group(2) else None
    return value, unit


def _scale_from_column(df: pd.DataFrame, std_col: str, unit_map: dict[str, float]) -> float | None:
    source = df.attrs.get(f"source_{std_col}", std_col)
    scale = _scale_from_name(source, unit_map)
    if scale is None:
        scale = _scale_from_name(std_col, unit_map)
    return scale


def apply_unit_conversions(df: pd.DataFrame, meta: dict, config: dict) -> pd.DataFrame:
    df = df.copy()

    units = config.get("units", {})
    current_map = units.get("current", {})
    voltage_map = units.get("voltage", {})
    field_map = units.get("field", {})
    temp_map = units.get("temperature", {})
    freq_map = units.get("freq", {})

    if "I_A" in df.columns:
        scale = _scale_from_column(df, "I_A", current_map)
        if scale:
            df["I_A"] = df["I_A"] * scale
    if "V_V" in df.columns:
        scale = _scale_from_column(df, "V_V", voltage_map)
        if scale:
            df["V_V"] = df["V_V"] * scale
    if "B_T" in df.columns:
        scale = _scale_from_column(df, "B_T", field_map)
        if scale:
            df["B_T"] = df["B_T"] * scale
    if "T_K" in df.columns:
        scale = _scale_from_column(df, "T_K", temp_map)
        if scale:
            df["T_K"] = df["T_K"] * scale
    if "freq_Hz" in df.columns:
        scale = _scale_from_column(df, "freq_Hz", freq_map)
        if scale:
            df["freq_Hz"] = df["freq_Hz"] * scale

    for key in ["t", "temp", "temperature"]:
        if key in meta:
            value, unit = _parse_value_with_unit(str(meta[key]))
            if value is not None and unit:
                unit = unit.lower()
                scale = temp_map.get(unit)
                if scale:
                    if "T_K" not in df.columns or df["T_K"].isna().all():
                        df["T_K"] = value * scale
                    meta["temperature_unit"] = unit

    return df


def infer_phase_radians(df: pd.DataFrame, meta: dict, config: dict) -> pd.DataFrame:
    df = df.copy()
    if "phi_rad" not in df.columns:
        return df

    series = pd.to_numeric(df["phi_rad"], errors="coerce")
    if series.dropna().empty:
        return df

    max_abs = float(np.nanmax(np.abs(series.to_numpy())))
    assume_deg = config.get("phase", {}).get("assume_degrees_if_abs_gt", 6.5)

    if max_abs > assume_deg and max_abs <= 360.0 * 2:
        LOGGER.warning("Assuming phase in degrees based on range; converting to radians.")
        df["phi_rad"] = np.deg2rad(series)
        meta["phase_unit_assumed"] = "deg"
    else:
        meta["phase_unit_assumed"] = "rad"

    return df
