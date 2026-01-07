from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jjparse.io import read_dat
from jjparse.metrics import fit_cpr
from jjparse.schema import coerce_types, standardize_columns


def test_read_dat_basic():
    path = Path(__file__).parent / "fixtures" / "sample.dat"
    df, meta = read_dat(path)

    assert len(df) == 5
    assert "device_name" in meta
    assert df.attrs.get("raw_columns")


def test_schema_coerce_types():
    df = pd.DataFrame({"I_A": ["1", "2"], "V_V": ["0.1", "0.2"]})
    df = standardize_columns(df, {})
    df = coerce_types(df)
    assert np.issubdtype(df["I_A"].dtype, np.number)
    assert np.issubdtype(df["V_V"].dtype, np.number)


def test_fit_cpr():
    phi = np.linspace(-np.pi, np.pi, 200)
    i1 = 2.0
    i2 = 0.5
    phi0 = 0.3
    offset = 0.1
    current = i1 * np.sin(phi + phi0) + i2 * np.sin(2 * (phi + phi0)) + offset

    result = fit_cpr(phi, current, [-np.pi, np.pi])
    assert abs(result["phi0_rad"] - phi0) < 0.2
