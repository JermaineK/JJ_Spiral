"""Configuration helpers for JJ parsing pipeline."""

from __future__ import annotations

import copy
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import yaml

DEFAULT_CONFIG = {
    "units": {
        "current": {"ma": 1e-3, "ua": 1e-6, "na": 1e-9, "pa": 1e-12},
        "voltage": {"mv": 1e-3, "uv": 1e-6, "nv": 1e-9},
        "field": {"mt": 1e-3, "ut": 1e-6, "nt": 1e-9, "gauss": 1e-4},
        "temperature": {"mk": 1e-3},
        "freq": {"khz": 1e3, "mhz": 1e6, "ghz": 1e9},
        "phase": {"deg": "deg"},
    },
    "phase": {
        "assume_degrees_if_abs_gt": 6.5,
    },
    "parity": {
        "grid_n": 200,
        "eps": 1e-12,
    },
    "knee": {
        "min_points": 6,
        "min_segment": 3,
        "use_ruptures": True,
        "ruptures_model": "rbf",
        "ruptures_penalty": 5.0,
        "bootstrap_iters": 100,
        "conf_slope_factor": 3.0,
        "conf_edge_frac": 0.05,
        "rank_cap_percentile": 99.0,
    },
    "qc": {
        "hist_bins": 50,
        "max_points": 5000,
    },
    "fit": {
        "phi0_bounds": [-3.14159265, 3.14159265],
    },
    "smoothing": {
        "method": "savgol",
        "window": 11,
        "polyorder": 3,
    },
    "metrics": {
        "dv_dx_spike_thresh": None,
        "dv_dx_spike_multiplier": 5.0,
        "eps": 1e-12,
    },
    "entropy": {
        "order": 3,
        "delay": 1,
        "max_points": 2000,
    },
    "incoherence": {
        "min_points": 16,
        "rolling_window": 11,
        "normalize": {
            "method": "percentile",
            "p_low": 1.0,
            "p_high": 99.0,
        },
    },
    "resample": {
        "grid_n": 200,
    },
    "segments": {
        "min_points": 5,
    },
    "pairing": {
        "csv": None,
        "regex": None,
    },
    "null_models": {
        "direction_n": 30,
        "phase_n": 5,
        "phase_max_segments": 600,
        "parity_boot": 200,
        "seed": 123,
        "phase_all_segments": False,
        "phase_full_pass": False,
    },
}


def _deep_update(base: dict, update: dict) -> dict:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base.get(key, {}), value)
        else:
            base[key] = value
    return base


def load_config(path: str | None) -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if not path:
        return config
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Config file must contain a mapping at top level.")
    return _deep_update(config, loaded)


def setup_logging(out_dir: str | Path, name: str = "jjparse") -> str:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = out_path / f"run_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return str(log_path)


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
