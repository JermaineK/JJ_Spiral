import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jjparse.metrics import spectral_flatness


def test_spectral_flatness_extremes():
    rng = np.random.default_rng(123)
    noise = rng.standard_normal(4096)
    sine = np.sin(np.linspace(0, 20 * np.pi, 4096))

    flat_noise = spectral_flatness(noise)
    flat_sine = spectral_flatness(sine)

    assert flat_noise is not None
    assert flat_sine is not None
    assert flat_noise > 0.45
    assert flat_sine < 0.2
