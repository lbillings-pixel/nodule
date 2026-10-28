"""Band count C6 and persistence C10 (ADR 0004) on drawn bands with known
answers, and the negative controls the ADR names for each."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from nodule import bands

PERIOD = 18.0  # pixels per light-dark couplet
SPACING = PERIOD / 2  # agatesim's band spacing is one band


def rings(radius: float, size: int, squeeze: float = 1.0, noise: float = 3.0):
    """Concentric cosine bands inside a disc (or an ellipse squeezed along y),
    with white noise; returns brightness and the face's mask."""
    y, x = np.mgrid[0:size, 0:size] - size / 2 + 0.5
    r = np.hypot(x, y / squeeze)
    rng = np.random.default_rng(1)
    lum = 128 + 40 * np.cos(2 * math.pi * r / PERIOD) + rng.normal(0, noise, r.shape)
    return lum.astype(np.float32), r < radius


def ar1(phi: float, n: int, seed: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, n)
    a = np.zeros(n)
    for i in range(1, n):
        a[i] = phi * a[i - 1] + e[i]
    return a


def test_rings_are_counted_from_margin_to_centre():
    lum, mask = rings(270.0, 600)
    b = bands.measure(lum, mask, 1.0, 3.0, SPACING, rng=np.random.default_rng(0))
    # One spacing of margin, then (270 - 9) / 18 = 14.5 couplets to the centre.
    assert b.profiles == bands.STARTS
    assert b.count == 14
    assert b.count_low is not None
    assert b.count_high is not None
    assert 13 <= b.count_low <= b.count_high <= 15
    assert b.rule_gap == 0


def test_count_follows_bands_round_an_elongated_face():
    lum, mask = rings(270.0, 600, squeeze=0.6)
    b = bands.measure(lum, mask, 1.0, 3.0, SPACING, rng=np.random.default_rng(0))
    assert b.count is not None
    assert 12 <= b.count <= 15


def test_scale_between_working_and_full_size():
    """A mask at half the brightness image's size gives the same count."""
    lum, mask = rings(270.0, 600)
    b = bands.measure(lum, mask[::2, ::2], 0.5, 3.0, SPACING, rng=np.random.default_rng(0))
    assert b.count == 14


def test_flat_profile_counts_zero():
    """Negative control: a uniform image has no bands."""
    c = bands.count(np.full(500, 128.0), 0.0, SPACING)
    assert (c.rule1, c.rule2, c.low, c.high) == (0, 0, 0, 0)


def test_uniform_face_gives_no_profiles():
    b = bands.measure(
        np.full((300, 300), 128, np.float32),
        np.ones((300, 300), bool),
        1.0,
        0.0,
        SPACING,
        rng=np.random.default_rng(0),
    )
    assert b.profiles == 0
    assert b.count is None
    assert b.persistence is None


def test_clean_bands_count_steadily_over_the_sweep():
    n = 600
    rng = np.random.default_rng(2)
    profile = 128 + 40 * np.cos(2 * math.pi * np.arange(n) / PERIOD) + rng.normal(0, 3, n)
    c = bands.count(profile, 3.0, SPACING)
    assert abs(c.rule1 - n / PERIOD) <= 1
    assert abs(c.rule2 - n / PERIOD) <= 1
    assert (c.high - c.low) / c.rule1 < 0.1


def test_red_noise_counts_swing_with_the_parameters():
    """Negative control: AR(1) red noise has no bands to agree on, so its count
    moves with the smoothing and threshold by more than the count itself."""
    for seed in range(10):
        profile = 128 + 10 * ar1(0.9, 600, seed)
        c = bands.count(profile, 1.0, SPACING)
        assert (c.high - c.low) / max(c.rule1, 1) > 0.8


def test_persistent_series_is_positive():
    rng = np.random.default_rng(5)
    series = [np.exp(2.5 + 0.2 * ar1(0.6, 15, int(s))) for s in rng.integers(0, 10**6, 20)]
    p = bands.persistence(series, np.random.default_rng(0))
    assert p is not None
    assert p > 0.2


def test_shuffled_series_is_about_zero():
    """Negative control: shuffling keeps the thickness distribution but must
    take the persistence to about 0."""
    rng = np.random.default_rng(5)
    series = [np.exp(2.5 + 0.2 * ar1(0.6, 15, int(s))) for s in rng.integers(0, 10**6, 20)]
    shuffled = [rng.permutation(w) for w in series]
    p = bands.persistence(shuffled, np.random.default_rng(0))
    assert p is not None
    assert abs(p) < 0.1


def test_persistence_needs_enough_widths():
    assert bands.persistence([np.ones(6) * 9.0] * 3, np.random.default_rng(0)) is None
