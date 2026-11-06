"""C26, the profile spectrum (ADR 0004 footnote 3)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from nodule import bands, spectrum

RADIUS = 300.0


def ar1(phi: float, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
    e = rng.normal(0, 1, n)
    a = np.zeros(n)
    for i in range(1, n):
        a[i] = phi * a[i - 1] + e[i]
    return a


def profiles(kind: str, seed: int) -> list[NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    k = np.arange(300)
    if kind == "white":
        return [rng.normal(0, 3, 300) for _ in range(24)]
    if kind == "periodic":
        return [
            10 * np.cos(2 * math.pi * k / 18 + rng.uniform(0, 7)) + rng.normal(0, 3, 300)
            for _ in range(24)
        ]
    return [ar1(float(kind), 300, rng) for _ in range(24)]


def ring_face(size: int, seed: int) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Concentric bands of random thickness (log-SD 0.35, 9 px median at 600 px
    across) drawn at 1200 px and block-averaged to `size`, with white noise."""
    y, x = np.mgrid[0:1200, 0:1200] - 599.5
    r: NDArray[np.float64] = np.hypot(x, y) / 2
    rng = np.random.default_rng(seed)
    edges: NDArray[np.float64] = np.concatenate(
        [[0.0], np.cumsum(9.0 * np.exp(rng.normal(0, 0.35, 200)))]
    )
    drawn = 128 + 40 * np.cos(np.interp(r, edges, np.arange(edges.size) * math.pi))
    f = 1200 // size
    lum = drawn.reshape(size, f, size, f).mean(axis=(1, 3)) + rng.normal(0, 3, (size, size))
    return lum.astype(np.float32), r[::f, ::f] < 270


def test_shares_are_of_the_total_power():
    s = spectrum.measure(profiles("white", 0), 1.0, RADIUS, None)
    assert s is not None
    assert 0.5 < sum(s.power) <= 1.0  # periods longer than MIN_CYCLES allow are left out
    assert list(s.period_share) == sorted(s.period_share)


def test_nothing_resolved_keeps_the_curve_without_summaries():
    s = spectrum.measure(profiles("white", 0), 1.0, RADIUS, None)
    assert s is not None
    assert len(s.power) > spectrum.MIN_BINS
    assert (s.tilt, s.peak, s.peak_share) == (None, None, None)


def test_too_few_bins_in_the_band_give_no_tilt():
    """The band must hold MIN_TILT_BINS; a limit near its top leaves the peak but no tilt.
    Control: the same profiles with a fine limit have one."""
    rng = np.random.default_rng(0)
    long = [rng.normal(0.0, 1.0, 2000) for _ in range(40)]  # periods up to 2 radii of 500 px
    fine = spectrum.measure(long, 1.0, 500.0, 2.0)
    assert fine is not None
    assert fine.tilt is not None
    near_top = spectrum.measure(
        long, 1.0, 500.0, spectrum.TILT_TOP_SHARE * 500.0 / spectrum.GRID_STEP
    )
    assert near_top is not None
    assert near_top.tilt is None
    assert near_top.peak is not None


@pytest.mark.parametrize("seed", range(5))
def test_white_noise_tilt_is_about_minus_one(seed: int):
    s = spectrum.measure(profiles("white", seed), 1.0, RADIUS, 2.0)
    assert s is not None
    assert s.tilt is not None
    assert -1.15 < s.tilt < -0.7  # 5th-95th percentile -1.01 to -0.85 over 20 seeds


@pytest.mark.parametrize("kind", ["white", "0.9", "0.99"])
@pytest.mark.parametrize("seed", range(5))
def test_negative_control_noise_without_a_period_gives_no_peak(kind: str, seed: int):
    """White and red noise have no preferred period: their peak over the fitted
    red-noise background stays low (5th-95th percentile 1.2-1.7 over 20 seeds)."""
    s = spectrum.measure(profiles(kind, seed), 1.0, RADIUS, 2.0)
    assert s is not None
    assert s.peak is not None
    assert s.peak < 2.0


def test_one_period_gives_a_peak_there():
    s = spectrum.measure(profiles("periodic", 0), 1.0, RADIUS, 2.0)
    assert s is not None
    assert s.peak is not None
    assert s.peak_share is not None
    assert s.peak > 50
    period = s.peak_share * RADIUS
    assert 18 / spectrum.GRID_STEP <= period <= 18 * spectrum.GRID_STEP


def test_red_noise_tilt_rises_with_its_memory():
    tilts: list[float] = []
    for kind in ("white", "0.9", "0.99"):
        s = spectrum.measure(profiles(kind, 0), 1.0, RADIUS, 2.0)
        assert s is not None
        assert s.tilt is not None
        tilts.append(s.tilt)
    assert tilts == sorted(tilts)


def test_two_resolutions_give_the_same_curve():
    """Negative control: one drawn stone at 600 and 300 px across. Shares that
    hold at least 1% agree within 25%, the peak sits in the same bin, and the
    tilt agrees within 1 (its band starts at L, which differs)."""
    fine, coarse = (
        bands.measure(
            *ring_face(n, 4), 1.0, 3.0, 9.0 * n / 600, rng=np.random.default_rng(0), ladder=False
        )
        for n in (600, 300)
    )
    a, c = fine.spectrum, coarse.spectrum
    assert a is not None
    assert c is not None
    assert coarse.limit_share is not None
    assert a.peak_share == c.peak_share
    assert a.peak is not None
    assert c.peak is not None
    assert abs(a.peak / c.peak - 1) < 0.2
    assert a.tilt is not None
    assert c.tilt is not None
    assert abs(a.tilt - c.tilt) < 1.0
    fa, fc = (
        dict(zip(a.period_share, a.power, strict=True)),
        dict(zip(c.period_share, c.power, strict=True)),
    )
    shared = [k for k in fc if k in fa and k >= coarse.limit_share and fc[k] >= 0.01]
    assert len(shared) >= 5
    for k in shared:
        assert abs(fa[k] / fc[k] - 1) < 0.25, k


def test_fine_structure_moves_the_tilt():
    """Negative control: bands drawn without fine structure (ADR 0006 section
    10's obvious fake) must read far from the same bands with sub-bands (measured
    -6.6 and -1.6). The smooth bands are the lower here: one narrow line, and the
    tilt's band runs past it (`nodule.spectrum`), so this is no rule of direction."""
    y, x = np.mgrid[0:600, 0:600] - 299.5
    r = np.hypot(x, y)
    rng = np.random.default_rng(1)
    smooth = 128 + 40 * np.cos(2 * math.pi * r / 18) + rng.normal(0, 3, r.shape)
    sub = smooth + 12 * np.cos(2 * math.pi * r / 4.5)
    tilts: list[float] = []
    for lum in (smooth, sub):
        b = bands.measure(
            lum.astype(np.float32),
            r < 270,
            1.0,
            3.0,
            9.0,
            rng=np.random.default_rng(0),
            ladder=False,
        )
        assert b.spectrum is not None
        assert b.spectrum.tilt is not None
        tilts.append(b.spectrum.tilt)
    assert tilts[0] < tilts[1] - 3.0
