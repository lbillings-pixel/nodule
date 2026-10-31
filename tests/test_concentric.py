"""C13, concentricity and completeness (ADR 0004)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from agatesim import _scipy as agatesim_scipy  # pyright: ignore[reportPrivateUsage]
from agatesim.specimen.shape import MARGIN_SPACINGS
from numpy.typing import NDArray

from nodule.bands import COUPLET_SPACINGS
from nodule.concentric import Concentric, judge, measure
from nodule.edges import edge_face


def face(
    n: int, spacing: float, kind: str, seed: int = 0, noise: float = 1.5
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """A round face n px across on a dark ground, with grey noise (levels), of
    couplet period `spacing`: concentric rings; rings whose centres drift
    ("ecc" and the shift per period of radius); two sets of rings about
    centres 0.36 n apart, inside shared outer bands; rings with one extra
    band inserted between two points (a wedging band); rings of three greys
    (light, middle, dark, middle, each band half a period wide); straight
    stripes; isotropic noise with power at the band spacing; or no bands."""
    y, x = np.mgrid[:n, :n].astype(np.float64)
    c = (n - 1) / 2
    yy, xx = y - c, x - c
    r = np.hypot(yy, xx)
    mask = r < 0.45 * n
    if kind == "noise":
        f = np.fft.fft2(np.random.default_rng(seed).normal(size=(n, n)))
        fy, fx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
        f *= np.exp(-0.5 * ((np.hypot(fy, fx) - 1 / spacing) / (0.15 / spacing)) ** 2)
        lum = np.fft.ifft2(f).real
        lum = 128 + 40 * lum / lum[mask].std()
    elif kind == "even":
        lum = np.full((n, n), 128.0)
    else:
        if kind in ("rings", "grey"):
            level = r
        elif kind.startswith("ecc"):
            level = r + float(kind[3:]) * xx
        elif kind == "two":
            d = 0.18 * n
            level = np.minimum(np.hypot(yy, xx - d), np.hypot(yy, xx + d))
        elif kind == "stripes":
            level = xx
        else:  # "wedge": a pair of opposite phase defects on a ring of radius n / 4
            z = xx + 1j * yy
            p1, p2 = 0.25 * n * np.exp(0.4j) + 0.3, 0.25 * n * np.exp(2.4j) + 0.3
            level = r + spacing * np.angle((z - p1) / (z - p2)) / (2 * np.pi)
        lum = 128 + 40 * np.cos(2 * np.pi * level / spacing)
        if kind == "grey":
            lum = 128 + np.array([40.0, 0.0, -40.0, 0.0])[(r // (spacing / 2)).astype(int) % 4]
    lum = lum + np.random.default_rng(seed + 1).normal(0, noise, lum.shape)
    lum[~mask] = 20
    return lum.astype(np.float32), mask


def run(n: int, spacing: float, kind: str, seed: int = 0) -> Concentric:
    lum, mask = face(n, spacing, kind, seed)
    # `spacing` is the couplet period; the gate reports one band, half of it.
    return measure(lum, mask, 1.0, spacing / COUPLET_SPACINGS)


@pytest.mark.parametrize(("n", "spacing"), [(400, 16.0), (800, 32.0)])
def test_concentric_rings_are_one_complete_set_with_no_drift(n: int, spacing: float) -> None:
    c = run(n, spacing, "rings")
    assert c.sets == 1
    # 0.45 n less the margin, over half a couplet per ring: about 18 rings.
    assert c.depth is not None
    assert 16 <= c.depth <= 20
    assert c.drift is not None
    assert c.drift < 0.02
    assert c.complete == 1.0
    assert c.wedges == 0


@pytest.mark.parametrize(("shift", "expected"), [(0.2, 0.104), (0.4, 0.238)])
def test_drifting_centres_give_their_drift(shift: float, expected: float) -> None:
    # Contours r + e x = L have centres e L / (1 - e^2) from the origin, and
    # successive rings (light, then dark) are half a period of L apart.
    for n, spacing in ((400, 16.0), (800, 32.0)):
        c = run(n, spacing, f"ecc{shift}")
        assert c.sets == 1
        assert c.drift == pytest.approx(expected, abs=0.015), n


def test_negative_control_two_separate_sets_report_two_centres() -> None:
    for n, spacing in ((400, 16.0), (800, 32.0)):
        c = run(n, spacing, "two")
        assert c.sets == 2, n
        assert c.complete == 1.0


def test_a_wedging_band_is_counted_and_lowers_completeness() -> None:
    for n, spacing in ((400, 16.0), (800, 32.0)):
        c = run(n, spacing, "wedge")
        assert c.sets == 1, n
        assert c.wedges == 1, n
        assert c.complete is not None
        assert 0.95 < c.complete < 1.0, n


def test_middle_grey_bands_are_rings_of_their_own() -> None:
    # Each grey band is half a period wide, as each band of "rings": the same
    # count of nested rings, not half of it (a middle band merged into a
    # neighbour).
    for n, spacing in ((400, 16.0), (800, 32.0)):
        c = run(n, spacing, "grey")
        assert c.sets == 1, n
        assert c.depth is not None
        assert 16 <= c.depth <= 20, n
        assert c.drift is not None
        assert c.drift < 0.02
        assert c.complete == 1.0


@pytest.mark.parametrize("kind", ["rings", "grey", "two", "ecc0.2"])
def test_results_hold_over_the_edge_smoothing(kind: str) -> None:
    lum, mask = face(400, 16.0, kind)
    seen: list[Concentric] = []
    for smooth in (0.08, 0.125, 0.2):
        ef = edge_face(lum, mask, 1.0, 16.0 / COUPLET_SPACINGS, smooth=smooth)
        judged = agatesim_scipy.erode(ef.face, math.ceil(MARGIN_SPACINGS * ef.spacing))
        seen.append(judge(ef.edges, judged, ef.spacing))
    assert len({c.sets for c in seen}) == 1, seen
    depths = [c.depth or 0 for c in seen]
    assert max(depths) - min(depths) <= 1, seen


def test_negative_control_stripes_have_no_centre() -> None:
    c = run(400, 16.0, "stripes")
    assert c == Concentric(0, 0, None, None, 0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_negative_control_noise_blobs_have_no_nesting_order(seed: int) -> None:
    for n, spacing in ((400, 16.0), (800, 32.0)):
        c = run(n, spacing, "noise", seed)
        assert c.sets == 0, (seed, n)
        assert c.depth is not None
        assert c.depth <= 2, (seed, n)
        assert c.drift is None
        assert c.complete is None


def test_an_even_face_has_no_set() -> None:
    c = run(400, 16.0, "even")
    assert c.sets == 0
    assert c.complete is None


def test_a_face_too_small_is_not_judged() -> None:
    lum, mask = face(40, 16.0, "rings")
    assert measure(lum, mask, 1.0, 8.0) == Concentric(None, None, None, None, None)
