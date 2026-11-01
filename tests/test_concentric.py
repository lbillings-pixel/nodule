"""C13, concentricity and completeness (ADR 0004 and its footnote 2)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from agatesim import _scipy as agatesim_scipy  # pyright: ignore[reportPrivateUsage]
from agatesim.specimen.shape import MARGIN_SPACINGS
from numpy.typing import NDArray

from nodule.bands import COUPLET_SPACINGS
from nodule.concentric import Concentric, find_sets, judge, measure
from nodule.edges import edge_face

NOISE = 1.5  # grey levels of the test faces' pixel noise, as the gate would report
SIZES = ((400, 16.0), (800, 32.0))  # face px across, couplet period px


def face(
    n: int, spacing: float, kind: str, seed: int = 0, noise: float = NOISE
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """A round face n px across on a dark ground, with grey noise (levels), of
    couplet period `spacing`: concentric rings; rings whose centres drift
    ("ecc" and the shift per period of radius; "whole" adds that the rings stop
    before the outline cuts any); two sets of rings about centres 0.36 n
    apart, inside shared outer bands; rings with one extra couplet inserted
    between two points (a wedging band); rings of three greys (light, middle,
    dark, middle, each band half a period wide); straight stripes; isotropic
    noise with power at the band spacing; or no bands."""
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
            level = r + float(kind[3:].removesuffix("whole")) * xx
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
        if kind.endswith("whole"):  # every ring below this level lies inside the face
            e = float(kind[3:].removesuffix("whole"))
            lum = np.where(level < 0.45 * n * (1 - e) * 0.85, lum, 128.0)
    lum = lum + np.random.default_rng(seed + 1).normal(0, noise, lum.shape)
    lum[~mask] = 20
    return lum.astype(np.float32), mask


def run(n: int, spacing: float, kind: str, seed: int = 0) -> Concentric:
    lum, mask = face(n, spacing, kind, seed)
    # `spacing` is the couplet period; the gate reports one band, half of it.
    return measure(lum, mask, 1.0, NOISE, spacing / COUPLET_SPACINGS)


@pytest.mark.parametrize(("n", "spacing"), SIZES)
def test_concentric_rings_are_one_complete_set_with_no_drift(n: int, spacing: float) -> None:
    c = run(n, spacing, "rings")
    assert c.sets == 1
    # 0.45 n less the margin, over half a couplet per band edge: about 20.
    assert c.depth is not None
    assert 18 <= c.depth <= 22
    assert c.drift is not None
    assert c.drift < 0.01
    assert c.complete == 1.0


@pytest.mark.parametrize(("n", "spacing"), SIZES)
def test_the_centre_is_found_where_the_rings_meet(n: int, spacing: float) -> None:
    lum, mask = face(n, spacing, "rings")
    ef = edge_face(lum, mask, 1.0, NOISE, spacing / COUPLET_SPACINGS)
    judged = agatesim_scipy.erode(ef.face, math.ceil(MARGIN_SPACINGS * ef.spacing))
    sets, _ = find_sets(ef.lum, ef.face, ef.edges, judged, ef.spacing)
    middle = (ef.lum.shape[0] - 1) / 2, (ef.lum.shape[1] - 1) / 2
    assert len(sets) == 1
    assert math.dist(sets[0].centre, middle) < 0.05 * ef.spacing


@pytest.mark.parametrize("shift", [0.2, 0.4])
def test_drift_of_whole_rings_is_their_centre_shift_per_radius(shift: float) -> None:
    # Contours r + e x = L are ellipses whose centres move e / (1 - e^2) per
    # unit L, while their mean radius (a + b) / 2 grows by
    # (1 / (1 - e^2) + 1 / sqrt(1 - e^2)) / 2 per unit L.
    e = shift
    expected = (e / (1 - e * e)) / ((1 / (1 - e * e) + 1 / math.sqrt(1 - e * e)) / 2)
    for n, spacing in SIZES:
        c = run(n, spacing, f"ecc{shift}whole")
        assert c.sets == 1, n
        assert c.drift == pytest.approx(expected, rel=0.06), n


@pytest.mark.parametrize(("shift", "low", "high"), [(0.2, 0.17, 0.2), (0.4, 0.29, 0.37)])
def test_drift_reads_low_where_the_outline_cuts_the_rings(
    shift: float, low: float, high: float
) -> None:
    # The module's limit (b): cut rings' normals meet short of their centres.
    for n, spacing in SIZES:
        c = run(n, spacing, f"ecc{shift}")
        assert c.sets == 1, n
        assert c.drift is not None
        assert low < c.drift < high, (n, c.drift)


def test_negative_control_two_separate_sets_report_two_centres() -> None:
    for n, spacing in SIZES:
        c = run(n, spacing, "two")
        assert c.sets == 2, n
        assert c.complete == 1.0
        assert c.drift is not None
        assert c.drift < 0.02, n


def test_a_wedging_band_leaves_one_set() -> None:
    for n, spacing in SIZES:
        c = run(n, spacing, "wedge")
        assert c.sets == 1, n
        assert c.complete == 1.0


def test_middle_grey_bands_count_as_bands_of_their_own() -> None:
    # Each grey band is half a period wide, as each band of "rings": the same
    # depth, not half of it (a middle band merged into a neighbour).
    for n, spacing in SIZES:
        c = run(n, spacing, "grey")
        assert c.sets == 1, n
        assert c.depth is not None
        assert 18 <= c.depth <= 22, n
        assert c.complete == 1.0


@pytest.mark.parametrize("kind", ["rings", "grey", "two", "ecc0.2"])
def test_results_hold_over_the_edge_smoothing(kind: str) -> None:
    lum, mask = face(400, 16.0, kind)
    seen: list[Concentric] = []
    for smooth in (0.08, 0.125, 0.2):
        ef = edge_face(lum, mask, 1.0, NOISE, 16.0 / COUPLET_SPACINGS, smooth=smooth)
        judged = agatesim_scipy.erode(ef.face, math.ceil(MARGIN_SPACINGS * ef.spacing))
        seen.append(judge(ef.lum, ef.face, ef.edges, judged, ef.spacing))
    assert len({c.sets for c in seen}) == 1, seen
    depths = [c.depth or 0 for c in seen]
    assert max(depths) - min(depths) <= 1, seen


def test_negative_control_stripes_have_no_centre() -> None:
    for n, spacing in SIZES:
        c = run(n, spacing, "stripes")
        assert c.sets == 0, n
        assert c.drift is None
        assert c.complete is None


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_negative_control_noise_has_no_set(seed: int) -> None:
    for n, spacing in SIZES:
        c = run(n, spacing, "noise", seed)
        assert c.sets == 0, (seed, n)
        assert c.depth is not None
        assert c.depth < 6, (seed, n)  # MIN_DEPTH
        assert c.drift is None
        assert c.complete is None


def test_an_even_face_has_no_set() -> None:
    c = run(400, 16.0, "even")
    assert c == Concentric(0, 0, None, None)


def test_a_face_too_small_is_not_judged() -> None:
    lum, mask = face(40, 16.0, "rings")
    assert measure(lum, mask, 1.0, NOISE, 8.0) == Concentric(None, None, None, None)
