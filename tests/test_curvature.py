"""C15, band curvature and sharp corners (ADR 0004 table row C15), with its
negative controls."""

from __future__ import annotations

import math

import numpy as np
import pytest
from agatesim import _scipy as agatesim_scipy  # pyright: ignore[reportPrivateUsage]
from agatesim.specimen.quality import MIN_COHERENCE
from agatesim.specimen.shape import MARGIN_SPACINGS
from numpy.typing import NDArray

from nodule.bands import COUPLET_SPACINGS, coarse_face, direction_field
from nodule.curvature import Curvature, even_face, measure, turn_rate
from nodule.orientation import doubled

ROUGH = 0.12  # rms roughening of the "rough" rings, spacings


def face(
    n: int, spacing: float, kind: str, seed: int = 0, noise: float = 1.5
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """A round face n px across on a dark ground, with grey noise (levels).
    Bands are level lines of a function f: rings (f = r), straight stripes,
    rings whose radius is roughened by `ROUGH` spacings (rms) of noise with
    its power at half a spacing, a single V ("vee<deflection deg>") or a zig-zag of
    nested V's ("zig<deg>", period 0.3 n); or isotropic noise with power at
    the band spacing, or no bands at all ("even")."""
    y, x = np.mgrid[:n, :n].astype(np.float64)
    c = (n - 1) / 2
    yy, xx = y - c, x - c
    r = np.hypot(yy, xx)
    mask = r < 0.45 * n
    rng = np.random.default_rng(seed)
    if kind == "noise":
        f = np.fft.fft2(rng.normal(size=(n, n)))
        fy, fx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
        f *= np.exp(-0.5 * ((np.hypot(fy, fx) - 1 / spacing) / (0.15 / spacing)) ** 2)
        lum = np.fft.ifft2(f).real
        lum = 128 + 40 * lum / lum[mask].std()
    elif kind == "even":
        lum = np.full((n, n), 128.0)
    else:
        if kind == "rings":
            level = r
        elif kind == "stripes":
            level = xx
        elif kind == "rough":
            g = np.fft.fft2(rng.normal(size=(n, n)))
            fy, fx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
            g *= np.exp(-0.5 * ((np.hypot(fy, fx) - 2 / spacing) / (0.3 / spacing)) ** 2)
            rough = np.fft.ifft2(g).real
            level = r + ROUGH * spacing * rough / rough.std()
        else:
            half = math.radians(float(kind[3:]) / 2)
            if kind.startswith("vee"):
                across = np.abs(xx)
            else:
                period = 0.3 * n
                across = np.abs((xx / period) % 1.0 - 0.5) * period
            level = yy * math.cos(half) + across * math.sin(half)
        lum = 128 + 40 * np.cos(2 * np.pi * level / spacing)
    lum = lum + np.random.default_rng(seed + 1).normal(0, noise, lum.shape)
    lum[~mask] = 20
    return lum.astype(np.float32), mask


def run(n: int, spacing: float, kind: str, seed: int = 0) -> Curvature:
    lum, mask = face(n, spacing, kind, seed)
    # `spacing` is the couplet period; the gate reports one band, half of it.
    return measure(lum, mask, 1.0, spacing / COUPLET_SPACINGS)


def ring_kappa_r(n: int, spacing: float, kind: str) -> NDArray[np.float64]:
    """kappa x r at each read pixel of a ring face (1 for exact rings)."""
    lum, mask = face(n, spacing, kind)
    coarse, crop, k = coarse_face(lum, mask, 1.0, spacing)
    step = (spacing / k) / 4.0 if spacing / k > 4.0 else 1.0  # even_face's resampling
    coarse, crop, s = even_face(coarse, crop, spacing / k)
    inner = agatesim_scipy.erode(crop, max(1, math.ceil(MARGIN_SPACINGS * s)))
    ny, nx, coh = direction_field(coarse, s, crop)
    banded = inner & (coh >= MIN_COHERENCE)
    kappa, read = turn_rate(doubled(ny, nx), banded, s)
    at = (np.arange(coarse.shape[0], dtype=np.float64) + 0.5) * step - 0.5  # coarse px
    centres = (at + 0.5) * k - 0.5 - (n - 1) / 2
    radius = np.hypot(centres[:, None], centres[None, :])  # full px
    keep = read & (radius > 3 * spacing)  # no clear direction near the centre
    # kappa x spacing, over the spacing in full px, times r in full px
    return np.asarray(kappa[keep] / spacing * radius[keep], dtype=np.float64)


def test_straight_stripes_have_no_curvature_and_no_corners() -> None:
    c = run(400, 10.0, "stripes")
    assert c.median is not None
    assert c.high is not None
    assert c.median < 0.005
    assert c.high < 0.02
    assert c.corners == 0


@pytest.mark.parametrize("kind", ["rings", "rough"])
def test_a_circle_gives_one_over_r(kind: str) -> None:
    # Rough: the radius is roughened at half a spacing, which the smoothing of
    # one spacing takes out.
    for n, spacing in ((400, 10.0), (800, 20.0)):
        kr = ring_kappa_r(n, spacing, kind)
        assert float(kr.mean()) == pytest.approx(1.0, abs=0.04), (kind, n)
        assert float(np.median(kr)) == pytest.approx(1.0, abs=0.08), (kind, n)
    assert run(400, 10.0, kind).corners == 0  # nor at the rings' centre


@pytest.mark.parametrize("deflection", [60, 90, 120])
def test_one_vee_is_one_corner_with_straight_arms(deflection: int) -> None:
    c = run(400, 10.0, f"vee{deflection}")
    assert c.corners == 1
    assert c.median is not None
    assert c.median < 0.01  # the arms are straight


@pytest.mark.parametrize("deflection", [60, 90, 120])
def test_a_zigzag_is_counted_corner_by_corner(deflection: int) -> None:
    # Period 0.3 n across a face 0.9 n wide: five apex lines reach the judged face.
    for n, spacing in ((400, 10.0), (800, 20.0)):
        assert run(n, spacing, f"zig{deflection}").corners == 5, (deflection, n)


def test_values_do_not_depend_on_resolution() -> None:
    for kind in ("rings", "rough", "zig90", "zig60"):
        a, b = run(400, 10.0, kind), run(560, 14.0, kind)
        for field in ("median", "high"):
            va, vb = getattr(a, field), getattr(b, field)
            # 2026-10-31: within 6% from 8 to 32 px per spacing, after even_face
            # (without it the zig-zag's upper quantile moved by up to a half).
            assert va == pytest.approx(vb, rel=0.08, abs=0.005), (kind, field)
        assert a.corners == b.corners, kind


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_negative_control_isotropic_noise_has_no_corners(seed: int) -> None:
    # Noise with power at the band spacing: most of it is not banded, and the
    # scraps of corner it leaves (2026-10-31: up to 4 spacings) are shorter
    # than a corner line must run.
    for n, spacing in ((400, 10.0), (800, 20.0)):
        c = run(n, spacing, "noise", seed)
        assert c.banded < 0.25
        assert c.corners in (None, 0), (seed, n)


def test_negative_control_outline_step_adds_no_curvature_or_corner() -> None:
    # An even face on a dark ground: its outline is a sharply curved, strongly
    # oriented step, but too little of the face is banded to be judged.
    for noise in (0.0, 0.1):
        lum, mask = face(400, 10.0, "even", noise=noise)
        c = measure(lum, mask, 1.0, 10.0 / COUPLET_SPACINGS)
        assert c.banded < 0.25, noise
        assert c.corners in (None, 0), noise
