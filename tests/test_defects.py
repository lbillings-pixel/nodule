"""C14, defect density (ADR 0004 table row C14), with its negative control."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from nodule.defects import Defects, classify, groom, measure, thin


def face(
    n: int, spacing: float, kind: str, seed: int = 0, noise: float = 1.5
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """A round face n px across on a dark ground, with grey noise (levels):
    concentric rings, straight stripes, stripes with one band inserted at the
    centre (a dislocation: one band ends, and the two beside it join beyond
    it), or isotropic noise with power at the band spacing."""
    y, x = np.mgrid[:n, :n].astype(np.float64)
    c = (n - 1) / 2
    yy, xx = y - c, x - c
    mask = np.hypot(yy, xx) < 0.45 * n
    rng = np.random.default_rng(seed)
    if kind == "noise":
        f = np.fft.fft2(rng.normal(size=(n, n)))
        fy, fx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
        f *= np.exp(-0.5 * ((np.hypot(fy, fx) - 1 / spacing) / (0.15 / spacing)) ** 2)
        lum = np.fft.ifft2(f).real
        lum = 128 + 40 * lum / lum[mask].std()
    else:
        phase = {
            "rings": 2 * np.pi * np.hypot(yy, xx) / spacing,
            "stripes": 2 * np.pi * xx / spacing,
            "dislocation": 2 * np.pi * xx / spacing + np.arctan2(yy, xx - 0.3),
        }[kind]
        lum = 128 + 40 * np.cos(phase)
    lum = lum + np.random.default_rng(seed + 1).normal(0, noise, lum.shape)
    lum[~mask] = 20
    return lum.astype(np.float32), mask


def run(n: int, spacing: float, kind: str, seed: int = 0) -> Defects:
    lum, mask = face(n, spacing, kind, seed)
    return measure(lum, mask, 1.0, spacing)


def test_thinning_and_classes_on_a_tee() -> None:
    # A bar 7 px thick with a stem: one junction, three ends.
    m = np.zeros((60, 60), dtype=np.bool_)
    m[10:17, 5:55] = True
    m[10:50, 27:34] = True
    skeleton = thin(m)
    assert skeleton.sum() < 0.2 * m.sum()
    ends, junction = classify(skeleton)
    assert int(ends.sum()) == 3
    assert junction.any()


def test_grooming_removes_short_spurs_only() -> None:
    m = np.zeros((40, 80), dtype=np.bool_)
    m[20, 5:75] = True  # a line
    m[16:20, 40] = True  # a spur 4 px long
    m[5:20, 20] = True  # a branch 15 px long
    out = groom(m, 6.0)
    assert not out[16:20, 40].any()
    assert out[5:20, 20].all()
    ends, _ = classify(out)
    assert int(ends.sum()) == 3


@pytest.mark.parametrize("kind", ["rings", "stripes"])
def test_perfect_bands_have_no_defects(kind: str) -> None:
    # Rings: the disc at their centre thins to a dot, which is left out.
    for n, spacing in ((400, 16.0), (800, 32.0), (500, 10.0)):
        d = run(n, spacing, kind)
        assert d.density == 0.0, (kind, n)


def test_one_dislocation_is_one_end_and_one_junction() -> None:
    for n, spacing in ((400, 16.0), (800, 32.0), (500, 10.0)):
        for seed in (0, 1):
            d = run(n, spacing, "dislocation", seed)
            assert (d.terminations, d.junctions) == (1, 1), (n, seed)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_negative_control_thresholded_noise_has_many_defects(seed: int) -> None:
    # 2026-10-31: 0.89-0.97 per spacing squared at both resolutions.
    for n, spacing in ((400, 16.0), (800, 32.0)):
        d = run(n, spacing, "noise", seed)
        assert d.density is not None
        assert d.density > 0.5, (seed, n)


def test_noise_density_does_not_depend_on_resolution() -> None:
    a = [run(400, 16.0, "noise", s).density or 0.0 for s in range(4)]
    b = [run(800, 32.0, "noise", s).density or 0.0 for s in range(4)]
    assert float(np.mean(a)) == pytest.approx(float(np.mean(b)), rel=0.1)


def test_a_face_too_small_is_not_judged() -> None:
    d = run(40, 16.0, "stripes")
    assert d == Defects(None, None, None)
