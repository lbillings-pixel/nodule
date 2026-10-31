"""C16, orientation order (ADR 0004 table row C16), with its negative controls."""

from __future__ import annotations

import numpy as np
import pytest
from agatesim.specimen.quality import MIN_COHERENCE
from numpy.typing import NDArray

from nodule.orientation import Order, measure


def face(
    n: int, spacing: float, kind: str, seed: int = 0, noise: float = 1.5
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """A round face n px across on a dark ground, with grey noise (levels):
    concentric rings, straight stripes, isotropic noise with power at the band
    spacing, or no bands at all."""
    y, x = np.mgrid[:n, :n].astype(np.float64)
    c = (n - 1) / 2
    r = np.hypot(y - c, x - c)
    mask = r < 0.45 * n
    if kind == "rings":
        lum = 128 + 40 * np.cos(2 * np.pi * r / spacing)
    elif kind == "stripes":
        lum = 128 + 40 * np.cos(2 * np.pi * x / spacing)
    elif kind == "noise":
        f = np.fft.fft2(np.random.default_rng(seed).normal(size=(n, n)))
        fy, fx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
        f *= np.exp(-0.5 * ((np.hypot(fy, fx) - 1 / spacing) / (0.15 / spacing)) ** 2)
        lum = np.fft.ifft2(f).real
        lum = 128 + 40 * lum / lum[mask].std()
    else:
        lum = np.full((n, n), 128.0)
    lum = lum + np.random.default_rng(seed + 1).normal(0, noise, lum.shape)
    lum[~mask] = 20
    return lum.astype(np.float32), mask


def run(n: int, spacing: float, kind: str, seed: int = 0) -> Order:
    lum, mask = face(n, spacing, kind, seed)
    return measure(lum, mask, 1.0, spacing)


def test_concentric_rings_have_no_global_order_and_no_parallel_share() -> None:
    o = run(400, 10.0, "rings")
    assert o.coherence is not None
    assert o.coherence > 0.9
    assert o.order is not None
    assert o.order < 0.05
    assert o.parallel is not None
    assert o.parallel < 0.02
    assert o.correlation is not None
    assert 0.1 < o.correlation < 0.3
    assert not o.censored


def test_straight_stripes_are_fully_ordered_and_parallel() -> None:
    o = run(400, 10.0, "stripes")
    assert o.order is not None
    assert o.order > 0.99
    assert o.parallel is not None
    assert o.parallel > 0.99
    assert o.censored  # C(r) never falls to 1/e: at least half the face


def test_values_do_not_depend_on_resolution() -> None:
    for kind in ("rings", "stripes"):
        a, b = run(400, 10.0, kind), run(800, 20.0, kind)
        for field in ("coherence", "order", "correlation", "parallel"):
            va, vb = getattr(a, field), getattr(b, field)
            assert va == pytest.approx(vb, abs=0.02), (kind, field)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_negative_control_isotropic_noise_is_not_read_as_ordered_bands(seed: int) -> None:
    # Noise with power at the band spacing (labyrinth-like, the hardest case):
    # its coherence stays under agatesim's MIN_COHERENCE (2026-10-31: 0.19-0.21,
    # bands 0.96-1.0), most of it is not banded, and what is shows little order.
    o = run(600, 15.0, "noise", seed)
    assert o.coherence is not None
    assert o.coherence < MIN_COHERENCE
    assert o.banded < 0.25
    assert o.order is None or o.order < 0.35
    assert o.correlation is None or o.correlation < 0.15
    assert o.parallel is None or o.parallel < 0.05


def test_negative_control_outline_step_adds_no_direction() -> None:
    # An even face on a dark ground: the step at its outline is strongly
    # oriented, but must change nothing inside the face. Compared with the same
    # face on a ground of its own level, noiseless and with faint noise.
    for noise in (0.0, 0.1):
        lum, mask = face(400, 10.0, "even", noise=noise)
        level = lum.copy()
        level[~mask] = 128.0
        a, b = measure(lum, mask, 1.0, 10.0), measure(level, mask, 1.0, 10.0)
        assert a.banded == pytest.approx(b.banded, abs=0.005), noise
        assert a.coherence == pytest.approx(b.coherence, abs=0.005), noise
    assert a.banded < 0.25  # pyright: ignore[reportPossiblyUnbound]


def test_negative_control_grey_noise_alone_is_not_banded() -> None:
    # The same face with its grey noise: the structure tensor finds some
    # direction in noise alone (2026-10-31: coherence 0.165), under the bands' level.
    o = run(400, 10.0, "even")
    assert o.coherence is not None
    assert o.coherence < MIN_COHERENCE
    assert o.banded < 0.25
    assert o.parallel is None or o.parallel < 0.05
