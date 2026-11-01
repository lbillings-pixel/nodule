"""nodule.render.mie (CPU; runs in CI). Each check has a control that must fail it."""

# pyright: basic

from __future__ import annotations

import math

import numpy as np
import pytest

from nodule._scipy import spherical_bessel
from nodule.render.mie import coefficients, grain_part, mie, pore_part, rgd_grain_mus

L = np.array([450.0, 550.0, 650.0])


def direct(m: complex, x: float, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """a_n, b_n straight from the Riccati-Bessel functions and their derivatives,
    without the logarithmic-derivative recurrence the module uses."""

    def jn(z, d=False):
        return spherical_bessel(n, z, d)

    def hn(z, d=False):
        return jn(z, d) + 1j * spherical_bessel(n, z, d, kind="y")

    def psi(z, d=False):
        return (jn(z) + z * jn(z, True)) if d else z * jn(z)

    def xi(z, d=False):
        return (hn(z) + z * hn(z, True)) if d else z * hn(z)

    mx = m * x
    a = (m * psi(mx) * psi(x, True) - psi(x) * psi(mx, True)) / (
        m * psi(mx) * xi(x, True) - xi(x) * psi(mx, True)
    )
    b = (psi(mx) * psi(x, True) - m * psi(x) * psi(mx, True)) / (
        psi(mx) * xi(x, True) - m * xi(x) * psi(mx, True)
    )
    return a, b


@pytest.mark.parametrize(("m", "x"), [(1.5, 1.0), (1.004, 17.6), (0.65, 0.7), (1.55 + 0.1j, 8.0)])
def testcoefficients_match_a_direct_bessel_evaluation(m, x) -> None:
    a, b = coefficients(complex(m), x)
    n = np.arange(1, len(a) + 1)
    da, db = direct(complex(m), x, n)
    assert np.allclose(a, da, rtol=1e-6, atol=1e-12)
    assert np.allclose(b, db, rtol=1e-6, atol=1e-12)
    # control: a_n with m and 1/m swapped is a different sphere and must not agree
    assert not np.allclose(coefficients(1 / complex(m), x)[0], da, rtol=1e-3)


def test_rayleigh_limit() -> None:
    m = 1.5
    k2 = abs((m * m - 1) / (m * m + 2)) ** 2
    ext, sca, g = mie(m, 0.01)
    assert sca == pytest.approx(8 / 3 * 0.01**4 * k2, rel=1e-4)
    assert ext == pytest.approx(sca, rel=1e-9)  # no absorption: all extinction is scattering
    assert abs(g) < 1e-3
    # control: at x = 2 the Rayleigh formula no longer holds (Mie gives 0.49 of it)
    assert abs(mie(m, 2.0)[1] / (8 / 3 * 2.0**4 * k2) - 1) > 0.3


def test_extinction_paradox() -> None:
    assert mie(1.33 + 0.01j, 500.0)[0] == pytest.approx(2.0, abs=0.05)  # measured 2.031
    assert mie(1.33 + 0.01j, 0.5)[0] < 0.5  # control: a small sphere is far from 2


@pytest.mark.parametrize("r", [1.0, 2.5])
def test_grains_reproduce_apetz_eq11_where_rgd_holds(r) -> None:
    """Apetz Eq. 11 [@apetz2003] is the RGD large-sphere limit (x >> 1, 4 pi r dn / lambda << 1)."""
    mus, _ = grain_part(r, L, 1.54)
    assert np.allclose(mus, rgd_grain_mus(r, L), rtol=0.02)  # measured < 1.2%


def test_control_eq11_overestimates_small_grains() -> None:
    """At r = 0.15 um, x ~ 2.6 is not >> 1, outside the range Pabst give as safe for the
    large-sphere form [@pabst2013]; Eq. 11 then comes out too high (computed here)."""
    mus, _ = grain_part(0.15, L, 1.54)
    assert (rgd_grain_mus(0.15, L) > 1.15 * mus).all()  # measured 1.20-1.40x


def test_grain_asymmetry_matches_the_notes_table() -> None:
    """notes/n2-reading/agate-optics2.md: g from Apetz Eq. 13 for quartz [inference]."""
    for r, want in ((0.15, (0.81, 0.77, 0.69)), (0.5, (0.975, 0.965, 0.954))):
        _, g = grain_part(r, L, 1.54)
        assert np.allclose(g, want, atol=0.005)
    _, g = grain_part(2.5, L, 1.54)
    assert (g > 0.996).all()


def test_pores_scale_with_fraction_and_vanish_without() -> None:
    one, g1 = pore_part(0.1, 1e-3, 1.0, L, 1.54)
    two, g2 = pore_part(0.1, 2e-3, 1.0, L, 1.54)
    assert np.allclose(two, 2 * one)
    assert np.allclose(g1, g2)
    assert (pore_part(0.1, 0.0, 1.0, L, 1.54)[0] == 0).all()
    water, _ = pore_part(0.1, 1e-3, 1.33, L, 1.54)
    assert (water < 0.3 * one).all()  # a smaller index step scatters far less
    assert one[0] > one[2]  # sub-wavelength pores: bluish


def test_size_parameter_must_be_positive() -> None:
    with pytest.raises(ValueError, match="size parameter"):
        mie(1.5, 0.0)
    with pytest.raises(ValueError, match="grain radius"):
        grain_part(0.0, L, 1.54)
    assert math.isfinite(mie(1.5, 1e-4)[2])
