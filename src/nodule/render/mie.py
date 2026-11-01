"""Mie scattering by a homogeneous sphere, and the two scattering parts of a band
(ADR 0007 choice 7a C: grains in Apetz's form plus pores).

`mie(m, x)` returns the efficiencies Q_ext, Q_sca and the asymmetry g of one sphere
of relative index m and size parameter x = 2 pi r n_med / lambda_0. The series is
the standard one (Lorenz-Mie coefficients a_n, b_n from the logarithmic derivative
D_n(mx) by downward recurrence and Riccati-Bessel functions of x); the algorithm is
Bohren & Huffman's [@bohren1983] (cited only). It is checked numerically, not from the
book: against a direct Bessel-function evaluation of a_n, b_n, the Rayleigh limit,
the extinction paradox (Q_ext -> 2) and Apetz's Eq. 11 in the RGD limit
(tests/test_mie.py).

`grain_part` is Apetz's virtual composite [@apetz2003]: a polycrystal behaves like
half its volume of index-matched spheres, radius r = half the mean grain size, with
index step Delta n = (2/3) Delta n_max. Apetz evaluate it with the RGD shortcut
(Eq. 11, gamma = 3 pi^2 r Delta n^2 / lambda_0^2, valid for 4 pi r Delta n / lambda_0
<< 1); here the same spheres go through the full Mie series, so the 5 um white-band
grains of Lorenzi [@lorenzi2022], near the edge of RGD, are covered too [inference].
Independent scattering at volume fraction 1/2 is Apetz's own assumption.

`pore_part` is a volume fraction of spheres of another index (air or water) in the
matrix [@apetz2003, pp. 485-486, where pores dominate backscatter]. Whether agate
holds such pores is *speculative* [@graetsch1985 saw none at 0.1 um by TEM].

All radii and fractions passed in are `effective` (ADR 0007 s.6, 7a); Delta n_max is
`physical`. Grades (ADR 0002): the Mie arithmetic is *verified* by the tests; its use
for agate is *plausible* (grains) and *speculative* (pores).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from nodule._scipy import riccati_bessel

__all__ = ["DN_QUARTZ", "coefficients", "grain_part", "mie", "pore_part", "rgd_grain_mus"]

F64 = NDArray[np.float64]

# Quartz birefringence at 550 nm (physical); Apetz's isotropic average is 2/3 of it,
# 0.0061 (notes/n2-reading/agate-optics2.md s.1). Dispersion of Delta n is neglected.
DN_MAX_QUARTZ = 0.0092
DN_QUARTZ = 2.0 / 3.0 * DN_MAX_QUARTZ
GRAIN_FILL = 0.5  # Apetz Eq. 8: the virtual composite is half spheres


def _terms(x: float) -> int:
    return int(x + 4.0 * x ** (1.0 / 3.0) + 2.0)  # Wiscombe's stopping rule, as in BHMIE


def coefficients(m: complex, x: float) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """a_n, b_n for n = 1..N."""
    nstop = _terms(x)
    mx = m * x
    nmx = max(nstop, int(abs(mx))) + 16
    d = np.zeros(nmx + 1, np.complex128)  # D_n(mx), downward recurrence from D = 0
    for n in range(nmx, 0, -1):
        d[n - 1] = n / mx - 1.0 / (d[n] + n / mx)
    n = np.arange(0, nstop + 1)
    psi, chi = riccati_bessel(n, x)  # psi_n(x) = x j_n(x); x y_n(x)
    xi = psi + 1j * chi  # xi_n = x h_n^(1)(x)
    k = np.arange(1, nstop + 1)
    dk = d[1 : nstop + 1]
    ta = dk / m + k / x
    tb = dk * m + k / x
    a = (ta * psi[1:] - psi[:-1]) / (ta * xi[1:] - xi[:-1])
    b = (tb * psi[1:] - psi[:-1]) / (tb * xi[1:] - xi[:-1])
    return a, b


def mie(m: complex, x: float) -> tuple[float, float, float]:
    """Q_ext, Q_sca, g of a sphere: relative index m (n + ik over the medium's index),
    size parameter x = 2 pi r n_med / lambda_0."""
    if x <= 0:
        raise ValueError("size parameter must be positive")
    a, b = coefficients(complex(m), float(x))
    n = np.arange(1, len(a) + 1, dtype=np.float64)
    qext = 2.0 / x**2 * float(np.sum((2 * n + 1) * (a + b).real))
    qsca = 2.0 / x**2 * float(np.sum((2 * n + 1) * (abs(a) ** 2 + abs(b) ** 2)))
    # g Q_sca: the standard sum over adjacent and crossed coefficients
    nn = n[:-1]
    s1 = np.sum(nn * (nn + 2) / (nn + 1) * (a[:-1] * np.conj(a[1:]) + b[:-1] * np.conj(b[1:])).real)
    s2 = np.sum((2 * n + 1) / (n * (n + 1)) * (a * np.conj(b)).real)
    g = 4.0 / x**2 * float(s1 + s2) / qsca if qsca > 0 else 0.0
    return qext, qsca, g


@lru_cache(maxsize=4096)
def _part(r_um: float, m: complex, fraction: float, n_med: float,
          wavelengths_nm: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:  # fmt: skip
    mus: list[float] = []
    gs: list[float] = []
    r_mm = r_um * 1e-3
    number = fraction / (4.0 / 3.0 * math.pi * r_mm**3)  # spheres per mm^3
    for lam in wavelengths_nm:
        x = 2.0 * math.pi * r_um * n_med / (lam * 1e-3)
        _, qsca, g = mie(m, x)
        mus.append(number * qsca * math.pi * r_mm**2)
        gs.append(g)
    return tuple(mus), tuple(gs)


def grain_part(r_um: float, wavelengths_nm: F64, n_med: float,
               dn: float = DN_QUARTZ) -> tuple[F64, F64]:  # fmt: skip
    """mu_s (per mm) and g per wavelength of Apetz's virtual composite with grain
    radius r_um (half the mean grain size)."""
    if r_um <= 0:
        raise ValueError("grain radius must be positive")
    mus, g = _part(float(r_um), complex((n_med + dn) / n_med), GRAIN_FILL, float(n_med),
                   tuple(float(v) for v in wavelengths_nm))  # fmt: skip
    return np.array(mus), np.array(g)


def pore_part(r_um: float, fraction: float, n_pore: float, wavelengths_nm: F64,
              n_med: float) -> tuple[F64, F64]:  # fmt: skip
    """mu_s (per mm) and g per wavelength of a volume fraction of spherical pores of
    radius r_um and index n_pore (1.0 air, 1.33 water)."""
    if fraction <= 0:
        return np.zeros(len(wavelengths_nm)), np.zeros(len(wavelengths_nm))
    if r_um <= 0:
        raise ValueError("pore radius must be positive")
    mus, g = _part(float(r_um), complex(n_pore / n_med), float(fraction), float(n_med),
                   tuple(float(v) for v in wavelengths_nm))  # fmt: skip
    return np.array(mus), np.array(g)


def rgd_grain_mus(r_um: float, wavelengths_nm: F64, dn: float = DN_QUARTZ) -> F64:
    """Apetz Eq. 11, gamma = 3 pi^2 r Delta n^2 / lambda_0^2 (per mm) [@apetz2003]."""
    lam_mm = np.asarray(wavelengths_nm, np.float64) * 1e-6
    return 3.0 * math.pi**2 * (r_um * 1e-3) * dn**2 / lam_mm**2
