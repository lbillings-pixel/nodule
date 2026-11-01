"""Exact and published values the renderer is checked against (CPU, numpy only).

Grade: the checks against these are *verified* (ADR 0007 section 5).
"""

from __future__ import annotations

import math

import numpy as np

# MCML [@wang1995a] Table 1 (p. 143): slab n 1, mua 1/mm, mus 9/mm, g 0.75, d 0.2 mm;
# van de Hulst values.
T1 = {"d": 0.2, "n": 1.0, "mua": 1.0, "mus": 9.0, "g": 0.75, "rd": 0.09739, "tt": 0.66096}
# Table 2: half-space n 1.5, g 0, same coefficients; Giovanelli Rd 0.2600. That value
# INCLUDES the specular 0.04: the diffuse part is 0.2200 (H function below, and
# notes/n2-proto/RESULTS.md: MC .21978 + .04 = .25978) [inference, numerically strong].
T2 = {"n": 1.5, "mua": 1.0, "mus": 9.0, "g": 0.0, "rd_with_specular": 0.2600}


def fresnel(ci: float, n1: float, n2: float) -> float:
    """Unpolarised Fresnel reflectance at |cos incidence| ci, n1 -> n2."""
    s2 = (n1 / n2) ** 2 * (1 - ci * ci)
    if s2 >= 1:
        return 1.0
    ct = math.sqrt(1 - s2)
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
    rp = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci)
    return 0.5 * (rs * rs + rp * rp)


def hg(g: float, c: float) -> float:
    """Henyey-Greenstein density per steradian at scattering cosine c."""
    return (1 - g * g) / (4 * math.pi * (1 + g * g - 2 * g * c) ** 1.5)


class ChandrasekharH:
    """Chandrasekhar's H function for isotropic scattering, by the standard
    iteration H = 1 / (1 - (w/2) mu int H(mu') / (mu + mu') dmu') on Gauss-
    Legendre nodes. Check: `integral` = 2 (1 - sqrt(1 - w)) / w."""

    def __init__(self, albedo: float, nodes: int = 400) -> None:
        x, wq = np.polynomial.legendre.leggauss(nodes)
        self.albedo = albedo
        self._mu, self._wq = 0.5 * (x + 1), 0.5 * wq
        h = np.ones(nodes)
        for _ in range(10_000):
            hn = 1 / (
                1 - 0.5 * albedo * self._mu * (self._wq * h / (self._mu[:, None] + self._mu)).sum(1)
            )
            done = bool(np.abs(hn - h).max() < 1e-14)
            h = hn
            if done:
                break
        self._h = h
        self.integral = float((self._wq * h).sum())

    def __call__(self, m: float) -> float:
        s = (self._wq * self._h / (m + self._mu)).sum()
        return float(1 / (1 - 0.5 * self.albedo * m * s))


def halfspace_albedo_iso(albedo: float) -> float:
    """Exact directional-hemispherical reflectance of a matched (n = 1) isotropic
    half-space for a normal beam: 1 - H(1) sqrt(1 - w)."""
    return 1 - ChandrasekharH(albedo)(1.0) * math.sqrt(1 - albedo)


def single_scatter_halfspace(n: float, mua: float, mus: float, g: float, theta_deg: float) -> float:
    """Radiance seen along the normal from single scattering in a half-space lit
    by unit irradiance (perpendicular to the beam) at polar angle theta, through a
    smooth face of index n: T(0)/n^2 * mus p(-cos th_t) E_in / (mu_t (1 + 1/cos th_t))."""
    th = math.radians(theta_deg)
    st = math.sin(th) / n
    ct = math.sqrt(1 - st * st)
    e_in = (1 - fresnel(math.cos(th), 1, n)) * math.cos(th) / ct
    mt = mua + mus
    return (1 - fresnel(1, 1, n)) / n**2 * mus * hg(g, -ct) * e_in / (mt * (1 + 1 / ct))
