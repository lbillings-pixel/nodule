"""Optical materials from agatesim's band fields (ADR 0007 section 3).

Two steps. `BandRecord` reads what fills each growth-time step from agatesim
`Fields` (phase, pigment, impurity, translucency), checking as it goes that these
depend on the step alone. In agatesim's 3D growth model they do: every band
property is drawn per band, and band = f(t_dep) (agatesim `growth_fields`). A
specimen that breaks this (an open cavity, hollow crystals, dendrites) is
refused rather than rendered wrong. `OpticsTable.materials` then turns each
distinct step into absorption, scattering and a phase function, the camera's
material table (camera.py: 0 host, 1 tube core, 2.. bands, band = profile[k]
for t in [k, k + 1) / STEPS).

THE MAPPING IS AN ASSUMPTION (ADR 0007 section 3): every coefficient below is
`effective` unless marked `physical`. Couplings between agatesim's fields and
optics are stated where they are made. Grades (ADR 0002): the Rayleigh
absorption arithmetic is *verified* against the reading notes; the pigment n, k
are `physical` within their sources; everything else is *speculative* until the
set-level comparison (ADR 0006 s.10). Renders are procedural and are never
evidence for what agate bands are made of.

Scattering is a STAND-IN until ADR 0007 choice 7a (grain + pores per band,
roadmap N2 step 4 item 2) replaces `stand_in_scattering`: log-interpolated
between the prototype's clear and milky values (notes/n2-proto/RESULTS.md),
which look too milky and soft-edged.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from agatesim.fields import Phase, Pigment
from numpy.typing import NDArray

from nodule.render.camera import N_CHALCEDONY, Materials
from nodule.render.record import FIELDS, STEPS, BandRecord

__all__ = ["STEPS", "BandRecord", "OpticsTable", "rayleigh_absorption"]

WAVELENGTH_NM = np.array([450.0, 550.0, 650.0])  # the renderer's channels (ADR 0007 s.2)

F64 = NDArray[np.float64]

# --- pigment optics -------------------------------------------------------------------------
# Complex index n + ik at 450/550/650 nm, `physical` within each source's spread.
# Hematite: the two sources disagree about 3x in the blue (notes/n2-reading/agate-optics2.md).
HEMATITE_NK = {
    # Go et al.'s retained set of five datasets, read from Fig. 3 (+-30% in k) [@go2022]
    "go2022": ((3.0, 3.1, 2.95), (1.0, 0.5, 0.2)),
    # ordinary ray, read from Fig. 3a (the 450 value is the 460 nm reading) [@bedidi1993]
    "bedidi1993": ((3.24, 3.255, 3.03), (0.31, 0.215, 0.185)),
}
# Goethite, Bedidi Fig. 3b [@bedidi1993]: n 2.42 (460)/2.28/2.21, k 0.068 (460)/0.085/0.08.
# Its k does not rise toward the blue, so it cannot give goethite's yellow; the 450 nm k
# below is raised past the absorption edge at 488 nm (479-493) [@scheinost1998]. That
# value is `effective` and *speculative* (ADR 0007 s.3).
GOETHITE_N = (2.42, 2.28, 2.21)
GOETHITE_K = (0.25, 0.085, 0.08)  # k(450) = 0.25 effective; 550, 650 physical
# Pigments without a measured k: absorption SHAPE at 450/550/650 (relative to 550), all
# `effective`. Celadonite: relative K/S 1.5/1.0/1.8 from Bishop et al.'s powder spectrum
# (Fig. 6) via Kubelka-Munk [@bishop2008]. Manganese oxide: no k found, a neutral black.
# Violet: no source; the shape of agatesim's 2D palette end colour (agatesim ADR 0050),
# optical density of linear sRGB (100, 62, 130): 1.50/3.04/2.06 -> 0.49/1/0.68.
SHAPE = {
    Pigment.CELADONITE: (1.5, 1.0, 1.8),
    Pigment.MANGANESE: (1.0, 1.0, 1.0),
    Pigment.VIOLET: (0.49, 1.0, 0.68),
}


def rayleigh_absorption(n: Sequence[float], k: Sequence[float], phi: float,
                        n_med: float = N_CHALCEDONY) -> F64:  # fmt: skip
    """Absorption (per mm) at the three channels of a volume fraction `phi` of small
    spheres of index n + ik in a medium n_med: the electrostatic (dipole) limit of
    Mie theory, mu_a = phi (6 pi n_med / lambda) Im[(m^2 - 1)/(m^2 + 2)], m = (n + ik)/n_med.
    Valid for particles well below ~50 nm, where Bedidi's Mie albedo is <~ 0.3 so the
    scattering they add is neglected [@bedidi1993] [inference]."""
    m = (np.asarray(n) + 1j * np.asarray(k)) / n_med
    lam_mm = WAVELENGTH_NM * 1e-6
    return phi * 6 * math.pi * n_med / lam_mm * np.imag((m * m - 1) / (m * m + 2))


# --- the table ------------------------------------------------------------------------------

Optics = tuple[F64, F64, float]  # mua (3,), mus (3,), g


@dataclass(frozen=True)
class OpticsTable:
    """Fields -> optics. Every number `effective` unless its comment says `physical`."""

    hematite: Literal["go2022", "bedidi1993"] = "go2022"  # owner's choice; see HEMATITE_NK
    # Host rock (material 0) and tube cores (material 1): the prototype's stand-ins, a dull
    # grey-brown rock and a dark green Fe-silicate core (agatesim reports tube cores as
    # celadonite-like).
    host: Optics = field(default=(np.array([1.6, 1.0, 0.6]), np.full(3, 15.0), 0.8))
    tube_core: Optics = field(default=(np.array([2.5, 0.8, 1.8]), np.full(3, 10.0), 0.8))
    # Clean chalcedony absorbs little: bounded by Lorenzi's total-transmittance loss of
    # ~0.13 per mm, which includes backscatter [@lorenzi2022]; 0.01 sits under the bound.
    matrix_mua: float = 0.01
    # STAND-IN scattering (until 7a): translucency 1 (clear) .. 0 (milky), log-interpolated.
    mus_clear: float = 1.5
    mus_milky: float = 20.0
    g_band: float = 0.9
    quartz_mus: float = 0.05  # macrocrystalline quartz: nearly clear
    # Pigment loading: volume fraction at impurity 1 for the iron oxides (impurity scales it
    # linearly, an assumption), and mu_a at 550 nm at impurity 1 for the shape-only pigments.
    phi_iron_oxide: float = 3e-4
    mua550_shape: float = 3.0
    # Agatesim's unnamed IRON stain runs ochre to deep red as impurity rises (agatesim
    # render.STAIN): assumed a goethite-hematite mixture with hematite share = impurity.
    # BLUE and WHITE pigments are scattering, not pigment (agatesim ADR 0050): extra mu_s at
    # impurity 1, Rayleigh-shaped (lambda^-4, g 0) for blue and flat (g 0.5) for white.
    mus_blue550: float = 2.0
    mus_white: float = 20.0

    def _hematite(self, phi: float) -> F64:
        n, k = HEMATITE_NK[self.hematite]
        return rayleigh_absorption(n, k, phi)

    def pigment_mua(self, pigment: int, impurity: float) -> F64:
        p, x = Pigment(pigment), float(impurity)
        phi = x * self.phi_iron_oxide
        if p is Pigment.HEMATITE:
            return self._hematite(phi)
        if p is Pigment.GOETHITE:
            return rayleigh_absorption(GOETHITE_N, GOETHITE_K, phi)
        if p is Pigment.IRON:
            h = min(max(x, 0.0), 1.0)
            goethite = rayleigh_absorption(GOETHITE_N, GOETHITE_K, (1 - h) * phi)
            return self._hematite(h * phi) + goethite
        if p in SHAPE:
            return x * self.mua550_shape * np.asarray(SHAPE[p], float)
        return np.zeros(3)  # BLUE, WHITE: scattering only

    def stand_in_scattering(self, translucency: float) -> tuple[F64, float]:
        a = min(max(float(translucency), 0.0), 1.0)
        mus = math.exp(a * math.log(self.mus_clear) + (1 - a) * math.log(self.mus_milky))
        return np.full(3, mus), self.g_band

    def band(self, phase: int, pigment: int, impurity: float, translucency: float) -> Optics:
        """Optics of one growth step's deposit."""
        if Phase(phase) is Phase.QUARTZ:
            # agatesim stains quartz by impurity as iron (render._quartz); same here.
            mua = self.matrix_mua + self.pigment_mua(Pigment.IRON, impurity)
            return mua, np.full(3, self.quartz_mus), self.g_band
        if Phase(phase) is not Phase.CHALCEDONY:
            raise ValueError(f"no optics for phase {Phase(phase).name} in a band")
        mus, g = self.stand_in_scattering(translucency)
        mua = self.matrix_mua + self.pigment_mua(pigment, impurity)
        extra, g_extra = np.zeros(3), 0.0
        if pigment == Pigment.BLUE:
            extra, g_extra = impurity * self.mus_blue550 * (550.0 / WAVELENGTH_NM) ** 4, 0.0
        elif pigment == Pigment.WHITE:
            extra, g_extra = np.full(3, impurity * self.mus_white), 0.5
        if extra.any():
            # Two scatterers in one Henyey-Greenstein lobe: g weighted by mu_s (assumption;
            # exact for the mean cosine, not the shape). Per channel, then averaged, since the
            # kernel takes one g per material.
            tot = mus + extra
            g = float(np.mean((mus * g + extra * g_extra) / tot))
            mus = tot
        return mua, mus, g

    def materials(self, record: BandRecord) -> Materials:
        """The camera's material table for a record: 0 host, 1 tube core, then one
        material per distinct deposit, and the step -> material profile (dt = 1/STEPS)."""
        src = record.nearest_seen()
        keys = np.stack([getattr(record, name)[src].astype(np.float64) for name in FIELDS], 1)
        uniq, prof = np.unique(keys, axis=0, return_inverse=True)
        rows = [self.host, self.tube_core] + [
            self.band(int(p), int(pg), float(x), float(tr)) for p, pg, x, tr in uniq
        ]
        return Materials.of(
            mua=[r[0] for r in rows], mus=[r[1] for r in rows], g=[r[2] for r in rows],
            profile=prof.ravel(), dt=1.0 / STEPS,
        )  # fmt: skip
