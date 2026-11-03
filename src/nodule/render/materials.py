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

Scattering follows ADR 0007 choice 7a C: two parts per band, kept as two
Henyey-Greenstein lobes (camera.Materials, lobes a and b). The grain part is
Apetz's virtual composite through the Mie series (mie.grain_part) [@apetz2003];
the pore part is water-filled spheres (mie.pore_part; ADR 0010). agatesim's translucency
(1 clear .. 0 milky) sets both: the grain radius runs log-linearly from Lorenzi's
blue-band to white-band grains [@lorenzi2022], and the pore fraction rises
toward milky as (1 - t)^4 (ADR 0012). Coupling translucency to both parts is an ASSUMPTION, and
every radius and fraction is `effective`, to be tuned by render-and-compare.
Nothing here claims which part makes agate's white bands white (ADR 0007 s.5).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, NamedTuple

import numpy as np
from agatesim.fields import Phase, Pigment
from numpy.typing import ArrayLike, NDArray

from nodule.render.camera import N_CHALCEDONY, Materials
from nodule.render.mie import grain_part, pore_part
from nodule.render.record import FIELDS, STEPS, BandRecord

__all__ = ["STEPS", "BandRecord", "Optics", "OpticsTable", "rayleigh_absorption"]

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


class Optics(NamedTuple):
    """One material per channel: absorption, total scattering, and the two lobes
    (asymmetry g of lobe a, share f of mus in lobe b, asymmetry gb of lobe b)."""

    mua: F64
    mus: F64
    g: F64
    f: F64
    gb: F64

    @classmethod
    def one_lobe(cls, mua: ArrayLike, mus: ArrayLike, g: float) -> Optics:
        return cls(np.asarray(mua, float) * np.ones(3), np.asarray(mus, float) * np.ones(3),
                   np.full(3, g), np.zeros(3), np.zeros(3))  # fmt: skip


@dataclass(frozen=True)
class OpticsTable:
    """Fields -> optics. Every number `effective` unless its comment says `physical`."""

    hematite: Literal["go2022", "bedidi1993"] = "go2022"  # owner's choice; see HEMATITE_NK
    # Host rock (material 0) and tube cores (material 1): the prototype's stand-ins, a dull
    # grey-brown rock and a dark green Fe-silicate core (agatesim reports tube cores as
    # celadonite-like).
    host: Optics = field(default=Optics.one_lobe([1.6, 1.0, 0.6], 15.0, 0.8))
    tube_core: Optics = field(default=Optics.one_lobe([2.5, 0.8, 1.8], 10.0, 0.8))
    # Clean chalcedony absorbs little: bounded by Lorenzi's total-transmittance loss of
    # ~0.13 per mm, which includes backscatter [@lorenzi2022]; 0.01 sits under the bound.
    matrix_mua: float = 0.01
    # 7a C, grain part: crystallite radius (um) at translucency 1 (clear) and 0 (milky),
    # log-interpolated. The ends are Lorenzi's grain sizes, ~300 nm in blue bands and
    # ~5 um in white bands, halved to radii [@lorenzi2022].
    grain_r_clear_um: float = 0.15
    grain_r_milky_um: float = 2.5
    # 7a C, pore part: radius (um), index, and volume fraction pore_milky * (1 - t)^pore_gamma
    # (t: translucency). Tuning step 1 (ADR 0010 10a A): water-filled pores of ~0.1 um,
    # as Folk and Weaver saw them, at 1.75% with gamma 1, near the ~2% their densities give
    # if the pores hold water [@folk1952]; disputed (no such pores by TEM [@graetsch1985]).
    # 1.75% water scatters like the 0.3% air of the tested setting (ADR 0007's was 0.1% air).
    # Tuning step 2 (ADR 0012 12a A): gamma 4, so clear bands reach the grains' floor, with
    # pore_milky rescaled to keep the step-1 fraction (1.29%) at the n2-slice record's median
    # band, t 0.264. The curve's shape is an ASSUMPTION: no source measures pores band by
    # band. All `effective`, *speculative* as optics.
    pore_r_um: float = 0.1
    pore_n: float = 1.333  # water, `physical`
    pore_gamma: float = 4.0
    pore_milky: float = 0.0175 / 0.736**3  # = 0.04389
    quartz_mus: float = 0.05  # macrocrystalline quartz: nearly clear (one lobe)
    quartz_g: float = 0.9
    # Pigment loading: volume fraction at impurity 1 for the iron oxides (impurity scales it
    # linearly, an assumption), and mu_a at 550 nm at impurity 1 for the shape-only pigments.
    phi_iron_oxide: float = 9e-4  # ADR 0010 step 1 (was 3e-4)
    mua550_shape: float = 3.0
    # Agatesim's unnamed IRON stain runs ochre to deep red as impurity rises (agatesim
    # render.STAIN): assumed a goethite-hematite mixture with hematite share = impurity.
    # BLUE and WHITE pigments are scattering, not pigment (agatesim ADR 0050), both in lobe
    # b: BLUE as Rayleigh scatterers (mu_s at 550 nm at impurity 1, lambda^-4, g 0), WHITE
    # as extra pores of the pore part's kind (volume fraction at impurity 1).
    mus_blue550: float = 2.0
    pore_white: float = 0.011667  # ADR 0010: 2e-3 scaled with pore_milky (0.0175 / 0.003)

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

    def grain_radius(self, translucency: float) -> float:
        a = min(max(float(translucency), 0.0), 1.0)
        lo, hi = math.log(self.grain_r_milky_um), math.log(self.grain_r_clear_um)
        return math.exp(a * hi + (1 - a) * lo)

    def scattering(self, pigment: int, impurity: float, translucency: float) -> Optics:
        """Scattering of a chalcedony deposit (mua left 0). Lobe a: the grains. Lobe b:
        the pores plus any BLUE scatterers, their g weighted by mu_s within the lobe (exact
        for the mean cosine; everything in lobe b is near-isotropic, g < ~0.65)."""
        a = min(max(float(translucency), 0.0), 1.0)
        mus_a, g_a = grain_part(self.grain_radius(a), WAVELENGTH_NM, N_CHALCEDONY)
        phi = (1 - a) ** self.pore_gamma * self.pore_milky
        if pigment == Pigment.WHITE:
            phi += max(float(impurity), 0.0) * self.pore_white
        mus_p, g_p = pore_part(self.pore_r_um, phi, self.pore_n, WAVELENGTH_NM, N_CHALCEDONY)
        mus_bl = np.zeros(3)
        if pigment == Pigment.BLUE:
            mus_bl = max(float(impurity), 0.0) * self.mus_blue550 * (550.0 / WAVELENGTH_NM) ** 4
        mus_b = mus_p + mus_bl
        gb = np.divide(mus_p * g_p, mus_b, out=np.zeros(3), where=mus_b > 0)
        mus = mus_a + mus_b
        return Optics(np.zeros(3), mus, g_a, mus_b / mus, gb)

    def band(self, phase: int, pigment: int, impurity: float, translucency: float) -> Optics:
        """Optics of one growth step's deposit."""
        if Phase(phase) is Phase.QUARTZ:
            # agatesim stains quartz by impurity as iron (render._quartz); same here.
            mua = self.matrix_mua + self.pigment_mua(Pigment.IRON, impurity)
            return Optics.one_lobe(mua, self.quartz_mus, self.quartz_g)
        if Phase(phase) is not Phase.CHALCEDONY:
            raise ValueError(f"no optics for phase {Phase(phase).name} in a band")
        mua = self.matrix_mua + self.pigment_mua(pigment, impurity)
        return self.scattering(pigment, impurity, translucency)._replace(mua=mua)

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
            mua=[r.mua for r in rows], mus=[r.mus for r in rows], g=[r.g for r in rows],
            profile=prof.ravel(), dt=1.0 / STEPS, f=[r.f for r in rows], gb=[r.gb for r in rows],
        )  # fmt: skip
