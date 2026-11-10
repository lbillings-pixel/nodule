"""The layers' place-to-place part, baked as absorption on the growth-time grid (ADR 0019).

Nodule's quick layers (`nodule.variety.VarietyLayers`: mottle, rim stain, crack
stains and fills, dust) vary along a band, so a growth step's record cannot hold them.
Their per-step part (white bands thinning inward, quartz low in iron) goes in the
record (`Specimen.step_fields`); the rest is baked here once per cut, as the extra
absorption (mm^-1, three channels) it gives at each node of the cut's growth-time grid,
which the camera adds to each sample's own (`camera.cu` matid, trilinear).

Absorption, not impurity, is averaged: the optics give iron pigments no absorption
below an impurity floor (ADR 0014 4l), so averaging a thin dust line's impurity over a
cell would drop it under the floor and lose it. Nodes the layers touch (read with area
coverage at the grid's spacing) are sampled at SUB^3 sub-points with the layers at the
sub-point spacing, and their absorption gains averaged; other nodes gain nothing.

Plumes (ADR 0018 s.13) are baked the same way. A plume point renders as the nearest
step's chalcedony (`record.plume_points`) and gains its mineral's absorption: agatesim's
tint (0 green Fe-silicate, 0.5 red-brown Fe oxide, 1 black Mn oxide) read through
PLUME_STOPS, the pigments at the stops of `nodule.plumes.INCLUSION_RAMP`, each scaled so
its strongest channel absorbs PLUME_MUA. A plume's clear halo (agatesim ADR 0079) is
lost colour: the band's own absorption minus the halo's (`Specimen.per_step` gives the
grid the band's values; `Specimen.placed` the halo's).

Limits (ADR 0019): scattering is not baked, so a clear-filled crack trace is clearer
in colour but not in haze, a halo loses its band's colour but not its milkiness, and a
plume scatters as the chalcedony round it, so it reads as a stain, not an opaque
mineral body; a pigment change takes the new pigment's absorption at the point; plume
branches thinner than the grid spacing blur. Appearance only (ADR 0002).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import numpy as np
from agatesim import progress
from agatesim.fields import Fields, Phase, Pigment
from agatesim.grid import FloatArray
from numpy.typing import NDArray

from nodule.render.materials import OpticsTable
from nodule.render.record import plume_points
from nodule.render.tgrid import CHUNK_POINTS, GrowthGrid, to_world
from nodule.specimen import Specimen

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]
SUB = 2  # sub-samples per node along each axis
LEVELS = 2000  # impurity steps in the absorption table (nearest; the floor is a step)
# A plume's mineral over agatesim's tint: celadonite (green), goethite (orange-gold,
# yellow-brown), hematite (red), manganese oxide (black); between stops the absorptions
# mix linearly (0.75: half hematite, half Mn, the ramp's dark brown). `effective`.
PLUME_STOPS = (
    (0.0, Pigment.CELADONITE),
    (0.3, Pigment.GOETHITE),
    (0.5, Pigment.HEMATITE),
    (1.0, Pigment.MANGANESE),
)
# Absorption (mm^-1) of a plume's strongest channel: `effective`, an ASSUMPTION by eye.
# Over a ~0.2 mm branch it passes ~e^-2 of that channel and more of the others, so the
# hue shows; much higher and every plume turns black.
PLUME_MUA = 10.0


def absorption_table(optics: OpticsTable) -> F64:
    """(pigment, impurity level, channel) -> pigment absorption, impurity in
    [0, 1] at LEVELS + 1 levels."""
    x = np.linspace(0.0, 1.0, LEVELS + 1)
    return np.array([[optics.pigment_mua(int(p), float(v)) for v in x] for p in Pigment])


def absorption(table: F64, f: Fields) -> F64:
    """Pigment absorption (n, 3) at each point of `f`, as the optics give a band of
    that pigment and impurity (quartz as IRON: `OpticsTable.band`); 0 elsewhere."""
    imp = np.clip(np.nan_to_num(f.impurity, nan=0.0), 0.0, 1.0)
    k = np.rint(imp * LEVELS).astype(np.intp)
    pig = (
        np.full(len(imp), int(Pigment.IRON), np.intp)
        if f.pigment is None
        else f.pigment.astype(np.intp)
    )
    pig = np.where(f.phase == Phase.QUARTZ, int(Pigment.IRON), pig)
    out = table[pig, k]
    keep = (f.phase == Phase.CHALCEDONY) | (f.phase == Phase.QUARTZ)
    return np.where(keep[:, None], out, 0.0)


def plume_stops(optics: OpticsTable) -> F64:
    """(stop, channel): each PLUME_STOPS pigment's absorption shape, its strongest
    channel at PLUME_MUA."""
    rows = [optics.pigment_mua(int(p), 1.0) for _, p in PLUME_STOPS]
    return np.array([PLUME_MUA * r / r.max() for r in rows])


def plume_absorption(stops: F64, f: Fields, inside: NDArray[np.bool_]) -> F64:
    """A plume's mineral absorption (n, 3) at its points (`record.plume_points`) from
    their tint; 0 elsewhere."""
    at = plume_points(f, inside)
    tint = np.clip(np.nan_to_num(f.impurity[at], nan=1.0), 0.0, 1.0)
    xs = [x for x, _ in PLUME_STOPS]
    out = np.zeros((len(at), 3))
    out[at] = np.stack([np.interp(tint, xs, stops[:, c]) for c in range(3)], axis=-1)
    return out


def _touched(a: Fields, b: Fields) -> NDArray[np.bool_]:
    out = (a.impurity != b.impurity) | (a.translucency != b.translucency) | (a.phase != b.phase)
    if a.pigment is not None and b.pigment is not None:
        out |= a.pigment != b.pigment
    return out


@dataclass
class Baker:
    """The bake, chunk by chunk, from the fields the grid fill reads anyway
    (`GrowthGrid.fill` with optics), so each node is read once."""

    specimen: Specimen
    table: F64
    stops: F64
    spacing_mm: float
    out: F32  # (3, n)

    @classmethod
    def of(cls, specimen: Specimen, optics: OpticsTable, spacing_mm: float, n: int) -> Baker:
        if specimen.layers is None:
            raise ValueError("no layers to bake")
        if specimen.fractures:
            raise ValueError("a layered specimen takes no slipping fractures")
        table, stops = absorption_table(optics), plume_stops(optics)
        return cls(specimen, table, stops, spacing_mm, np.zeros((3, n), np.float32))

    def gain(self, p: FloatArray, raw: Fields, inside: NDArray[np.bool_], h: float) -> F64:
        """Absorption (n, 3) of what lies at each point less that of its step's own
        material, which the camera gives it: `raw` the growth model's fields there
        (`Specimen.restored_fields`), layers drawn at pixel h."""
        sp = self.specimen
        step, placed = sp.per_step(raw), sp.placed(p, raw, inside, h)
        lies = absorption(self.table, placed) + plume_absorption(self.stops, placed, inside)
        return lies - absorption(self.table, step)

    def chunk(
        self, start: int, p: FloatArray, raw: Fields, step: Fields, inside: NDArray[np.bool_]
    ) -> None:
        """Nodes start .. start + len(p): their places, the growth model's fields there
        and their per-step view (`Specimen.per_step`, which the grid holds)."""
        h = self.spacing_mm
        placed = self.specimen.placed(p, raw, inside, h)
        touched = inside & (_touched(placed, step) | plume_points(placed, inside))
        if not touched.any():
            return
        q = p[touched]
        gain = np.zeros((len(q), 3))
        offsets = list(product((np.arange(SUB) + 0.5) / SUB - 0.5, repeat=3))
        for off in offsets:
            sub = q + np.asarray(off) * h
            fr, ib = self.specimen.restored_fields(sub)
            gain += self.gain(sub, fr, ib, h / SUB)
        idx = start + np.flatnonzero(touched)
        self.out[:, idx] = (gain / len(offsets)).T.astype(np.float32)


def bake_absorption(specimen: Specimen, grid: GrowthGrid, optics: OpticsTable) -> F32 | None:
    """The layers' added absorption (3, nw, nt, ns) at the grid's nodes (RESTORED
    coordinates, as the grid; layered specimens carry no slipping fractures); None
    for a specimen without layers. A separate pass over a filled grid; the camera
    bakes during the fill (`GrowthGrid.fill` with optics), reading each node once."""
    if specimen.layers is None:
        return None
    shape = (len(grid.w), len(grid.t), len(grid.s))
    n = math.prod(shape)
    baker = Baker.of(specimen, optics, grid.spacing_mm, n)
    for a in range(0, n, CHUNK_POINTS):
        progress.report(a / n, "baking the layers")
        idx = np.arange(a, min(a + CHUNK_POINTS, n))
        k, j, i = np.unravel_index(idx, shape)
        p = to_world(grid.plane, grid.s[i], grid.t[j], grid.w[k])
        raw, inside = specimen.restored_fields(p)
        baker.chunk(a, p, raw, specimen.per_step(raw), inside)
    progress.report(1.0, "baking the layers")
    return baker.out.reshape(3, *shape)
