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

Limits (ADR 0019): scattering is not baked, so a clear-filled crack trace is clearer
in colour but not in haze; a pigment change takes the new pigment's absorption at the
point. Appearance only (ADR 0002).
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
from nodule.render.tgrid import CHUNK_POINTS, GrowthGrid, to_world
from nodule.specimen import Layers, Specimen

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]
SUB = 2  # sub-samples per node along each axis
LEVELS = 2000  # impurity steps in the absorption table (nearest; the floor is a step)


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


def _touched(a: Fields, b: Fields) -> NDArray[np.bool_]:
    out = (a.impurity != b.impurity) | (a.translucency != b.translucency) | (a.phase != b.phase)
    if a.pigment is not None and b.pigment is not None:
        out |= a.pigment != b.pigment
    return out


@dataclass
class Baker:
    """The bake, chunk by chunk, from the base fields the grid fill reads anyway
    (`GrowthGrid.fill` with optics), so each node is read once."""

    specimen: Specimen
    layers: Layers
    table: F64
    spacing_mm: float
    out: F32  # (3, n)

    @classmethod
    def of(cls, specimen: Specimen, optics: OpticsTable, spacing_mm: float, n: int) -> Baker:
        if specimen.layers is None:
            raise ValueError("no layers to bake")
        if specimen.fractures:
            raise ValueError("a layered specimen takes no slipping fractures")
        table = absorption_table(optics)
        return cls(specimen, specimen.layers, table, spacing_mm, np.zeros((3, n), np.float32))

    def chunk(self, start: int, p: FloatArray, base: Fields, inside: NDArray[np.bool_]) -> None:
        """Nodes start .. start + len(p): their places and base fields."""
        h = self.spacing_mm
        touched = inside & _touched(self.layers(p, base, inside, h), base)
        if not touched.any():
            return
        q = p[touched]
        gain = np.zeros((len(q), 3))
        offsets = list(product((np.arange(SUB) + 0.5) / SUB - 0.5, repeat=3))
        for off in offsets:
            sub = q + np.asarray(off) * h
            fb, ib = self.specimen.restored_step_fields(sub)
            fl = self.layers(sub, fb, ib, h / SUB)
            gain += absorption(self.table, fl) - absorption(self.table, fb)
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
        base, inside = specimen.restored_step_fields(p)
        baker.chunk(a, p, base, inside)
    progress.report(1.0, "baking the layers")
    return baker.out.reshape(3, *shape)
