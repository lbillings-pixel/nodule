"""The hollow's void box: where a crystal-lined centre is air (ADR 0019 s.6.12, step 3).

agatesim's last crystal stage leaves a point open when its earliest crystal arrives after
the stage's last reach (agatesim ADR 0090, `CrystalWall`). The box holds, at world nodes
i * `ID_MM` around the whole hollow, each node's crystal (`wall.arrival`) and a gate: the
nodes where the level's sign marks the hollow (the stage's own quartz, by grain, or open; elsewhere,
e.g. chalcedony behind the crystals' sites, the level is positive but no hollow). At a point
the wall test takes the earliest of its cell's 8 corner crystals in closed form
(`wall.times`, on the GPU agatesim's `CRYSTAL_TIME_CUDA`), so the wall stays sharp at any
id spacing. One box per stone, shared by every cut and view: its nodes do not depend on the
box's bounds (resolution-independence rule, CLAUDE.md).

The box holds no material: each view reads it on its own grid, continued under the hollow
(the owner's option (b), s.6.12 step 3). Not wired into `pack` or `cutface` yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from agatesim.fields import Phase
from agatesim.volume.growth import CrystalWall
from agatesim.volume.simulate import VolumeSimulation
from numpy.typing import NDArray

from nodule._scipy import distance_inside
from nodule.render.record import void_points

ID_MM = 0.2  # effective: id spacing; the wall's shape comes from the closed form (s.6.12)
COARSE_MM = 0.4  # effective: the pass that finds the hollow
SHELL_MM = 3.0  # effective: gated lining outside the hollow (s.6.12 step 3)
INNER_MM = 0.8  # effective: band evaluated inside the coarse hollow (2 coarse cells)
CHUNK = 200_000  # points a call to agatesim (memory)
CORNERS = np.array([(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)])

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def _nodes(a: I64, shape: tuple[int, ...], flat: I64, h: float) -> F64:
    """World coordinates of the box nodes with these flat (C-order) indices."""
    return np.asarray((np.stack(np.unravel_index(flat, shape), axis=-1) + a) * h, np.float64)


def _lattice(lo: F64, hi: F64, h: float) -> tuple[I64, I64]:
    """Node ranges of the world lattice (node i at i * h) covering lo..hi."""
    return np.floor(lo / h).astype(np.int64), np.ceil(hi / h).astype(np.int64)


def coarse_hollow(volume: VolumeSimulation, wall: CrystalWall) -> tuple[BoolArray, I64]:
    """The crystal hollow on the world lattice at `COARSE_MM` over the solid's bounds:
    open points (`void_points`) where the crystal level is positive. Hollow crystals and
    tubes are open too; they enter only where the level is positive (s.6.12 step 3 build).
    Returns the mask and its first node's index."""
    lo, hi = (np.asarray(x, dtype=np.float64) for x in volume.solid.bounds())
    ca, cb = _lattice(lo, hi, COARSE_MM)
    shape = tuple((cb - ca + 1).tolist())
    out = np.zeros(int(np.prod(shape)), bool)
    for s in range(0, len(out), CHUNK):
        k = np.arange(s, min(s + CHUNK, len(out)))
        p = _nodes(ca, shape, k, COARSE_MM)
        ins = volume.solid.contains(p)
        if not ins.any():
            continue
        void = void_points(volume.evaluate(p, ins), ins)
        if void.any():
            void[void] = wall.level(p[void]) > 0
        out[k] = void
    return out.reshape(shape), ca


@dataclass(frozen=True, eq=False)
class VoidBox:
    """Crystal ids and the gate at world nodes (a + index) * h, C-order (x, y, z): the
    layout the GPU reads (ids[i, j, k] at i * ny * nz + j * nz + k)."""

    a: I64  # world node index of ids[0, 0, 0]
    h: float
    ids: NDArray[np.int32]  # each node's crystal, as agatesim's `assign` gives it
    gate: BoolArray  # the level's sign marks the hollow here (the stage's quartz, or open)
    wall: CrystalWall

    @property
    def shape(self) -> tuple[int, int, int]:
        x, y, z = self.ids.shape
        return x, y, z

    def _cells(self, points: F64) -> tuple[I64, BoolArray]:
        """Each point's cell (its lowest corner, box index) and whether the box holds it."""
        base = np.floor(points / self.h).astype(np.int64) - self.a
        ok = np.all((base >= 0) & (base < np.array(self.shape) - 1), axis=1)
        return np.where(ok[:, None], base, 0), ok

    def level(self, points: F64) -> F64:
        """The box's level (mm): the earliest of the cell's 8 corner crystals, minus the
        stage's last reach (the GPU's wall test). NaN outside the box. Ungated."""
        base, ok = self._cells(points)
        out = np.full(len(points), np.nan)
        if not ok.any():
            return out
        idx = tuple((base[ok][:, None, :] + CORNERS[None]).transpose(2, 0, 1))
        cid = self.ids[idx].astype(np.int64)
        out[ok] = self.wall.times(points[ok], cid).min(axis=1) - self.wall.last_reach
        return out

    def gated(self, points: F64) -> BoolArray:
        """All 8 corners of the point's cell gated (*assumption*, s.6.12 step 3 build: a
        cell reaching outside the stage's quartz and the hollow is solid)."""
        base, ok = self._cells(points)
        out = np.zeros(len(points), bool)
        if ok.any():
            idx = tuple((base[ok][:, None, :] + CORNERS[None]).transpose(2, 0, 1))
            out[ok] = self.gate[idx].all(axis=1)
        return out

    def air(self, points: F64) -> BoolArray:
        """Points the renderer draws as air: gated and the box's level positive."""
        lev = self.level(points)
        return self.gated(points) & (np.nan_to_num(lev, nan=-1.0) > 0)


def build_void_box(
    volume: VolumeSimulation,
    *,
    shell_mm: float = SHELL_MM,
    pad_mm: float | None = None,
    h: float = ID_MM,
) -> VoidBox | None:
    """The stone's void box, or None without a crystal wall or a hollow. The box reaches
    `pad_mm` (default the shell plus a coarse cell) past the coarse hollow; ids and gate
    at a node do not depend on it (only on the world node, the hollow and `shell_mm`)."""
    wall = volume.crystal_wall
    if wall is None:
        return None
    hollow, ca = coarse_hollow(volume, wall)
    found = np.argwhere(hollow)
    if not len(found):
        return None
    pad = shell_mm + COARSE_MM if pad_mm is None else pad_mm
    if pad < shell_mm + COARSE_MM:
        raise ValueError("the box must hold the whole shell")
    lo = np.asarray((found.min(0) + ca) * COARSE_MM - pad, np.float64)
    hi = np.asarray((found.max(0) + ca) * COARSE_MM + pad, np.float64)
    a, b = _lattice(lo, hi, h)
    shape = tuple((b - a + 1).tolist())
    n = int(np.prod(shape))

    ids = np.empty(n, np.int32)
    for s in range(0, n, CHUNK):
        k = np.arange(s, min(s + CHUNK, n))
        ids[k] = wall.arrival(_nodes(a, shape, k, h))[1]

    # Distances to the coarse hollow, read at each box node from its nearest coarse node.
    # The hollow sits wholly inside the box, so they do not depend on the bounds.
    hol = np.zeros(n, bool)
    for s in range(0, n, CHUNK):
        k = np.arange(s, min(s + CHUNK, n))
        c = np.rint(_nodes(a, shape, k, h) / COARSE_MM).astype(np.int64) - ca
        inside = np.all((c >= 0) & (c < np.array(hollow.shape)), axis=1)
        c = c[inside]
        hol[k[inside]] = hollow[c[:, 0], c[:, 1], c[:, 2]]
    hol3 = hol.reshape(shape)
    d_out = distance_inside(~hol3).ravel() * h
    d_in = distance_inside(hol3).ravel() * h
    band = np.flatnonzero((d_out <= shell_mm) & ((d_out > 0) | (d_in <= INNER_MM)))
    gate = hol & (d_in > INNER_MM)  # deep in the hollow: open, or a crystal's tip
    del d_out, d_in
    for s in range(0, len(band), CHUNK):
        k = band[s : s + CHUNK]
        p = _nodes(a, shape, k, h)
        ins = volume.solid.contains(p)
        f = volume.evaluate(p, ins)
        stage = (f.grain >= wall.grain_offset) & (f.grain < wall.grain_offset + len(wall.sites))
        gate[k] = ins & (((f.phase == Phase.QUARTZ) & stage) | (f.phase == Phase.VOID))
    return VoidBox(a=a, h=h, ids=ids.reshape(shape), gate=gate.reshape(shape), wall=wall)
