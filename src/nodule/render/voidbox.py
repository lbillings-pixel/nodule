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

Where the hollow meets bare chalcedony (thin crystal linings), the box also holds the
chalcedony's last surface: agatesim's `CrystalWall.chalcedony_level` (phi, agatesim ADR
0091) at world nodes i * `PHI_MM`, read trilinearly; air needs phi > 0 too, and cell
corners may then be chalcedony near that surface (the owner's option (A), rule R2 of the
s.6.12 probe). Without phi (agatesim returns None) the box is the gate-only box.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from agatesim.fields import Phase
from agatesim.volume.growth import CrystalWall
from agatesim.volume.simulate import VolumeSimulation
from agatesim.volume.slicing import Plane
from numpy.typing import NDArray

from nodule._scipy import distance_inside
from nodule.render.record import void_points
from nodule.render.tgrid import Window, to_plane

ID_MM = 0.2  # effective: id spacing; the wall's shape comes from the closed form (s.6.12)
COARSE_MM = 0.4  # effective: the pass that finds the hollow
SHELL_MM = 3.0  # effective: gated lining outside the hollow (s.6.12 step 3)
INNER_MM = 0.8  # effective: band evaluated inside the coarse hollow (2 coarse cells)
PHI_MM = ID_MM / 2  # effective: phi spacing (s.6.12 option A probe: 0.2 mm loses 2-3 %)
PHI_BAND_MM = 0.6  # effective: phi evaluated exactly on PHI_MM nodes this near the surface
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
    # The chalcedony's last surface (None: agatesim gives no phi; the gate alone decides).
    phi: NDArray[np.float32] | None = None  # at world nodes (2 a + index) * h / 2
    chal: BoolArray | None = None  # chalcedony at nodes with |phi| < PHI_BAND_MM (id grid)
    # Nodes within the shell (shell_mm of the coarse hollow, the hollow included): below the
    # slab the camera reads material here and loses light leaving it (s.6.12 step 4)
    shell: BoolArray | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        x, y, z = self.ids.shape
        return x, y, z

    def _cells(self, points: F64) -> tuple[I64, BoolArray]:
        """Each point's cell (its lowest corner, box index) and whether the box holds it."""
        base = np.floor(points / self.h).astype(np.int64) - self.a
        ok = np.all((base >= 0) & (base < np.array(self.shape) - 1), axis=1)
        return np.where(ok[:, None], base, 0), ok

    def in_shell(self, points: NDArray[np.floating]) -> BoolArray:
        """Points whose nearest id node lies in the shell (the GPU's rule); False off the box."""
        if self.shell is None:
            raise ValueError("this box holds no shell")
        i = np.rint(points / self.h).astype(np.int64) - self.a
        ok = np.all((i >= 0) & (i < np.array(self.shape)), axis=1)
        out = np.zeros(len(points), bool)
        j = i[ok]
        out[ok] = self.shell[j[:, 0], j[:, 1], j[:, 2]]
        return out

    def chalcedony_level(self, points: F64) -> F64:
        """phi read trilinearly from its grid (NaN outside the box). Requires phi."""
        if self.phi is None:
            raise ValueError("this box holds no chalcedony level")
        return trilinear(self.phi, 2 * self.a, self.h / 2, points)

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
        cell reaching outside the stage's quartz and the hollow is solid). With phi, a
        corner may also be chalcedony near its surface (phi then decides there)."""
        base, ok = self._cells(points)
        out = np.zeros(len(points), bool)
        if ok.any():
            idx = tuple((base[ok][:, None, :] + CORNERS[None]).transpose(2, 0, 1))
            g = self.gate[idx]
            if self.chal is not None:
                g = g | self.chal[idx]
            out[ok] = g.all(axis=1)
        return out

    def air(self, points: F64) -> BoolArray:
        """Points the renderer draws as air: gated, the box's level positive and, with
        phi, beyond the chalcedony's last surface."""
        out = self.gated(points) & (np.nan_to_num(self.level(points), nan=-1.0) > 0)
        if self.phi is not None and out.any():
            out[out] = self.chalcedony_level(points[out]) > 0
        return out


def dilated_shell(box: VoidBox) -> VoidBox:
    """The box with its shell grown by one id node (all 26 neighbours, so no corner of a
    cell next to the shell is left unfilled): a view grid's nodes within a cell of the
    shell, which the camera's trilinear reads there may touch."""
    if box.shell is None:
        raise ValueError("this box holds no shell")
    s = box.shell.copy()
    for ax in range(3):  # separable: grown along each axis in turn gives the 3 x 3 x 3 block
        grown = s.copy()
        for sh in (1, -1):
            r = np.roll(s, sh, axis=ax)
            edge = [slice(None)] * 3
            edge[ax] = slice(0, 1) if sh > 0 else slice(-1, None)
            r[tuple(edge)] = False
            grown |= r
        s = grown
    return replace(box, shell=s)


def under_slab(
    box: VoidBox, plane: Plane, window: Window, spacing_mm: float
) -> tuple[float, Callable[[NDArray[np.floating]], BoolArray]]:
    """How deep a view's grid must reach below its face to hold the shell under its window
    (the owner's option (b), s.6.12 step 3: the view's own grid continued down), and the
    keep test for the nodes below its slab: the shell grown by one node."""
    grown = dilated_shell(box)
    assert grown.shell is not None
    nodes = (np.argwhere(grown.shell) + box.a) * box.h
    s, t, w = to_plane(plane, nodes.astype(np.float64))
    (s0, s1), (t0, t1) = window
    under = (s >= s0) & (s <= s1) & (t >= t0) & (t <= t1)
    deep = float(w[under].max()) + spacing_mm if under.any() else 0.0
    return deep, grown.in_shell


def hollow_nodes(box: VoidBox, points: NDArray[np.floating]) -> BoolArray:
    """Points of the crystal hollow (agatesim's crystal level positive) within the shell:
    drawn as air, or as their lining where the box's wall is a hair off (slivers), so the
    void's stand-in absorption is left off there (s.6.12 step 4)."""
    out = box.in_shell(points)
    if out.any():
        out[out] = box.wall.level(points[out]) > 0
    return out


def trilinear(grid: NDArray[np.floating], a: I64, h: float, points: F64) -> F64:
    """grid (node (a + index) * h) read trilinearly at points; NaN outside it. The cell
    and weights as the GPU will take them: base = floor(p / h), weight = p / h - base."""
    q = points / h
    fl = np.floor(q)
    base = fl.astype(np.int64) - a
    ok = np.all((base >= 0) & (base < np.array(grid.shape) - 1), axis=1)
    out = np.full(len(points), np.nan)
    if not ok.any():
        return out
    b, w = base[ok], q[ok] - fl[ok]
    acc = np.zeros(len(b))
    for c in CORNERS:
        wt = np.prod(np.where(c[None] == 1, w, 1.0 - w), axis=1)
        acc += wt * grid[b[:, 0] + c[0], b[:, 1] + c[1], b[:, 2] + c[2]]
    out[ok] = acc
    return out


def _upsample[T: np.generic](arr: NDArray[T], op: str) -> NDArray[T]:
    """Double a node grid's density (2 n - 1 nodes an axis): a new node takes the mean
    ('mean', i.e. trilinear), any ('or') or all ('and') of the old nodes around it."""
    for ax in range(3):
        n = int(arr.shape[ax])
        shape = [int(x) for x in arr.shape]
        shape[ax] = 2 * n - 1
        out = np.empty(shape, arr.dtype)
        even = [slice(None)] * 3
        odd = [slice(None)] * 3
        even[ax], odd[ax] = slice(0, None, 2), slice(1, None, 2)
        lo: NDArray[Any] = np.take(arr, np.arange(n - 1), axis=ax)
        hi: NDArray[Any] = np.take(arr, np.arange(1, n), axis=ax)
        out[tuple(even)] = arr
        if op == "mean":
            out[tuple(odd)] = (lo + hi) * arr.dtype.type(0.5)
        elif op == "or":
            out[tuple(odd)] = lo | hi
        else:
            out[tuple(odd)] = lo & hi
        arr = out
    return arr


def _phi_grid(
    volume: VolumeSimulation,
    level: Callable[[F64, BoolArray], NDArray[np.floating]],
    a: I64,
    shape: tuple[int, ...],
    h: float,
) -> tuple[NDArray[np.float32], I64]:
    """phi on the id lattice (exact), doubled to h / 2: exact where its parent nodes lie
    within PHI_BAND_MM of the surface or differ in sign (a tube core's jump), elsewhere
    the parents' trilinear mean. Both rules read only world nodes, so the grid does not
    depend on the box's bounds. Also returns the id-lattice nodes near the surface."""
    n = int(np.prod(shape))
    phi = np.empty(n, np.float32)
    for s in range(0, n, CHUNK):
        k = np.arange(s, min(s + CHUNK, n))
        p = _nodes(a, shape, k, h)
        phi[k] = level(p, volume.solid.contains(p))
    phi3 = phi.reshape(shape)
    near = np.abs(phi3) < PHI_BAND_MM
    pos = phi3 > 0
    fine = _upsample(phi3, "mean")
    exact = _upsample(near, "or") | (_upsample(pos, "or") & ~_upsample(pos, "and"))
    del pos
    exact[::2, ::2, ::2] = False  # the id nodes are exact already
    todo = np.flatnonzero(exact)
    del exact
    fshape, flat, a2 = fine.shape, fine.reshape(-1), 2 * a
    for s in range(0, len(todo), CHUNK):
        k = todo[s : s + CHUNK]
        p = _nodes(a2, fshape, k, h / 2)
        flat[k] = level(p, volume.solid.contains(p))
    return fine, np.flatnonzero(near)


def build_void_box(
    volume: VolumeSimulation,
    *,
    shell_mm: float = SHELL_MM,
    pad_mm: float | None = None,
    h: float = ID_MM,
    with_phi: bool = True,
) -> VoidBox | None:
    """The stone's void box, or None without a crystal wall or a hollow. The box reaches
    `pad_mm` (default the shell plus a coarse cell) past the coarse hollow; ids, gate and
    phi at a node do not depend on it (only on the world node, the hollow and `shell_mm`).
    phi (at h / 2) comes with the wall's `chalcedony_level` unless `with_phi` is False."""
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
    shell = d_out <= shell_mm
    del d_out, d_in

    phi, near = None, np.zeros(0, np.int64)
    if with_phi and wall.chalcedony_level is not None:
        phi, near = _phi_grid(volume, wall.chalcedony_level, a, shape, h)
    in_band = np.zeros(n, bool)
    in_band[band] = True
    is_near = np.zeros(n, bool)
    is_near[near] = True
    chal = np.zeros(n, bool)
    todo = np.flatnonzero(in_band | is_near)  # one evaluate for both
    for s in range(0, len(todo), CHUNK):
        k = todo[s : s + CHUNK]
        p = _nodes(a, shape, k, h)
        ins = volume.solid.contains(p)
        f = volume.evaluate(p, ins)
        stage = (f.grain >= wall.grain_offset) & (f.grain < wall.grain_offset + len(wall.sites))
        g = ins & (((f.phase == Phase.QUARTZ) & stage) | (f.phase == Phase.VOID))
        b = in_band[k]
        gate[k[b]] = g[b]
        c = is_near[k]
        chal[k[c]] = (ins & (f.phase == Phase.CHALCEDONY))[c]
    return VoidBox(
        a=a,
        h=h,
        ids=ids.reshape(shape),
        gate=gate.reshape(shape),
        wall=wall,
        phi=phi,
        chal=None if phi is None else chal.reshape(shape),
        shell=shell.reshape(shape),
    )
