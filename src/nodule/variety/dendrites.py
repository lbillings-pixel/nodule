"""Growing a stone's Mn dendrites (`nodule.dendrites`; work plan item 14; ADR 0018 s.17).

Each dendrite is grown by agatesim's 3D diffusion-limited aggregation
(`agatesim.volume.dendrites.grow`, after [@hou2023]) on a lattice fixed in a local frame
of world coordinates, from its own spawned stream, so the same dendrite appears in any
cut at any spacing (CLAUDE.md). Moved from the local probe `runs/dendrite-probe/`
(`grow.py`, `thin.py`), whose look sheets the owner judged. Two carriers:

- "wall": into the chalcedony from a patch of the wall, grown once at full reach, then
  only the largest eighth of its trees kept (the owner's pick, 2026-11-17 night: sparse;
  dropping a tree drops its seed cells too, so the seed patch does not show);
- "crack": on a crack disc, from where it meets the wall, inward along it; a lattice
  one cell thick that follows the crack's bowl. Not thinned (shown unthinned to the
  owner).

Growth is lazy: a dendrite is grown, whole, the first time points come near its box,
and kept; the result never depends on which points asked first. Sites are steered onto
the cut (2026-11-18), so a dendrite stone's cut grows all of them; a stone without
dendrites pays nothing. Costs (probe, measured): a wall dendrite 16-33 s and about
1.2 GB host RAM; a crack up to about 12 s.

A crack's film is shown `CrackGrowth.seen_mm` thick across the crack (`Grown.across`),
its pattern within the crack unchanged.

All sizes `effective` (no source gives them). Couplings, each an assumption: dendrites
keep out of the last chalcedony and the quartz (CAVITY_DEPTH); they lie over whatever
the bands and later stains left (drawn last). Appearance only (ADR 0002).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from agatesim.grid import FloatArray
from agatesim.volume.dendrites import Aggregate3, Lattice3, grow
from agatesim.volume.slicing import Plane
from agatesim.volume.solids import Solid
from numpy.typing import NDArray

from nodule.dendrites import Dendrites
from nodule.rind import wall_distance
from nodule.variety.recipe import FractureLayer, rim_point

CAVITY_DEPTH = 0.9  # depth (0 wall, 1 last chalcedony) a dendrite stops short of
MN_IMPURITY = 0.9  # the preview's Mn black-brown, as on the probe's sheets
MIN_TREE = 5  # cells: a tree that grew at all
CHUNK = 1_000_000  # lattice cells read per pass (memory)
NO_GRAVITY = np.array([0.0, 0.0, -1.0])  # the grower's bias axis; unused at upward 0


@dataclass(frozen=True)
class WallGrowth:
    """`grow_3d` of the probe at its defaults (the look the owner picked from)."""

    reach_mm: float = 5.0
    patch_mm: float = 0.8
    h_mm: float = 0.06
    amount: float = 0.06
    density: float = 0.04
    sticking: float = 0.5


@dataclass(frozen=True)
class CrackGrowth:
    """`grow_crack` of the probe at its defaults; `seen_mm` the thickness the film
    shows across the crack (`effective`, the owner's "darker", 2026-11-18): grown one
    cell (0.08 mm) thick, its trace on a cut covered under half of each 0.1 mm render
    pixel (84-1: mean 0.39) and rendered faint grey."""

    reach_mm: float = 6.0
    h_mm: float = 0.05
    amount: float = 0.25
    density: float = 0.04
    sticking: float = 0.5
    seen_mm: float = 0.2


WALL, CRACK = WallGrowth(), CrackGrowth()


@dataclass(frozen=True)
class Frame:
    """A local frame: local = (world - origin) @ axes.T (rows e1, e2, e3), with the
    third coordinate measured from a bowl rising by `sagitta` at radius `radius` (a
    crack's bend, as `FractureLayer.distance`); sagitta 0 is flat."""

    origin: FloatArray
    axes: FloatArray
    sagitta: float = 0.0
    radius: float = 1.0

    def to_local(self, p: FloatArray) -> FloatArray:
        q = (p - self.origin) @ self.axes.T
        if self.sagitta:
            rho = np.hypot(q[:, 0], q[:, 1])
            q[:, 2] -= self.sagitta * (np.minimum(rho, self.radius) / self.radius) ** 2
        return q

    def to_world(self, q: FloatArray) -> FloatArray:
        q = q.copy()
        if self.sagitta:
            rho = np.hypot(q[:, 0], q[:, 1])
            q[:, 2] += self.sagitta * (np.minimum(rho, self.radius) / self.radius) ** 2
        return self.origin + q @ self.axes


@dataclass(frozen=True)
class Grown:
    """A grown dendrite. `across` (<= 1) scales the local third coordinate before the
    distance, so a film on a crack shows 1/across times thicker across it and alike
    within its plane (the face-on pattern stands)."""

    frame: Frame
    agg: Aggregate3
    across: float = 1.0

    def distance(self, points: FloatArray) -> FloatArray:
        """Distance (mm) to the dendrite's surface (negative inside); inf away from
        its lattice box."""
        out = np.full(len(points), np.inf)
        if not len(self.agg.cells):
            return out
        lat = self.agg.lattice
        q = self.frame.to_local(points)
        q[:, 2] *= self.across
        lo, hi = lat.origin - lat.h, lat.origin + (np.asarray(lat.shape) + 1) * lat.h
        idx = np.flatnonzero(np.all((q > lo) & (q < hi), axis=1))
        if len(idx):
            out[idx] = self.agg.distance(q[idx])
        return out


@dataclass(frozen=True)
class Site:
    """One dendrite to grow: its world box (anything it can reach lies inside) and how."""

    lo: FloatArray
    hi: FloatArray
    grow: Callable[[], Grown]

    def near(self, points: FloatArray, margin: float) -> bool:
        return bool(np.any(np.all((points > self.lo - margin) & (points < self.hi + margin), 1)))


def unit(v: FloatArray) -> FloatArray:
    return v / float(np.linalg.norm(v))


def chalcedony(
    solid: Solid,
    depth: Callable[[FloatArray], FloatArray],
    pts: FloatArray,
    reach: float,
) -> tuple[NDArray[np.bool_], FloatArray]:
    """Points inside the stone within `reach` mm of the wall, in chalcedony short of
    CAVITY_DEPTH; and their distance (mm) in from the wall."""
    ok = np.zeros(len(pts), bool)
    inward = np.full(len(pts), np.inf)
    for a in range(0, len(pts), CHUNK):
        p = pts[a : a + CHUNK]
        d, _ = wall_distance(solid, p, reach + 1.0)
        inn = -d
        o = (inn > 0) & (inn < reach)
        idx = np.flatnonzero(o)
        if len(idx):
            o[idx[depth(p[idx]) >= CAVITY_DEPTH]] = False
        ok[a : a + CHUNK] = o
        inward[a : a + CHUNK] = inn
    return ok, inward


def _aggregate(
    lat: Lattice3, free: NDArray[np.bool_], seeds: NDArray[np.int64], g: WallGrowth | CrackGrowth,
    rng: np.random.Generator,
) -> Aggregate3:  # fmt: skip
    return grow(lat, free, seeds, density=g.density, amount=g.amount, sticking=g.sticking,
                upward=0.0, gravity=NO_GRAVITY, rng=rng)  # fmt: skip


def largest_trees(agg: Aggregate3, share: float) -> Aggregate3:
    """The aggregate with only the largest `share` of its trees (at least one) of at
    least MIN_TREE cells; agatesim traces which seed each cell grew from."""
    if agg.plume is None or not len(agg.cells):
        return agg
    sizes = np.bincount(agg.plume)
    grown = np.flatnonzero(sizes >= MIN_TREE)
    if not len(grown):
        return Aggregate3(agg.lattice, agg.cells[:0], agg.order[:0], agg.plume[:0],
                          None if agg.along is None else agg.along[:0])  # fmt: skip
    ranked = grown[np.argsort(sizes[grown], kind="stable")[::-1]]
    keep = np.isin(agg.plume, ranked[: max(1, round(len(ranked) * share))])
    return Aggregate3(agg.lattice, agg.cells[keep], agg.order[keep], agg.plume[keep],
                      None if agg.along is None else agg.along[keep])  # fmt: skip


def wall_site(
    solid: Solid,
    depth: Callable[[FloatArray], FloatArray],
    direction: FloatArray,
    rng: np.random.Generator,
    share: float,
    g: WallGrowth,
) -> Site:
    """Into the chalcedony from the wall patch where the ray from the solid's centre
    in `direction` leaves the stone; world axes (a world-fixed lattice)."""
    site = np.asarray(solid.centre(), float) + rim_point(solid, unit(np.asarray(direction)))
    frame = Frame(site, np.eye(3))
    span = g.reach_mm + 2 * g.h_mm

    def build() -> Grown:
        lat = Lattice3.around(np.full(3, -g.reach_mm), np.full(3, g.reach_mm), g.h_mm)
        q = lat.centres()
        ok, inward = chalcedony(solid, depth, frame.to_world(q), g.reach_mm)
        near = ok & (inward < 1.5 * lat.h) & (np.linalg.norm(q, axis=1) < g.patch_mm)
        agg = _aggregate(lat, ok, np.flatnonzero(near), g, rng)
        return Grown(frame, largest_trees(agg, share))

    return Site(site - span, site + span, build)


def meets_wall(solid: Solid, crack: FractureLayer) -> bool:
    """Whether the crack's disc reaches the stone's wall (some of its edge lies
    outside the stone), so dendrites can start on it."""
    n = unit(np.asarray(crack.normal, float))
    e1 = unit(np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0]))
    e2 = np.cross(n, e1)
    phi = np.linspace(0.0, 2 * np.pi, 72, endpoint=False)[:, None]
    edge = crack.radius_mm * (np.cos(phi) * e1 + np.sin(phi) * e2) + crack.sagitta_mm * n
    pts = np.asarray(solid.centre(), float) + np.asarray(crack.centre_mm) + edge
    return not bool(solid.contains(pts).all())


def crack_frame(solid: Solid, crack: FractureLayer) -> Frame:
    """The crack's local frame (bowl-following; e1 points away from the solid's centre)."""
    c0 = np.asarray(solid.centre(), float)
    centre = c0 + np.asarray(crack.centre_mm)
    n = unit(np.asarray(crack.normal, float))
    out_dir = centre - c0 if np.linalg.norm(centre - c0) > 1e-9 else np.array([1.0, 0.0, 0.0])
    e1 = out_dir - (out_dir @ n) * n
    e1 = unit(e1 if np.linalg.norm(e1) > 1e-9 else np.cross(n, [0.0, 0.0, 1.0]))
    return Frame(centre, np.stack([e1, np.cross(n, e1), n]), crack.sagitta_mm, crack.radius_mm)


ON_CUT_MM = 0.1  # a crack point this near the cut's plane lies on the cut
WALL_END_MM = 1.0  # the cut crosses a crack within this of where it meets the wall


def meets_wall_on_cut(solid: Solid, crack: FractureLayer, plane: Plane) -> bool:
    """Whether the cut crosses the crack where it meets the wall: crack points on the
    cut's plane lie both outside the stone and inside within WALL_END_MM of the wall,
    so dendrites grown inward from that edge show on the cut. Sampled every ~0.1 mm."""
    r = crack.radius_mm
    rho = np.linspace(0.0, r, max(int(r / 0.1), 2))
    phi = np.linspace(0.0, 2 * np.pi, max(int(2 * np.pi * r / 0.1), 8), endpoint=False)
    rr, pp = (a.ravel() for a in np.meshgrid(rho, phi))
    q = np.stack([rr * np.cos(pp), rr * np.sin(pp), np.zeros_like(rr)], axis=1)
    pts = crack_frame(solid, crack).to_world(q)
    on = np.abs((pts - plane.origin) @ np.asarray(plane.normal)) < ON_CUT_MM
    pts = pts[on]
    if not len(pts):
        return False
    d, _ = wall_distance(solid, pts, WALL_END_MM + 1.0)
    return bool(np.any(d > 0)) and bool(np.any((d < 0) & (d > -WALL_END_MM)))


def cut_site(solid: Solid, plane: Plane, v: FloatArray) -> tuple[float, float, float] | None:
    """A wall site on the cut (the owner, 2026-11-18): the unit direction from the
    solid's centre to where the ray in the cut's plane, from its origin along v (put
    into the plane), leaves the stone. None where the plane's origin lies outside the
    stone, v is nearly normal to the plane, or the ray from the centre along that
    direction leaves elsewhere (`rim_point` lands over SITE_TOL_MM away: another lobe)."""
    m = np.asarray(plane.normal, float)
    o = np.asarray(plane.origin, float)
    u = v - (v @ m) * m
    if np.linalg.norm(u) < 1e-3 or not solid.contains(o[None, :])[0]:
        return None
    u = unit(u)
    lo, hi = 0.0, float(np.linalg.norm(np.asarray(solid.bounds()[1]) - solid.bounds()[0]))
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if solid.contains((o + mid * u)[None, :])[0]:
            lo = mid
        else:
            hi = mid
    c = np.asarray(solid.centre(), float)
    d = unit(o + lo * u - c)
    if np.linalg.norm(c + rim_point(solid, d) - (o + lo * u)) > SITE_TOL_MM:
        return None
    x, y, z = d.tolist()
    return (x, y, z)


SITE_TOL_MM = 0.5


def crack_site(
    solid: Solid,
    depth: Callable[[FloatArray], FloatArray],
    crack: FractureLayer,
    rng: np.random.Generator,
    g: CrackGrowth,
) -> Site:
    """On the crack, from the whole edge where it meets the wall, inward along it."""
    frame = crack_frame(solid, crack)
    centre, n = frame.origin, frame.axes[2]
    reach = (np.sqrt(np.clip(1 - n**2, 0, 1)) * crack.radius_mm + crack.sagitta_mm
             + 2 * g.h_mm + g.seen_mm)  # fmt: skip

    def build() -> Grown:
        rad = crack.radius_mm + g.h_mm
        lat = Lattice3.around(np.array([-rad, -rad, -g.h_mm]), np.array([rad, rad, g.h_mm]),
                              g.h_mm)  # fmt: skip
        q = lat.centres()
        on = (np.abs(q[:, 2]) < 0.75 * lat.h) & (np.hypot(q[:, 0], q[:, 1]) <= crack.radius_mm)
        ok = np.zeros(len(q), bool)
        inward = np.full(len(q), np.inf)
        idx = np.flatnonzero(on)
        ok[idx], inward[idx] = chalcedony(solid, depth, frame.to_world(q[idx]), g.reach_mm)
        near = ok & (inward < 1.5 * lat.h)
        agg = _aggregate(lat, ok, np.flatnonzero(near), g, rng)
        if not len(agg.cells):
            return Grown(frame, agg)
        half = float(np.abs(agg.centres()[:, 2]).max()) + agg.radius  # the film's half-thickness
        return Grown(frame, agg, 2 * half / g.seen_mm if g.seen_mm > 2 * half else 1.0)

    return Site(centre - reach, centre + reach, build)


@dataclass(eq=False)
class DendriteLayer:
    """A stone's dendrites, grown when first needed (module docstring)."""

    sites: list[Site]
    _grown: dict[int, Grown] = field(default_factory=dict[int, Grown])

    @classmethod
    def of(
        cls,
        d: Dendrites,
        fractures: tuple[FractureLayer, ...],
        solid: Solid,
        depth: Callable[[FloatArray], FloatArray],
        wall: WallGrowth = WALL,
        crack: CrackGrowth = CRACK,
    ) -> DendriteLayer:
        if d.carrier == "wall":
            seqs = np.random.SeedSequence(d.key).spawn(len(d.sites))
            sites = [wall_site(solid, depth, np.asarray(s), np.random.default_rng(q),
                               d.keep_share, wall)
                     for s, q in zip(d.sites, seqs, strict=True)]  # fmt: skip
        else:
            seqs = np.random.SeedSequence(d.key).spawn(len(d.cracks))
            sites = [crack_site(solid, depth, fractures[i], np.random.default_rng(q), crack)
                     for i, q in zip(d.cracks, seqs, strict=True)]  # fmt: skip
        return cls(sites)

    def grown_sites(self) -> list[int]:
        """The sites grown so far."""
        return sorted(self._grown)

    def grown(self, k: int) -> Grown:
        if k not in self._grown:
            self._grown[k] = self.sites[k].grow()
        return self._grown[k]

    def distance(self, points: FloatArray, margin: float = 0.0) -> FloatArray:
        """Distance (mm) to the nearest dendrite; inf far from all of them. Grows
        those whose box comes within `margin` of the points."""
        out = np.full(len(points), np.inf)
        for k, s in enumerate(self.sites):
            if s.near(points, margin + 0.1):
                out = np.minimum(out, self.grown(k).distance(points))
        return out

    def coverage(self, points: FloatArray, pixel_mm: float = 0.0) -> FloatArray:
        """Share of each pixel covered (a half-space box filter on the distance); a
        point sample (pixel 0) is covered or not."""
        d = self.distance(points, pixel_mm)
        if pixel_mm <= 0:
            return (d <= 0).astype(np.float64)
        return np.clip(0.5 - d / pixel_mm, 0.0, 1.0)
