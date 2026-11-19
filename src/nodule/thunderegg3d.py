"""Thunder-egg star cavities drawn in 3D (ADR 0018 s.19-20; agatesim ADR 0089).

The 2D star (`nodule.thunderegg`) extruded into a prism shows the same star on
every cut. Here each winding arm is a fin instead: a sheet with its own height,
centre height, lean and twist, after Ross's "triangular shear planes", which make
the arms sections of sheets [@ross1941, p. 729]. A cut sees only the fins it
crosses, so cuts gain and lose arms with height (the owner's request, 2026-11-08;
build chosen 2026-11-18). The arms in plan are the approved 2D ones
(`thunderegg.arm_path`, side buds included). The look is *plausible* (one private
specimen and the sources' descriptions); never evidence for how cavities form
(ADR 0002). Every size and share here is `effective`.

The star is a union of balls on a grid fixed in shape units (so it does not depend
on any preview or render resolution), blurred as in 2D to round the embayments,
and handed to agatesim as a level on a lattice: positive inside and close to the
signed distance, in cells, to the blurred wall (agatesim ADR 0089). Arrays here are
indexed [x, y, z]; the star lies in the x-y plane, the cut's plane (tilt 0).

A jagged core (ADR 0018 s.21; owner's choice 2026-11-18: zigzag walls) is the 2D
jagged method in 3D: a faceted core and kinked wedge arms, their walls flat runs
between points pushed in or out, built as a union of flat-sided (convex) pieces
after Ross's planar shear faces [@ross1941, p. 729] and the stepped walls of the
Saxon lithophysae [@goetze2023, Fig. 2; sections only, so the 3D form is an
assumption]. Its level is exact, not blurred: the largest, over pieces, of the
least distance to a piece's face planes.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from agatesim._scipy import distance_to_false, gaussian, label, linear_at
from agatesim.grid import FloatArray as AnyFloatArray
from agatesim.specimen.segment import trace_outline
from agatesim.volume.config import SampledLevel, SolidConfig
from agatesim.volume.sampled import CLAMP_CELLS, QUANT, Sampled
from numpy.typing import NDArray

from nodule._scipy import hull_planes, largest_component_3d, nearest_within
from nodule.thunderegg import ARMS_BASE, EXTENT, JAGGED_ARMS, ArmShape, arm_path

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

GRID3 = 192  # cells across the drawing box on each axis (shape units, as the 2D grid)
CELL3 = 2 * EXTENT / GRID3
_C = (np.arange(GRID3) + 0.5) * CELL3 - EXTENT  # cell centres
BLUR = 0.025  # shape units, as the 2D star
PAD = 6  # lattice nodes kept beyond the wall's box (agatesim needs 2 outside)
CLAMP = CLAMP_CELLS  # cells: the level agatesim stores is clamped to this
BAND = 5.0  # cells: nodes nearer a wall crossing than this get a measured distance
TRIES = 8  # redraws (same stream) if agatesim refuses a drawing


@dataclass(frozen=True)
class Fins:
    """How the arms rise out of the plan; all `effective`, shape units. Set so that
    cuts through the deepest point show arms about as many and as long as the
    approved 2D stars (2026-11-18, `runs/thunderegg-3d/arm_length.py`): arms
    crossing circles at 1.5 / 2 / 2.5 / 3 core radii, mean of 16 cavities 6.6 /
    5.7 / 2.6 / 0.9 against 6.5 / 5.4 / 2.8 / 0.6 for 40 2D stars. The probe's
    fins (centre and lean 0.2, tip 0.4) gave 6.1 / 4.4 / 2.1 / 0.6: stubby arms.
    Cuts 4 mm off the middle still change by about one arm, 8 mm off by about
    three (all 16 cavities change)."""

    half_height: tuple[float, float] = (0.12, 0.45)  # at the root
    tip_height: float = 0.7  # half-height at the tip, x the root's
    centre: float = 0.1  # fin centre height, drawn within +-this of the body's
    lean: float = 0.1  # rise per unit length along the arm, within +-this
    twist: float = 0.4  # sideways shift per unit height, within +-this
    body_flat: tuple[float, float] = (0.55, 0.9)  # body's height / width


FINS = Fins()


def _ball(m: BoolArray, x: float, y: float, z: float, r: float) -> None:
    def span(c: float) -> slice:
        lo = max(0, int((c - r + EXTENT) / CELL3) - 1)
        return slice(lo, min(GRID3, int((c + r + EXTENT) / CELL3) + 2))

    sx, sy, sz = span(x), span(y), span(z)
    dx = _C[sx][:, None, None] - x
    dy = _C[sy][None, :, None] - y
    dz = _C[sz][None, None, :] - z
    m[sx, sy, sz] |= dx * dx + dy * dy + dz * dz <= r * r


def _body(rng: np.random.Generator, radius: float, flat: float) -> BoolArray:
    """A lumpy flattened body (the 2D body's lumps, as a spheroid)."""
    x, y, z = np.meshgrid(_C, _C, _C, indexing="ij", sparse=True)
    th = np.arctan2(y, x)
    lump = sum(
        rng.uniform(0, 0.08) * np.cos(k * th + rng.uniform(0, 2 * math.pi)) for k in (2, 3, 4)
    )
    rr = radius * (1 + lump)
    return np.asarray((x * x + y * y) / (rr * rr) + z * z / (radius * flat) ** 2 <= 1)


def _sheet(
    m: BoolArray, pts: FloatArray, w: FloatArray, hz: FloatArray, z0: FloatArray, twist: float
) -> None:  # fmt: skip
    """Stack balls across a fin's height at every second centreline point: radius
    w at mid-height, 0.35 w at the edges (a lens-shaped cross-section)."""
    for i in range(0, len(pts), 2):
        t = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]
        nrm = np.array([-t[1], t[0]]) / max(float(np.hypot(t[0], t[1])), 1e-9)
        k = max(2, int(2 * hz[i] / (0.7 * w[i])))
        top = float(hz[i])
        for dz in (top * (2 * j / (k - 1) - 1) for j in range(k)):
            f = math.sqrt(max(0.0, 1 - (dz / hz[i]) ** 2))
            p = pts[i] + twist * dz * nrm
            _ball(m, float(p[0]), float(p[1]), float(z0[i] + dz), float(w[i]) * (0.35 + 0.65 * f))


def _fin(
    m: BoolArray, rng: np.random.Generator, angle: float, body: float, shape: ArmShape,
    fins: Fins,
) -> None:  # fmt: skip
    """One winding arm (the 2D arm and its side buds, in plan) raised into a fin."""
    reach = rng.uniform(0.55, 1.0)
    start = 0.6 * body
    half_width = shape.width * rng.uniform(0.08, 0.14)
    meander = shape.bend * rng.uniform(0.5, 1.0)
    pts, w = arm_path(rng, angle, start, reach - start, half_width, meander)
    s = np.arange(len(pts)) * 0.008
    root = rng.uniform(*fins.half_height)
    zc = rng.uniform(-fins.centre, fins.centre)
    lean = rng.uniform(-fins.lean, fins.lean)
    twist = rng.uniform(-fins.twist, fins.twist)
    hz = np.maximum(root * (1 - (1 - fins.tip_height) * s / s[-1]), w)
    z0 = np.asarray(zc + lean * s, dtype=np.float64)
    _sheet(m, pts, w, hz, z0, twist)
    for bud_rng in rng.spawn(int(rng.integers(0, 3))):  # side buds, as in 2D
        i = int(bud_rng.integers(len(pts) // 5, max(len(pts) // 5 + 1, 3 * len(pts) // 4)))
        d = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]
        side = math.atan2(float(d[1]), float(d[0]))
        turn = bud_rng.choice([-1, 1]) * math.radians(bud_rng.uniform(40, 80))
        bud_len = bud_rng.uniform(0.08, 0.22)
        t = np.linspace(0, bud_len, max(3, int(bud_len / 0.008)))
        bud = pts[i] + np.outer(t, [math.cos(side + turn), math.sin(side + turn)])
        bw = np.maximum(0.6 * w[i] * (1 - t / bud_len) ** 0.7, 0.012)
        bh = np.maximum(hz[i] * (1 - 0.6 * t / bud_len), bw)
        _sheet(m, bud, bw, bh, np.full(len(t), z0[i]), twist)


def sinuous_field(
    rng: np.random.Generator, shape: ArmShape, fins: Fins = FINS
) -> NDArray[np.float32]:  # fmt: skip
    """A winding 3D star on the drawing grid, [x, y, z]: the blurred union of body
    and fins, its largest piece only; the wall is where it crosses 0.5. Each arm
    draws from its own spawned stream."""
    body = rng.uniform(0.22, 0.32)
    m = _body(rng, body, rng.uniform(*fins.body_flat))
    arms = int(rng.integers(ARMS_BASE[0], ARMS_BASE[1] + 1)) + shape.extra_arms
    base = rng.uniform(0, 2 * math.pi) + np.arange(arms) * 2 * math.pi / arms
    base = base + rng.uniform(-0.25, 0.25, arms) * 2 * math.pi / arms
    for a, arm_rng in zip(base, rng.spawn(arms), strict=True):
        _fin(m, arm_rng, float(a), body, shape, fins)
    soft = gaussian(m.astype(np.float32), sigma=BLUR / CELL3).astype(np.float32)
    return soft * largest_component_3d(soft > 0.5)


def level_cells(field: NDArray[np.float32]) -> FloatArray:
    """The level agatesim stores for a field whose wall is its 0.5 crossing: the
    signed distance in cells (positive inside, clamped to CLAMP), on the drawing
    grid cropped to the wall's box plus PAD nodes. Distance at a node near the wall:
    to the tangent plane at the nearest wall crossing on a grid edge (the crossings
    are found by linear interpolation, so they fall anywhere along the edges)."""
    inside = field > 0.5
    idx = np.argwhere(inside)
    lo = np.maximum(idx.min(axis=0) - PAD, 0)
    hi = np.minimum(idx.max(axis=0) + PAD + 1, GRID3)
    f = field[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].astype(np.float64) - 0.5
    grad = np.stack(cast("list[FloatArray]", np.gradient(f)), axis=-1)
    pts: list[FloatArray] = []
    nrm: list[FloatArray] = []
    for ax in range(3):
        s0 = np.moveaxis(f, ax, 0)[:-1]
        s1 = np.moveaxis(f, ax, 0)[1:]
        g0 = np.moveaxis(grad, ax, 0)[:-1]
        g1 = np.moveaxis(grad, ax, 0)[1:]
        hit = np.nonzero((s0 > 0) != (s1 > 0))
        t = s0[hit] / (s0[hit] - s1[hit])
        p = np.stack(hit, axis=1).astype(np.float64)
        p[:, 0] += t
        order = [ax] + [i for i in range(3) if i != ax]  # moved axes back to x, y, z
        back = np.empty_like(p)
        back[:, order] = p
        pts.append(back)
        nrm.append(g0[hit] * (1 - t)[:, None] + g1[hit] * t[:, None])
    p_all = np.concatenate(pts)
    n_all = np.concatenate(nrm)
    n_len = np.linalg.norm(n_all, axis=1, keepdims=True)
    n_all = n_all / np.maximum(n_len, 1e-12)
    nodes = np.argwhere(np.ones(f.shape, dtype=bool)).astype(np.float64)
    dist, k = nearest_within(p_all, nodes, BAND)
    near = np.isfinite(dist)
    off = nodes[near] - p_all[k[near]]
    plane = np.abs(np.einsum("ij,ij->i", off, n_all[k[near]]))
    good = n_len[k[near], 0] > 1e-9
    d = np.full(len(nodes), CLAMP)
    d[near] = np.minimum(np.where(good, plane, dist[near]), CLAMP)
    return _one_stored_piece(np.where(f > 0, 1.0, -1.0) * d.reshape(f.shape))


def _one_stored_piece(level: FloatArray) -> FloatArray:
    """agatesim counts pieces on the stored 8-bit level, where a node just inside
    the wall rounds to 0: a bridge of such nodes splits the cavity. Keep the
    largest piece as stored and turn the rest outside."""
    stored_inside = np.rint(level / CLAMP_CELLS * QUANT) > 0
    stray = stored_inside & ~largest_component_3d(stored_inside)
    return np.where(stray, -np.abs(level), level)


def sections(field: NDArray[np.float32], heights: tuple[float, ...]) -> list[BoolArray]:
    """Horizontal cuts at `heights` (shape units) above the field's deepest point,
    on the 2D drawing grid ([y, x], 2x finer, as `thunderegg.GRID`)."""
    inside = field > 0.5
    kz = int(np.unravel_index(int(np.argmax(distance_to_false(inside, 1.0))), inside.shape)[2])
    out: list[BoolArray] = []
    for h in heights:
        k = int(np.clip(round(kz + h / CELL3), 0, GRID3 - 1))
        plane = np.asarray(field[:, :, k], dtype=np.float64).T
        fine = np.repeat(np.repeat(plane, 2, axis=0), 2, axis=1)  # nearest, then smoothed
        out.append(gaussian(fine, sigma=0.7) > 0.5)
    return out


def arms_crossing(mask: BoolArray, k: float = 2.0) -> int:
    """Arms of a cut crossing the circle about its core (deepest point) at k x the
    core's inscribed radius; 0 if the cut misses the cavity."""
    if not mask.any():
        return 0
    depth = distance_to_false(mask, 1.0)
    i0, j0 = np.unravel_index(int(np.argmax(depth)), mask.shape)
    ang = np.linspace(0, 2 * math.pi, 1440, endpoint=False)
    rad = k * float(depth.max())
    i = np.clip(np.rint(i0 + rad * np.sin(ang)).astype(int), 0, mask.shape[0] - 1)
    j = np.clip(np.rint(j0 + rad * np.cos(ang)).astype(int), 0, mask.shape[1] - 1)
    on = mask[i, j]
    if on.all():
        return 1
    return int(np.count_nonzero(np.diff(np.r_[on, on[:1]].astype(np.int8)) == 1))


@dataclass(frozen=True)
class Jagged:
    """A jagged core in 3D; all `effective`, shape units. Runs, pushes, core radii,
    arm reach, kink and widths as the approved 2D jagged cores (`thunderegg.
    jagged_mask`, `jagged_outline`). Heights after the winding fins (centre and lean
    within ±0.1, tip keeping 0.7 of the root's height). Measured 2026-11-18 on the
    stored solid's middle cut (20 cavities, `runs/thunderegg-3d/jagged_probe.py`):
    arms crossing circles at 1.5 / 2 / 2.5 / 3 core radii 5.0 / 3.4 / 1.2 / 0.1,
    against 4.75 / 3.0 / 0.9 / 0 for 40 approved 2D cores (a core 1.1x as large
    gave 4.2 / 2.1 / 0.3)."""

    step: tuple[float, float] = (0.06, 0.12)  # wall runs between pushed points
    amp: tuple[float, float] = (0.02, 0.04)  # push in or out, at most
    core_radius: tuple[float, float] = (0.32, 0.52)  # corners of the core's middle ring
    core_height: tuple[float, float] = (0.44, 0.72)  # core half-height, x mean radius
    root_height: tuple[float, float] = (0.15, 0.4)  # arm half-height at the centre
    tip_height: float = 0.7  # x the root's, at the tip
    centre: float = 0.1  # arm centre height within ±this
    lean: float = 0.1  # rise per unit length along the arm, within ±this
    overlap: float = 0.015  # pieces run on past their joints (~1 cell), so no seams


JAGGED = Jagged()
_UP = np.array([0.0, 0.0, 1.0])
_MARGIN = CLAMP + 2  # cells evaluated beyond a piece's box


class _Union:
    """A union of convex pieces as a level on the drawing grid (shape units,
    positive inside): each piece's level is the least distance to its face planes
    (exact inside; the box's planes keep it at least the distance to its box
    outside), and the union takes the largest. Slope at most 1, wall exact."""

    def __init__(self) -> None:
        self.level = np.full((GRID3, GRID3, GRID3), -_MARGIN * CELL3, dtype=np.float32)

    def add(self, pts: FloatArray) -> None:
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        box = np.c_[np.eye(3), -hi], np.c_[-np.eye(3), lo]
        planes = np.vstack([hull_planes(pts), *box])
        sl = [
            slice(
                max(0, int((a + EXTENT) / CELL3 - _MARGIN)),
                min(GRID3, int((b + EXTENT) / CELL3 + _MARGIN) + 2),
            )
            for a, b in zip(lo, hi, strict=True)
        ]
        x, y, z = np.meshgrid(_C[sl[0]], _C[sl[1]], _C[sl[2]], indexing="ij", sparse=True)
        val = np.full((len(_C[sl[0]]), len(_C[sl[1]]), len(_C[sl[2]])), np.inf, dtype=np.float32)
        for a, b, c, d in planes:
            np.minimum(val, -(a * x + b * y + c * z + d), out=val)
        sub = self.level[sl[0], sl[1], sl[2]]
        np.maximum(sub, val, out=sub)


def _ext(a: FloatArray, b: FloatArray, d: float) -> tuple[FloatArray, FloatArray]:
    """Points a, b moved apart along their line by the share d each."""
    return a - d * (b - a), b + d * (b - a)


def _jagged_core(u: _Union, rng: np.random.Generator, step: float, amp: float, jg: Jagged) -> None:
    """A faceted core: an irregular 5-8-cornered middle ring (as the 2D core) and
    smaller rings above and below. Its side walls are flat runs between points
    spaced about `step` round the core at a few heights, each pushed in or out by
    up to `amp` (less near the top and bottom, where the core narrows), drawn as
    sector pieces from the axis over the core shrunk so inward pushes show; top
    and bottom stay the hull's facets."""
    corners = int(rng.integers(5, 9))
    th = np.sort((np.arange(corners) + rng.uniform(-0.3, 0.3, corners)) * 2 * math.pi / corners)
    rad = rng.uniform(*jg.core_radius, corners)
    hc = float(rng.uniform(*jg.core_height) * rad.mean())
    ring = np.stack([rad * np.cos(th), rad * np.sin(th), rng.uniform(-0.15, 0.15, corners) * hc], 1)
    pts = [ring]
    for sign in (-1, 1):
        k = int(rng.integers(3, 6))
        tt = (
            rng.uniform(0, 2 * math.pi) + np.arange(k) * 2 * math.pi / k + rng.uniform(-0.3, 0.3, k)
        )
        rr = rad.mean() * rng.uniform(0.35, 0.7, k)
        zz = sign * hc * rng.uniform(0.75, 1.0, k)
        pts.append(np.stack([rr * np.cos(tt), rr * np.sin(tt), zz], 1))
    p = np.concatenate(pts)
    planes = hull_planes(p)
    shrink = 1 - (amp + 0.02) / float(rad.min())
    u.add(p * np.array([shrink, shrink, 1.0]))
    n_ang = max(10, int(2 * math.pi * float(rad.mean()) / step))
    phis = rng.uniform(0, 2 * math.pi) + np.arange(n_ang) * 2 * math.pi / n_ang
    zs = np.linspace(-0.7 * hc, 0.7 * hc, max(2, int(1.4 * hc / step) + 1))

    def wall(phi: float, z: float) -> FloatArray:
        out = np.array([math.cos(phi), math.sin(phi), 0.0])
        o = np.array([0.0, 0.0, z])
        den = planes[:, :3] @ out
        t = np.where(
            den > 1e-9, -(planes[:, :3] @ o + planes[:, 3]) / np.maximum(den, 1e-9), np.inf
        )
        f = int(np.argmin(t))
        local = min(1.0, float(t[f]) / float(rad.mean()))  # narrow near top and bottom
        return o + t[f] * out + rng.uniform(-amp, amp) * local * planes[f, :3]

    w = np.array([[wall(float(ph), float(z)) for z in zs] for ph in phis])  # [phi, z, 3]
    d = jg.overlap / step
    for j in range(n_ang):
        jn = (j + 1) % n_ang
        for k in range(len(zs) - 1):
            a0, a1 = _ext(w[j, k], w[jn, k], d)
            b0, b1 = _ext(w[j, k + 1], w[jn, k + 1], d)
            a0, b0 = _ext(a0, b0, d)
            a1, b1 = _ext(a1, b1, d)
            axis = [[0.0, 0.0, zs[k] - jg.overlap], [0.0, 0.0, zs[k + 1] + jg.overlap]]
            u.add(np.array([a0, a1, b0, b1, *axis]))


def _jagged_arm(
    u: _Union, rng: np.random.Generator, angle: float, step: float, amp: float, jg: Jagged
) -> None:  # fmt: skip
    """A 2D jagged wedge (broad at the centre, kinked, narrow at the tip) raised
    into a fin of six-sided section. At stations about `step` apart its two sides,
    top and bottom are each pushed in or out by up to `amp` (at most 0.4 of the
    local half-width or half-height); the walls between are flat."""
    reach = rng.uniform(0.55, 0.95)
    f = rng.uniform(0.4, 0.7)  # where the arm kinks
    turn = math.radians(rng.uniform(-35, 35))
    h0 = rng.uniform(0.15, 0.3)  # half-width at the centre
    h1 = h0 * rng.uniform(0.5, 0.85)  # at the kink
    h2 = rng.uniform(0.02, 0.05)  # at the tip
    v0 = rng.uniform(*jg.root_height)
    zc = rng.uniform(-jg.centre, jg.centre)
    lean = rng.uniform(-jg.lean, jg.lean)
    d0 = np.array([math.cos(angle), math.sin(angle), 0.0])
    d1 = np.array([math.cos(angle + turn), math.sin(angle + turn), 0.0])
    kink = reach * f
    ss = [0.0]
    while ss[-1] < reach - 1e-9:
        ss.append(min(reach, ss[-1] + step * rng.uniform(0.7, 1.3)))

    def section(s: float, last: bool) -> FloatArray:
        if s <= kink:
            c, d, hw = s * d0, d0, h0 + (h1 - h0) * s / kink
        else:
            c, d = kink * d0 + (s - kink) * d1, d1
            hw = h1 + (h2 - h1) * (s - kink) / (reach - kink)
        vh = max(v0 * (1 - (1 - jg.tip_height) * s / reach), hw)
        c = c + (zc + lean * s) * _UP
        side = np.cross(_UP, d)
        keep = 0.0 if last else 1.0  # the tip is not pushed
        pl, pr = rng.uniform(-1, 1, 2) * min(amp, 0.4 * hw) * keep
        pt, pb = rng.uniform(-1, 1, 2) * min(amp, 0.4 * vh) * keep
        return np.array([
            c + (hw + pr) * side, c - (hw + pl) * side,
            c + 0.4 * hw * side + (vh + pt) * _UP, c - 0.4 * hw * side + (vh + pt) * _UP,
            c + 0.4 * hw * side - (vh + pb) * _UP, c - 0.4 * hw * side - (vh + pb) * _UP,
        ])  # fmt: skip

    rings = [section(s, k == len(ss) - 1) for k, s in enumerate(ss)]
    for k in range(len(rings) - 1):
        r0, r1 = rings[k], rings[k + 1]
        d = jg.overlap / max(ss[k + 1] - ss[k], 1e-6)
        e0 = r0 - d * (r1 - r0) if k > 0 else r0
        last = k == len(rings) - 2
        e1 = r1 if last else r1 + d * (r1 - r0)
        pts = [*e0, *e1]
        if last:  # a point beyond the tip, as the 2D wedge
            pts.append(r1.mean(axis=0) + 0.5 * h2 * (d1 if reach > kink else d0))
        u.add(np.array(pts))


def jagged_union(rng: np.random.Generator, jg: Jagged = JAGGED) -> NDArray[np.float32]:
    """A jagged 3D core's level on the drawing grid (shape units, [x, y, z],
    positive inside): the core and 3-6 arms [@goetze2023, Fig. 2], each arm from
    its own spawned stream."""
    u = _Union()
    step = rng.uniform(*jg.step)
    amp = rng.uniform(*jg.amp)
    _jagged_core(u, rng, step, amp, jg)
    arms = int(rng.integers(JAGGED_ARMS[0], JAGGED_ARMS[1] + 1))
    base = rng.uniform(0, 2 * math.pi) + np.arange(arms) * 2 * math.pi / arms
    base = base + rng.uniform(-0.3, 0.3, arms) * 2 * math.pi / arms
    for a, arm_rng in zip(base, rng.spawn(arms), strict=True):
        _jagged_arm(u, arm_rng, float(a), step, amp, jg)
    return u.level


def union_cells(level: NDArray[np.float32]) -> FloatArray:
    """The level agatesim stores for a union level (shape units, wall at 0): in
    cells, clamped to CLAMP, on the grid cropped to the largest piece's box plus
    PAD nodes, with stray pieces turned outside."""
    inside = largest_component_3d(level > 0)
    idx = np.argwhere(inside)
    lo = np.maximum(idx.min(axis=0) - PAD, 0)
    hi = np.minimum(idx.max(axis=0) + PAD + 1, GRID3)
    sub = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
    cells = np.clip(level[sub].astype(np.float64) / CELL3, -CLAMP, CLAMP)
    cells = np.where((cells > 0) & ~inside[sub], -cells, cells)
    return _one_stored_piece(cells)


CORNER_HEIGHTS_MM = (-4.0, 0.0, 4.0)  # cuts pooled by `cut_corners`, about the deepest point
CORNER_RADIUS = 3.0  # lattice cells: turns tighter than this count as corners
# `cut_corners` above this reads as jagged; set as a group, not stone by stone.
# 2026-11-18, `runs/thunderegg-3d/corner_check.py`: seeds 2117 (20 + 20) jagged
# 0.446-0.607, winding 0.202-0.418 (threshold between); fresh seeds 2118 jagged
# 0.394-0.658 (median 0.53, 19 of 20 above), winding 0.280-0.405 (all below).
CORNER_MIN = 0.43


def cut_corners(solid: Sampled, deep_z_mm: float) -> float:
    """How cornered a drawn cavity's walls are: the share of the wall's turning
    that happens at turns tighter than CORNER_RADIUS, over horizontal cuts at
    CORNER_HEIGHTS_MM about the deepest point (as `thunderegg.wall_corners` for
    2D outlines). The wall is ordered by tracing each cut's largest piece on a
    grid of half lattice cells, but its direction is the stored level's gradient,
    so the grid's staircase does not count as turning. Turning is summed over
    three traced points at a time."""
    lo, hi = (np.asarray(b, dtype=np.float64) for b in solid.bounds())
    h = solid.cell_mm / 2
    xs = np.arange(lo[0] - 4 * h, hi[0] + 4 * h, h)
    ys = np.arange(lo[1] - 4 * h, hi[1] + 4 * h, h)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    steps: list[FloatArray] = []
    turns: list[FloatArray] = []
    for dz in CORNER_HEIGHTS_MM:
        pts = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, deep_z_mm + dz)], axis=1)
        lev = solid.level(pts).reshape(xx.shape)
        found = label(lev > 0, eight_connected=False)
        sizes = np.bincount(found.ravel())
        sizes[0] = 0
        if sizes.max() < 16:
            continue
        xy = trace_outline(found == int(np.argmax(sizes))).astype(np.float64)
        gy, gx = (np.asarray(g, dtype=np.float64) for g in np.gradient(lev))
        at: list[AnyFloatArray] = [xy[:, 1], xy[:, 0]]
        ang = np.unwrap(np.arctan2(linear_at(gy, at), linear_at(gx, at)))
        closing = ang[0] + 2 * math.pi * round(float(ang[-1] - ang[0]) / (2 * math.pi))
        turns.append(np.abs(np.diff(np.r_[ang, closing])))
        steps.append(np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1))
    if not turns:
        return 0.0
    win = 3
    sharp_pts: list[BoolArray] = []
    for ds, dth in zip(steps, turns, strict=True):
        k = np.ones(win)
        t = np.convolve(np.r_[dth, dth[: win - 1]], k, "valid")[: len(dth)]
        s = np.convolve(np.r_[ds, ds[: win - 1]], k, "valid")[: len(ds)]
        sharp = t > s / (2 * CORNER_RADIUS)  # radius in grid (half lattice) cells
        held = np.zeros(len(dth), dtype=bool)
        for j in range(win):
            held |= np.roll(sharp, j)
        sharp_pts.append(held)
    every = np.concatenate(turns)
    return float(every[np.concatenate(sharp_pts)].sum() / every.sum())


def sinuous_solid(
    rng: np.random.Generator, size_mm: float, fins: Fins = FINS
) -> tuple[ArmShape, SolidConfig]:  # fmt: skip
    """A winding 3D thunder-egg cavity, longest side `size_mm`, as agatesim's drawn
    3D solid. Redrawn from the same stream if agatesim refuses the drawing (pieces
    joined only through a too-thin neck, say). A fresh stream (nothing drawn from
    it yet) is remembered by its seed: a cavity takes a few seconds to draw, and
    sheets are drawn again and again."""
    key = _fresh_key(rng)
    if key is not None:
        return _remembered_sinuous(key, size_mm, fins)
    return _draw_sinuous(rng, size_mm, fins)


def jagged_solid(rng: np.random.Generator, size_mm: float, jg: Jagged = JAGGED) -> SolidConfig:
    """A jagged 3D thunder-egg core, longest side `size_mm`, as agatesim's drawn
    3D solid; redrawn and remembered as `sinuous_solid`."""
    key = _fresh_key(rng)
    if key is not None:
        return _remembered_jagged(key, size_mm, jg)
    return _accepted(lambda: union_cells(jagged_union(rng, jg)), size_mm)


_Key = tuple[int | tuple[int, ...], tuple[int, ...], int]


def _fresh_key(rng: np.random.Generator) -> _Key | None:
    """The seed of a stream nothing has been drawn from yet, else None."""
    seq = rng.bit_generator.seed_seq
    if isinstance(seq, np.random.SeedSequence):
        fresh = np.random.default_rng(seq).bit_generator.state == rng.bit_generator.state
        if fresh and seq.entropy is not None:
            ent = seq.entropy
            entropy = int(ent) if isinstance(ent, int) else tuple(int(e) for e in ent)
            return (entropy, tuple(seq.spawn_key), seq.pool_size)
    return None


def _stream(key: _Key) -> np.random.Generator:
    entropy, spawn_key, pool = key
    return np.random.default_rng(
        np.random.SeedSequence(entropy, spawn_key=spawn_key, pool_size=pool)
    )


@functools.lru_cache(maxsize=64)
def _remembered_sinuous(key: _Key, size_mm: float, fins: Fins) -> tuple[ArmShape, SolidConfig]:
    return _draw_sinuous(_stream(key), size_mm, fins)


@functools.lru_cache(maxsize=64)
def _remembered_jagged(key: _Key, size_mm: float, jg: Jagged) -> SolidConfig:
    rng = _stream(key)
    return _accepted(lambda: union_cells(jagged_union(rng, jg)), size_mm)


def _draw_sinuous(
    rng: np.random.Generator, size_mm: float, fins: Fins
) -> tuple[ArmShape, SolidConfig]:  # fmt: skip
    shape = ArmShape.draw(rng)
    return shape, _accepted(lambda: level_cells(sinuous_field(rng, shape, fins)), size_mm)


def _accepted(draw: Callable[[], FloatArray], size_mm: float) -> SolidConfig:
    """agatesim's solid for the first of up to TRIES drawings it accepts."""
    for _ in range(TRIES):
        try:
            return SolidConfig(
                sampled=SampledLevel.of(draw()),
                host="volcanic",  # type: ignore[arg-type]
                size_mm=size_mm,
            )
        except ValueError:
            continue
    raise RuntimeError(f"no 3D thunder-egg cavity accepted in {TRIES} draws")
