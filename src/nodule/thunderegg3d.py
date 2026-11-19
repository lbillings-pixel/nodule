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
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import cast

import numpy as np
from agatesim._scipy import distance_to_false, gaussian
from agatesim.volume.config import SampledLevel, SolidConfig
from agatesim.volume.sampled import CLAMP_CELLS, QUANT
from numpy.typing import NDArray

from nodule._scipy import largest_component_3d, nearest_within
from nodule.thunderegg import ARMS_BASE, EXTENT, ArmShape, arm_path

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
    level = np.where(f > 0, 1.0, -1.0) * d.reshape(f.shape)
    # agatesim counts pieces on the stored 8-bit level, where a node just inside
    # the wall rounds to 0: a bridge of such nodes splits the cavity. Keep the
    # largest piece as stored and turn the rest outside.
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


def sinuous_solid(
    rng: np.random.Generator, size_mm: float, fins: Fins = FINS
) -> tuple[ArmShape, SolidConfig]:  # fmt: skip
    """A winding 3D thunder-egg cavity, longest side `size_mm`, as agatesim's drawn
    3D solid. Redrawn from the same stream if agatesim refuses the drawing (pieces
    joined only through a too-thin neck, say). A fresh stream (nothing drawn from
    it yet) is remembered by its seed: a cavity takes a few seconds to draw, and
    sheets are drawn again and again."""
    seq = rng.bit_generator.seed_seq
    if isinstance(seq, np.random.SeedSequence):
        fresh = np.random.default_rng(seq).bit_generator.state == rng.bit_generator.state
        if fresh:
            key = (str(seq.entropy), tuple(seq.spawn_key), seq.pool_size)
            return _remembered(key, size_mm, fins)
    return _draw_solid(rng, size_mm, fins)


@functools.lru_cache(maxsize=64)
def _remembered(
    key: tuple[str, tuple[int, ...], int], size_mm: float, fins: Fins
) -> tuple[ArmShape, SolidConfig]:  # fmt: skip
    entropy, spawn_key, pool = key
    seq = np.random.SeedSequence(int(entropy), spawn_key=spawn_key, pool_size=pool)
    return _draw_solid(np.random.default_rng(seq), size_mm, fins)


def _draw_solid(
    rng: np.random.Generator, size_mm: float, fins: Fins
) -> tuple[ArmShape, SolidConfig]:  # fmt: skip
    shape = ArmShape.draw(rng)
    for _ in range(TRIES):
        level = level_cells(sinuous_field(rng, shape, fins))
        try:
            solid = SolidConfig(
                sampled=SampledLevel.of(level),
                host="volcanic",  # type: ignore[arg-type]
                size_mm=size_mm,
            )
        except ValueError:
            continue
        return shape, solid
    raise RuntimeError(f"no 3D thunder-egg cavity accepted in {TRIES} draws")
