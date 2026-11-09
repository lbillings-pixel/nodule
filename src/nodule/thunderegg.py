"""Thunder-egg star cavities, drawn as 2D outlines for agatesim (ADR 0018 s.10).

Three styles exist because the sources disagree on how arms look. The owner
chose (2026-11-08) sinuous on most stones and jagged on about 1 in 3; angular
stays as the negative control of the sinuous check and is never drawn on a sheet.

- *angular*: straight-sided arms, as sections through Ross's rupture model, a
  pyritohedron of inward pentagonal pyramids made by shear planes [@ross1941,
  p. 729]. This was the first drawing (ADR 0018 s.8).
- *sinuous*: lobed, winding arms with rounded embayments, "like fluid that
  infiltrated an irregular void", from the owner's private reference
  (thunderegg-1, agatesim `specimens/observations.md`; one specimen, so the look
  is *plausible*, never evidence for a mechanism; ADR 0002).
- *jagged*: irregular, asymmetric cores of 3-6 broad arms with kinks and stepped,
  zigzag walls, as in five Saxon lithophysae [@goetze2023, Fig. 2, p. 26]; also
  *plausible* as a look only.

A sinuous star is a union of discs along winding arm centrelines around a lumpy
body, rasterised on a grid fixed in shape units (so it does not depend on any
render or preview resolution), blurred to round the embayments, and traced. A
jagged star is a union of kinked wedges around an irregular polygon, traced, then
cut into short straight runs whose corners are pushed in or out. Every size and
share here is `effective`.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from agatesim._scipy import distance_to_false, fill_holes, gaussian, label
from agatesim.geometry import outline_problem
from agatesim.specimen.segment import trace_outline
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

ArmStyle = Literal["angular", "sinuous", "jagged"]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

GRID = 384  # raster cells across the drawing box (shape units, not pixels of a cut)
EXTENT = 1.3  # half-width of the drawing box; arms reach at most about 1.0
OUTLINE_POINTS = 480  # resampled outline length (agatesim allows up to 2000)
# `arm_meander` above this reads as sinuous. Over 80 draws of each style: angular
# 0.008-0.031, sinuous 0.046-0.165 (2026-11-08).
SINUOUS_MIN = 0.04
# Per-stone spread of a sinuous star (owner, 2026-11-08: moderately more and less
# bending and thickness; arms from one fewer to two more). `effective`.
BEND_RANGE = (0.7, 1.4)  # multiplies each arm's meander, drawn log-uniform
WIDTH_RANGE = (0.75, 1.3)  # multiplies each arm's half-width, drawn log-uniform
ARMS_BASE = (5, 8)  # arms before the spread [@ross1941, p. 727: five-pointed stars]
EXTRA_ARMS_RANGE = (-1, 2)  # added to the base count: 4-10 arms, the ends rare
JAGGED_SHARE = 1 / 3  # owner, 2026-11-08 (Claude's suggested share); `effective`
JAGGED_ARMS = (3, 6)  # arms of a jagged core [@goetze2023, Fig. 2]
# `wall_corners` above this reads as jagged. Over 40 draws each (2026-11-08):
# jagged 0.70-0.87, sinuous 0.07-0.36, angular 0.40-0.62.
JAGGED_MIN = 0.66


class ArmShape(BaseModel):
    """How one sinuous star departs from the typical one; all `effective`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bend: float = Field(default=1.0, gt=0, description="Meander, x the typical.")
    width: float = Field(default=1.0, gt=0, description="Arm thickness, x the typical.")
    extra_arms: int = Field(default=0, description="Arms added to the base count.")

    @classmethod
    def draw(cls, rng: np.random.Generator) -> ArmShape:
        def log_uniform(lo: float, hi: float) -> float:
            return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))

        return cls(
            bend=log_uniform(*BEND_RANGE),
            width=log_uniform(*WIDTH_RANGE),
            extra_arms=int(rng.integers(EXTRA_ARMS_RANGE[0], EXTRA_ARMS_RANGE[1] + 1)),
        )

    def summary(self) -> str:
        return f"bend {self.bend:.2f}, width {self.width:.2f}, arms {self.extra_arms:+d}"


def star_outline(rng: np.random.Generator, arms: int) -> list[tuple[float, float]]:
    """The angular star [@ross1941; @pabian1994, p. 16]: arms of uneven length
    with straight sides (the first drawing, kept as an option)."""
    lengths = rng.uniform(0.6, 1.0, arms)
    phase = rng.uniform(0, 2 * math.pi)
    theta = np.linspace(0, 2 * math.pi, 24 * arms, endpoint=False)
    k = np.floor((theta - phase) % (2 * math.pi) / (2 * math.pi) * arms).astype(int) % arms
    frac = ((theta - phase) % (2 * math.pi / arms)) / (2 * math.pi / arms)
    peak = (0.5 + 0.5 * np.cos(2 * math.pi * (frac - 0.5))) ** 3  # 1 at an arm's tip
    r = 0.32 + (lengths[k] - 0.32) * peak
    return [(float(a), float(b)) for a, b in zip(r * np.cos(theta), r * np.sin(theta), strict=True)]


def _grid() -> tuple[FloatArray, FloatArray]:
    c = (np.arange(GRID) + 0.5) / GRID * 2 * EXTENT - EXTENT
    y, x = np.meshgrid(c, c, indexing="ij")
    return x, y


def _discs(mask: BoolArray, x: FloatArray, y: FloatArray, pts: FloatArray, w: FloatArray) -> None:
    """Add discs (centres `pts`, radii `w`) to `mask`, each only in its bounding box."""
    cell = 2 * EXTENT / GRID
    for (px, py), r in zip(pts, w, strict=True):
        i0 = max(0, int((py - r + EXTENT) / cell) - 1)
        i1 = min(GRID, int((py + r + EXTENT) / cell) + 2)
        j0 = max(0, int((px - r + EXTENT) / cell) - 1)
        j1 = min(GRID, int((px + r + EXTENT) / cell) + 2)
        sub = (x[i0:i1, j0:j1] - px) ** 2 + (y[i0:i1, j0:j1] - py) ** 2 <= r * r
        mask[i0:i1, j0:j1] |= sub


def _arm(
    rng: np.random.Generator, angle: float, start: float, length: float, half_width: float,
    meander: float,
) -> tuple[FloatArray, FloatArray]:  # fmt: skip
    """Centreline points and half-widths of one winding arm. Its heading departs
    from the outward radial by a sum of two sines (`meander` radians at most,
    capped at 70 degrees so an arm never turns back toward the centre); its width
    tapers to the tip and swells in places (the lobes)."""
    ds = 0.008
    n = max(4, int(length / ds))
    s = np.arange(n) * ds
    wavelengths = rng.uniform(0.25, 0.6, 2)
    phases = rng.uniform(0, 2 * math.pi, 2)
    weights = rng.dirichlet([2.0, 2.0])
    dev = meander * sum(
        wt * np.sin(2 * math.pi * s / lam + ph)
        for wt, lam, ph in zip(weights, wavelengths, phases, strict=True)
    )
    dev = np.clip(dev, -math.radians(70), math.radians(70))
    pts = np.empty((n, 2))
    pos = np.array([start * math.cos(angle), start * math.sin(angle)])
    for i in range(n):
        pts[i] = pos
        radial = math.atan2(pos[1], pos[0])
        heading = radial + float(dev[i])
        pos = pos + ds * np.array([math.cos(heading), math.sin(heading)])
    taper = (1 - s / length) ** 0.7
    swell = 1 + rng.uniform(0.2, 0.45) * np.sin(
        2 * math.pi * s / rng.uniform(0.15, 0.35) + rng.uniform(0, 2 * math.pi)
    )
    w = np.maximum(half_width * taper * swell, 0.012)
    return pts, w


def sinuous_mask(rng: np.random.Generator, arms: int, shape: ArmShape | None = None) -> BoolArray:
    """A sinuous star rasterised on the fixed drawing grid. Each arm draws from
    its own spawned stream, so a change in one arm's side lobes leaves the other
    arms as they were."""
    shape = shape or ArmShape()
    x, y = _grid()
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    body = rng.uniform(0.22, 0.32)
    lump = sum(
        rng.uniform(0, 0.08) * np.cos(k * th + rng.uniform(0, 2 * math.pi)) for k in (2, 3, 4)
    )
    mask = r <= body * (1 + lump)
    base = rng.uniform(0, 2 * math.pi) + np.arange(arms) * 2 * math.pi / arms
    base = base + rng.uniform(-0.25, 0.25, arms) * 2 * math.pi / arms
    for a, arm_rng in zip(base, rng.spawn(arms), strict=True):
        reach = arm_rng.uniform(0.55, 1.0)
        start = 0.6 * body
        half_width = shape.width * arm_rng.uniform(0.08, 0.14)
        meander = shape.bend * arm_rng.uniform(0.5, 1.0)
        pts, w = _arm(arm_rng, float(a), start, reach - start, half_width, meander)
        _discs(mask, x, y, pts, w)
        buds = int(arm_rng.integers(0, 3))
        bud_rngs = arm_rng.spawn(buds)
        for bud_rng in bud_rngs:  # side buds: short lobes off an arm
            i = int(bud_rng.integers(len(pts) // 5, max(len(pts) // 5 + 1, 3 * len(pts) // 4)))
            side = math.atan2(*(pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)])[::-1])
            turn = bud_rng.choice([-1, 1]) * math.radians(bud_rng.uniform(40, 80))
            bud_len = bud_rng.uniform(0.08, 0.22)
            t = np.linspace(0, bud_len, max(3, int(bud_len / 0.008)))
            bud = pts[i] + np.outer(t, [math.cos(side + turn), math.sin(side + turn)])
            _discs(mask, x, y, bud, np.maximum(0.6 * w[i] * (1 - t / bud_len) ** 0.7, 0.012))
    soft = gaussian(mask.astype(np.float64), sigma=0.025 / (2 * EXTENT / GRID))
    found = label(fill_holes(soft > 0.5), eight_connected=False)
    sizes = np.bincount(found.ravel())
    sizes[0] = 0
    return found == int(np.argmax(sizes))


def mask_outline(mask: BoolArray, points: int = OUTLINE_POINTS) -> FloatArray:
    """The traced boundary of `mask` in shape units, resampled evenly along its
    length and lightly smoothed (removes the pixel staircase)."""
    traced = trace_outline(mask)  # (x, y) pixel centres
    cell = 2 * EXTENT / GRID
    xy = (traced + 0.5) * cell - EXTENT
    closed = np.vstack([xy, xy[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    t: FloatArray = np.linspace(0.0, float(arc[-1]), points, endpoint=False)
    v: FloatArray = np.stack(
        [np.interp(t, arc, closed[:, 0]), np.interp(t, arc, closed[:, 1])], axis=1
    )
    for _ in range(3):
        v = 0.25 * np.roll(v, 1, axis=0) + 0.5 * v + 0.25 * np.roll(v, -1, axis=0)
    return v


def sinuous_outline(
    rng: np.random.Generator, arms: int, shape: ArmShape | None = None
) -> list[tuple[float, float]]:
    """A sinuous star's outline; redrawn (from the same stream) in the rare case
    that tracing gives a self-touching polygon."""
    for _ in range(8):
        v = mask_outline(sinuous_mask(rng, arms, shape))
        if outline_problem(v) is None:
            return [(float(a), float(b)) for a, b in v]
    raise RuntimeError("no valid sinuous outline in 8 draws")


def thunder_egg_outline(
    rng: np.random.Generator, style: ArmStyle | None = None, shape: ArmShape | None = None
) -> tuple[ArmStyle, ArmShape | None, list[tuple[float, float]]]:
    """A thunder egg's star cavity. The style is drawn per stone unless given
    (jagged on `JAGGED_SHARE`, else sinuous); a sinuous star's shape, including
    its arm count (`ARMS_BASE` plus its extra arms), likewise."""
    base = int(rng.integers(ARMS_BASE[0], ARMS_BASE[1] + 1))
    if style is None:
        style = "jagged" if rng.random() < JAGGED_SHARE else "sinuous"
    match style:
        case "angular":
            return style, None, star_outline(rng, base)
        case "jagged":
            arms = int(rng.integers(JAGGED_ARMS[0], JAGGED_ARMS[1] + 1))
            return style, None, jagged_outline(rng, arms)
        case "sinuous":
            if shape is None:
                shape = ArmShape.draw(rng)
            return style, shape, sinuous_outline(rng, base + shape.extra_arms, shape)


def jagged_mask(rng: np.random.Generator, arms: int) -> BoolArray:
    """A jagged core before its walls are stepped: kinked wedges, broad at the
    base, around an irregular polygon [@goetze2023, Fig. 2]. Each arm draws from
    its own spawned stream."""
    corners = int(rng.integers(5, 9))
    th = np.sort((np.arange(corners) + rng.uniform(-0.3, 0.3, corners)) * 2 * math.pi / corners)
    rad = rng.uniform(0.32, 0.52, corners)
    mask = outline_mask(np.stack([rad * np.cos(th), rad * np.sin(th)], axis=1))
    base = rng.uniform(0, 2 * math.pi) + np.arange(arms) * 2 * math.pi / arms
    base = base + rng.uniform(-0.3, 0.3, arms) * 2 * math.pi / arms
    for a, arm_rng in zip(base, rng.spawn(arms), strict=True):
        reach = arm_rng.uniform(0.55, 0.95)
        f = arm_rng.uniform(0.4, 0.7)  # where the arm kinks
        turn = math.radians(arm_rng.uniform(-35, 35))
        h0 = arm_rng.uniform(0.15, 0.3)  # half-width at the body
        h1 = h0 * arm_rng.uniform(0.5, 0.85)  # at the kink
        h2 = arm_rng.uniform(0.02, 0.05)  # at the tip
        d0 = np.array([math.cos(a), math.sin(a)])
        d1 = np.array([math.cos(a + turn), math.sin(a + turn)])
        p1 = reach * f * d0
        p2 = p1 + reach * (1 - f) * d1
        n0, n1 = _perp(d0), _perp(d1)
        mid = _perp(d0 + d1) / float(np.linalg.norm(d0 + d1))
        wedge = [h0 * n0, p1 + h1 * mid, p2 + h2 * n1, p2 + 0.5 * h2 * d1]
        wedge += [p2 - h2 * n1, p1 - h1 * mid, -h0 * n0]
        mask |= outline_mask(np.array(wedge))
    found = label(fill_holes(mask), eight_connected=False)
    sizes = np.bincount(found.ravel())
    sizes[0] = 0
    return found == int(np.argmax(sizes))


def _perp(d: FloatArray) -> FloatArray:
    return np.array([-d[1], d[0]])


def _resample(v: FloatArray, step: float) -> FloatArray:
    """A closed polygon resampled at about `step` along its length."""
    closed = np.vstack([v, v[:1]])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    n = max(8, int(arc[-1] / step))
    t: FloatArray = np.linspace(0.0, float(arc[-1]), n, endpoint=False)
    return np.stack([np.interp(t, arc, closed[:, 0]), np.interp(t, arc, closed[:, 1])], axis=1)


def jagged_outline(rng: np.random.Generator, arms: int) -> list[tuple[float, float]]:
    """A jagged core's outline: the traced wedges cut into straight runs of
    0.06-0.12 (shape units), each corner pushed in or out by up to `amp`, which
    gives stepped, zigzag walls. Halves the push if the result crosses itself."""
    smooth = mask_outline(jagged_mask(rng, arms))
    step = rng.uniform(0.06, 0.12)
    amp = rng.uniform(0.02, 0.04)
    v = _resample(smooth, step)
    tangent = np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    push = rng.uniform(-1, 1, len(v))
    for _ in range(6):
        out = v + (amp * push)[:, None] * normal
        if outline_problem(out) is None:
            return [(float(a), float(b)) for a, b in out]
        amp *= 0.5
    return [(float(a), float(b)) for a, b in v]


def wall_corners(outline: FloatArray, step: float = 0.005, turn_deg: float = 15.0) -> float:
    """Share of the wall's turning that happens at sharp corners: with the
    outline resampled at `step`, the summed turn at points turning by more than
    `turn_deg`, over the summed turn everywhere. A stepped, zigzag wall turns
    mostly at corners; a smooth winding wall turns gradually."""
    v = _resample(np.asarray(outline, dtype=np.float64), step)
    a = v - np.roll(v, 1, axis=0)
    b = np.roll(v, -1, axis=0) - v
    turn = np.abs(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(axis=1)))
    return float(turn[turn > math.radians(turn_deg)].sum() / turn.sum())


def outline_mask(outline: FloatArray) -> BoolArray:
    """Rasterise a closed outline on the drawing grid (even-odd rule)."""
    x, y = _grid()
    inside = np.zeros(x.shape, dtype=bool)
    v = np.asarray(outline, dtype=np.float64)
    a, b = v, np.roll(v, -1, axis=0)
    for (x0, y0), (x1, y1) in zip(a, b, strict=True):
        if y0 == y1:
            continue
        cross = (y >= min(y0, y1)) & (y < max(y0, y1))
        xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= cross & (x < xc)
    return inside


def arm_meander(mask: BoolArray, q: float = 75.0) -> float:
    """How far arms wander from straight lines: the upper quartile (`q`), over
    arms, of the
    largest sideways offset of an arm's midline from its best straight line,
    divided by the radial span over which the arm was followed.

    Arms are followed outward on circles about the core (the point deepest
    inside, so uneven arms do not pull the centre off), from 0.45 to 0.92 of the
    farthest reach: each run of inside points on a circle is joined to the
    overlapping run on the next (a split or merge starts new tracks). A straight
    star (angular arms, symmetric about their axes) gives about 0; a winding one
    gives clearly more."""
    x, y = _grid()
    deep = np.unravel_index(int(np.argmax(distance_to_false(mask, 1.0))), mask.shape)
    cx, cy = float(x[deep]), float(y[deep])
    reach = float(np.hypot(x[mask] - cx, y[mask] - cy).max())
    cell = 2 * EXTENT / GRID
    n_ang = 1440
    ang = np.arange(n_ang) * 2 * math.pi / n_ang
    tracks: list[list[tuple[float, float]]] = []
    live: list[tuple[int, int, int]] = []  # (track index, first, last) angle indices
    for rad in np.linspace(0.45 * reach, 0.92 * reach, 40):
        px = cx + rad * np.cos(ang)
        py = cy + rad * np.sin(ang)
        j = np.clip(((px + EXTENT) / cell).astype(int), 0, GRID - 1)
        i = np.clip(((py + EXTENT) / cell).astype(int), 0, GRID - 1)
        on = mask[i, j]
        if on.all() or not on.any():
            live = []
            continue
        shift = int(np.argmin(on))  # start the scan outside an arm
        rolled = np.roll(on, -shift)
        edges = np.flatnonzero(np.diff(rolled.astype(np.int8)))
        runs = [
            (int(edges[k]) + 1 + shift, int(edges[k + 1]) + shift)
            for k in range(0, len(edges) - 1, 2)
        ]
        matches = [[t for t, a, b in live if _overlap(lo, hi, a, b, n_ang)] for lo, hi in runs]
        claims = [t for m in matches for t in m]
        new_live: list[tuple[int, int, int]] = []
        for (lo, hi), match in zip(runs, matches, strict=True):
            mid = 0.5 * (lo + hi) * 2 * math.pi / n_ang
            point = (cx + rad * math.cos(mid), cy + rad * math.sin(mid))
            # Continue a track only one-to-one: a split or a merge starts afresh.
            if len(match) == 1 and claims.count(match[0]) == 1:
                tracks[match[0]].append(point)
                new_live.append((match[0], lo, hi))
            else:
                tracks.append([point])
                new_live.append((len(tracks) - 1, lo, hi))
        live = new_live
    scores: list[float] = []
    for tr in tracks:
        if len(tr) < 8:
            continue
        p = np.asarray(tr)
        span = float(np.hypot(*(p[-1] - p[0])))
        centred = p - p.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        offset = np.abs(centred @ vt[1])
        scores.append(float(offset.max()) / max(span, 1e-9))
    return float(np.percentile(scores, q)) if scores else 0.0


def _overlap(lo: int, hi: int, a: int, b: int, n: int) -> bool:
    """Whether circular index ranges [lo, hi] and [a, b] (mod n) share a point."""
    start = (a - lo) % n  # [a, b] measured from lo
    return start <= hi - lo or start + (b - a) >= n
