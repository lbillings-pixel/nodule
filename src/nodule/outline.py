"""Outline shape of one view, without a scale (ADR 0004, C1-C3).

Every value is a ratio of lengths, so no ruler is needed. All three are read
from the view's filled mask: holes such as an open cavity are filled first,
since these describe the outline only.

- **C1, aspect ratio and regularity** [@shea2010, App. C]: b/a and A/(πab),
  where a and b are the semi-axes of an ellipse fitted to the outline. Shea et
  al. call A/(πab) "regularity": it accounts for irregularities in the
  outline while disregarding elongation. How their ellipse is fitted is not
  recorded in our reading of the paper, so here a and b are those of the
  ellipse with the same second moments as the mask (an assumption). With that
  fit an ellipse gives exactly 1 at any aspect ratio, a square 3/π ≈ 0.955,
  and lobed outlines less.
- **C2, Riley circularity** √(Di/Dc) [@blott2008, Table 12]: Di is the
  diameter of the largest inscribed circle, Dc of the smallest circumscribing
  circle. A circle gives 1, an octagon 0.961, a square 0.841.
- **C3, concavity irregularity** I(2D) = Σ (y − x)/y over the concavities
  [@blott2008, p. 60]. x runs from the centre of the largest inscribed circle
  to the nearest point of a concavity, y from that centre to the convex hull
  in the same direction. Any convex outline gives 0; more and deeper
  concavities give more. Blott and Pye's classes: 0 perfectly regular, up to
  0.5 slightly, 0.5-1 moderately, 1-2 very, above 2 extremely irregular.

A digitised outline has pixel-sized notches, which are not concavities: a
concavity counts only if it is at least `MIN_CONCAVITY` of the hull distance
deep and `MIN_CONCAVITY_PX` pixels deep (both `effective`).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from agatesim import _scipy as agatesim_scipy  # pyright: ignore[reportPrivateUsage]
from numpy.typing import NDArray

from nodule._scipy import hull_2d

MIN_CONCAVITY = 0.02  # effective: (y − x)/y below this is digitisation, not shape
MIN_CONCAVITY_PX = 2.0  # effective: and so is anything shallower than this


@dataclass(frozen=True)
class Outline:
    """Scale-free outline shape of one view."""

    aspect: float  # C1: b/a of the moment ellipse
    regularity: float  # C1: A/(πab)
    circularity: float  # C2: Riley √(Di/Dc)
    irregularity: float  # C3: I(2D)
    concavities: int  # how many concavities C3 summed over

    def to_json(self) -> dict[str, float | int]:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(self).items()}


def filled(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """The largest connected region of `mask`, with its holes filled."""
    labels = agatesim_scipy.label(mask, eight_connected=False)
    if labels.max() == 0:
        return np.zeros_like(mask)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return agatesim_scipy.fill_holes(labels == int(np.argmax(sizes)))


def moment_ellipse(mask: NDArray[np.bool_]) -> tuple[float, float]:
    """Semi-axes (a, b), a ≥ b, of the ellipse with the mask's second moments.
    Each pixel counts as a unit square (variance 1/12 added per axis)."""
    rows, cols = np.nonzero(mask)
    cov = np.cov(np.stack([cols, rows]).astype(np.float64)) + np.eye(2) / 12.0
    low, high = np.linalg.eigvalsh(cov)
    return 2.0 * math.sqrt(high), 2.0 * math.sqrt(max(low, 0.0))


def inscribed_circle(mask: NDArray[np.bool_]) -> tuple[NDArray[np.float64], float]:
    """Centre (x, y) and radius of the largest circle inside the mask, in pixels."""
    distance = agatesim_scipy.distance_to_false(np.pad(mask, 1), 1.0)[1:-1, 1:-1]
    row, col = np.unravel_index(int(np.argmax(distance)), distance.shape)
    # Distances run between pixel centres; the region's edge is half a pixel out.
    return np.array([col, row], dtype=np.float64), float(distance[row, col]) - 0.5


def _corners(mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Corners (x, y) of the mask's edge pixels: the region as a union of squares."""
    edge = mask & ~agatesim_scipy.erode(mask, 1)
    rows, cols = np.nonzero(edge)
    centres = np.stack([cols, rows], axis=1).astype(np.float64)
    offsets = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    return np.unique((centres[:, None, :] + offsets[None, :, :]).reshape(-1, 2), axis=0)


def _circle_two(
    p: NDArray[np.float64], q: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    return (p + q) / 2.0, float(np.linalg.norm(p - q)) / 2.0


def _circle_three(
    p: NDArray[np.float64], q: NDArray[np.float64], r: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float] | None:
    """The circle through three points, or None if they are in a line."""
    d = 2.0 * (p[0] * (q[1] - r[1]) + q[0] * (r[1] - p[1]) + r[0] * (p[1] - q[1]))
    if abs(d) < 1e-12:
        return None
    pp, qq, rr = p @ p, q @ q, r @ r
    ux = (pp * (q[1] - r[1]) + qq * (r[1] - p[1]) + rr * (p[1] - q[1])) / d
    uy = (pp * (r[0] - q[0]) + qq * (p[0] - r[0]) + rr * (q[0] - p[0])) / d
    centre = np.array([ux, uy])
    return centre, float(np.linalg.norm(p - centre))


def enclosing_circle(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    """Centre and radius of the smallest circle holding all points (Welzl's
    incremental method). The points are visited in a fixed shuffled order; the
    circle itself is unique, so the order changes only the running time."""
    pts = points[np.random.default_rng(0).permutation(len(points))]
    tol = 1e-9 * (1.0 + float(np.abs(pts).max()))

    def outside(p: NDArray[np.float64], c: NDArray[np.float64], r: float) -> bool:
        return float(np.linalg.norm(p - c)) > r + tol

    centre, radius = pts[0].copy(), 0.0
    for i in range(1, len(pts)):
        if not outside(pts[i], centre, radius):
            continue
        centre, radius = pts[i].copy(), 0.0
        for j in range(i):
            if not outside(pts[j], centre, radius):
                continue
            centre, radius = _circle_two(pts[i], pts[j])
            for k in range(j):
                if outside(pts[k], centre, radius):
                    three = _circle_three(pts[i], pts[j], pts[k])
                    if three is not None:
                        centre, radius = three
    return centre, radius


def _hull_mask(vertices: NDArray[np.float64], shape: tuple[int, int]) -> NDArray[np.bool_]:
    """Pixels whose centres lie inside a convex polygon, row by row: a convex
    polygon crosses each row in one interval."""
    a, b = vertices, np.roll(vertices, -1, axis=0)
    ys = np.arange(shape[0], dtype=np.float64)[:, None]
    lo_y, hi_y = np.minimum(a[:, 1], b[:, 1]), np.maximum(a[:, 1], b[:, 1])
    crosses = (ys >= lo_y) & (ys <= hi_y) & (hi_y > lo_y)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ys - a[:, 1]) / (b[:, 1] - a[:, 1])
    xs = a[:, 0] + t * (b[:, 0] - a[:, 0])
    x_lo = np.where(crosses, xs, np.inf).min(axis=1)
    x_hi = np.where(crosses, xs, -np.inf).max(axis=1)
    cols = np.arange(shape[1], dtype=np.float64)[None, :]
    eps = 1e-9
    return (cols >= x_lo[:, None] - eps) & (cols <= x_hi[:, None] + eps)


def _to_hull(
    centre: NDArray[np.float64], direction: NDArray[np.float64], edges: NDArray[np.float64]
) -> float:
    """Distance from `centre` (inside) along a unit `direction` to the hull."""
    normals, offsets = edges[:, :2], edges[:, 2]
    along = normals @ direction
    ahead = along > 1e-12
    return float(np.min(-(normals[ahead] @ centre + offsets[ahead]) / along[ahead]))


def concavities(mask: NDArray[np.bool_], centre: NDArray[np.float64]) -> list[tuple[float, float]]:
    """(x, y) of each concavity of a filled mask, as Blott and Pye measure them
    from `centre`, the inscribed circle's centre."""
    vertices, edges = hull_2d(_corners(mask))
    pockets = agatesim_scipy.label(_hull_mask(vertices, mask.shape) & ~mask, eight_connected=False)
    out: list[tuple[float, float]] = []
    for n in range(1, int(pockets.max()) + 1):
        rows, cols = np.nonzero(pockets == n)
        pts = np.stack([cols, rows], axis=1).astype(np.float64)
        dist = np.linalg.norm(pts - centre, axis=1)
        i = int(np.argmin(dist))
        # The pocket's nearest pixel lies just outside the region's edge.
        x = float(dist[i]) - 0.5
        y = _to_hull(centre, (pts[i] - centre) / dist[i], edges)
        if y - x >= max(MIN_CONCAVITY * y, MIN_CONCAVITY_PX):
            out.append((x, y))
    return out


def touches_frame(mask: NDArray[np.bool_]) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def measure(mask: NDArray[np.bool_]) -> Outline | None:
    """Outline shape of the region in `mask`, or None if it is too small to judge
    or runs off the frame (so its whole outline is not seen)."""
    region = filled(mask)
    if region.sum() < 9 or touches_frame(region):
        return None
    a, b = moment_ellipse(region)
    centre, r_in = inscribed_circle(region)
    _, r_out = enclosing_circle(hull_2d(_corners(region))[0])
    found = concavities(region, centre)
    return Outline(
        aspect=b / a,
        regularity=float(region.sum()) / (math.pi * a * b),
        circularity=math.sqrt(r_in / r_out),
        irregularity=sum((y - x) / y for x, y in found),
        concavities=len(found),
    )
