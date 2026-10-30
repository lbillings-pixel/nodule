"""The drawn rim-to-centre line of C24-C25, round 2 (ADR 0006, section 9, choice 7o).

Every tagger reads C25's zones and C24's thirds along one straight line per
face, from the rim (0) to the centre of the largest band set (1), with ticks
every 0.1. The computer drafts the line; the owner checks it before helpers
see it.

- **Centre.** The band-following paths of `nodule.bands.trace` run inward
  along the band normal, so where bands close round a centre their last
  stretches, carried on as straight lines, meet near it, even where a path
  stopped at a quartz core or a cavity first. Every crossing of two such
  lines inside the face, and every path's end, is a candidate; the one the
  most lines pass near (within `NEAR_SPACINGS` band spacings or
  `NEAR_SHARE` of the face's size, and not behind the path) wins, and the
  centre is the least-squares point of those lines. With several band sets
  the set with the most paths wins: the largest set, if paths start evenly
  round the rim. Too little support (`MIN_SUPPORT`), or supporting lines
  that share one axis (`MAX_AXIAL`: parallel bands meet nowhere), gives no
  draft: then the
  face's deepest point stands in, marked "deepest point", for the owner to
  move. An assumption, not from a source; tested on synthetic faces with a
  known centre, and with parallel bands and a plain face that must give no
  draft (the negative controls).
- **Direction.** An angle drawn from a stream spawned per face from
  `zones.SEED` (keyed by the library id, so the same at any resolution); the
  rim end is where the ray from the centre leaves the face. The owner may
  take the next draw for a stated reason (`REASONS`).

Points are stored as fractions of the picture's width and height, so they
hold at any size. Appearance only: never evidence for a formation mechanism
(ADR 0002).
"""

from __future__ import annotations

import itertools
import math
import zlib
from dataclasses import dataclass

import numpy as np
from agatesim import _scipy as agatesim_scipy  # pyright: ignore[reportPrivateUsage]
from numpy.typing import NDArray

from nodule import bands, zones

TAIL_SPACINGS = 2.0  # effective: a path's last stretch, in band spacings, gives its direction
NEAR_SPACINGS = 2.0  # effective: a line this close to a candidate supports it ...
NEAR_SHARE = 0.03  # effective: ... or this share of the face's size, whichever is larger
MIN_SUPPORT = 4  # effective: fewest supporting paths for a draft ...
MIN_SUPPORT_SHARE = 0.25  # effective: ... and fewest as a share of the paths traced
# effective: supporting lines must spread beyond about ±32° of one axis (1 = all
# parallel); on the synthetic faces 0.05-0.76 with a centre, 1.0 for parallel bands
MAX_AXIAL = 0.9
MIN_ANGLE = math.radians(10)  # effective: two lines meeting at less than this give no crossing
TICKS = 10  # ticks along the line, one every 0.1
LINE_STREAM = 2  # child of zones.SEED's streams (0 and 1 pick faces for the owner)
REASONS = zones.LINE_REASONS


@dataclass(frozen=True)
class Draft:
    """A drafted centre: (row, col) in full px, how many path lines support it
    and how many paths there were; `centre` is None where support was too
    little."""

    centre: tuple[float, float] | None
    support: int
    paths: int


@dataclass(frozen=True)
class Line:
    """A drawn line, as fractions (x of width, y of height) of the picture."""

    rim: tuple[float, float]
    centre: tuple[float, float]
    source: str  # "paths" or "deepest point"
    draw: int  # which direction draw of the face's stream


def tails(
    traced: bands.Traced, spacing_px: float
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Each path's end point and unit direction over its last stretch."""
    out: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    for full, arc in traced.paths:
        first = int(np.searchsorted(arc, float(arc[-1]) - TAIL_SPACINGS * spacing_px))
        d = full[-1] - full[min(first, full.shape[0] - 2)]
        norm = float(np.hypot(*d))
        if norm > 0:
            out.append((full[-1], d / norm))
    return out


def inside(face: NDArray[np.bool_], k: float, point: NDArray[np.float64]) -> bool:
    """Whether a full-px point lies in `face`, held at 1/k of full size."""
    i, j = int(point[0] // k), int(point[1] // k)
    return 0 <= i < face.shape[0] and 0 <= j < face.shape[1] and bool(face[i, j])


def draft_centre(traced: bands.Traced, spacing_px: float) -> Draft:
    """Where the paths' last stretches, carried on, meet (module docstring)."""
    lines = tails(traced, spacing_px)
    n = len(lines)
    if n < MIN_SUPPORT:
        return Draft(None, 0, n)
    size = math.sqrt(float(traced.face.sum())) * traced.k
    near = max(NEAR_SPACINGS * spacing_px, NEAR_SHARE * size)
    points = np.array([p for p, _ in lines])
    dirs = np.array([d for _, d in lines])
    candidates = [p for p in points]
    for (p1, d1), (p2, d2) in itertools.combinations(lines, 2):
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < math.sin(MIN_ANGLE):
            continue
        t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / cross
        candidates.append(p1 + t * d1)
    candidates = [c for c in candidates if inside(traced.face, traced.k, c)]
    if not candidates:
        return Draft(None, 0, n)
    cand = np.array(candidates)  # (m, 2)
    rel = cand[:, None, :] - points[None, :, :]  # (m, n, 2)
    along = np.einsum("mnk,nk->mn", rel, dirs)
    off = np.abs(rel[..., 0] * dirs[None, :, 1] - rel[..., 1] * dirs[None, :, 0])
    supports = (off < near) & (along > -near)
    count = supports.sum(axis=1)
    best = int(np.lexsort((np.where(supports, off, 0.0).sum(axis=1), -count))[0])
    support = int(count[best])
    if support < max(MIN_SUPPORT, MIN_SUPPORT_SHARE * n):
        return Draft(None, support, n)
    use = supports[best]
    doubled = 2 * np.arctan2(dirs[use, 0], dirs[use, 1])
    if float(np.hypot(np.cos(doubled).mean(), np.sin(doubled).mean())) > MAX_AXIAL:
        return Draft(None, support, n)
    # The point nearest, in least squares, to every supporting line.
    a = np.zeros((2, 2))
    b = np.zeros(2)
    for p, d in zip(points[use], dirs[use], strict=True):
        proj = np.eye(2) - np.outer(d, d)
        a += proj
        b += proj @ p
    centre = np.linalg.solve(a, b) if abs(np.linalg.det(a)) > 1e-9 else cand[best]
    if not inside(traced.face, traced.k, centre):
        centre = cand[best]
    return Draft((float(centre[0]), float(centre[1])), support, n)


def deepest_point(face: NDArray[np.bool_], k: float) -> tuple[float, float]:
    """The point of the face farthest from its outline, in full px."""
    depth = agatesim_scipy.distance_to_false(face, 1.0)
    i, j = np.unravel_index(int(np.argmax(depth)), depth.shape)
    return (float((i + 0.5) * k - 0.5), float((j + 0.5) * k - 0.5))


def direction(image_id: str, draw: int = 0) -> float:
    """The `draw`th angle (radians) of the face's own stream."""
    key = (LINE_STREAM, zlib.crc32(image_id.encode("utf-8")))
    rng = np.random.default_rng(np.random.SeedSequence(zones.SEED, spawn_key=key))
    return float(rng.uniform(0, 2 * math.pi, draw + 1)[draw])


def rim_point(
    face: NDArray[np.bool_], k: float, centre: tuple[float, float], angle: float
) -> tuple[float, float]:
    """Where the ray from `centre` at `angle` last lies in the face (full px)."""
    dr, dc = -math.sin(angle), math.cos(angle)  # angle counted anticlockwise as seen
    step = 0.5 * k
    r, c = centre
    last = centre
    for _ in range(int(2 * math.hypot(*face.shape) * k / step) + 1):
        r, c = r + step * dr, c + step * dc
        if not inside(face, k, np.array([r, c])):
            break
        last = (float(r), float(c))
    return last


def draw_line(
    traced: bands.Traced,
    spacing_px: float,
    image_id: str,
    shape: tuple[int, int],
    draw: int = 0,
    centre: tuple[float, float] | None = None,
) -> tuple[Line, Draft]:
    """The face's line on a picture of `shape` (rows, cols) full px: from the
    drafted centre (or a given one, full px) at the `draw`th angle."""
    found = draft_centre(traced, spacing_px)
    source = "paths"
    if centre is None:
        centre = found.centre
    else:
        source = "owner"
    if centre is None:
        centre, source = deepest_point(traced.face, traced.k), "deepest point"
    rim = rim_point(traced.face, traced.k, centre, direction(image_id, draw))
    h, w = shape

    def frac(p: tuple[float, float]) -> tuple[float, float]:
        return ((p[1] + 0.5) / w, (p[0] + 0.5) / h)

    return Line(frac(rim), frac(centre), source, draw), found


def outline_line(
    mask: NDArray[np.bool_], image_id: str, shape: tuple[int, int], draw: int = 0
) -> Line:
    """A line for a face with no band paths (no band spacing known): from the
    deepest point of `mask` (any size) on a picture of `shape` (rows, cols)."""
    k = shape[1] / mask.shape[1]
    centre = deepest_point(mask, k)
    rim = rim_point(mask, k, centre, direction(image_id, draw))
    h, w = shape
    return Line(
        ((rim[1] + 0.5) / w, (rim[0] + 0.5) / h),
        ((centre[1] + 0.5) / w, (centre[0] + 0.5) / h),
        "deepest point",
        draw,
    )
