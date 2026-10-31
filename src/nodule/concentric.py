"""C13, concentricity and completeness of a face's bands (ADR 0004, N2 step 3b
step 2).

The face is read through its band edges (`nodule.edges`, Canny-like, as
Marichal et al. find tree-ring boundaries): each connected piece (4-connected)
of the judged face between edges is one *band*, whatever its grey level, so a
middle-grey band between a light and a dark one is a band of its own. This
follows the ring model of Marichal et al. [@marichal2025]: rings are "roughly
concentric, even if their shape is irregular", do not intersect, and a ring
that disappears "without forming a closed curve" stays open; and of Gillert et
al. [@gillert2023], whose wedging rings form "incomplete circles".

- A band is a **ring** where it encloses a hole at least one spacing across
  (a hole that holds at least another band, not a speck of noise): closed
  edge loops bound it on both sides. Its centre is the centroid of the area
  it encloses (the band with its holes filled).
- A band is **cut** where it touches the edge of the judged face: the photo or
  the outline ends it, so it is left out. A band that spans less than
  `DOT_SPACINGS` is a dot and is left out too.
- A band that is neither ring, cut nor dot ends inside the face: a **wedging
  band**, counted if it spans at least `WEDGE_SPACINGS`, unless it holds the
  centre of a set (the solid eye inside the innermost ring).
- Rings nest: ring B lies inside ring A where B sits in A's holes. Each ring's
  parent is the smallest ring that holds it, which makes a tree. Its height is
  the number of rings on the longest chain down from it.
- A **concentric set** is a tree of height at least `MIN_DEPTH`. Where a ring
  holds two or more subtrees each `BRANCH_DEPTH` high or more, each is a set
  of its own (two eyes inside shared outer bands are two sets).

Reported per face:

- **sets**: the number of concentric sets (0 for stripes or noise).
- **depth**: rings on the longest nested chain.
- **drift**: the median distance between the centres of a ring and a ring
  just inside it, in spacings, over the rings of the sets (0 for concentric
  circles). The drift of successive band centres over the spacing that
  ADR 0004 asks for.
- **complete**: the share of 360 degrees present, averaged over the bands of
  the sets and the wedging bands: a ring counts 1; a wedging band the share of
  `ANGLE_BINS` angle bins it covers about the centre of the innermost ring of
  the nearest set.
- **wedges**: the number of wedging bands.

All are None where the judged face is too small; drift and complete are None
where there is no set.

Limit: a band that ends inside the face, with its edges, joins the bands
beside it where its edges stop, so they read as one piece. Where the bands
round one eye open where two sets meet, that set is lost in its neighbour.
Rings drawn as edge curves (a closed curve is a ring, a curve that ends is
open) avoid this but were tried and rejected 2026-10-31. A fault crossing the
bands closes false loops, the sets of three specimen cuts swung between 2
and 3 over the edge parameters, and a band that pinches out became a ring.

Measured 2026-10-31: the tests' faces (16 and 32 px per spacing) as asserted
there, middle-grey bands included. On three random cuts of the `n2-slice`
preset made to look like library photos (`nodule.descriptors`, seed 0),
sets 2, 2 and 1 and depth 6, 19 and 19. Over the edge smoothing (0.08-0.2
spacings), hysteresis (`HIGH` 0.2-0.4, `LOW` 0.05-0.15) and `EDGE_PX` (9-16),
the sets held except at smoothing 0.2 on the first cut (2 to 1), depth moved
by 3 at most, and drift by 0.14 at most (0.04 apart from smoothing). The
blob draft this replaces split the face by sign about a local mean. Its sets
and depth swung with that mean's width, as middle-grey bands merged. 0.3-0.4 s
and 85-136 MB traced per cut of 1000-1500 px.

Units: here a *spacing* is one light-dark couplet, `bands.COUPLET_SPACINGS`
(2) times the quality gate's band spacing, which is one band (`measure`
converts).

Grade (ADR 0004): plausible; the ring model is borrowed from analogues (tree
rings). The tree, set and wedge rules are this project's own. Every parameter
here is `effective`.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from agatesim import _scipy as agatesim_scipy  # pyright: ignore[reportPrivateUsage]
from agatesim.specimen.shape import MARGIN_SPACINGS
from numpy.typing import NDArray

from nodule.defects import MIN_SPACINGS2
from nodule.edges import edge_face

DOT_SPACINGS = 0.75  # effective: a band spanning less is a dot (C14's dot, 1.5 band widths)
HOLE_SPACINGS = 1.0  # effective: the least hole that makes a band a ring
WEDGE_SPACINGS = 2.0  # effective: the least span of a wedging band
MIN_DEPTH = 4  # effective: rings nested in a concentric set (two couplets)
BRANCH_DEPTH = 2  # effective: rings a subtree needs of its own to be a set of its own
ANGLE_BINS = 72  # effective: 5-degree bins for a wedging band's share of 360 degrees


@dataclass(frozen=True)
class Concentric:
    """C13 of one face; None where it could not be judged."""

    sets: int | None
    depth: int | None
    drift: float | None  # spacings
    complete: float | None  # share of 360 degrees, mean over bands
    wedges: int | None


@dataclass
class Band:
    """One connected piece of one phase."""

    rows: NDArray[np.intp]
    cols: NDArray[np.intp]
    filled: NDArray[np.bool_]  # the band with its holes filled, in its box
    top: int  # the box's first row and column in the face
    left: int
    ring: bool
    centre: tuple[float, float]  # of the filled area (rings) or of the band
    children: list[int] = field(default_factory=lambda: [])
    parent: int | None = None
    height: int = 1


def pieces(labels: NDArray[np.int32]) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Rows and columns of each labelled piece, label 1 first."""
    flat = labels.ravel()
    order = np.argsort(flat, kind="stable")
    cuts = np.searchsorted(flat[order], np.arange(1, int(flat.max()) + 2))
    width = labels.shape[1]
    return [(order[a:b] // width, order[a:b] % width) for a, b in itertools.pairwise(cuts)]


def bands_of(between: NDArray[np.bool_], edge: NDArray[np.bool_], spacing: float) -> list[Band]:
    """The rings and the bands that may be wedging bands among the regions
    between edges (module docstring)."""
    labels = agatesim_scipy.label(between, eight_connected=False)
    out: list[Band] = []
    for rows, cols in pieces(labels):
        span = max(int(np.ptp(rows)), int(np.ptp(cols))) + 1
        if span < DOT_SPACINGS * spacing or bool(edge[rows, cols].any()):
            continue
        top, left = int(rows.min()), int(cols.min())
        box = np.zeros((int(np.ptp(rows)) + 3, int(np.ptp(cols)) + 3), dtype=np.bool_)
        box[rows - top + 1, cols - left + 1] = True
        filled = agatesim_scipy.fill_holes(box)
        holes = filled & ~box
        ring = False
        if holes.any():
            for hr, hc in pieces(agatesim_scipy.label(holes, eight_connected=False)):
                if max(int(np.ptp(hr)), int(np.ptp(hc))) + 1 >= HOLE_SPACINGS * spacing:
                    ring = True
                    break
        if ring:
            fr, fc = np.nonzero(filled)
            centre = (float(fr.mean()) + top - 1, float(fc.mean()) + left - 1)
        elif span >= WEDGE_SPACINGS * spacing:
            centre = (float(rows.mean()), float(cols.mean()))
        else:
            continue
        out.append(Band(rows, cols, filled, top - 1, left - 1, ring, centre))
    return out


def holds(band: Band, point: tuple[int, int]) -> bool:
    """Whether `point` lies in `band` or in its holes."""
    y, x = point[0] - band.top, point[1] - band.left
    h, w = band.filled.shape
    return 0 <= y < h and 0 <= x < w and bool(band.filled[y, x])


def nest(rings: list[Band]) -> None:
    """Parents, children and heights of `rings`, in place (module docstring)."""
    order = sorted(range(len(rings)), key=lambda i: int(rings[i].filled.sum()))
    for a, i in enumerate(order):
        point = (int(rings[i].rows[0]), int(rings[i].cols[0]))
        for j in order[a + 1 :]:  # larger rings, smallest first
            if holds(rings[j], point):
                rings[i].parent = j
                rings[j].children.append(i)
                break
    for i in order:  # smallest first, so children are done before their parent
        kids = rings[i].children
        rings[i].height = 1 + max((rings[c].height for c in kids), default=0)


def eyes(rings: list[Band], i: int) -> list[int]:
    """The innermost ring of each concentric set under ring `i`: a set splits
    where two or more subtrees are `BRANCH_DEPTH` high."""
    big = [c for c in rings[i].children if rings[c].height >= BRANCH_DEPTH]
    if len(big) >= 2:
        return [e for c in big for e in eyes(rings, c)]
    if rings[i].children:
        tallest = max(rings[i].children, key=lambda c: rings[c].height)
        return eyes(rings, tallest)
    return [i]


def coverage(band: Band, centre: tuple[float, float]) -> float:
    """Share of `ANGLE_BINS` angle bins about `centre` that `band` covers."""
    angle = np.arctan2(band.rows - centre[0], band.cols - centre[1])
    bins = np.floor((angle + math.pi) / (2 * math.pi) * ANGLE_BINS).astype(np.int64)
    return float(np.unique(np.clip(bins, 0, ANGLE_BINS - 1)).size / ANGLE_BINS)


def judge(edges: NDArray[np.bool_], judged: NDArray[np.bool_], spacing: float) -> Concentric:
    """C13 from a face's edge map and its judged pixels, `spacing` in px."""
    border = judged & ~agatesim_scipy.erode(judged, 1)
    found = bands_of(judged & ~edges, border, spacing)
    rings = [b for b in found if b.ring]
    nest(rings)
    roots = [i for i, r in enumerate(rings) if r.parent is None and r.height >= MIN_DEPTH]
    depth = max((r.height for r in rings), default=0)
    centres = [e for i in roots for e in eyes(rings, i)]
    points = {(round(rings[e].centre[0]), round(rings[e].centre[1])) for e in centres}
    open_bands = [b for b in found if not b.ring and not any(holds(b, p) for p in points)]
    wedges = len(open_bands)
    if not centres:
        return Concentric(0, depth, None, None, wedges)
    members: list[int] = []
    stack = list(roots)
    while stack:
        i = stack.pop()
        members.append(i)
        stack += rings[i].children
    steps = [
        math.dist(rings[i].centre, rings[p].centre) / spacing
        for i in members
        if (p := rings[i].parent) is not None
    ]
    shares = [1.0] * len(members)
    for b in open_bands:
        near = min(centres, key=lambda e: math.dist(rings[e].centre, b.centre))
        shares.append(coverage(b, rings[near].centre))
    drift = float(np.median(steps)) if steps else None
    return Concentric(len(centres), depth, drift, float(np.mean(shares)), wedges)


def measure(
    lum: NDArray[np.float32], mask: NDArray[np.bool_], scale: float, spacing_px: float
) -> Concentric:
    """C13 of a face. Arguments as `nodule.orientation.measure`."""
    ef = edge_face(lum, mask, scale, spacing_px)
    judged = agatesim_scipy.erode(ef.face, max(1, math.ceil(MARGIN_SPACINGS * ef.spacing)))
    if judged.sum() / ef.spacing**2 < MIN_SPACINGS2:
        return Concentric(None, None, None, None, None)
    return judge(ef.edges, judged, ef.spacing)
