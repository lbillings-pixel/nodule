"""C14, defect density of a face's bands (ADR 0004, N2 step 3b step 2).

Band terminations (a band that ends) and junctions (a band that splits or two
that join), counted on band skeletons as Murphy et al. [@murphy2015] do for
block-copolymer stripes, and given per band spacing squared of the judged
face (per period squared, as ADR 0004 asks).

The bands are those C13 reads (`nodule.concentric`): the pieces of the judged
face between the edges of `nodule.edges`, whatever their grey level. Murphy
et al. split a two-phase image by sign and thin both phases; a middle-grey
band then merges with a light or a dark one, depending on the width of the
local mean, and the counts swung with it (14-271 junctions on three specimen
cuts). Read through edges, a band that pinches out between two of the same
tone is one end and one junction, as in a split by sign; one light-dark
couplet inserted is two ends (both its bands end, and the bands beside them,
of opposite tone, never join), where a split by sign reads one end and one
junction: the density is the same.

- Edges stop short where the contrast fades (at a band's tip, between bands
  too thin to resolve). Each edge end is joined by a straight line to the
  nearest pixel of another edge piece within `LINK_SPACINGS`, so the bands
  either side stay apart; an end with none so near stays open.
- Pixels of a piece that touch another piece across a diagonal step of an
  edge are dropped, so thinning cannot join them. The pieces are enlarged
  `UPSAMPLE` times (each pixel repeated), so a band is about 10 px wide:
  Murphy et al. find skeleton centres biased for lines narrower than about
  7 px. They are thinned to one-pixel skeletons (Zhang and Suen's
  two-sub-pass thinning [@zhang1984], method only: its paper is cited, not
  read).
- **terminations**: skeleton ends (one neighbour, or two touching ones).
- **junctions**: skeleton pixels where three or more branches meet (crossing
  number of three or more), joined into one where they touch.
- Grooming, as Murphy et al.: a branch from an end that reaches a junction
  within `SPUR_WIDTHS` band widths is a spur of the thinning and is removed,
  and the junction is judged again. A skeleton piece with no junction that
  spans less than that is a dot (Murphy et al. count dots separately); dots are
  left out, so the centre of a set of rings adds nothing.
- Bridges, this project's own rule: a branch from one junction to another
  shorter than `BRIDGE_WIDTHS` band widths is two bands meeting through a gap
  in the edge between them (a dashed edge between thin bands gives a ladder
  of them), and is removed, then the skeleton is groomed again.
- Only defects at least `BORDER_SPACINGS` inside the judged face (itself
  `MARGIN_SPACINGS` inside the outline) are counted, as the judged face's
  edge cuts every band that reaches it (Murphy et al. leave out features
  touching the image edge); the area is that inner face.

Measured 2026-10-31 (test faces at 10, 16 and 32 px per spacing): perfect rings
and stripes 0; one couplet inserted, two terminations; a middle-grey band that
pinches out between light ones, one termination and one junction; isotropic
noise with power at the band spacing 0.76-0.93 per spacing squared at both 16
and 32 px. An even face with pixel noise only has no edges (`nodule.edges`,
noise gate), so no bands, and is not judged (None); before the gate its edges
followed the noise and it read about 1.0-1.2.

On three random cuts of the `n2-slice` preset made to look like library
photos (`nodule.descriptors`, seed 0): 0.10, 0.07 and 0.002, mostly
terminations. Not stable over the parameters: on the first cut the edge
smoothing (0.08-0.2 spacings) moves it 0.03-0.28, `LINK_SPACINGS` (0.35-0.75)
0.04-0.33, `EDGE_PX` (9-16) 0.04-0.11, `BRIDGE_WIDTHS` (2-4) 0.07-0.15;
hysteresis and `UPSAMPLE` (1-3) move it by 0.02 at most. The cause is one
spacing for the whole face, where the outer bands are about half as wide as
the face's typical band: their edges break into dashes that the parameters
close or leave open. So C14 is compared only set against set, measured the
same way on both sides, until the edges follow the local band width
(owner's choice, 2026-10-31). 0.9-3.4 s and 85-136 MB traced per cut of
1000-1500 px. On four library photos (1100-1950 px, band spacings 6-14
px) 0.9-7.3 s, after dots were found in one pass (92 s before on the
finest, with the same counts).

Units: here a *spacing* is one light-dark couplet, `bands.COUPLET_SPACINGS`
(2) times the quality gate's band spacing, which is one band (`edge_face`
converts).

Grade (ADR 0004): plausible; the method is borrowed from analogues
(block-copolymer stripes, tree-ring edges). The bridge and linking rules are
this project's own. Every parameter here is `effective`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from agatesim import _scipy as agatesim_scipy  # pyright: ignore[reportPrivateUsage]
from agatesim.specimen.shape import MARGIN_SPACINGS
from numpy.typing import NDArray

from nodule.edges import edge_face

UPSAMPLE = 2  # effective: px per edge-map px for the skeletons (24 px per spacing)
LINK_SPACINGS = 0.5  # effective: an edge end joins another edge piece this near, spacings
SPUR_WIDTHS = 1.5  # effective: spur and dot length, band widths (half a spacing) [@murphy2015]
BRIDGE_WIDTHS = 3.0  # effective: least junction-to-junction branch, band widths
BORDER_SPACINGS = 0.5  # effective: ends are counted this far inside the judged face
MIN_SPACINGS2 = 4.0  # effective: least judged area, spacings squared

# The eight neighbours in ring order (N, NE, E, SE, S, SW, W, NW), as offsets.
RING = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


@dataclass(frozen=True)
class Defects:
    """C14 of one face; None where it could not be judged."""

    density: float | None  # (terminations + junctions) per spacing squared
    terminations: int | None
    junctions: int | None


def _tables() -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.int8], NDArray[np.int8]]:
    """For each of the 256 neighbourhoods (bit i set where neighbour `RING[i]`
    is on): whether Zhang-Suen's first and second sub-pass delete the centre,
    the count of neighbours that are on, and the crossing number (0 -> 1 steps
    around the ring)."""
    codes = np.arange(256, dtype=np.int64)
    bits = np.stack([(codes >> i) & 1 for i in range(8)])
    p = list(bits)  # p[0] = N (Zhang-Suen's P2) ... p[7] = NW (P9)
    n = bits.sum(axis=0)
    a = ((bits == 0) & (np.roll(bits, -1, axis=0) == 1)).sum(axis=0)
    base = (n >= 2) & (n <= 6) & (a == 1)
    first = base & (p[0] * p[2] * p[4] == 0) & (p[2] * p[4] * p[6] == 0)
    second = base & (p[0] * p[2] * p[6] == 0) & (p[0] * p[4] * p[6] == 0)
    return first, second, n.astype(np.int8), a.astype(np.int8)


FIRST, SECOND, NEIGHBOURS, CROSSINGS = _tables()


def _codes(flat: NDArray[np.uint8], at: NDArray[np.intp], width: int) -> NDArray[np.intp]:
    """Neighbourhood codes (`_tables`) of pixels `at` of a flattened image
    `width` px wide, padded by one pixel."""
    code = np.zeros(at.size, dtype=np.intp)
    for i, (dr, dc) in enumerate(RING):
        code |= flat[at + (dr * width + dc)].astype(np.intp) << i
    return code


def thin(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Zhang-Suen thinning of `mask` to an 8-connected one-pixel skeleton.
    A pixel's test depends only on its neighbours, so after the first pass
    each sub-pass tests only pixels next to one removed in the two sub-passes
    before it (since it last ran): the same skeleton as testing every pixel
    each time."""
    img = np.pad(mask, 1).astype(np.uint8)
    width = img.shape[1]
    flat = img.ravel()
    steps = np.array([dr * width + dc for dr, dc in RING], dtype=np.intp)
    at = np.flatnonzero(flat)
    mark = np.zeros(flat.size, dtype=np.bool_)  # pixels to test next
    recent: list[NDArray[np.intp]] = []
    sub = 0
    while True:
        at = at[flat[at] == 1]
        gone = at[(FIRST if sub == 0 else SECOND)[_codes(flat, at, width)]]
        flat[gone] = 0
        recent = [*recent[-1:], gone]
        if len(recent) == 1:  # the second sub-pass has not run yet: it tests every pixel
            at = np.flatnonzero(flat)
        else:
            both = np.concatenate(recent)
            if both.size == 0:
                break
            mark[(both[:, None] + steps[None, :]).ravel()] = True
            at = np.flatnonzero(mark)
            mark[at] = False
        sub = 1 - sub
    return img[1:-1, 1:-1].astype(np.bool_)


def classify(skeleton: NDArray[np.bool_]) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Skeleton ends and junction pixels (module docstring)."""
    img = np.pad(skeleton, 1).astype(np.uint8)
    width = img.shape[1]
    at = np.flatnonzero(img)
    code = _codes(img.ravel(), at, width)
    n, a = NEIGHBOURS[code], CROSSINGS[code]
    ends = np.zeros(img.size, dtype=np.bool_)
    joins = np.zeros(img.size, dtype=np.bool_)
    ends[at] = (n == 1) | ((n == 2) & (a == 1))
    joins[at] = a >= 3
    shape = img.shape
    return ends.reshape(shape)[1:-1, 1:-1], joins.reshape(shape)[1:-1, 1:-1]


def _walk(
    skeleton: NDArray[np.bool_], junction: NDArray[np.bool_], start: tuple[int, int], most: int
) -> list[tuple[int, int]] | None:
    """The pixels from an end up to (not including) a junction, if one is
    reached within `most` steps; None otherwise."""
    h, w = skeleton.shape
    path = [start]
    seen = {start}
    r, c = start
    for _ in range(most):
        nxt: list[tuple[int, int]] = []
        for dr, dc in RING:
            q = (r + dr, c + dc)
            if 0 <= q[0] < h and 0 <= q[1] < w and skeleton[q] and q not in seen:
                if junction[q]:
                    return path
                nxt.append(q)
        if not nxt:
            return None
        if len(nxt) > 1:  # a staircase step: take the side neighbour
            side = [q for q in nxt if q[0] == r or q[1] == c]
            if len(side) != 1:
                return None
            nxt = side
        r, c = nxt[0]
        seen.add((r, c))
        path.append((r, c))
    return None


def groom(skeleton: NDArray[np.bool_], spur_px: float) -> NDArray[np.bool_]:
    """`skeleton` less the spurs (module docstring), until none is left."""
    out = skeleton.copy()
    most = max(1, math.ceil(spur_px))
    while True:
        ends, junction = classify(out)
        spurs = [_walk(out, junction, (int(r), int(c)), most) for r, c in np.argwhere(ends)]
        spurs = [s for s in spurs if s is not None]
        if not spurs:
            return out
        for s in spurs:
            for q in s:
                out[q] = False


def count(skeleton: NDArray[np.bool_], judged: NDArray[np.bool_], dot_px: float) -> tuple[int, int]:
    """Terminations and junctions of a groomed skeleton at `judged` pixels,
    dots left out (module docstring)."""
    ends, junction = classify(skeleton)
    labels = agatesim_scipy.label(skeleton, eight_connected=True)
    has_junction = np.zeros(int(labels.max()) + 1, dtype=np.bool_)
    has_junction[labels[junction]] = True
    # Each piece's span in rows and columns, in one pass over the skeleton.
    n = has_junction.size
    rows, cols = np.nonzero(labels)
    of = labels[rows, cols]
    h, w = skeleton.shape
    low_r, low_c = np.full(n, h, dtype=np.intp), np.full(n, w, dtype=np.intp)
    high_r, high_c = np.full(n, -1, dtype=np.intp), np.full(n, -1, dtype=np.intp)
    np.minimum.at(low_r, of, rows)
    np.minimum.at(low_c, of, cols)
    np.maximum.at(high_r, of, rows)
    np.maximum.at(high_c, of, cols)
    span = np.maximum(high_r - low_r, high_c - low_c)
    dot = np.zeros(n, dtype=np.bool_)
    with_ends = np.unique(labels[ends])
    dot[with_ends] = ~has_junction[with_ends] & (span[with_ends] < dot_px)
    keep = ends & judged & ~dot[labels]
    groups = agatesim_scipy.label(
        agatesim_scipy.dilate(junction, 1) & skeleton, eight_connected=True
    )
    joined = np.unique(groups[junction & judged])
    return int(keep.sum()), int(np.count_nonzero(joined))


def link_ends(edges: NDArray[np.bool_], most: float) -> NDArray[np.bool_]:
    """`edges` thinned, each end joined by a straight line to the nearest
    pixel of another edge piece within `most` px (module docstring). The
    pieces are those before any line is drawn."""
    out = thin(edges)
    ends, _ = classify(out)
    at = np.argwhere(ends)
    reach = max(1, math.floor(most))
    dr, dc = np.mgrid[-reach : reach + 1, -reach : reach + 1]
    dist = np.hypot(dr, dc).ravel()
    order = np.argsort(dist, kind="stable")
    order = order[(dist[order] > 0) & (dist[order] <= most)]
    steps = np.stack([dr.ravel()[order], dc.ravel()[order]], axis=1)  # nearest first
    if at.size == 0 or steps.size == 0:
        return out
    labels = np.pad(agatesim_scipy.label(out, eight_connected=True), reach)
    own = labels[at[:, 0] + reach, at[:, 1] + reach]
    near = labels[
        at[:, 0, None] + reach + steps[None, :, 0], at[:, 1, None] + reach + steps[None, :, 1]
    ]
    other = (near > 0) & (near != own[:, None])
    found = other.any(axis=1)
    first = other.argmax(axis=1)
    h, w = out.shape
    for i in np.flatnonzero(found):
        r, c, k = int(at[i, 0]), int(at[i, 1]), int(first[i])
        tr, tc = r + int(steps[k, 0]), c + int(steps[k, 1])
        n = 2 * math.ceil(float(dist[order][k])) + 2
        t = np.linspace(0.0, 1.0, n)
        rr = np.clip(np.round(r + t * (tr - r)).astype(np.intp), 0, h - 1)
        cc = np.clip(np.round(c + t * (tc - c)).astype(np.intp), 0, w - 1)
        out[rr, cc] = True
    return out


def separate(labels: NDArray[np.int32]) -> NDArray[np.bool_]:
    """The labelled pieces less the pixels that touch another piece (8
    neighbours), so thinning cannot join two bands across a diagonal step
    of the edge between them."""
    h, w = labels.shape
    padded = np.pad(labels, 1)
    clash = np.zeros(labels.shape, dtype=np.bool_)
    for dr, dc in RING:
        n = padded[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
        clash |= (n != 0) & (n != labels)
    return (labels != 0) & ~clash


def unbridge(skeleton: NDArray[np.bool_], most: float, spur_px: float) -> NDArray[np.bool_]:
    """`skeleton` less its bridges: branches shorter than `most` px from
    one junction to another (module docstring), regroomed, until none is
    left."""
    out = skeleton.copy()
    h, w = out.shape
    while True:
        _, junction = classify(out)
        if not junction.any():
            return out
        hub = agatesim_scipy.dilate(junction, 1) & out
        groups = np.pad(agatesim_scipy.label(hub, eight_connected=True), 1)
        branches = agatesim_scipy.label(out & ~hub, eight_connected=True)
        sizes = np.bincount(branches.ravel())
        pairs: list[NDArray[np.int64]] = []
        for dr, dc in RING:  # (branch, junction group) pairs that touch
            g = groups[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
            both = (branches > 0) & (g > 0)
            pairs.append(branches[both].astype(np.int64) * (int(groups.max()) + 1) + g[both])
        touching = np.unique(np.concatenate(pairs)) // (int(groups.max()) + 1)
        ids, hubs = np.unique(touching, return_counts=True)
        short = ids[(hubs >= 2) & (sizes[ids] < most)]
        if short.size == 0:
            return out
        out &= ~np.isin(branches, short)
        out = groom(thin(out), spur_px)


def measure(
    lum: NDArray[np.float32],
    mask: NDArray[np.bool_],
    scale: float,
    noise: float,
    spacing_px: float,
) -> Defects:
    """C14 of a face. Arguments as `nodule.edges.edge_face`."""
    ef = edge_face(lum, mask, scale, noise, spacing_px)
    spacing = ef.spacing
    judged = agatesim_scipy.erode(ef.face, max(1, math.ceil(MARGIN_SPACINGS * spacing)))
    inner = agatesim_scipy.erode(judged, max(1, math.ceil(BORDER_SPACINGS * spacing)))
    area = inner.sum() / spacing**2
    if area < MIN_SPACINGS2:
        return Defects(None, None, None)
    if not (ef.edges & judged).any():  # no bands, so none to judge
        return Defects(None, None, None)
    edges = link_ends(ef.edges, LINK_SPACINGS * spacing)
    bands = separate(agatesim_scipy.label(judged & ~edges, eight_connected=False))
    bands = bands.repeat(UPSAMPLE, axis=0).repeat(UPSAMPLE, axis=1)
    inner = inner.repeat(UPSAMPLE, axis=0).repeat(UPSAMPLE, axis=1)
    width = 0.5 * spacing * UPSAMPLE  # one band
    spur = SPUR_WIDTHS * width
    skeleton = unbridge(groom(thin(bands), spur), BRIDGE_WIDTHS * width, spur)
    ends, joins = count(skeleton, inner, spur)
    return Defects(float((ends + joins) / area), ends, joins)
