"""C13, concentricity and completeness of a face's bands (ADR 0004, N2 step 3b
step 2; the rule of ADR 0004 footnote 2).

The face is read through its band edges (`nodule.edges`) and the direction
across the bands at each edge px, the normal of a structure tensor. As in
pith detection on log ends [@marichal2024; @norell2008; @schraml2013], the
centre of a set lies where the lines through the band edges along their
normals meet; as in CS-TRD [@marichal2025], bands are then read along 360 rays
from that centre, keeping the edge points whose normal lies within 30 degrees
of the ray.

- **Edges**: edge pieces (8-connected) under `PIECE_SPACINGS` long are left
  out: texture breaks into short pieces, a band edge runs along the band.
  An edge px votes where the tensor's coherence is at least `COHERENCE`.
- **Centre**: each voting px draws its normal line across an accumulator of
  `CELL_SPACINGS` cells (every k-th px where more than `VOTERS` vote), blurred
  by `BLUR_SPACINGS`; the peak inside the judged face seeds a least-squares
  point of the lines that pass within `REFINE_SPACINGS` of it, refitted until
  it settles, as APD refines its centre on the edges near it
  [@marichal2024]. Line voting and a blurred peak follow Norell and
  Borgefors [@norell2008] and Schraml and Uhl [@schraml2013].
- **Rays**: `RAYS` rays from the centre run until they leave the judged face.
  A crossing is a run of edge px (dilated by 1 px, so a ray cannot slip
  between diagonal px) along a ray; it *faces* the centre where the normal
  there lies within `TOLERANCE_DEG` of the ray. Each ray scores its longest
  run of facing crossings with no other crossing between them: on band-scale
  noise a third of the crossings face any point by chance, so a plain count
  grows with the ray's length, and a run does not.
- **Set**: the centre makes a concentric set where at least `SET_SHARE` of
  all rays (not of those the face leaves room for) score `MIN_DEPTH` or more.
  The edge px whose normal faces the set's centre are then taken out, and the
  search repeats on the rest for a further set, up to `MAX_SETS`. The papers
  above assume one centre per face; the run, the set rule and the further sets
  are this project's own.

Reported per face:

- **sets**: the number of concentric sets (0 for stripes or noise).
- **depth**: the deepest set's median score over its judged rays (the upper
  median, so a set's depth is at least `MIN_DEPTH`), in band edges: about
  the number of bands crossed from the eye outward. Where there is no set,
  the median score from the first centre tried (0 where none).
  A ray is judged where the face leaves it room for `MIN_DEPTH` bands.
- **drift**: the centre shift per unit radius, the median over the sets: the
  set's facing edge px inside the largest circle about its centre that the
  face does not cut are split at their median radius, a centre fitted to each
  half as above (inliers within `DRIFT_SPACINGS`), and their distance divided
  by the difference of the halves' mean radii (0 for concentric circles).
  The drift of successive band centres over the spacing that ADR 0004 asks
  for, as a ratio.
- **complete**: the share of each set's judged rays that score `MIN_DEPTH`,
  averaged over the sets: the share of 360 degrees where the set is present.

All are None where the judged face is too small; drift and complete are None
where there is no set.

Limits. (a) A fortification whose bands turn at sharp corners can read as two
sets, its inner bands meeting at one point and its outer ones at the corners'
tips (seen by eye on one of the library photos with two sets). Two rules that
merged such a centre into the set round it also merged the two separate eyes
of the tests, and were dropped. (b) Drift reads low where the outline cuts
the rings, as a cut ring's normals meet short of its centre: on the tests'
drifting rings 0.19 and 0.33 against 0.20 and 0.42 for the same rings whole.
(c) The old count of wedging bands is gone (ADR 0004 footnote 2): counting
open arcs round the centre gave 4 for drifting rings and 0 for one wedge.

Measured 2026-11-01: the tests' faces (16 and 32 px per spacing) as asserted
there. On the 72 library photos whose rim outline is judged "yes"
(`nodule.tags`): a set on 39 of the 63 tagged fortification (two sets on 4),
on none of the 3 tagged moss alone, and on 4 of the 6 others (an eye, a tube
and two quartz geodes, the geodes at the threshold). The old closed-ring rule
found a set on 2 fortification faces. Over the piece length (0, 1, 2
spacings) fortification faces with a set were 43, 39 and 33 and moss stayed
at 0. At a 45 degree tolerance stripes and noise read as sets. 0.1 s
for a 400 px test face; 0.7 s per photo on average, 3.0 s and 258 MiB traced
for the largest (a 1900 px edge map), 0.15 s and 20 MiB for the smallest;
the 72 photos took 51 s.

Units: here a *spacing* is one light-dark couplet, `bands.COUPLET_SPACINGS`
(2) times the quality gate's band spacing, which is one band (`measure`
converts).

Grade (ADR 0004): plausible; the centre and ray reading are borrowed from
analogues (tree rings). The set, run, piece and further-set rules are this
project's own. Every parameter here is `effective`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from agatesim import _scipy as agatesim_scipy  # pyright: ignore[reportPrivateUsage]
from agatesim.specimen.shape import MARGIN_SPACINGS
from numpy.typing import NDArray

from nodule.defects import MIN_SPACINGS2
from nodule.edges import edge_face, face_core, smoothed_face

TENSOR_SPACINGS = 0.25  # effective: structure-tensor window (Gaussian sigma)
COHERENCE = 0.5  # effective: least coherence of a voting edge px
PIECE_SPACINGS = 1.0  # effective: least length (px) of an edge piece kept
CELL_SPACINGS = 0.5  # effective: accumulator cell side
BLUR_SPACINGS = 0.5  # effective: accumulator blur (Gaussian sigma)
VOTERS = 30_000  # effective: most edge px that vote; the vote only seeds the fit
REFINE_SPACINGS = 1.0  # effective: least-squares inliers, line-to-centre distance
DRIFT_SPACINGS = 2.0  # effective: the same for the centres of drift's halves
TOLERANCE_DEG = 30.0  # effective: a crossing faces the centre (CS-TRD's 30 degrees)
RAYS = 360  # effective: rays from a centre (as CS-TRD)
MIN_DEPTH = 6  # effective: least score of a ray in a set (three couplets)
SET_SHARE = 0.5  # effective: least share of all rays that score MIN_DEPTH
JUDGED_SHARE = 0.25  # effective: least share of rays with room, to judge a centre
MAX_SETS = 4  # effective: most sets looked for
MIN_VOTERS = 20  # effective: fewer voting px look for no further set
STEP = 0.5  # px between samples along a ray
ITERATIONS = 10  # least-squares refits at most


@dataclass(frozen=True)
class Concentric:
    """C13 of one face; None where it could not be judged."""

    sets: int | None
    depth: int | None  # band edges
    drift: float | None  # centre shift per unit radius
    complete: float | None  # share of 360 degrees, mean over sets


@dataclass(frozen=True)
class Normals:
    """The direction across the bands at every px, and its coherence."""

    ny: NDArray[np.float64]
    nx: NDArray[np.float64]
    coherence: NDArray[np.float64]


@dataclass(frozen=True)
class Voters:
    """Edge px (row, col) and their normals."""

    py: NDArray[np.float64]
    px: NDArray[np.float64]
    ny: NDArray[np.float64]
    nx: NDArray[np.float64]

    def __getitem__(self, keep: NDArray[np.bool_]) -> Voters:
        return Voters(self.py[keep], self.px[keep], self.ny[keep], self.nx[keep])


@dataclass(frozen=True)
class Scores:
    """Each ray's score, whether the face leaves it room, and its reach (px)."""

    score: NDArray[np.int64]
    judged: NDArray[np.bool_]
    reach: NDArray[np.float64]


@dataclass(frozen=True)
class Set:
    """One concentric set."""

    centre: tuple[float, float]
    depth: int
    complete: float
    drift: float | None


def normals(field: NDArray[np.float64], face: NDArray[np.bool_], spacing: float) -> Normals:
    """Structure-tensor normals of `field` smoothed as the edge map is."""
    grad = np.gradient(smoothed_face(field, face_core(face, spacing), spacing))
    gy: NDArray[np.float64] = grad[0]
    gx: NDArray[np.float64] = grad[1]
    sigma = max(1.0, TENSOR_SPACINGS * spacing)
    jxx = agatesim_scipy.gaussian(gx * gx, sigma)
    jyy = agatesim_scipy.gaussian(gy * gy, sigma)
    jxy = agatesim_scipy.gaussian(gx * gy, sigma)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    coherence = np.hypot(jxx - jyy, 2 * jxy) / np.maximum(jxx + jyy, 1e-12)
    return Normals(np.sin(theta), np.cos(theta), coherence)


def long_pieces(edges: NDArray[np.bool_], spacing: float) -> NDArray[np.bool_]:
    """`edges` without its pieces under `PIECE_SPACINGS` long (px count)."""
    labels = agatesim_scipy.label(edges, eight_connected=True)
    size = np.bincount(labels.ravel())
    size[0] = 0
    keep: NDArray[np.bool_] = size >= PIECE_SPACINGS * spacing
    return keep[labels]


def vote(v: Voters, judged: NDArray[np.bool_], spacing: float) -> tuple[float, float]:
    """The blurred accumulator's peak inside `judged` (module docstring)."""
    cell = CELL_SPACINGS * spacing
    h, w = judged.shape
    rows, cols = math.ceil(h / cell), math.ceil(w / cell)
    if v.py.size > VOTERS:
        v = v[np.arange(v.py.size) % math.ceil(v.py.size / VOTERS) == 0]
    t = np.arange(-math.hypot(h, w), math.hypot(h, w), cell)
    acc = np.zeros(rows * cols)
    chunk = 2000
    for a in range(0, v.py.size, chunk):
        yy = v.py[a : a + chunk, None] + t * v.ny[a : a + chunk, None]
        xx = v.px[a : a + chunk, None] + t * v.nx[a : a + chunk, None]
        ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        cells = (yy[ok] // cell).astype(np.int64) * cols + (xx[ok] // cell).astype(np.int64)
        acc += np.bincount(cells, minlength=rows * cols)
    blurred = agatesim_scipy.gaussian(acc.reshape(rows, cols), BLUR_SPACINGS / CELL_SPACINGS)
    mid_r: NDArray[np.float64] = (np.arange(rows, dtype=np.float64) + 0.5) * cell
    mid_c: NDArray[np.float64] = (np.arange(cols, dtype=np.float64) + 0.5) * cell
    at_r = np.clip(mid_r.astype(np.int64), 0, h - 1)
    at_c = np.clip(mid_c.astype(np.int64), 0, w - 1)
    blurred[~judged[np.ix_(at_r, at_c)]] = -1.0
    i, j = np.unravel_index(int(np.argmax(blurred)), blurred.shape)
    return ((float(i) + 0.5) * cell, (float(j) + 0.5) * cell)


def fit(c: tuple[float, float], v: Voters, within: float) -> tuple[float, float]:
    """The point nearest (least squares) to the normal lines that pass within
    `within` px of it, refitted from `c` until it settles."""
    for _ in range(ITERATIONS):
        ty, tx = v.nx, -v.ny  # along each line's normal to it
        k = np.abs((c[0] - v.py) * ty + (c[1] - v.px) * tx) < within
        if int(k.sum()) < 3:
            break
        a, b = ty[k], tx[k]
        m = np.array([[a @ a, a @ b], [a @ b, b @ b]]) + 1e-9 * np.eye(2)
        rhs = np.array([a @ (a * v.py[k] + b * v.px[k]), b @ (a * v.py[k] + b * v.px[k])])
        y, x = np.linalg.solve(m, rhs)
        moved = math.dist((float(y), float(x)), c)
        c = (float(y), float(x))
        if moved < 1e-3:
            break
    return c


def facing(c: tuple[float, float], v: Voters) -> NDArray[np.bool_]:
    """Voters whose normal lies within `TOLERANCE_DEG` of the line to `c`."""
    dy, dx = c[0] - v.py, c[1] - v.px
    r = np.maximum(np.hypot(dy, dx), 1e-9)
    return np.abs(dy * v.ny + dx * v.nx) / r >= math.cos(math.radians(TOLERANCE_DEG))


def score(
    c: tuple[float, float],
    edges: NDArray[np.bool_],
    n: Normals,
    judged: NDArray[np.bool_],
    spacing: float,
) -> Scores:
    """Each ray's longest run of facing crossings (module docstring)."""
    h, w = judged.shape
    angle = (np.arange(RAYS) + 0.5) * 2 * math.pi / RAYS
    s = np.arange(STEP, math.hypot(h, w), STEP)
    sy, sx = np.sin(angle)[:, None], np.cos(angle)[:, None]
    iy = np.rint(c[0] + s * sy).astype(np.int64)
    ix = np.rint(c[1] + s * sx).astype(np.int64)
    inside = (iy >= 0) & (iy < h) & (ix >= 0) & (ix < w)
    iy, ix = np.clip(iy, 0, h - 1), np.clip(ix, 0, w - 1)
    inside = np.logical_and.accumulate(inside & judged[iy, ix], axis=1)
    hit = inside & edges[iy, ix]
    start = hit & ~np.pad(hit[:, :-1], ((0, 0), (1, 0)))
    faces = np.abs(n.ny[iy, ix] * sy + n.nx[iy, ix] * sx) >= math.cos(math.radians(TOLERANCE_DEG))
    good = np.cumsum(start & faces, axis=1)
    reset = np.maximum.accumulate(np.where(start & ~faces, good, 0), axis=1)
    reach = inside.sum(axis=1) * STEP
    return Scores((good - reset).max(axis=1), reach >= MIN_DEPTH * spacing / 2, reach)


def drift(c: tuple[float, float], v: Voters, radius: float, spacing: float) -> float | None:
    """Centre shift per unit radius between the inner and outer half of `v`
    inside `radius` (module docstring)."""
    r = np.hypot(v.py - c[0], v.px - c[1])
    v, r = v[r < radius], r[r < radius]
    if r.size < MIN_VOTERS:
        return None
    halves = (r <= np.median(r), r > np.median(r))
    centres = [fit(c, v[k], DRIFT_SPACINGS * spacing) for k in halves]
    span = float(r[halves[1]].mean() - r[halves[0]].mean())
    return math.dist(centres[0], centres[1]) / span if span >= spacing else None


def depth_of(s: Scores) -> int:
    return int(np.percentile(s.score[s.judged], 50, method="higher"))


def find_sets(
    field: NDArray[np.float64],
    face: NDArray[np.bool_],
    edges: NDArray[np.bool_],
    judged: NDArray[np.bool_],
    spacing: float,
) -> tuple[list[Set], int]:
    """The face's sets, and the first centre's depth (0 where none was tried)."""
    n = normals(field, face, spacing)
    kept = long_pieces(edges & judged, spacing)
    crossed = agatesim_scipy.dilate(kept, 1) & judged
    py, px = np.nonzero(kept & (n.coherence >= COHERENCE))
    v = Voters(py.astype(np.float64), px.astype(np.float64), n.ny[py, px], n.nx[py, px])
    left = np.ones(py.size, dtype=np.bool_)
    remaining = crossed.copy()
    sets: list[Set] = []
    first = 0
    for k in range(MAX_SETS):
        if int(left.sum()) < MIN_VOTERS:
            break
        c = fit(vote(v[left], judged, spacing), v[left], REFINE_SPACINGS * spacing)
        if not (0 <= c[0] < judged.shape[0] and 0 <= c[1] < judged.shape[1]):
            break
        test = score(c, remaining, n, judged, spacing)
        if test.judged.mean() < JUDGED_SHARE:
            break
        if k == 0:
            first = depth_of(test)
        if (test.score >= MIN_DEPTH).mean() < SET_SHARE:
            break
        mine = left & facing(c, v)
        full = score(c, crossed, n, judged, spacing)
        complete = float((full.score[full.judged] >= MIN_DEPTH).mean())
        dr = drift(c, v[mine], float(full.reach.min()), spacing)
        sets.append(Set(c, depth_of(full), complete, dr))
        gone = np.zeros_like(kept)
        gone[py[mine], px[mine]] = True
        remaining &= ~agatesim_scipy.dilate(gone, 1)
        left &= ~mine
    return sets, first


def judge(
    field: NDArray[np.float64],
    face: NDArray[np.bool_],
    edges: NDArray[np.bool_],
    judged: NDArray[np.bool_],
    spacing: float,
) -> Concentric:
    """C13 from a face's resampled field, face, edge map and judged px,
    `spacing` in px."""
    sets, first = find_sets(field, face, edges, judged, spacing)
    if not sets:
        return Concentric(0, first, None, None)
    drifts = [s.drift for s in sets if s.drift is not None]
    return Concentric(
        len(sets),
        max(s.depth for s in sets),
        float(np.median(drifts)) if drifts else None,
        float(np.mean([s.complete for s in sets])),
    )


def measure(
    lum: NDArray[np.float32],
    mask: NDArray[np.bool_],
    scale: float,
    noise: float,
    spacing_px: float,
) -> Concentric:
    """C13 of a face. Arguments as `nodule.edges.edge_face`."""
    ef = edge_face(lum, mask, scale, noise, spacing_px)
    judged = agatesim_scipy.erode(ef.face, max(1, math.ceil(MARGIN_SPACINGS * ef.spacing)))
    if judged.sum() / ef.spacing**2 < MIN_SPACINGS2:
        return Concentric(None, None, None, None)
    return judge(ef.lum, ef.face, ef.edges, judged, ef.spacing)
