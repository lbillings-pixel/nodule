"""C14, defect density of a face's bands (ADR 0004, N2 step 3b step 2).

Band terminations (a band that ends) and junctions (a band that splits or two
that join), counted on the skeletons of both band phases as Murphy et al.
[@murphy2015] do for block-copolymer stripes, and given per band spacing
squared of the judged face.

The face is the coarse copy C15 reads (`bands.coarse_face`, resampled by
`curvature.even_face` to exactly `FIELD_SPACING_PX` px per spacing). It is
band-passed there (a Gaussian of `FINE_SPACINGS` less one of
`BROAD_SPACINGS`, both averaged over the face only, so the outline adds no
step), then resampled `UPSAMPLE` times finer, so a band (half a spacing) is
about 6 px wide: Murphy et al. find skeleton centres biased for lines
narrower than about 7 px. The sign of the band-passed face splits it into
light and dark bands; each is thinned to a one-pixel skeleton (Zhang and Suen's
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
- Only defects at least `MARGIN_SPACINGS` inside the outline are counted
  (Murphy et al. leave out features touching the image edge), and the area is
  that judged face. A face with no bands still has a sign everywhere: its
  texture gives defects too, so C14 is read with C16's banded share.

Measured 2026-10-31 (test faces at 10, 16 and 32 px per spacing): perfect rings
and stripes 0; stripes with one band inserted, one termination and one
junction; isotropic noise with power at the band spacing 0.89-0.97 per spacing
squared at both 16 and 32 px; an even face with pixel noise only, about 0.7.
Restricting the count to the banded area (C16) does not separate that even face:
its noise reads as 25-37% banded and keeps about 0.7.

Grade (ADR 0004): plausible; the method is borrowed from an analogue
(block-copolymer stripes). Every parameter here is `effective`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from agatesim import _scipy as agatesim_scipy  # pyright: ignore[reportPrivateUsage]
from agatesim.specimen.shape import MARGIN_SPACINGS
from numpy.typing import NDArray

from nodule.bands import OUTLINE_PX, coarse_face
from nodule.curvature import even_face

FINE_SPACINGS = 0.25  # effective: band-pass, fine Gaussian sigma, spacings
BROAD_SPACINGS = 1.0  # effective: band-pass, local-mean Gaussian sigma, spacings
UPSAMPLE = 3  # effective: px per coarse px for the skeletons (12 px per spacing)
SPUR_WIDTHS = 1.5  # effective: spur and dot length, band widths (half a spacing) [@murphy2015]
MIN_SPACINGS2 = 4.0  # effective: least judged area, spacings squared

# The eight neighbours in ring order (N, NE, E, SE, S, SW, W, NW), as offsets.
RING = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


@dataclass(frozen=True)
class Defects:
    """C14 of one face; None where it could not be judged."""

    density: float | None  # (terminations + junctions) per spacing squared
    terminations: int | None
    junctions: int | None


def band_pass(
    coarse: NDArray[np.float64], face: NDArray[np.bool_], spacing: float
) -> NDArray[np.float64]:
    """The face less its local mean, smoothed a little (module docstring);
    both Gaussians average over the face's core only."""
    core = agatesim_scipy.erode(face, OUTLINE_PX)
    weight = core.astype(np.float64)

    def mean(sigma: float) -> NDArray[np.float64]:
        total = agatesim_scipy.gaussian(coarse * weight, sigma)
        return total / np.maximum(agatesim_scipy.gaussian(weight, sigma), 1e-12)

    out = mean(max(1.0, FINE_SPACINGS * spacing)) - mean(BROAD_SPACINGS * spacing)
    return np.where(face, out, 0.0)


def upsample(field: NDArray[np.float64], factor: int) -> NDArray[np.float64]:
    """`field` resampled `factor` times finer (bilinear, pixel centres kept)."""
    h, w = field.shape
    rows = (np.arange(h * factor, dtype=np.float64) + 0.5) / factor - 0.5
    cols = (np.arange(w * factor, dtype=np.float64) + 0.5) / factor - 0.5
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return agatesim_scipy.bilinear(field, rr, cc)


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
    dots: set[int] = set()
    for i in np.unique(labels[ends]):
        if has_junction[i]:
            continue
        rows, cols = np.nonzero(labels == i)
        if max(np.ptp(rows), np.ptp(cols)) < dot_px:
            dots.add(int(i))
    keep = ends & judged & ~np.isin(labels, list(dots))
    groups = agatesim_scipy.label(
        agatesim_scipy.dilate(junction, 1) & skeleton, eight_connected=True
    )
    joined = np.unique(groups[junction & judged])
    return int(keep.sum()), int(np.count_nonzero(joined))


def measure(
    lum: NDArray[np.float32], mask: NDArray[np.bool_], scale: float, spacing_px: float
) -> Defects:
    """C14 of a face. Arguments as `nodule.orientation.measure`."""
    coarse, face, k = coarse_face(lum, mask, scale, spacing_px)
    coarse, face, spacing = even_face(coarse, face, spacing_px / k)
    signal = upsample(band_pass(coarse, face, spacing), UPSAMPLE)
    fine = upsample(face.astype(np.float64), UPSAMPLE) >= 0.5
    spacing *= UPSAMPLE
    judged = agatesim_scipy.erode(fine, max(1, math.ceil(MARGIN_SPACINGS * spacing)))
    area = judged.sum() / spacing**2
    if area < MIN_SPACINGS2:
        return Defects(None, None, None)
    spur = SPUR_WIDTHS * 0.5 * spacing
    ends = joins = 0
    for phase in (fine & (signal > 0), fine & (signal <= 0)):
        e, j = count(groom(thin(phase), spur), judged, spur)
        ends, joins = ends + e, joins + j
    return Defects(float((ends + joins) / area), ends, joins)
