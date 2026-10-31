"""C15, band curvature and sharp corners of a face (ADR 0004, N2 step 3b step 2).

Both parts read the band direction of `nodule.bands.direction_field` on the
coarse copy of the face that C16 and the band profiles use
(`bands.coarse_face`), resampled so one band spacing spans exactly
`FIELD_SPACING_PX` px: block-averaging by whole pixels leaves 4-8 px per
spacing, and at corners the curvature's upper quantile moved with that by up
to a half (2026-10-31: zig-zag, 0.093 at 5 px, 0.143 at 4 px), while at
exactly 4 px it is the same from 8 to 32 full px per spacing. The outline
step is left out and only pixels at least `MARGIN_SPACINGS` inside the
outline are judged, as for C16 (`nodule.orientation`);
a pixel is *banded* where its coherence reaches agatesim's `MIN_COHERENCE`.

- **curvature**: how fast the band direction turns along the bands, kappa =
  |t . grad theta|, with t the band tangent and theta the band normal's angle.
  This is the curvature of the band contours as the direction field sees them
  at the scale of one spacing; Riecke and Madruga [@riecke2006] read the same
  quantity from traced, diffusion-smoothed contours with quadratic fits. Here
  the doubled direction exp(2i theta) of the banded pixels is smoothed over
  `SMOOTH_SPACINGS` first (in place of their contour smoothing), which also
  takes out most of the 3-5 degree ripple that block-averaging leaves in the
  direction. kappa is given times the band spacing (dimensionless): 0 for
  straight bands, spacing / r on a ring of radius r. Unsigned: a photo does
  not say which side of a band is the inside. Reported over the banded pixels
  as the median and the 90th percentile; equally spaced bands make that close
  to the distribution along the contours.
- **corners**: the count of sharp corners (fortification angles). A judged
  pixel, banded or not (at a right-angled apex the two arms cancel in the
  structure tensor), lies on a corner where, along some direction b (`DIRECTIONS` tried
  over 180 degrees), the bands `ARM_SPACINGS` before and after it differ in
  direction by at least `CORNER_DEG`, each of those two arms runs straight
  (within `ARM_TURN_DEG` of the band twice as far out), and across b the bands
  turn by no more than `ACROSS_DEG`. The arm test rejects rings at any radius:
  their bands keep turning, so the arms are not straight. The across test
  rejects the centre of a set of rings, which turns in every direction, while
  nested corners line up along their bisector and do not turn across it.
  Corner pixels join (8-connected) into corner lines; a line counts as one
  corner if it runs at least `CORNER_RUN` spacings, an apex seen on about five
  nested bands. Isotropic noise with power at the band spacing leaves only
  scraps of corner up to 4 spacings long (2026-10-31, five seeds at two
  resolutions); corner lines of a zig-zag run 19-33.

Grade (ADR 0004): plausible; the curvature method is borrowed from an
analogue (convection stripes) and the corner rule is this project's own.
Every parameter here is `effective`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from agatesim import _scipy as agatesim_scipy  # pyright: ignore[reportPrivateUsage]
from agatesim.specimen.quality import MIN_COHERENCE
from agatesim.specimen.shape import MARGIN_SPACINGS
from numpy.typing import NDArray

from nodule.bands import FIELD_SPACING_PX, coarse_face, direction_field
from nodule.orientation import doubled

SMOOTH_SPACINGS = 1.0  # effective: the doubled direction is smoothed over this before kappa
HIGH_QUANTILE = 0.9  # effective: the "high" curvature reported beside the median
ARM_SPACINGS = 1.5  # effective: distance from a corner pixel to the arms read
CORNER_DEG = 45.0  # effective: the least turn between the two arms of a corner
ARM_TURN_DEG = 10.0  # effective: the most an arm may turn between 1 and 2 arm lengths
ACROSS_DEG = 20.0  # effective: the most the bands may turn across the corner
DIRECTIONS = 16  # effective: directions b tried over 180 degrees
CORNER_RUN = 5.0  # effective: the least length of a corner line, spacings
MIN_PIXELS = 50  # effective: banded coarse pixels needed, as for C16


@dataclass(frozen=True)
class Curvature:
    """C15 of one face; None where it could not be judged."""

    median: float | None  # kappa x spacing
    high: float | None  # its HIGH_QUANTILE quantile
    corners: int | None
    banded: float  # share of the judged face that is banded


def even_face(
    coarse: NDArray[np.float64], face: NDArray[np.bool_], spacing: float
) -> tuple[NDArray[np.float64], NDArray[np.bool_], float]:
    """The coarse face resampled (linear) so a spacing spans exactly
    `FIELD_SPACING_PX` px, and that spacing (module docstring)."""
    step = spacing / FIELD_SPACING_PX  # coarse px per resampled px, 1 to 2
    if step <= 1.0 + 1e-9:
        return coarse, face, spacing
    h, w = (int(n / step) for n in coarse.shape)
    rows = (np.arange(h, dtype=np.float64) + 0.5) * step - 0.5
    cols = (np.arange(w, dtype=np.float64) + 0.5) * step - 0.5
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    lum = agatesim_scipy.bilinear(coarse, rr, cc)
    inside = agatesim_scipy.bilinear(face.astype(np.float64), rr, cc) >= 0.5
    return lum, inside, FIELD_SPACING_PX


def turn_rate(
    u: NDArray[np.complex128], banded: NDArray[np.bool_], spacing: float
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """kappa x spacing at each pixel, and where it is read: banded pixels
    whose neighbours are banded too (module docstring)."""
    w = banded.astype(np.float64)
    sigma = SMOOTH_SPACINGS * spacing
    re = agatesim_scipy.gaussian(np.real(u) * w, sigma)
    im = agatesim_scipy.gaussian(np.imag(u) * w, sigma)
    s = re + 1j * im
    s = s / np.maximum(np.abs(s), 1e-12)
    d_row = np.imag(np.conj(s) * np.gradient(s, axis=0))  # d(2 theta)/d row
    d_col = np.imag(np.conj(s) * np.gradient(s, axis=1))
    theta = 0.5 * np.angle(s)
    t_row, t_col = np.cos(theta), -np.sin(theta)  # band tangent (row, column)
    kappa = 0.5 * np.abs(t_row * d_row + t_col * d_col)
    return np.asarray(kappa * spacing, dtype=np.float64), agatesim_scipy.erode(banded, 1)


def _read(
    u: NDArray[np.complex128],
    banded: NDArray[np.bool_],
    rows: NDArray[np.int64],
    cols: NDArray[np.int64],
    step_r: float,
    step_c: float,
) -> tuple[NDArray[np.complex128], NDArray[np.bool_]]:
    """u at (rows, cols) + step, and whether that point is banded."""
    h, w = banded.shape
    pr = np.rint(rows + step_r).astype(np.int64)
    pc = np.rint(cols + step_c).astype(np.int64)
    ok = (pr >= 0) & (pr < h) & (pc >= 0) & (pc < w)
    pr, pc = np.clip(pr, 0, h - 1), np.clip(pc, 0, w - 1)
    return u[pr, pc], ok & banded[pr, pc]


def _angle_deg(a: NDArray[np.complex128], b: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Angle between two band directions given as doubled angles, degrees."""
    return np.asarray(0.5 * np.degrees(np.abs(np.angle(a * np.conj(b)))), dtype=np.float64)


def corner_pixels(
    u: NDArray[np.complex128],
    banded: NDArray[np.bool_],
    inner: NDArray[np.bool_],
    spacing: float,
) -> NDArray[np.bool_]:
    """Judged (`inner`) pixels that lie on a corner (module docstring)."""
    rows, cols = np.nonzero(inner)
    rows, cols = rows.astype(np.int64), cols.astype(np.int64)
    arm = ARM_SPACINGS * spacing
    hit = np.zeros(rows.size, dtype=np.bool_)
    for i in range(DIRECTIONS):
        phi = math.pi * i / DIRECTIONS
        br, bc = math.sin(phi), math.cos(phi)
        a1, ok_a1 = _read(u, banded, rows, cols, arm * br, arm * bc)
        b1, ok_b1 = _read(u, banded, rows, cols, -arm * br, -arm * bc)
        a2, ok_a2 = _read(u, banded, rows, cols, 2 * arm * br, 2 * arm * bc)
        b2, ok_b2 = _read(u, banded, rows, cols, -2 * arm * br, -2 * arm * bc)
        xa, ok_xa = _read(u, banded, rows, cols, arm * bc, -arm * br)  # across b
        xb, ok_xb = _read(u, banded, rows, cols, -arm * bc, arm * br)
        judged = ok_a1 & ok_b1 & ok_a2 & ok_b2
        across_ok = ~(ok_xa & ok_xb) | (_angle_deg(xa, xb) <= ACROSS_DEG)
        hit |= (
            judged
            & (_angle_deg(a1, b1) >= CORNER_DEG)
            & (_angle_deg(a1, a2) <= ARM_TURN_DEG)
            & (_angle_deg(b1, b2) <= ARM_TURN_DEG)
            & across_ok
        )
    out = np.zeros(inner.shape, dtype=np.bool_)
    out[rows[hit], cols[hit]] = True
    return out


def run_length(rows: NDArray[np.int64], cols: NDArray[np.int64]) -> float:
    """The longest extent of a patch, px: its widest projection over
    `DIRECTIONS` directions (within 0.5% of the longest pixel-to-pixel distance)."""
    phi = np.pi * np.arange(DIRECTIONS) / DIRECTIONS
    proj = np.outer(rows, np.sin(phi)) + np.outer(cols, np.cos(phi))
    return float(np.ptp(proj, axis=0).max())


def count_corners(pixels: NDArray[np.bool_], spacing: float) -> int:
    """Corner lines at least `CORNER_RUN` spacings long."""
    labels = agatesim_scipy.label(pixels, eight_connected=True)
    flat = np.flatnonzero(labels)
    if flat.size == 0:
        return 0
    order = np.argsort(labels.ravel()[flat], kind="stable")
    flat = flat[order]
    ids = labels.ravel()[flat]
    starts = np.flatnonzero(np.diff(ids, prepend=0))
    rows, cols = np.divmod(flat.astype(np.int64), labels.shape[1])
    ends = [*starts[1:], flat.size]
    return sum(
        run_length(rows[a:b], cols[a:b]) >= CORNER_RUN * spacing
        for a, b in zip(starts, ends, strict=True)
    )


def measure(
    lum: NDArray[np.float32], mask: NDArray[np.bool_], scale: float, spacing_px: float
) -> Curvature:
    """C15 of a face. Arguments as `nodule.orientation.measure`."""
    coarse, face, k = coarse_face(lum, mask, scale, spacing_px)
    coarse, face, spacing = even_face(coarse, face, spacing_px / k)
    inner = agatesim_scipy.erode(face, max(1, math.ceil(MARGIN_SPACINGS * spacing)))
    if not inner.any():
        return Curvature(None, None, None, 0.0)
    ny, nx, coh = direction_field(coarse, spacing, face)
    banded = inner & (coh >= MIN_COHERENCE)
    share = float(banded.sum() / inner.sum())
    if banded.sum() < MIN_PIXELS:
        return Curvature(None, None, None, share)
    u = doubled(ny, nx)
    kappa, read = turn_rate(u, banded, spacing)
    values = kappa[read]
    if values.size < MIN_PIXELS:
        return Curvature(None, None, None, share)
    corners = count_corners(corner_pixels(u, banded, inner, spacing), spacing)
    return Curvature(
        float(np.median(values)), float(np.quantile(values, HIGH_QUANTILE)), corners, share
    )
