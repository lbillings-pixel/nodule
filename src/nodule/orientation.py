"""C16, orientation order of a face's bands (ADR 0004, N2 step 3b step 2).

All four parts read the band direction of `nodule.bands.direction_field`
(the structure tensor [@bigun1987], method only: its paper is cited, not read),
on the coarse copy of the face that the band profiles use
(`bands.coarse_face`: about `bands.FIELD_SPACING_PX` px per couplet), so
a face gives the same values at any resolution that shows its bands. Gradients
next to the outline are left out, so the step from stone to background adds no
direction, and only pixels at least `MARGIN_SPACINGS` inside the outline are
judged, as for C6-C11. A pixel is *banded* where its coherence reaches
agatesim's `MIN_COHERENCE`, the level at which agatesim's profiles skip a
place as having no clear band direction.

- **coherence**: the mean structure-tensor coherence over the judged face (0
  for no preferred direction, 1 for perfectly parallel bands at the scale of
  one spacing).
- **order**: Herman's order parameter, S = 2<cos^2 phi> - 1, over the banded
  pixels, phi measured from their mean direction, so S = |<exp(2i theta)>|
  [@murphy2015]. 0 for directions spread evenly (concentric rings), 1 for one
  direction (straight stripes). Murphy et al. find it "depends strongly" on the
  area sampled; here the area is the whole face.
- **correlation length**: C(r) = <cos 2(theta - theta')> over pairs of banded
  pixels r apart [@murphy2015]; kappa is where C first falls below 1/e
  (Murphy et al. fit exp(-r/kappa); the first crossing is the same for an
  exponential and is not pulled by the modulation at the band period they
  report). Given relative to the face's equivalent diameter D = sqrt(4A/pi).
  Pairs thin out past D/2, so a C still above 1/e there is reported as D/2
  with `censored` set: "at least half the face".
- **parallel share**: the share of judged banded pixels whose bands run
  straight for `PARALLEL_LENGTH` of the face: on a straight segment that long
  along the pixel's band, centred on it, all `PARALLEL_SAMPLES` points read are
  banded and none turns from the centre's direction by more than
  `PARALLEL_TURN_DEG`. Seen from the end of such a segment, a circle of radius r
  has turned by atan(length / 2r); that is within 10 degrees only for r of at
  least 0.71 face diameters, so wall-parallel rings that fill the face never
  pass. Stripes pass wherever the segment fits in the face. The sides of a
  fortification angle can pass where they run straight that far. Pixels whose
  segment leaves the judged face are not judged.

Units: here a *spacing* is one light-dark couplet, `bands.COUPLET_SPACINGS`
(2) times the quality gate's band spacing, which is one band (`measure`
converts). The tests and the values below were measured in that unit.

Grade (ADR 0004): plausible; the methods are borrowed from an analogue
(block-copolymer stripes). Every parameter here is `effective`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from agatesim import _scipy as agatesim_scipy  # pyright: ignore[reportPrivateUsage]
from agatesim.specimen.quality import MIN_COHERENCE
from agatesim.specimen.shape import MARGIN_SPACINGS
from numpy.typing import NDArray

from nodule.bands import COUPLET_SPACINGS, coarse_face, direction_field

PARALLEL_LENGTH = 0.25  # effective: segment length, face diameters
PARALLEL_TURN_DEG = 10.0  # effective: most turn from the centre to any sample, degrees
PARALLEL_SAMPLES = 9  # effective: points read along the segment, ends included
MAX_LAG = 0.5  # effective: C(r) is read out to this many face diameters
MIN_PIXELS = 50  # effective: banded coarse pixels needed for order and correlation


@dataclass(frozen=True)
class Order:
    """C16 of one face; None where it could not be judged."""

    coherence: float | None
    order: float | None
    correlation: float | None  # kappa / D
    censored: bool  # C(r) still above 1/e at MAX_LAG: correlation is a lower bound
    parallel: float | None
    banded: float  # share of the judged face that is banded


def doubled(ny: NDArray[np.float64], nx: NDArray[np.float64]) -> NDArray[np.complex128]:
    """exp(2i theta) of the band normal (column part nx, row part ny)."""
    return (nx * nx - ny * ny) + 2j * nx * ny


def correlation(
    u: NDArray[np.complex128], banded: NDArray[np.bool_], most: float
) -> tuple[float, bool]:
    """kappa (px) where C(r) first falls below 1/e, and whether it was cut
    off at `most` px (module docstring)."""
    h, w = int(u.shape[0]), int(u.shape[1])
    shape = (2 * h, 2 * w)  # zero padding: no wrap-round pairs
    z = np.fft.fft2(np.where(banded, u, 0), shape)
    b = np.fft.fft2(banded.astype(np.float64), shape)
    pairs_z: NDArray[np.float64] = np.real(np.fft.ifft2(z * np.conj(z)))
    pairs_b: NDArray[np.float64] = np.real(np.fft.ifft2(b * np.conj(b)))
    ir, ic = np.arange(shape[0]), np.arange(shape[1])
    dr, dc = np.minimum(ir, shape[0] - ir), np.minimum(ic, shape[1] - ic)  # |lag|, px
    r = np.rint(np.hypot(dr[:, None], dc[None, :])).astype(np.int64)
    top = math.floor(most)
    keep = r <= top
    num = np.bincount(r[keep], weights=pairs_z[keep], minlength=top + 1).astype(np.float64)
    den = np.bincount(r[keep], weights=pairs_b[keep], minlength=top + 1).astype(np.float64)
    c = np.where(den > 0.5, num / np.maximum(den, 1e-12), np.nan)
    level = math.exp(-1.0)
    for i in range(1, top + 1):
        if np.isfinite(c[i]) and c[i] < level:
            before = float(c[i - 1])
            return (i - 1) + (before - level) / (before - float(c[i])), False
    return float(top), True


def parallel_share(
    u: NDArray[np.complex128],
    ny: NDArray[np.float64],
    nx: NDArray[np.float64],
    banded: NDArray[np.bool_],
    length: float,
) -> float | None:
    """Share of banded pixels whose bands run straight for `length` px
    (module docstring); None if no pixel could be judged."""
    rows, cols = np.nonzero(banded)
    if rows.size == 0:
        return None
    tr, tc = nx[rows, cols], -ny[rows, cols]  # band tangent (row, column)
    h, w = banded.shape
    centre = u[rows, cols]
    most = math.cos(2.0 * math.radians(PARALLEL_TURN_DEG))  # on doubled angles
    judged = np.ones(rows.size, dtype=np.bool_)
    straight = np.ones(rows.size, dtype=np.bool_)
    for f in np.linspace(-0.5, 0.5, PARALLEL_SAMPLES):
        pr = np.rint(rows + f * length * tr).astype(np.int64)
        pc = np.rint(cols + f * length * tc).astype(np.int64)
        ok = (pr >= 0) & (pr < h) & (pc >= 0) & (pc < w)
        pr, pc = np.clip(pr, 0, h - 1), np.clip(pc, 0, w - 1)
        judged &= ok & banded[pr, pc]
        straight &= np.real(centre * np.conj(u[pr, pc])) >= most
    if not judged.any():
        return None
    return float((straight & judged).sum() / judged.sum())


def measure(
    lum: NDArray[np.float32], mask: NDArray[np.bool_], scale: float, spacing_px: float
) -> Order:
    """C16 of a face. Arguments as `nodule.bands.measure`: `lum` is brightness
    at full resolution, `mask` the face at the working size (`scale` working px
    per full px), `spacing_px` the quality gate's band spacing (full px, one
    band; read here as a couplet of `COUPLET_SPACINGS` bands)."""
    period_px = COUPLET_SPACINGS * spacing_px
    coarse, face, k = coarse_face(lum, mask, scale, period_px)
    spacing = period_px / k
    inner = agatesim_scipy.erode(face, max(1, math.ceil(MARGIN_SPACINGS * spacing)))
    if not inner.any():
        return Order(None, None, None, False, None, 0.0)
    ny, nx, coh = direction_field(coarse, spacing, face)
    banded = inner & (coh >= MIN_COHERENCE)
    share = float(banded.sum() / inner.sum())
    mean_coherence = float(coh[inner].mean())
    if banded.sum() < MIN_PIXELS:
        return Order(mean_coherence, None, None, False, None, share)
    u = doubled(ny, nx)
    order = float(abs(u[banded].mean()))
    diameter = math.sqrt(4.0 * float(face.sum()) / math.pi)
    kappa, censored = correlation(u, banded, MAX_LAG * diameter)
    parallel = parallel_share(u, ny, nx, banded, PARALLEL_LENGTH * diameter)
    return Order(mean_coherence, order, kappa / diameter, censored, parallel, share)
