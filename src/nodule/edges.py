"""Band edges of a face, a Canny-like edge map (ADR 0004, N2 step 3b step 2).

Marichal et al. [@marichal2025] find tree-ring boundaries as Canny-Devernay
edges; here the edges are found the same way at pixel precision (Canny's
method, cited through Marichal et al., not read). Each grey step between two
bands is an edge, whatever the bands' grey levels, so a band of a middle grey
between a light and a dark one stays a band of its own. A split by sign about
the local mean (the first drafts of C13 and C14) merges it with one of them,
depending on the width of that mean. C13 (`nodule.concentric`) and C14
(`nodule.defects`) read the bands between these edges.

- The face is block-averaged and resampled (linear) so one light-dark couplet
  spans exactly `EDGE_PX` px; a band is then about 6 px wide.
- It is smoothed by a Gaussian of `SMOOTH_SPACINGS`, averaged over the face's
  core only, so the outline adds no step.
- Edge pixels are local maxima of the gradient's size along the gradient
  (non-maximum suppression, neighbours read by linear interpolation).
- Hysteresis: maxima at least `LOW` times the reference are kept where they
  connect (8-connected) to one at least `HIGH` times it. The reference is the
  `REF_QUANTILE` quantile of the maxima's sizes over the core: the steps of
  the face's own strongest bands, so the thresholds follow the face's
  contrast, not a fixed grey level.

Limit: on flat ground the reference falls to the pixel noise, so noise keeps
edges there too (an even face with noise is all short edges). A gate from the
photo's noise (`q.noise`) would remove them. It is not used yet: C13 rejects
such edges, as they do not nest.

Units: here a *spacing* is one light-dark couplet, `bands.COUPLET_SPACINGS`
(2) times the quality gate's band spacing, which is one band (`edge_face`
converts).

Grade (ADR 0004): plausible; the method is borrowed from an analogue (tree
rings). Every parameter here is `effective`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from agatesim import _scipy as agatesim_scipy  # pyright: ignore[reportPrivateUsage]
from numpy.typing import NDArray

from nodule.bands import COUPLET_SPACINGS, OUTLINE_PX

EDGE_PX = 12.0  # effective: px per spacing (couplet) of the edge map
SMOOTH_SPACINGS = 0.125  # effective: Gaussian sigma before the gradient, spacings
REF_QUANTILE = 0.9  # effective: the reference step among the gradient maxima
HIGH = 0.3  # effective: hysteresis, a piece needs one maximum this share of the reference
LOW = 0.1  # effective: hysteresis, the least maximum kept in a piece


@dataclass(frozen=True)
class EdgeFace:
    """A face resampled to `EDGE_PX` px per spacing, and its edges."""

    lum: NDArray[np.float64]
    face: NDArray[np.bool_]
    edges: NDArray[np.bool_]
    spacing: float  # px per spacing, `EDGE_PX`


def resample(
    lum: NDArray[np.float32], mask: NDArray[np.bool_], scale: float, period_px: float
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """`lum` block-averaged by whole px, then resampled (linear) so `period_px`
    full px span `EDGE_PX` px; `mask` (`scale` working px per full px) read
    at each block's centre, then resampled the same way."""
    k = max(1, int(period_px // EDGE_PX))
    h, w = lum.shape[0] // k, lum.shape[1] // k
    block = lum[: h * k, : w * k].astype(np.float64).reshape(h, k, w, k).mean(axis=(1, 3))
    centres_r: NDArray[np.float64] = (np.arange(h, dtype=np.float64) + 0.5) * (k * scale)
    centres_c: NDArray[np.float64] = (np.arange(w, dtype=np.float64) + 0.5) * (k * scale)
    rows = np.clip(centres_r.astype(np.int64), 0, mask.shape[0] - 1)
    cols = np.clip(centres_c.astype(np.int64), 0, mask.shape[1] - 1)
    inside = mask[np.ix_(rows, cols)].astype(np.float64)
    step = period_px / k / EDGE_PX  # block px per edge-map px
    big, wide = int(h / step), int(w / step)
    rr, cc = np.meshgrid(
        (np.arange(big, dtype=np.float64) + 0.5) * step - 0.5,
        (np.arange(wide, dtype=np.float64) + 0.5) * step - 0.5,
        indexing="ij",
    )
    rr, cc = np.clip(rr, 0, h - 1), np.clip(cc, 0, w - 1)
    return agatesim_scipy.bilinear(block, rr, cc), agatesim_scipy.bilinear(inside, rr, cc) >= 0.5


def edge_map(
    field: NDArray[np.float64],
    face: NDArray[np.bool_],
    spacing: float,
    *,
    smooth: float = SMOOTH_SPACINGS,
    high: float = HIGH,
    low: float = LOW,
) -> NDArray[np.bool_]:
    """The edges of `field` inside `face`, `spacing` in px (module docstring)."""
    core = agatesim_scipy.erode(face, max(1, round(OUTLINE_PX * spacing / 4)))
    weight = core.astype(np.float64)
    sigma = max(1.0, smooth * spacing)
    total = agatesim_scipy.gaussian(field * weight, sigma)
    smoothed = total / np.maximum(agatesim_scipy.gaussian(weight, sigma), 1e-12)
    gy, gx = np.gradient(smoothed)
    size = np.where(core, np.hypot(gx, gy), 0.0)
    h, w = size.shape
    rr, cc = np.mgrid[:h, :w].astype(np.float64)
    uy, ux = gy / np.maximum(size, 1e-12), gx / np.maximum(size, 1e-12)
    ahead = agatesim_scipy.bilinear(size, np.clip(rr + uy, 0, h - 1), np.clip(cc + ux, 0, w - 1))
    behind = agatesim_scipy.bilinear(size, np.clip(rr - uy, 0, h - 1), np.clip(cc - ux, 0, w - 1))
    peak = core & (size > ahead) & (size >= behind)
    if not peak.any():
        return peak
    ref = float(np.quantile(size[peak], REF_QUANTILE))
    labels = agatesim_scipy.label(peak & (size >= low * ref), eight_connected=True)
    strong = np.unique(labels[peak & (size >= high * ref)])
    return np.isin(labels, strong[strong > 0])


def edge_face(
    lum: NDArray[np.float32], mask: NDArray[np.bool_], scale: float, spacing_px: float, **kw: float
) -> EdgeFace:
    """The edge map of a face. Arguments as `nodule.orientation.measure`;
    keywords as `edge_map`."""
    field, face = resample(lum, mask, scale, COUPLET_SPACINGS * spacing_px)
    return EdgeFace(field, face, edge_map(field, face, EDGE_PX, **kw), EDGE_PX)
