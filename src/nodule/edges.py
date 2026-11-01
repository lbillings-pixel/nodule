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
- Noise gate: on flat ground the reference falls to the pixel noise, and
  without a gate an even face with noise is all short edges. So neither
  threshold falls below `NOISE_HIGH` (seed) and `NOISE_LOW` (kept) times the
  spread of one gradient component that white noise at the quality gate's
  level (`q.noise`) leaves after the same block averaging, resampling and
  smoothing (`noise_gradient`). That spread is read off a fixed draw of
  `NOISE_PATCH` px (1-2% apart between draws), as the linear resampling
  makes it awkward to work out exactly. It is Canny's noise-based threshold
  (cited through Marichal et al., not read). The block averaging cuts the
  noise by the block's side, so the floor halves at twice the px per spacing.

Measured 2026-10-31: an even face with noise of 1.5 or 5 grey levels keeps no
edges at 16 or 32 px per spacing (about 16,800 edge px without the gate);
rings whose swing is twice the noise keep at least 90% of their noise-free
edges. On three random cuts of the `n2-slice` preset made to look like
library photos (`nodule.descriptors`, seed 0), the face's own thresholds lie
18-68 times above the noise floor, so the gate removes nothing and C13 and
C14 are unchanged there; it acts on flat ground and on bands faint against
the noise. 0.01 s per face.

Limit: the gate takes the noise as white. A phone's processing (demosaicing,
denoising, JPEG) leaves noise correlated between neighbouring px, which the
block averaging cuts by less than the block's side, so on such photos the
floor is low and some noise edges may remain.

Units: here a *spacing* is one light-dark couplet, `bands.COUPLET_SPACINGS`
(2) times the quality gate's band spacing, which is one band (`edge_face`
converts).

Grade (ADR 0004): plausible; the method is borrowed from an analogue (tree
rings). Every parameter here is `effective`.
"""

from __future__ import annotations

import math
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
NOISE_HIGH = 5.0  # effective: noise gate, a piece needs one maximum this many noise spreads
NOISE_LOW = 3.0  # effective: noise gate, the least maximum kept, in noise spreads
NOISE_PATCH = 192  # edge-map px per side of the noise draw that calibrates the gate
NOISE_SEED = 0  # its fixed draw


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
    k = block_px(period_px)
    h, w = lum.shape[0] // k, lum.shape[1] // k
    block = lum[: h * k, : w * k].astype(np.float64).reshape(h, k, w, k).mean(axis=(1, 3))
    centres_r: NDArray[np.float64] = (np.arange(h, dtype=np.float64) + 0.5) * (k * scale)
    centres_c: NDArray[np.float64] = (np.arange(w, dtype=np.float64) + 0.5) * (k * scale)
    rows = np.clip(centres_r.astype(np.int64), 0, mask.shape[0] - 1)
    cols = np.clip(centres_c.astype(np.int64), 0, mask.shape[1] - 1)
    inside = mask[np.ix_(rows, cols)].astype(np.float64)
    step = period_px / k / EDGE_PX  # block px per edge-map px
    return onto_grid(block, step), onto_grid(inside, step) >= 0.5


def block_px(period_px: float) -> int:
    """Full px per side of the blocks `resample` averages."""
    return max(1, int(period_px // EDGE_PX))


def onto_grid(block: NDArray[np.float64], step: float) -> NDArray[np.float64]:
    """`block` read (linear) every `step` of its px, at the px centres."""
    h, w = block.shape
    big, wide = int(h / step), int(w / step)
    rr, cc = np.meshgrid(
        (np.arange(big, dtype=np.float64) + 0.5) * step - 0.5,
        (np.arange(wide, dtype=np.float64) + 0.5) * step - 0.5,
        indexing="ij",
    )
    rr, cc = np.clip(rr, 0, h - 1), np.clip(cc, 0, w - 1)
    return agatesim_scipy.bilinear(block, rr, cc)


def noise_gradient(noise: float, period_px: float, smooth: float = SMOOTH_SPACINGS) -> float:
    """The spread of one gradient component that white pixel noise of `noise`
    grey levels (per full px) leaves in the edge map of a face whose couplet
    spans `period_px` full px (module docstring, noise gate)."""
    if noise <= 0:
        return 0.0
    k = block_px(period_px)
    step = period_px / k / EDGE_PX
    # A block's mean of k * k px of white noise is white noise of noise / k.
    # A fixed draw: it calibrates a threshold and is not part of an image.
    rng = np.random.default_rng(NOISE_SEED)
    side = math.ceil(NOISE_PATCH * step) + 2
    field = onto_grid(rng.normal(0.0, noise / k, (side, side)), step)
    smoothed = agatesim_scipy.gaussian(field, max(1.0, smooth * EDGE_PX))
    gy, gx = np.gradient(smoothed)
    m = NOISE_PATCH // 8  # clear of the patch's edge
    size: NDArray[np.float64] = np.hypot(gx, gy)[m:-m, m:-m]
    return float(np.sqrt(0.5 * np.mean(size**2)))  # per component; the noise has mean 0


def edge_map(
    field: NDArray[np.float64],
    face: NDArray[np.bool_],
    spacing: float,
    *,
    smooth: float = SMOOTH_SPACINGS,
    high: float = HIGH,
    low: float = LOW,
    floor: float = 0.0,
) -> NDArray[np.bool_]:
    """The edges of `field` inside `face`, `spacing` in px (module docstring);
    `floor` is the noise's gradient spread (`noise_gradient`), 0 for no gate."""
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
    keep, seed = max(low * ref, NOISE_LOW * floor), max(high * ref, NOISE_HIGH * floor)
    labels = agatesim_scipy.label(peak & (size >= keep), eight_connected=True)
    strong = np.unique(labels[peak & (size >= seed)])
    return np.isin(labels, strong[strong > 0])


def edge_face(
    lum: NDArray[np.float32],
    mask: NDArray[np.bool_],
    scale: float,
    noise: float,
    spacing_px: float,
    **kw: float,
) -> EdgeFace:
    """The edge map of a face. Arguments as `nodule.bands.measure` (`noise`
    0 for no gate); keywords as `edge_map`."""
    period = COUPLET_SPACINGS * spacing_px
    field, face = resample(lum, mask, scale, period)
    floor = noise_gradient(noise, period, kw.get("smooth", SMOOTH_SPACINGS))
    return EdgeFace(field, face, edge_map(field, face, EDGE_PX, floor=floor, **kw), EDGE_PX)
