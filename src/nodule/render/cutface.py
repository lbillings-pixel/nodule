"""Render a measured cut for the set-level compare (ADR 0008 section 2, steps 3-5).

The cut (`SpecimenCut`) is sized by the stand-in probe (`nodule.descriptors`);
only its final image is rendered here, pixel for pixel on the cut's own grid,
so the face mask and quartz phase of the cut still apply.

- **Grid**: the growth-time grid under the face at `grid_mm` (ADR 0007 7c:
  0.2 mm), `depth_mm` deep. L3 renders a slab of that depth (3 mm by default,
  `effective`, ADR 0007 amendment 5). L1 renders the face of a stone whose
  depth below the face is cut off at `depth_mm` (10 mm by default,
  `effective`, an assumption: ADR 0007's prototype depth; its effect is
  measured, not assumed small).
- **Materials**: from a band record passed in, the stone's or a fake's
  (`nodule.render.fakes`), so the stone and its fakes differ only there.
- **Samples per pixel**: the fewest that keep the render's own Monte Carlo
  noise at or below a third of the matched photo's noise (ADR 0008 s. 2.5).
  Two pilot renders at `PILOT_SPP` on a coarser view (larger pixels, so the
  estimate errs high) give the noise; it falls as 1/sqrt(spp). Capped at
  `most_spp` for cost; a capped cut is flagged.
- **Exposure**: one fixed value per run (`effective`; ADR 0008 7b tunes it).

Procedural images: never evidence for a formation mechanism (ADR 0002).
"""

# pyright: basic

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from agatesim.specimen.quality import luminance
from numpy.typing import NDArray

from nodule.render.camera import Light, View, reflectance_factor, render, srgb
from nodule.render.materials import OpticsTable
from nodule.render.record import BandRecord
from nodule.render.scene import pack, upload
from nodule.render.tgrid import GrowthGrid
from nodule.specimen import Specimen, SpecimenCut

PILOT_SPP = 8  # samples per pixel of each pilot render
PILOT_PX = 256  # about this many pixels across the pilot view
NOISE_SHARE = 1 / 3  # render noise at most this share of the photo's (ADR 0008 s. 2.5)
GRID_MM = 0.2  # ADR 0007 7c
L1_DEPTH_MM = 10.0  # effective, assumption (module docstring)
L3_DEPTH_MM = 3.0  # effective, owner (ADR 0007 amendment 5)
MOST_SPP = 128  # cost cap (effective); capped cuts are flagged


@dataclass(frozen=True)
class Rendering:
    """How the cuts of one run are rendered."""

    light: Light
    record: BandRecord  # the stone's, or a fake's
    depth_mm: float
    optics: OpticsTable = field(default_factory=OpticsTable)
    grid_mm: float = GRID_MM
    exposure: float = 1.0  # effective, one per run
    most_spp: int = MOST_SPP
    undegraded: bool = False  # F5: no photo-like blur or noise (ADR 0008 amendment 4)

    @classmethod
    def reflected(cls, record: BandRecord, **kw: Any) -> Rendering:
        """L1, the dome, for reflected cut faces (ADR 0008 7c)."""
        return cls(Light("dome"), record, kw.pop("depth_mm", L1_DEPTH_MM), **kw)

    @classmethod
    def transmitted(cls, record: BandRecord, **kw: Any) -> Rendering:
        """L3, the back light, for slabs in transmitted light (ADR 0008 7c)."""
        return cls(Light("back"), record, kw.pop("depth_mm", L3_DEPTH_MM), **kw)


@dataclass(frozen=True)
class CutRender:
    rgb: NDArray[np.uint8]  # rows and columns as the cut's arrays
    spp: int
    noise: float  # the render's own noise (grey levels), estimated from the pilots
    capped: bool  # spp hit `most_spp`, so `noise` may exceed the share
    fill_s: float
    render_s: float


def view_of(cut: SpecimenCut) -> View:
    """The camera view whose pixel centres are the cut's sample points."""
    p = cut.spacing_mm
    return View(float(cut.s[0]) - p / 2, float(cut.t[0]) - p / 2, p, len(cut.s), len(cut.t))


def to_rgb(image: NDArray[np.float64], light: Light, exposure: float) -> NDArray[np.uint8]:
    """8-bit sRGB in the cut's row order (`srgb` flips rows for display)."""
    return srgb(reflectance_factor(image, light), exposure)[::-1]


def coarse_mask(inside: NDArray[np.bool_], k: int, ny: int, nx: int) -> NDArray[np.bool_]:
    """Coarse pixels lying wholly inside the face."""
    return inside[: ny * k, : nx * k].reshape(ny, k, nx, k).all((1, 3))


def spp_for(pilot_noise: float, photo_noise: float, most: int) -> int:
    """Fewest samples per pixel keeping the noise at or below NOISE_SHARE of the photo's."""
    target = NOISE_SHARE * photo_noise
    if not target > 0:
        return most
    want = math.ceil(PILOT_SPP * (pilot_noise / target) ** 2)
    return int(min(max(want, 1), most))


def render_cut(
    specimen: Specimen,
    cut: SpecimenCut,
    rendering: Rendering,
    photo_noise: float,
    seed: int,
) -> CutRender:
    """The cut rendered under `rendering`'s light, with as many samples per pixel
    as `photo_noise` (grey levels) asks (module docstring). GPU."""
    r = rendering
    t0 = time.perf_counter()
    grid = GrowthGrid.fill(specimen, cut.plane, r.grid_mm, r.depth_mm)
    dev = upload(pack(specimen, grid))
    fill_s = time.perf_counter() - t0
    mats = r.optics.materials(r.record)

    def shot(view: View, spp: int, seed: int) -> NDArray[np.uint8]:
        image = render(dev, view, mats, r.light, spp=spp, seed=seed, depth_mm=r.depth_mm).image
        return to_rgb(image, r.light, r.exposure)

    view = view_of(cut)
    k = max(1, round(view.nx / PILOT_PX))
    pv = view.coarser(k)
    t1 = time.perf_counter()
    a, b = (luminance(shot(pv, PILOT_SPP, s)).astype(np.float64) for s in (seed + 1, seed + 2))
    face = coarse_mask(cut.inside, k, pv.ny, pv.nx)
    diff = (a - b)[face] if face.any() else (a - b).ravel()
    pilot_noise = float(np.std(diff) / math.sqrt(2))
    spp = spp_for(pilot_noise, photo_noise, r.most_spp)
    rgb = shot(view, spp, seed)
    render_s = time.perf_counter() - t1
    noise = pilot_noise * math.sqrt(PILOT_SPP / spp)
    capped = noise > NOISE_SHARE * photo_noise + 1e-9
    return CutRender(rgb, spp, noise, capped, fill_s, render_s)
