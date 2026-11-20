"""Render a measured cut for the set-level compare (ADR 0008 section 2, steps 3-5,
as amended: amendment 5).

The cut (`SpecimenCut`) is sized by the stand-in probe (`nodule.descriptors`);
only its final image is rendered here, pixel for pixel on the cut's own grid,
so the face mask and quartz phase of the cut still apply.

- **Grid**: the growth-time grid under the face at `grid_mm` (ADR 0007 7c:
  0.2 mm), `depth_mm` deep. L3 renders a slab of that depth (3 mm by default,
  `effective`, ADR 0007 amendment 5). L1 renders the face of a stone whose
  depth below the face is cut off at `depth_mm` (2 mm by default since
  ADR 0010 10b, `effective`; on four n2-slice cuts 2 mm changed the image by
  0.14 grey levels against 10 mm and saved ~22% of the time; earlier, 20 mm
  changed one cut by 0.02 grey levels against 9.5 between seeds). Both checks had the
  slab's smooth bottom over black air, which reflects much of the light reaching it
  back up; `bottom="black"` instead loses it, as into the rest of a stone (ADR 0014,
  `effective`). Reflected renderings default to "black" since ADR 0014 closed; the
  back light keeps "air".
- **Materials**: from a band record passed in, the stone's or a fake's
  (`nodule.render.fakes`), so the stone and its fakes differ only there.
- **Samples and noise**: samples are added in batches, alternately to two
  halves drawn on independent seeds. The image is their mean; their
  difference measures the image's own noise (grey levels of luminance, per
  pixel, white) at no extra cost. How many samples a cut needs depends on
  the photo's blur, which only the measuring code finds, so
  `nodule.descriptors` asks for more (`CutRenderer.more`) until the render's
  noise after that blur is below `NOISE_SHARE` of the photo's, or `most_spp`
  is reached.
- **Exposure**: one fixed value per run (`effective`; ADR 0008 7b tunes it).
- **The owner's look** (`Rendering.look`, ADR 0019 B3): the dome without the face's
  mirror reflection (as a polariser or dark surround gives), exposed so the face's mean
  lightness equals the photo-compare rendering's (`look_exposure`), over a grey matte
  bottom (an index-matched Lambertian reflector of LOOK_BOTTOM_ALBEDO, so clear quartz
  no longer renders black), LOOK_DEPTH_MM below the face so clear bands show the stone
  under them (ADR 0019 s.6.10). For the eye only; the compare's renderings are unchanged.
- **One fill, several lights** (`CutRenderer.on`): a grid filled for the deepest slab
  serves every rendering of the same record and optics, e.g. the look and a backlit slice.
- **Open cavities** (ADR 0019 A1): a void in the stone renders as its lining's material
  plus a stand-in absorption, `Rendering.void_mua`: VOID_MUA under the dome, so the
  cavity reads dark, as a lined geode's face often photographs; 0 under the back light,
  so a slice's hole passes the light box. Not air: no refraction at the cavity's lining
  (A2). The look's lightness is matched over the stone's face without the void. A milky
  lining's stand-in void (ADR 0019 s.6.12) takes its last step's material without the
  milk, where the trilinear void mask passes 1/2 (`CutRenderer._void_mask`,
  `Scene.voidf`), so a slice's hole stays clear and the growth time stays the lining's.

Procedural images: never evidence for a formation mechanism (ADR 0002).
"""

# pyright: basic

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from agatesim.specimen.quality import luminance
from numpy.typing import NDArray

from nodule.render import voidgpu
from nodule.render.camera import (
    N_CHALCEDONY,
    Flag,
    Light,
    View,
    reflectance_factor,
    render,
    rolloff,
    srgb,
)
from nodule.render.materials import OpticsTable
from nodule.render.record import BandRecord, void_points
from nodule.render.scene import pack, upload, with_void
from nodule.render.shift import ShiftArrays, bake_drift, variation_of
from nodule.render.shift import upload as shift_upload
from nodule.render.tgrid import GrowthGrid, Window, to_world
from nodule.render.voidbox import VoidBox, hollow_nodes, under_slab
from nodule.specimen import Specimen, SpecimenCut

GRID_MM = 0.2  # ADR 0007 7c
L1_DEPTH_MM = 2.0  # effective, ADR 0010 10b (module docstring)
# The look's grey matte bottom (owner's pick 2026-11-09, ADR 0019 s.6.5): `effective`;
# 0.35 washed fine bands out, 0.15 lights clear quartz and keeps them
LOOK_BOTTOM_ALBEDO = 0.15
# The look's slab: clear bands show the stone up to ~10 mm below (diffusion length 7-18 mm,
# ADR 0014); 5 mm gives most of that while quartz centres keep some tone (owner's pick
# 2026-11-18, ADR 0019 s.6.10): `effective`
LOOK_DEPTH_MM = 5.0
L3_DEPTH_MM = 3.0  # effective, owner (ADR 0007 amendment 5)
# Matches the reflected photos' median lightness (L* 62); effective, a camera
# setting, not a property of the stone (ADR 0013). Refitted for the ADR 0014 optics on
# four default (2 mm) cuts (4m; was 2.2). L3 keeps 1.0 until calibrated.
REFLECTED_EXPOSURE = 1.54
# The open cavity's stand-in absorption under reflected light (ADR 0019 A1), mm^-1:
# `effective`. Over the clear lining (mus 0.05) it leaves an albedo under 1 %; it is
# added to the majorant (the bands' largest mu_t is ~16-28), so a larger value slows
# every sample of a stone with a void.
VOID_MUA = 10.0
FIRST_SPP = 256  # samples per pixel before the photo's blur is known (effective)
MOST_SPP = 1024  # cost cap (effective); cuts left noisier than the photo are flagged
# The render's noise after the photo's blur is brought below this share of the photo's,
# leaving room for the photo's noise to be added on top (effective; ADR 0008 amendment 5).
NOISE_SHARE = 0.9


@dataclass(frozen=True)
class Rendering:
    """How the cuts of one run are rendered."""

    light: Light
    record: BandRecord  # the stone's, or a fake's
    depth_mm: float
    optics: OpticsTable = field(default_factory=OpticsTable)
    grid_mm: float = GRID_MM
    exposure: float = 1.0  # effective, one per run and light (ADR 0013)
    first_spp: int = FIRST_SPP
    most_spp: int = MOST_SPP
    undegraded: bool = False  # F5: no photo-like blur or noise (ADR 0008 amendment 4)
    # below the slab (module docstring; ADR 0014); "matte": the black bottom made an
    # index-matched Lambertian reflector of bottom_albedo (the look's grey bottom, ADR 0019)
    bottom: Literal["air", "black", "matte"] = "air"
    bottom_albedo: tuple[float, float, float] = (0.0, 0.0, 0.0)  # effective
    # the drift along the bands (ADR 0016 s.9): baked once per cut on the growth-time grid,
    # or worked out at each sample ("exact", the reference); lenses are always exact
    drift: Literal["baked", "exact"] = "baked"
    # the face's mirror reflection of the light (dome, or a disc holding the normal);
    # False: left out, the owner's look (ADR 0019 B3)
    glare: bool = True
    # True: exposure per render by `look_exposure` (the photo compare's face lightness)
    match_lightness: bool = False
    # back light only: highlight roll-off from this knee, the bare light box white
    # (`camera.rolloff`; the backlit view's, ADR 0019 s.6.6); None: plain clipping
    rolloff_knee: float | None = None
    # the open cavity's stand-in absorption (module docstring, ADR 0019 A1), mm^-1
    void_mua: float = VOID_MUA
    # a crystal-lined hollow: the stand-in, or drawn as air when the renderer is given the
    # stone's void box (ADR 0019 s.6.12 step 4; the dome without glare, black or matte bottom)
    hollow: Literal["stand-in", "air"] = "stand-in"

    def __post_init__(self) -> None:
        if self.hollow not in ("stand-in", "air"):
            raise ValueError(f"hollow must be 'stand-in' or 'air', not {self.hollow!r}")
        if self.hollow == "air" and (
            self.light.kind not in ("dome", "disc")
            or self.glare
            or self.bottom not in ("black", "matte")
        ):
            raise ValueError("the hollow as air: dome or lamp, no glare, black or matte bottom")
        if self.match_lightness and self.light.kind != "dome":
            raise ValueError("lightness is matched to the dome's photo-compare rendering")
        if self.bottom not in ("air", "black", "matte"):
            raise ValueError(f"bottom must be 'air', 'black' or 'matte', not {self.bottom!r}")
        if (self.bottom == "matte") != any(a > 0.0 for a in self.bottom_albedo):
            raise ValueError("a matte bottom, and only it, has a nonzero bottom_albedo")
        if self.bottom == "matte" and self.light.kind != "dome":
            raise ValueError("a matte bottom is for the dome only")
        if self.rolloff_knee is not None:
            if self.light.kind != "back":
                raise ValueError("the highlight roll-off is for the back light only")
            if not 0.0 < self.rolloff_knee < 1.0 < self.exposure:
                raise ValueError("roll-off needs 0 < knee < 1 < exposure")
        if self.drift not in ("baked", "exact"):
            raise ValueError(f"drift must be 'baked' or 'exact', not {self.drift!r}")
        if self.void_mua < 0.0:
            raise ValueError("void_mua must be >= 0")
        if self.bottom == "black" and self.light.kind == "back":
            raise ValueError("the back light shines through the bottom: no black bottom")

    @property
    def flags(self) -> int:
        """Kernel flags of this rendering."""
        return int(Flag.BLACK_BOTTOM) if self.bottom in ("black", "matte") else 0

    @classmethod
    def reflected(cls, record: BandRecord, **kw: Any) -> Rendering:
        """L1, the dome, for reflected cut faces (ADR 0008 7c)."""
        kw.setdefault("exposure", REFLECTED_EXPOSURE)
        kw.setdefault("bottom", "black")  # ADR 0014
        return cls(Light("dome"), record, kw.pop("depth_mm", L1_DEPTH_MM), **kw)

    @classmethod
    def look(cls, record: BandRecord, **kw: Any) -> Rendering:
        """The owner's look (ADR 0019 B3): the dome of `reflected` without the face's
        mirror reflection, at the lightness `reflected` gives, over a grey matte bottom
        of LOOK_BOTTOM_ALBEDO (ADR 0019 s.6.5: clear quartz rendered black over black),
        LOOK_DEPTH_MM deep (s.6.10)."""
        kw |= {"glare": False, "match_lightness": True}
        kw.setdefault("depth_mm", LOOK_DEPTH_MM)
        if "bottom" not in kw:
            kw["bottom"] = "matte"
            kw.setdefault("bottom_albedo", (LOOK_BOTTOM_ALBEDO,) * 3)
        return cls.reflected(record, **kw)

    @classmethod
    def lamp(cls, record: BandRecord, light: Light, **kw: Any) -> Rendering:
        """A small lamp (a disc, L2 of ADR 0007 7b) over the look's slab, for the sparkle
        view (ADR 0019 s.6.12 step 5): no face glare, as the look; the black bottom, since
        the matte one takes no shadow rays. Added to the look, not shown alone."""
        if light.kind != "disc":
            raise ValueError("the sparkle view's lamp is a disc")
        kw |= {"glare": False}
        kw.setdefault("depth_mm", LOOK_DEPTH_MM)
        kw.setdefault("bottom", "black")
        return cls(light, record, kw.pop("depth_mm"), **kw)

    @classmethod
    def transmitted(cls, record: BandRecord, **kw: Any) -> Rendering:
        """L3, the back light, for slabs in transmitted light (ADR 0008 7c); an open
        cavity passes the light, as a hole through the slice does (void_mua 0)."""
        kw.setdefault("void_mua", 0.0)
        return cls(Light("back"), record, kw.pop("depth_mm", L3_DEPTH_MM), **kw)


def window_of(cut: SpecimenCut, pad_mm: float = 0.0) -> Window:
    """The part of the plane a cut's pixels cover (mm), padded by pad_mm."""
    p = cut.spacing_mm / 2 + pad_mm
    return (float(cut.s[0]) - p, float(cut.s[-1]) + p), (float(cut.t[0]) - p, float(cut.t[-1]) + p)


def view_of(cut: SpecimenCut) -> View:
    """The camera view whose pixel centres are the cut's sample points."""
    p = cut.spacing_mm
    return View(float(cut.s[0]) - p / 2, float(cut.t[0]) - p / 2, p, len(cut.s), len(cut.t))


def to_rgb(
    image: NDArray[np.float64], light: Light, exposure: float, knee: float | None = None
) -> NDArray[np.uint8]:
    """8-bit sRGB in the cut's row order (`srgb` flips rows for display). knee: the
    highlight roll-off (`camera.rolloff`), the bare light (factor 1) mapping to white."""
    x = reflectance_factor(image, light) * exposure
    if knee is not None:
        x = rolloff(x, knee, exposure)
    return srgb(x, 1.0)[::-1]


F0 = ((N_CHALCEDONY - 1) / (N_CHALCEDONY + 1)) ** 2  # the face's reflectance at normal incidence


def mean_lightness(factor: NDArray[np.float64], face: NDArray[np.bool_], exposure: float) -> float:
    """Mean sRGB-encoded luminance (0-1) over the face of a reflectance-factor image
    (cut row order)."""
    y = srgb(factor, exposure)[::-1].astype(np.float64) / 255
    lum = 0.2126 * y[..., 0] + 0.7152 * y[..., 1] + 0.0722 * y[..., 2]
    return float(lum[face].mean() if face.any() else lum.mean())


def look_exposure(bare: NDArray[np.float64], face: NDArray[np.bool_]) -> float:
    """Exposure giving a dome render without its face glare (reflectance factor `bare`)
    the face's mean lightness of the same render with the glare (F0 added) at
    REFLECTED_EXPOSURE: the photo compare's lightness, without its glare (ADR 0019 B3)."""
    target = mean_lightness(bare + F0, face, REFLECTED_EXPOSURE)
    lo, hi = 0.01, 1000.0
    for _ in range(50):  # bisection in log exposure; lightness rises with exposure
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if mean_lightness(bare, face, mid) < target else (lo, mid)
    return math.sqrt(lo * hi)


def half_noise(a: NDArray[np.uint8], b: NDArray[np.uint8], face: NDArray[np.bool_]) -> float:
    """Noise (grey levels) of the mean of two equal, independent renders, from
    their difference over the face: std(a - b) / 2."""
    d = luminance(a).astype(np.float64) - luminance(b)
    return float(np.std(d[face] if face.any() else d) / 2)


def spp_for(noise: float, spp: int, wanted: float, most: int) -> int:
    """Total samples per pixel bringing `noise` (at `spp`) to `wanted`, as noise
    falls as 1/sqrt(spp); even (two halves), at least `spp`, at most `most`."""
    if not wanted > 0:
        return most
    need = math.ceil(spp * (noise / wanted) ** 2)
    need += need % 2
    return int(min(max(need, spp), most))


class CutRenderer:
    """One cut under one rendering: the grid filled once, samples added in batches
    (module docstring). GPU."""

    def __init__(
        self,
        specimen: Specimen,
        cut: SpecimenCut,
        rendering: Rendering,
        rng: np.random.Generator,
        fill_depth_mm: float | None = None,
        box: VoidBox | None = None,
    ) -> None:
        """fill_depth_mm: fill the grid this deep (default the rendering's slab), so
        renderings with deeper slabs can share it (`on`). box: the stone's void box, for a
        rendering with hollow="air" (the grid then continues down under the hollow)."""
        r = self.rendering = rendering
        t0 = time.perf_counter()
        depth = r.depth_mm if fill_depth_mm is None else fill_depth_mm
        if depth < r.depth_mm:
            raise ValueError("the grid must be filled at least as deep as the slab")
        # the grid covers the view, padded by the slab's depth for light that wanders
        # sideways (a close-up's grid then covers its window, not the whole stone)
        # with the layers' place-to-place part baked as absorption in the same pass (ADR 0019)
        window = window_of(cut, depth)
        self._box = box if r.hollow == "air" else None
        deep, keep = None, None
        if self._box is not None:  # the view's grid continued down under the hollow
            deep, keep = under_slab(self._box, cut.plane, window, r.grid_mm)
        grid = GrowthGrid.fill(
            specimen, cut.plane, r.grid_mm, depth, window, r.optics, deep_mm=deep, keep=keep
        )
        self._fill_depth = depth
        self._grid = grid
        # the hollow's own void nodes take no stand-in when it is drawn as air
        self._air = None
        if self._box is not None and grid.void is not None:
            k, j, i = np.nonzero(grid.void)
            p = to_world(cut.plane, grid.s[i], grid.t[j], grid.w[k])
            self._air = np.zeros_like(grid.void)
            self._air[k, j, i] = hollow_nodes(self._box, p)
        packed = self._packed = pack(
            specimen, grid, absorb=grid.absorb, void_mua=r.void_mua, air=self._air,
            voidf=self._void_mask(r),
        )  # fmt: skip
        self._dev = upload(packed)
        self._void = None if self._box is None else voidgpu.upload(voidgpu.pack_box(self._box))
        # materials that vary along the bands (ADR 0016), when the stone has them and the
        # record marks steps that take them (a fake's record carries them with its runs)
        variation = variation_of(specimen) if r.record.shifts.any() else None
        host = ShiftArrays.of(variation) if variation is not None else None
        self._mats = r.optics.materials(r.record, host)
        self._shift = None
        if host is not None:
            baked = host.drift is not None and r.drift == "baked"
            dgrid = bake_drift(host, grid, packed.grid) if baked else None
            self._shift = shift_upload(host, self._mats, dgrid)
        self.fill_s = time.perf_counter() - t0  # the grid, and the drift baked on it
        self._view = view_of(cut)
        # the stone's face without its open cavity: for the look's lightness and the noise
        self._face = cut.inside & ~void_points(cut.fields, cut.inside)
        self._rng = rng
        self._sums = [np.zeros((self._view.ny, self._view.nx, 3)) for _ in range(2)]
        self.spp = 0  # in all, both halves
        self.render_s = 0.0
        self.rgb: NDArray[np.uint8] = np.zeros((self._view.ny, self._view.nx, 3), np.uint8)
        self.noise = math.inf  # of `rgb`, grey levels per pixel, before any blur
        self.exposure = r.exposure  # as used for `rgb` (the look sets it per render)
        self.crossings = self.guessed = 0.0  # the hollow's wall, last batch (camera.Rendered)

    def _void_mask(self, rendering: Rendering) -> NDArray[np.float32] | None:
        """The void nodes (1, else 0) for a stand-in hollow inside a milky lining: there the
        camera takes the lining's last material without the milk (`Scene.voidf`, the
        profile's last entry; ADR 0019 s.6.12), so a slice's hole stays clear. None for a
        hollow drawn as air (its stone near the wall keeps the lining's material) or a stone
        without milk (bit for bit as before)."""
        void = self._grid.void
        if rendering.optics.milk_mus <= 0 or rendering.hollow == "air" or void is None:
            return None
        return void.astype(np.float32)

    def on(self, rendering: Rendering) -> CutRenderer:
        """A renderer for another rendering on this one's filled grid (no new fill):
        same record, optics, grid spacing and drift; a slab no deeper than the fill.
        Starts with no samples; shares this one's random generator."""
        r = self.rendering
        if rendering.record is not r.record or rendering.optics != r.optics:
            raise ValueError("another record or optics needs its own fill")
        if rendering.grid_mm != r.grid_mm or rendering.drift != r.drift:
            raise ValueError("another grid spacing or drift needs its own fill")
        if rendering.depth_mm > self._fill_depth + 1e-9:
            raise ValueError(f"the grid was filled {self._fill_depth} mm deep, "
                             f"not {rendering.depth_mm}")  # fmt: skip
        other = copy.copy(self)
        other.rendering = rendering
        other._sums = [np.zeros_like(s) for s in self._sums]
        other.spp, other.render_s, other.fill_s = 0, 0.0, 0.0
        other.rgb = np.zeros_like(self.rgb)
        other.noise = math.inf
        other.exposure = rendering.exposure
        if rendering.hollow == "air" and self._void is None:
            raise ValueError("the hollow as air needs a renderer made with the void box")
        air_here = self._void is not None and r.hollow == "air"
        air_there = self._void is not None and rendering.hollow == "air"
        if not air_there:
            other._void = None  # e.g. the backlit slice keeps the stand-in
        mask = self._void_mask(rendering)
        mask_changes = (mask is None) != (self._packed.voidf is None)
        if (rendering.void_mua != r.void_mua or air_here != air_there or mask_changes) and (
            self._grid.void is not None
        ):
            # another stand-in for the void: only the absorption grid (and the milky
            # lining's void mask) change, but the scene is uploaded whole (one more copy of
            # the grid on the GPU)
            g = self._grid
            a = with_void(g.absorb, g, rendering.void_mua, self._air if air_there else None)
            amax = 0.0 if a is None else max(float(a.max()), 0.0)
            other._dev = upload(replace(self._packed, absorb=a, absorb_max=amax, voidf=mask))
        return other

    def more(self, spp: int) -> None:
        """Bring the total to `spp` samples per pixel (even; split between halves)."""
        add = (spp - self.spp) // 2
        if add <= 0:
            return
        r = self.rendering
        t0 = time.perf_counter()
        for k in range(2):
            seed = int(self._rng.integers(1, 2**32 - 1))  # the kernel's seed is uint32
            out = render(self._dev, self._view, self._mats, r.light, spp=add, seed=seed,
                         depth_mm=r.depth_mm, flags=r.flags, shift=self._shift,
                         glare=r.glare, bottom_albedo=r.bottom_albedo,
                         void=self._void)  # fmt: skip
            self._sums[k] += out.image * add
            self.crossings, self.guessed = out.crossings, out.guessed
        self.spp += 2 * add
        half = self.spp / 2
        mean = (self._sums[0] + self._sums[1]) / self.spp
        if r.match_lightness:
            self.exposure = look_exposure(reflectance_factor(mean, r.light), self._face)
        knee = r.rolloff_knee
        a, b = (to_rgb(s / half, r.light, self.exposure, knee) for s in self._sums)
        self.rgb = to_rgb(mean, r.light, self.exposure, knee)
        self.noise = half_noise(a, b, self._face)
        self.render_s += time.perf_counter() - t0

    def linear(self) -> NDArray[np.float64]:
        """The mean linear image so far (radiance per channel, rows as `rgb` before any
        flip)."""
        return (self._sums[0] + self._sums[1]) / max(self.spp, 1)


def sparkle_rgb(look: CutRenderer, lamp: CutRenderer) -> NDArray[np.uint8]:
    """The sparkle view (ADR 0019 s.6.12 step 5): the look plus a lamp pass on the same
    fill (`Rendering.lamp`; its strength is the lamp's irradiance against the dome's),
    exposed so the stone's face without its hollow keeps the look's mean lightness (mean
    sRGB-encoded luminance, as `look_exposure` matches it). Facets that mirror the lamp
    clip to white, as glints do in a photograph."""
    light = look.rendering.light
    face = look._face
    a = reflectance_factor(look.linear(), light)
    total = a + reflectance_factor(lamp.linear(), light)
    target = mean_lightness(a, face, look.exposure)
    exposure = look.exposure
    if mean_lightness(total, face, exposure) != target:  # (a lamp that adds nothing keeps it)
        lo, hi = exposure / 100.0, exposure * 100.0
        for _ in range(50):  # bisection in log exposure; lightness rises with exposure
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if mean_lightness(total, face, mid) < target else (lo, mid)
        exposure = math.sqrt(lo * hi)
    return to_rgb(total * light.white(), light, exposure, None)


@dataclass(frozen=True)
class CutRender:
    """What a rendered cut records: its samples, noise (before and after the
    photo's blur), whether the photo's noise could be matched, and its times."""

    spp: int
    noise: float  # grey levels per pixel, before the blur
    blurred_noise: float | None  # after the photo's blur; None where no blur was found
    matched_noise: bool | None  # blurred noise at or below the photo's; None as above
    fill_s: float
    render_s: float
