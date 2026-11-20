"""A stone's recipe: its vocabulary, shares per sheet, the recipe models and the
small helpers on solids and draws that the other variety modules share (ADR 0018).

Appearance only: never evidence for a formation mechanism (ADR 0002).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from agatesim.grid import FloatArray
from agatesim.volume.config import SliceConfig, SolidConfig
from agatesim.volume.sampled import Sampled
from agatesim.volume.solids import Solid
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nodule.crystals import Crystals
from nodule.dendrites import Dendrites
from nodule.dust import (
    Dust,
)
from nodule.eyes import Eyes, eligible
from nodule.moss import Moss
from nodule.plumes import Plumes
from nodule.rind import SEDIMENT, VOLCANIC, Rind
from nodule.thunderegg import ARMS_BASE, JAGGED_SHARE, ArmShape, ArmStyle
from nodule.thunderegg3d import LENS_SHARE, LensStyle, jagged_solid, lens_solid, sinuous_solid
from nodule.tubes import NO_TUBES, Channel
from nodule.variety.lumps import lumps_solid, sampled_key

Centre = Literal["solid", "quartz-filled", "quartz-lined"]


Colour = Literal["neutral", "iron", "onyx", "stained", "carnelian"]


Outline = Literal["vesicle", "thunder-egg", "prolate", "irregular", "pebble"]


Tint = Literal["none", "amethyst", "smoky"]


Warp = Literal["mild", "crenulated", "lobed"]


Fill = Literal["open", "clear", "iron"]


# Shares per sheet: the first sheet recipe (ADR 0018 s.7), all `effective`.
CENTRE_SHARES: dict[str, float] = {"solid": 1, "quartz-filled": 1, "quartz-lined": 1}


COLOUR_SHARES: dict[str, float] = {
    "neutral": 6,
    "iron": 3,
    "onyx": 1,
    "stained": 1,
    "carnelian": 1,
}


OUTLINE_SHARES: dict[str, float] = {
    "vesicle": 7,
    "thunder-egg": 2,
    "prolate": 1,
    "irregular": 1,
    "pebble": 1,
}


WARP_SHARES: dict[str, float] = {"mild": 2, "crenulated": 1, "lobed": 1}


FLOOR_STONES = (2, 3)  # per 12 stones


FRACTURED_STONES = (1, 2)  # per 12 stones


TINT_CHANCE = 0.15  # amethyst or smoky in 10-20 % of quartz centres


# Quartz (agatesim's quartz-geode values for threshold and supply).
QUARTZ_THRESHOLD = 0.35


QUARTZ_SUPPLY = 0.3


# Nucleation spacing: 0.5 mm, coarser than the recipe's 0.25 mm start, because a
# 45 mm stone took 124 s to build at 0.25 mm and 32 s at 0.5 mm (2026-11-08).
QUARTZ_GRAIN_MM = 0.5


# Wide open centres (owner 2026-11-19, ADR 0018 s.24): a share of quartz-lined stones
# leave a wide hollow. Set by two shares of the cavity depth, open and quartz, so the
# lining keeps a few crystals' thickness: centre_width = open + quartz, lining =
# quartz / centre_width. Sources give the frequency in words only ("most",
# [@howard2018, p. 99]; "sometimes", [@goetze2020, p. 9]); the owner chose 1 in 3 and
# the open range from a look sheet. All `effective`, *assumption*.
WIDE_CENTRE_CHANCE = 1 / 3
WIDE_OPEN_SHARE = (0.2, 0.5)
WIDE_QUARTZ_SHARE = (0.08, 0.16)
WIDE_QUARTZ_MIN_MM = 1.5  # three nucleation spacings (QUARTZ_GRAIN_MM)
# Thunder eggs keep small centres: their druse is described as microcrystals in a
# small central cavity [@kile2002, p. 255] (*plausible*, a magazine source).
NARROW_CENTRE_OUTLINES = ("thunder-egg",)


# Chalcedony to quartz, sharp or graded (owner 2026-11-19, ADR 0018 s.24). Sources
# disagree: sharp "nearly always" [@heaney1993, p. 72], "mostly" [@howard2018, p. 99];
# graded in [@goetze2009, p. 677], [@goetze2020, p. 35], [@conte2022, p. 8]. A graded
# stone's last chalcedony clears toward the quartz over GRADE_MM: translucency up to
# GRADE_CLEAR of the way to clear, iron down to QUARTZ_IRON_RATIO, linearly, so the
# layering stays faintly visible at the contact (conte2022). All `effective`,
# *assumption*; the zone takes at most GRADE_MAX_SHARE of the chalcedony's time.
GRADED_CHANCE = 1 / 3
GRADE_MM = (2.0, 4.0)
GRADE_CLEAR = 0.8
GRADE_MAX_SHARE = 0.5


# `physical`: macro-quartz holds 78-141 ppm Fe against 262-969 in the bands
# [@mrozik2023, Table 5]; 0.2 lies inside the range of ratios (about 0.08-0.54).
QUARTZ_IRON_RATIO = 0.2


# Milky hollow linings (owner 2026-11-20, ADR 0019 s.6.12): levels on the even-lightness
# scale of `nodule.varietyrender.milk_mus` (0 near-clear, 1 the owner's D). The owner
# found 2/3 and 1 both right on 4/4, so most stones draw between MILK_HIGH_FROM and 1 and
# MILK_LOW_SHARE of them below it, keeping a few darker, clearer hollows. Tinted linings
# (amethyst, smoky) stay near-clear, level 0 (owner's pick; colour-centre quartz is
# called clear, never milky [@gilg2003, p. 1009]). All `effective`.
MILK_HIGH_FROM = 2 / 3
MILK_LOW_SHARE = 0.2


# The no-fold slope cap on band wobble (ADR 0005 s.3: below 1, bands cannot fold).
# 0.5 is the look preset's; the owner asked to try higher (2026-11-08, ADR 0018 s.8).
MAX_SLOPE = 0.5


# Largest wobble amplitude per mm of wobble scale, per unit of cap: agatesim's slope
# bound is about 4.2 / scale per mm of amplitude (measured 2026-11-08), so this
# keeps the wobble's own slope near two thirds of the cap.
WOBBLE_PER_SCALE_PER_CAP = 0.16


# Mild warps at half the drawn wobble (the owner, 2026-11-08: a wide band droops).
# agatesim's wobble is independent at each depth, so a wide band's two edges drift
# apart; over sheets 1-6 the 28 mild stones without floors swell by a median 1.10
# (P99 band width over the same stone without wobble), 1.03 at half (all 28 less).
# Crenulated and lobed warps keep theirs: they sit at the no-fold clamp, and there a
# weaker wobble fades in more steeply from the wall, so lobed stones swell more
# (median 1.11 to 1.17; ADR 0018 s.10). `effective`.
MILD_WOBBLE_SHARE = 0.5


FULL_WOBBLE_STONES = (1, 1)  # per 12 stones: a mild stone keeps its full wobble


# The owner's notes on the roughness trial and the plain cap-0.5 stones
# (2026-11-08, ADR 0018 s.9): keep such features, but make them rare.
# Floors: silica supply while the floor settles. The floor gets the same height
# in 1/supply as many bands; at 1 its bands are about 5 times the wall bands
# (the floor rises faster than the walls grow), which the owner judged too thick
# for most stones.
FLOOR_SUPPLY = (0.12, 0.35)


THICK_FLOOR_CHANCE = 1 / 8  # supply 1: the thick floor bands, kept but rare


# Floor height, as a share of the cavity's height: lower than the first draw
# (0.3-0.7), which the owner judged too high (2026-11-08); `effective`.
FLOOR_HEIGHT = (0.2, 0.5)


# Band roughness, smooth form only: none on most stones, little on many, more on
# a few (the owner: the trial's 0.4 mm was a bit too strong).
ROUGHNESS_LIMIT_MM = 0.25


ROUGHNESS_SHARES: dict[str, float] = {"none": 0.5, "faint": 0.35, "marked": 0.15}


ROUGHNESS_MM = {"faint": (0.04, 0.12), "marked": (0.12, ROUGHNESS_LIMIT_MM)}


class FractureLayer(BaseModel):
    """A flat crack without offset (ADR 0018 s.4, option C; s.7). A disc in 3D."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    centre_mm: tuple[float, float, float] = Field(description="From the solid's centre.")
    normal: tuple[float, float, float]
    radius_mm: float = Field(gt=0)
    aperture_mm: float = Field(gt=0, description="Trace width, 0.05-0.4 (plausible, figures).")
    fill: Fill
    stain: float = Field(ge=0, le=1, description="Iron added at the trace.")
    halo_mm: float = Field(ge=0, description="Stain reach along the trace, 0-0.5.")
    pool_mm: float = Field(ge=0, description="Stain reach at the cavity and porous bands.")
    sagitta_mm: float = Field(
        default=0.0, ge=0, description="Rise along the normal at the edge; 0 is flat."
    )

    def distance(self, points: FloatArray, solid_centre: FloatArray) -> FloatArray:
        """Distance (mm) from points (n, 3) to the disc, bent into a bowl that rises
        by the sagitta at its edge (a paraboloid; slopes stay below 0.2, so the flat
        formula stays near true)."""
        n = np.asarray(self.normal, dtype=np.float64)
        n = n / float(np.linalg.norm(n))
        rel = points - (solid_centre + np.asarray(self.centre_mm))
        z = rel @ n
        r = np.linalg.norm(rel - z[:, None] * n, axis=1)
        if self.sagitta_mm:
            z = z - self.sagitta_mm * (np.minimum(r, self.radius_mm) / self.radius_mm) ** 2
        return np.hypot(z, np.maximum(r - self.radius_mm, 0.0))


class Mottle(BaseModel):
    """Red patches with darker rims (ADR 0018 s.7: patches 5-30 mm, rims 0.6-1.2 mm)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scale_mm: float = Field(gt=0)
    cover: float = Field(gt=0, lt=1)
    rim_mm: float = Field(gt=0)
    strength: float = Field(ge=0, le=1)
    large: bool = Field(default=False, description="Drawn to show a patch of LARGE_PATCH.")
    variant: int = Field(default=0, ge=0, description="Which noise draw; 0 is the first.")
    edge: Literal["hard", "soft", "mixed"] = "hard"
    soft_width: float = Field(default=0.75, gt=0, description="Soft ramp, noise units.")


class Roughness(BaseModel):
    """Band roughness that grows inward, band by band (ADR 0018 s.8; the owner's
    trial after the slope-cap pairs). All `effective`. Sizes from one thunder egg's
    close-ups, scaled by fingertips (*plausible*, +-40 %): chevron arms 1.5-3 mm."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    amplitude_mm: float = Field(
        gt=0, le=ROUGHNESS_LIMIT_MM, description="Largest shift of a band, reached inward."
    )
    fade_mm: float = Field(default=5.0, gt=0, description="Depth over which it builds.")
    scale_mm: float = Field(gt=0, description="Size of the roughness along a band.")
    renew_mm: float = Field(default=4.0, gt=0, description="Depth over which it renews.")
    ridged: bool = Field(default=True, description="Sharp creases (chevrons), else smooth.")


class StoneRecipe(BaseModel):
    """One stone's features. Lengths in mm; all values `effective` unless the
    module says otherwise."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int
    seed: int = Field(ge=0)
    outline: Outline
    solid: SolidConfig
    arms: ArmStyle | LensStyle | None = None  # a thunder egg's centre: star style, or a lens
    arm_shape: ArmShape | None = None  # a sinuous star's bend, width and side lobes
    centre: Centre
    centre_width: float = Field(ge=0, le=1, description="Share of the inradius.")
    lining: float = Field(ge=0, le=1, description="Quartz-lined: share of the centre in quartz.")
    grade_mm: float = Field(
        default=0.0,
        ge=0,
        description="Chalcedony to quartz: 0 sharp, else the depth (mm) over which the last "
        "chalcedony clears toward the quartz (ADR 0018 s.24).",
    )
    tint: Tint = "none"
    milk: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="A quartz-lined hollow's milkiness: 0 near-clear to 1 the owner's D (ADR "
        "0019 s.6.12; `nodule.varietyrender.milk_mus`). 0 on tinted linings (near-clear) "
        "and other centres (unused).",
    )
    floor: float = Field(ge=0, le=1, description="0 = none, else the floor's target height.")
    floor_supply: float = Field(
        default=1.0, gt=0, le=1, description="Silica supply while the floor settles."
    )
    colour: Colour
    band_mm: float = Field(gt=0, description="Mean band width.")
    warp: Warp
    wobble_mm: float = Field(ge=0)
    wobble_scale_mm: float = Field(gt=0)
    full_wobble: bool = False  # a mild stone kept at its full drawn wobble (rare)
    white_fade: float = Field(ge=0, le=1, description="White bands thin inward.")
    rim_stain: float = Field(ge=0, le=1)
    mottle: Mottle | None = None
    fractures: tuple[FractureLayer, ...] = ()
    roughness: Roughness | None = None
    dust: Dust | None = None  # band-following dust (`nodule.dust`, ADR 0018 s.11)
    plumes: Plumes | None = None  # grown before the bands (`nodule.plumes`, ADR 0018 s.13)
    moss: Moss | None = None  # grown down before the bands (`nodule.moss`, ADR 0018 s.14)
    # on the wall before the bands (`nodule.crystals`, ADR 0018 s.15)
    crystals: Crystals | None = None
    cut: SliceConfig
    # outside the wall, previews only (`nodule.rind`, ADR 0018 s.16)
    rind: Rind | None = None
    # an escape tube: agatesim's wall channel (`nodule.tubes`, ADR 0018 s.18)
    channel: Channel | None = None
    # black Mn dendrites, from the wall or on cracks (`nodule.dendrites`, ADR 0018 s.17)
    dendrites: Dendrites | None = None
    # banded domes on the wall: agatesim's spherulitic stage (`nodule.eyes`, ADR 0018 s.18)
    eyes: Eyes | None = None

    @model_validator(mode="after")
    def _eyes_on_their_own(self) -> StoneRecipe:
        early = self.crystals is not None or self.inclusions is not None
        if self.eyes is not None and not eligible(self.outline, early):
            raise ValueError(
                "eyes only on vesicle, prolate or thunder-egg stones without crystals, "
                "plumes or moss"
            )
        return self

    @model_validator(mode="after")
    def _dendrite_cracks_exist(self) -> StoneRecipe:
        d = self.dendrites
        if d is not None and any(not 0 <= i < len(self.fractures) for i in d.cracks):
            raise ValueError("a dendrite carrier crack must be one of the stone's cracks")
        return self

    @model_validator(mode="after")
    def _one_dendrite_stage(self) -> StoneRecipe:
        if self.plumes is not None and self.moss is not None:
            raise ValueError("plumes or moss, not both: agatesim runs one dendrite stage")
        return self

    @model_validator(mode="after")
    def _rind_fits_the_host(self) -> StoneRecipe:
        r = self.rind
        if r is None:
            return self
        if r.kind == "volcanic" and self.outline not in VOLCANIC:
            raise ValueError("a volcanic rind belongs on a vesicle or prolate stone")
        if r.kind == "pale" and self.outline not in SEDIMENT:
            raise ValueError("the pale rind belongs on a sediment stone (irregular, pebble)")
        if r.window is not None and r.kind != "volcanic":
            raise ValueError("windows only on volcanic rinds")
        return self

    @model_validator(mode="after")
    def _graded_needs_quartz(self) -> StoneRecipe:
        if self.grade_mm and self.centre == "solid":
            raise ValueError("a graded transition needs a quartz centre")
        return self

    @model_validator(mode="after")
    def _milk_lines_a_hollow(self) -> StoneRecipe:
        if self.milk and self.centre != "quartz-lined":
            raise ValueError("milkiness belongs to a quartz-lined hollow's lining")
        if self.milk and self.tint != "none":
            raise ValueError("a tinted lining stays near-clear (milk 0)")
        return self

    @model_validator(mode="after")
    def _no_tube_on_a_pebble(self) -> StoneRecipe:
        if self.channel is not None and self.outline in NO_TUBES:
            raise ValueError("no escape tube on a pebble (Walger: none below cherry size)")
        return self

    @property
    def inclusions(self) -> Plumes | Moss | None:
        """The stone's branched inclusions (plumes or moss), if any."""
        return self.plumes if self.plumes is not None else self.moss

    def summary(self) -> str:
        parts = [
            self.outline if self.arms is None else f"{self.outline} ({self.arms})",
            self.centre if self.centre == "solid" else f"{self.centre} {self.centre_width:.2f}",
            self.colour,
        ]
        if self.tint != "none":
            parts.append(self.tint)
        if self.centre == "quartz-lined":
            parts.append(f"milk {self.milk:.2f}")
        if self.grade_mm:
            parts.append(f"graded {self.grade_mm:.1f} mm")
        if self.floor:
            thick = " thick" if self.floor_supply >= 1 else ""
            parts.append(f"floor {self.floor:.2f}{thick}")
        if self.warp != "mild":
            parts.append(self.warp)
        elif self.full_wobble:
            parts.append("full wobble")
        if self.mottle is not None:
            parts.append("mottle")
        if self.fractures:
            parts.append(f"{len(self.fractures)} cracks")
        if self.roughness is not None:
            kind = "ridged" if self.roughness.ridged else "rough"
            parts.append(f"{kind} {self.roughness.amplitude_mm:.2f} mm")
        if self.dust is not None:
            parts.append(self.dust.summary())
        if self.inclusions is not None:
            parts.append(self.inclusions.summary())
        if self.crystals is not None:
            parts.append(self.crystals.summary())
        if self.rind is not None:
            parts.append(self.rind.summary())
        if self.channel is not None:
            parts.append(self.channel.summary())
        if self.dendrites is not None:
            parts.append(self.dendrites.summary())
        if self.eyes is not None:
            parts.append(self.eyes.summary())
        parts.append(f"bands {self.band_mm:.2f} mm")
        return ", ".join(parts)


def quota(rng: np.random.Generator, shares: dict[str, float], n: int) -> list[str]:
    """n labels in proportion to shares (largest remainder), in random order."""
    total = sum(shares.values())
    exact = {k: v / total * n for k, v in shares.items()}
    counts = {k: math.floor(x) for k, x in exact.items()}
    by_rest = sorted(exact, key=lambda k: exact[k] - counts[k], reverse=True)
    for k in by_rest[: n - sum(counts.values())]:
        counts[k] += 1
    labels = [k for k, c in counts.items() for _ in range(c)]
    return [labels[i] for i in rng.permutation(n)]


def scaled_count(rng: np.random.Generator, per_12: tuple[int, int], n: int) -> int:
    return round(int(rng.integers(per_12[0], per_12[1] + 1)) * n / 12)


DustForms = tuple[bool, bool, bool]  # lines, specks, blobs


def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


# Irregular and lobate stones draw their own lumps (owner 2026-11-19 night, ADR 0018
# s.25; `nodule.variety.lumps`) instead of agatesim's one fixed union each. False
# gives the presets back (a control for tests).
OWN_LUMPS = True


def solid_for(
    rng: np.random.Generator, outline: Outline
) -> tuple[SolidConfig, ArmStyle | LensStyle | None, ArmShape | None]:
    """The outline follows the host [@pabian1994, pp. 11, 16] (ADR 0018 s.7); a
    thunder egg's star also has an arm style and, if sinuous, an arm shape
    (`nodule.thunderegg`). Lobate vesicles and irregular nodules draw their own
    lumps from a stream spawned off `rng`, so `rng`'s draws stay as they were."""
    yaw = float(rng.uniform(0, 360))
    match outline:
        case "vesicle":
            preset = str(rng.choice(["vesicle-round", "vesicle-lobate"]))
            size = float(rng.uniform(30, 65))
            if preset == "vesicle-lobate" and OWN_LUMPS:  # its own lobes (`lumps`)
                return lumps_solid(rng, "lobate", size), None, None
            return SolidConfig(preset=preset, size_mm=size, yaw_deg=yaw), None, None
        case "prolate":
            return (
                SolidConfig(
                    preset="vesicle-almond",
                    size_mm=float(rng.uniform(45, 70)),
                    yaw_deg=yaw,
                    pitch_deg=float(rng.uniform(-15, 15)),
                ),
                None,
                None,
            )
        case "irregular":
            size = float(rng.uniform(35, 60))
            if OWN_LUMPS:  # its own lumps (`lumps`)
                return lumps_solid(rng, "irregular", size), None, None
            return SolidConfig(preset="nodule-irregular", size_mm=size, yaw_deg=yaw), None, None
        case "pebble":
            return (
                SolidConfig(preset="nodule-round", size_mm=float(rng.uniform(12, 20)), yaw_deg=yaw),
                None,
                None,
            )
        case "thunder-egg":
            # The style is drawn as in `thunder_egg_outline`; both styles are drawn in
            # 3D from the first spawned stream (ADR 0018 s.20-21), then the size. A lens
            # centre is chosen on the second spawned stream (s.22), so every star stone
            # draws as before; a lens replaces the star before it is drawn.
            rng.integers(ARMS_BASE[0], ARMS_BASE[1] + 1)  # the 2D arm count, unused now
            jagged = rng.random() < JAGGED_SHARE
            child, lens_rng = rng.spawn(2)
            size = float(rng.uniform(40, 70))
            if lens_rng.random() < LENS_SHARE:
                return lens_solid(lens_rng.spawn(1)[0], size), "lens", None
            if jagged:
                return jagged_solid(child, size), "jagged", None
            shape, solid = sinuous_solid(child, size)
            return solid, "sinuous", shape


def inradius(solid: Solid) -> float:
    """Half the smallest side of the bounding box: an upper bound on how far
    the wall lining must grow to fill the cavity."""
    lo, hi = solid.bounds()
    return 0.5 * float(np.min(np.asarray(hi) - np.asarray(lo)))


# `effective`: a star thunder egg's inradius as a share of its size (sinuous 0.25-0.40,
# median 0.33; jagged 0.21-0.32, median 0.27; ADR 0018 s.22). A lens's own inradius is
# only half its depth (0.11-0.19 of its size), which left its cracks barely showing.
LENS_CRACK_SHARE = 0.3


def crack_scale(config: SolidConfig, solid: Solid, arms: str | None) -> float:
    """The length (mm) a stone's cracks are sized by (ADR 0018 s.10, s.22): its
    inradius, except on a lens-centred thunder egg, where it is the inradius a star egg
    of that size typically has, so the cracks are sized to the egg (owner, 2026-11-19).
    How far the cracks keep from an open centre still uses the true inradius."""
    if arms == "lens":
        return LENS_CRACK_SHARE * config.size_mm
    return inradius(solid)


def body_height(solid: Solid) -> float:
    """How far (mm, along z) the deepest point of a drawn 3D cavity lies from the
    centre of its box: where a thunder egg's cutter aims (ADR 0018 s.20). 0 for
    other solids (a prism's deepest points span its height). Found on a 48-point
    grid a side, so to within about 2 % of the box."""
    if not isinstance(solid, Sampled):
        return 0.0
    deep, _ = _deepest(solid)
    return float(deep[2] - np.asarray(solid.centre())[2])


def _deepest(solid: Solid, n: int = 48) -> tuple[FloatArray, FloatArray]:
    """The deepest of an n-point grid a side over the solid's box, and the grid's step."""
    bounds = solid.bounds()
    lo = np.asarray(bounds[0], dtype=np.float64)
    hi = np.asarray(bounds[1], dtype=np.float64)
    axes: list[FloatArray] = [np.linspace(float(lo[i]), float(hi[i]), n) for i in range(3)]
    grids: list[FloatArray] = list(np.meshgrid(*axes, indexing="ij"))
    pts: FloatArray = np.stack(grids, axis=-1).reshape(-1, 3)
    inside = pts[solid.contains(pts)]
    return inside[int(np.argmax(solid.wall_distance(inside)))], (hi - lo) / (n - 1)


def cavity_depth(solid: Solid) -> float:
    """How far (mm) the wall lining must grow to fill the cavity: the wall distance of
    a drawn 3D cavity's deepest point (found on `_deepest`'s grid, then on a 9-point
    grid a side within one step of it), and the `inradius` of other solids, as before.
    A winding star's box is 2-2.7 x as thick as its deepest point lies from the wall
    (jagged cores 1.3-1.6 x, lenses 1.03-1.06 x), so sizing its bands by `inradius`
    grew chalcedony over every quartz centre (ADR 0018 s.23). Other solids keep their
    `inradius` unless the measured depth is below DEPTH_GAP of it: the irregular
    nodules (0.82-0.85) are measured, the round, lobed and almond vesicles (0.96-1.0)
    are not, so they build as before. A drawn cavity's depth is remembered by its stored
    level (the same value; agatesim keeps only a few drawn solids' distances, so drawing a
    sheet asked for each lumpy stone's several times)."""
    key = None
    if isinstance(solid, Sampled):
        key = sampled_key(solid)
        known = _SAMPLED_DEPTHS.get(key)
        if known is not None:
            return known
    deep, step = _deepest(solid)
    f = np.linspace(-1.0, 1.0, 9)
    near: FloatArray = np.stack(np.meshgrid(f, f, f, indexing="ij"), axis=-1).reshape(-1, 3)
    pts: FloatArray = deep + near * step
    depth = float(np.max(solid.wall_distance(pts[solid.contains(pts)])))
    if key is not None:
        if len(_SAMPLED_DEPTHS) >= 1024:
            _SAMPLED_DEPTHS.clear()
        _SAMPLED_DEPTHS[key] = depth
        return depth
    r = inradius(solid)
    return depth if depth < DEPTH_GAP * r else r


_SAMPLED_DEPTHS: dict[tuple[object, ...], float] = {}  # `cavity_depth` of drawn cavities


# `effective`: below this share of the inradius, a solid's measured depth sizes its
# bands (ADR 0018 s.23); a few per cent is left to the bands' wobble.
DEPTH_GAP = 0.95


def recipe_size(solid: Solid) -> float:
    """The longest side of the solid's bounding box (mm)."""
    lo, hi = solid.bounds()
    return float(np.max(np.asarray(hi) - np.asarray(lo)))


def rim_point(solid: Solid, direction: FloatArray) -> FloatArray:
    """Where a ray from the solid's centre leaves it (bisection), as an offset."""
    c = solid.centre()
    lo, hi = 0.0, float(np.linalg.norm(np.asarray(solid.bounds()[1]) - solid.bounds()[0]))
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if solid.contains((c + mid * direction)[None, :])[0]:
            lo = mid
        else:
            hi = mid
    return lo * direction


def unit_vector(rng: np.random.Generator) -> FloatArray:
    v = rng.standard_normal(3)
    return v / float(np.linalg.norm(v))


def share_weights(shares: dict[str, float]) -> list[float]:
    total = sum(shares.values())
    return [v / total for v in shares.values()]
