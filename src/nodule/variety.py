"""Broad mock-ups varied per stone: recipes and quick feature layers (ADR 0018).

A sheet of stones is drawn from one seed. Each stone gets its own spawned stream
and its own recipe: a varied subset of features 1 to 5 of ADR 0018 (centre,
fractures, warp and mottle, colour, outline and band density), with counts per
sheet fixed by quota as in the first sheet recipe (ADR 0018 section 7). No stone
carries every feature.

Most parts come from agatesim, chosen per stone: the outline (solid presets, or
a drawn star for thunder eggs), the quartz centre (a silica supply that drops
below agatesim's quartz threshold), the open cavity (supply that stops), the
floor (a settling pulse at the start: floor first [@moxon2020, p. 16]; Kigai's
alternation [@kigai2019] is the option not taken here), warp (agatesim's wobble,
slope-capped so bands never fold) and the palette. What agatesim lacks starts
here as layers on the fields (`VarietyLayers`): fracture traces with stain pools,
mottle patches with darker rims, a rim stain that crosses bands, quartz kept low
in iron, and white bands that thin inward.

Every share, size and strength is `effective` unless marked `physical`; the
plausible ranges come from figures read in ADR 0018 section 7. Couplings between
features are listed on `VarietyLayers`, each with its source or as an assumption.
Random choices use spawned streams, and noise lives on lattices fixed to the
stone, so a preview matches a full-resolution cut (CLAUDE.md).

Appearance only: never evidence for a formation mechanism (ADR 0002).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from statistics import NormalDist
from typing import Literal

import numpy as np
from agatesim.fields import Fields, Phase, Pigment
from agatesim.grid import FloatArray
from agatesim.volume.config import SliceConfig, SolidConfig, VolumeConfig
from agatesim.volume.models import SmoothNoise3
from agatesim.volume.solids import BoolArray, Solid
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from nodule.specimen import Specimen, SpecimenConfig, SpecimenCut

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
MOTTLE_SHARE = 1 / 3
TINT_CHANCE = 0.15  # amethyst or smoky in 10-20 % of quartz centres

# Quartz (agatesim's quartz-geode values for threshold and supply).
QUARTZ_THRESHOLD = 0.35
QUARTZ_SUPPLY = 0.3
# Nucleation spacing: 0.5 mm, coarser than the recipe's 0.25 mm start, because a
# 45 mm stone took 124 s to build at 0.25 mm and 32 s at 0.5 mm (2026-11-08).
QUARTZ_GRAIN_MM = 0.5
# `physical`: macro-quartz holds 78-141 ppm Fe against 262-969 in the bands
# [@mrozik2023, Table 5]; 0.2 lies inside the range of ratios (about 0.08-0.54).
QUARTZ_IRON_RATIO = 0.2
# The no-fold slope cap on band wobble (ADR 0005 s.3: below 1, bands cannot fold).
# 0.5 is the look preset's; the owner asked to try higher (2026-11-08, ADR 0018 s.8).
MAX_SLOPE = 0.5
# Largest wobble amplitude per mm of wobble scale, per unit of cap: agatesim's slope
# bound is about 4.2 / scale per mm of amplitude (measured 2026-11-08), so this
# keeps the wobble's own slope near two thirds of the cap.
WOBBLE_PER_SCALE_PER_CAP = 0.16


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

    def distance(self, points: FloatArray, solid_centre: FloatArray) -> FloatArray:
        """Distance (mm) from points (n, 3) to the disc."""
        n = np.asarray(self.normal, dtype=np.float64)
        n = n / float(np.linalg.norm(n))
        rel = points - (solid_centre + np.asarray(self.centre_mm))
        z = rel @ n
        r = np.linalg.norm(rel - z[:, None] * n, axis=1)
        return np.hypot(z, np.maximum(r - self.radius_mm, 0.0))


class Mottle(BaseModel):
    """Red patches with darker rims (ADR 0018 s.7: patches 5-30 mm, rims 0.6-1.2 mm)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scale_mm: float = Field(gt=0)
    cover: float = Field(gt=0, lt=1)
    rim_mm: float = Field(gt=0)
    strength: float = Field(ge=0, le=1)


class Roughness(BaseModel):
    """Band roughness that grows inward, band by band (ADR 0018 s.8; the owner's
    trial after the slope-cap pairs). All `effective`. Sizes from one thunder egg's
    close-ups, scaled by fingertips (*plausible*, +-40 %): chevron arms 1.5-3 mm."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    amplitude_mm: float = Field(gt=0, description="Largest shift of a band, reached inward.")
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
    centre: Centre
    centre_width: float = Field(ge=0, le=1, description="Share of the inradius.")
    lining: float = Field(ge=0, le=1, description="Quartz-lined: share of the centre in quartz.")
    tint: Tint = "none"
    floor: float = Field(ge=0, le=1, description="0 = none, else the floor's target height.")
    colour: Colour
    band_mm: float = Field(gt=0, description="Mean band width.")
    warp: Warp
    wobble_mm: float = Field(ge=0)
    wobble_scale_mm: float = Field(gt=0)
    white_fade: float = Field(ge=0, le=1, description="White bands thin inward.")
    rim_stain: float = Field(ge=0, le=1)
    mottle: Mottle | None = None
    fractures: tuple[FractureLayer, ...] = ()
    roughness: Roughness | None = None
    cut: SliceConfig

    def summary(self) -> str:
        parts = [
            self.outline,
            self.centre if self.centre == "solid" else f"{self.centre} {self.centre_width:.2f}",
            self.colour,
        ]
        if self.tint != "none":
            parts.append(self.tint)
        if self.floor:
            parts.append(f"floor {self.floor:.2f}")
        if self.warp != "mild":
            parts.append(self.warp)
        if self.mottle is not None:
            parts.append("mottle")
        if self.fractures:
            parts.append(f"{len(self.fractures)} cracks")
        if self.roughness is not None:
            parts.append("ridged" if self.roughness.ridged else "rough")
        parts.append(f"bands {self.band_mm:.2f} mm")
        return ", ".join(parts)


# --- drawing a sheet -------------------------------------------------------------------


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


def _scaled(rng: np.random.Generator, per_12: tuple[int, int], n: int) -> int:
    return round(int(rng.integers(per_12[0], per_12[1] + 1)) * n / 12)


def draw_sheet(seed: int, n: int = 12) -> list[StoneRecipe]:
    """n stone recipes from one seed (ADR 0018 s.7)."""
    root = np.random.SeedSequence(seed)
    plan_seq, *stone_seqs = root.spawn(n + 1)
    plan = np.random.default_rng(plan_seq)
    outlines = quota(plan, OUTLINE_SHARES, n)
    centres = quota(plan, CENTRE_SHARES, n)
    for i, o in enumerate(outlines):  # a pebble is small and solid (s.7)
        if o == "pebble" and centres[i] != "solid" and "solid" in centres:
            j = next(j for j, c in enumerate(centres) if c == "solid" and outlines[j] != "pebble")
            centres[i], centres[j] = centres[j], centres[i]
    colours = quota(plan, COLOUR_SHARES, n)
    warps = quota(plan, WARP_SHARES, n)
    # Floors need a "down" across a wide cavity: not in thunder eggs or pebbles.
    floor_ok = [i for i, o in enumerate(outlines) if o not in ("thunder-egg", "pebble")]
    floors = set(plan.permutation(floor_ok)[: _scaled(plan, FLOOR_STONES, n)].tolist())
    crack_ok = [i for i, o in enumerate(outlines) if o != "pebble"]
    cracked = set(plan.permutation(crack_ok)[: _scaled(plan, FRACTURED_STONES, n)].tolist())
    mottled = set(plan.permutation(n)[: round(MOTTLE_SHARE * n)].tolist())
    return [
        draw_stone(
            i,
            np.random.default_rng(stone_seqs[i]),
            outlines[i],
            centres[i],
            colours[i],
            warps[i],
            floor=i in floors,
            cracked=i in cracked,
            mottled=i in mottled,
        )
        for i in range(n)
    ]


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def star_outline(rng: np.random.Generator, arms: int) -> list[tuple[float, float]]:
    """A thunder egg's star cavity [@pabian1994, p. 16]: arms of uneven length."""
    lengths = rng.uniform(0.6, 1.0, arms)
    phase = rng.uniform(0, 2 * math.pi)
    theta = np.linspace(0, 2 * math.pi, 24 * arms, endpoint=False)
    k = np.floor((theta - phase) % (2 * math.pi) / (2 * math.pi) * arms).astype(int) % arms
    frac = ((theta - phase) % (2 * math.pi / arms)) / (2 * math.pi / arms)
    peak = (0.5 + 0.5 * np.cos(2 * math.pi * (frac - 0.5))) ** 3  # 1 at an arm's tip
    r = 0.32 + (lengths[k] - 0.32) * peak
    return [(float(a), float(b)) for a, b in zip(r * np.cos(theta), r * np.sin(theta), strict=True)]


def solid_for(rng: np.random.Generator, outline: Outline) -> SolidConfig:
    """The outline follows the host [@pabian1994, pp. 11, 16] (ADR 0018 s.7)."""
    yaw = float(rng.uniform(0, 360))
    match outline:
        case "vesicle":
            preset = str(rng.choice(["vesicle-round", "vesicle-lobate"]))
            return SolidConfig(preset=preset, size_mm=float(rng.uniform(30, 65)), yaw_deg=yaw)
        case "prolate":
            return SolidConfig(
                preset="vesicle-almond",
                size_mm=float(rng.uniform(45, 70)),
                yaw_deg=yaw,
                pitch_deg=float(rng.uniform(-15, 15)),
            )
        case "irregular":
            return SolidConfig(
                preset="nodule-irregular", size_mm=float(rng.uniform(35, 60)), yaw_deg=yaw
            )
        case "pebble":
            return SolidConfig(
                preset="nodule-round", size_mm=float(rng.uniform(12, 20)), yaw_deg=yaw
            )
        case "thunder-egg":
            arms = int(rng.integers(5, 9))
            return SolidConfig(
                outline=star_outline(rng, arms),
                host="volcanic",  # type: ignore[arg-type]
                size_mm=float(rng.uniform(40, 70)),
                depth_fraction=0.6,
            )


def inradius(solid: Solid) -> float:
    """Half the smallest side of the bounding box: an upper bound on how far
    the wall lining must grow to fill the cavity."""
    lo, hi = solid.bounds()
    return 0.5 * float(np.min(np.asarray(hi) - np.asarray(lo)))


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


def _unit(rng: np.random.Generator) -> FloatArray:
    v = rng.standard_normal(3)
    return v / float(np.linalg.norm(v))


def draw_fractures(rng: np.random.Generator, solid: Solid, red: bool) -> tuple[FractureLayer, ...]:
    """1-3 sheets at high angle to the bands (through the centre region), and 0-3
    short rim-only cracks 1-4 mm long (ADR 0018 s.7, plausible from figures). On a
    red stone, iron fill hardly shows, so most cracks there are open or clear
    (`effective`, for the sheet's eye)."""
    r = inradius(solid)
    fills, p = ["open", "clear", "iron"], [0.4, 0.5, 0.1] if red else [0.3, 0.3, 0.4]
    out: list[FractureLayer] = []
    for _ in range(int(rng.integers(1, 4))):
        fill: Fill = str(rng.choice(fills, p=p))  # type: ignore[assignment]
        out.append(
            FractureLayer(
                centre_mm=tuple((0.3 * r * rng.uniform() ** (1 / 3) * _unit(rng)).tolist()),
                normal=tuple(_unit(rng).tolist()),
                radius_mm=float(rng.uniform(0.6, 1.5) * r),
                aperture_mm=_log_uniform(rng, 0.05, 0.4),
                fill=fill,
                stain=float(rng.uniform(0.4, 1.0)),
                halo_mm=float(rng.uniform(0.0, 0.5)),
                pool_mm=float(rng.uniform(1.0, 3.0)),  # half the 2-6 mm pools
            )
        )
    for _ in range(int(rng.integers(0, 4))):
        radial = _unit(rng)
        normal = np.cross(radial, _unit(rng))  # the crack runs inward from the rim
        out.append(
            FractureLayer(
                centre_mm=tuple(rim_point(solid, radial).tolist()),
                normal=tuple((normal / float(np.linalg.norm(normal))).tolist()),
                radius_mm=float(rng.uniform(1.0, 4.0)),
                aperture_mm=_log_uniform(rng, 0.05, 0.2),
                fill="clear" if red else "iron",
                stain=float(rng.uniform(0.3, 0.8)),
                halo_mm=float(rng.uniform(0.0, 0.3)),
                pool_mm=float(rng.uniform(0.5, 1.5)),
            )
        )
    return tuple(out)


def draw_stone(
    index: int,
    rng: np.random.Generator,
    outline: str,
    centre: str,
    colour: str,
    warp: str,
    *,
    floor: bool,
    cracked: bool,
    mottled: bool,
) -> StoneRecipe:
    solid_config = solid_for(rng, outline)  # type: ignore[arg-type]
    solid = solid_config.build()
    r = inradius(solid)
    width = 0.0 if centre == "solid" else float(np.clip(rng.gamma(4.0, 0.05), 0.08, 0.6))
    tint: Tint = "none"
    if centre != "solid" and rng.uniform() < TINT_CHANCE:
        tint = str(rng.choice(["amethyst", "smoky"]))  # type: ignore[assignment]
    match warp:
        case "crenulated":
            wobble, scale = float(rng.uniform(0.2, 1.0)), float(rng.uniform(2.0, 5.0))
        case "lobed":
            wobble, scale = float(rng.uniform(0.5, 1.5)), float(rng.uniform(8.0, 15.0))
        case _:
            wobble, scale = float(rng.uniform(0.05, 0.25)), float(rng.uniform(3.0, 8.0))
    mottle = None
    if mottled:
        # Patches 5-30 mm, but at most about a third of the stone, so a face shows
        # several patches rather than one film (`effective`).
        top = max(5.0, min(30.0, 0.35 * recipe_size(solid)))
        mottle = Mottle(
            scale_mm=_log_uniform(rng, min(5.0, top), top),
            cover=float(rng.uniform(0.15, 0.35)),
            rim_mm=float(rng.uniform(0.6, 1.2)),
            strength=float(rng.uniform(0.3, 0.7)),
        )
    if outline == "thunder-egg":
        cut = SliceConfig(
            tilt_deg=0.0, azimuth_deg=0.0, offset_mm=float(rng.uniform(-0.1, 0.1) * r)
        )
    else:
        cut = SliceConfig(
            tilt_deg=float(rng.uniform(70, 90)),
            azimuth_deg=float(rng.uniform(0, 360)),
            offset_mm=float(rng.uniform(-0.15, 0.15) * r),
        )
    return StoneRecipe(
        index=index,
        seed=int(rng.integers(0, 2**31)),
        outline=outline,  # type: ignore[arg-type]
        solid=solid_config,
        centre=centre,  # type: ignore[arg-type]
        centre_width=width,
        lining=float(rng.uniform(0.3, 0.7)),
        tint=tint,
        floor=float(rng.uniform(0.3, 0.7)) if floor else 0.0,
        colour=colour,  # type: ignore[arg-type]
        band_mm=0.12 if outline == "pebble" else _log_uniform(rng, 0.2, 1.0),
        warp=warp,  # type: ignore[arg-type]
        wobble_mm=wobble,
        wobble_scale_mm=scale,
        white_fade=float(rng.uniform(0.2, 0.5)),
        rim_stain=float(rng.uniform(0.5, 0.9)) if colour == "stained" else 0.0,
        mottle=mottle,
        fractures=draw_fractures(rng, solid, colour in ("iron", "carnelian")) if cracked else (),
        cut=cut,
    )


# --- recipe to specimen ------------------------------------------------------------------


@dataclass(frozen=True)
class Growth:
    """The growth run that gives the recipe's centre: total growth and the time
    (share of the run) at which silica drops to quartz supply."""

    growth_mm: float
    quartz_at: float | None

    @property
    def chalcedony_end(self) -> float:
        """Growth time at which the last chalcedony band lies down."""
        return self.quartz_at if self.quartz_at is not None else 1.0


def growth_for(recipe: StoneRecipe, r: float) -> Growth:
    """Chalcedony to (1 - w) r, then quartz (assumes the agatesim growth rate
    scales with supply, as its quartz-geode preset does)."""
    w = recipe.centre_width
    match recipe.centre:
        case "solid":
            # 1.3 r fills any cavity whose inradius is at most r; the last band
            # lies down at about 1/1.3 of the run.
            return Growth(1.3 * r, None)
        case "quartz-filled":
            g = 3.0 * r  # quartz at 0.3 supply then outgrows the centre for w <= 0.6
            return Growth(g, (1 - w) * r / g)
        case "quartz-lined":
            # The run ends as the lining reaches its thickness, leaving the rest
            # open (agatesim allows no stage after quartz, so supply cannot stop).
            quartz = recipe.lining * w * r
            g = (1 - w) * r + quartz / QUARTZ_SUPPLY
            return Growth(g, (1 - w) * r / g)


def volume_config(recipe: StoneRecipe, max_slope: float = MAX_SLOPE) -> tuple[VolumeConfig, Growth]:
    solid = recipe.solid.build()
    r = inradius(solid)
    growth = growth_for(recipe, r)
    g = growth.growth_mm
    model: dict[str, object] = {
        "kind": "procedural-growth",
        "growth_mm": g,
        "band_period": min(recipe.band_mm / g, 0.5),
        "period_variability": 0.7,  # ADR 0011, kept from the look preset
        "alternation": 0.7,
        # Clamped under the no-fold cap, so at 0.5 crenulation is gentler than the
        # figures show (ADR 0018 s.8).
        "wobble_mm": min(
            recipe.wobble_mm, WOBBLE_PER_SCALE_PER_CAP * max_slope * recipe.wobble_scale_mm
        ),
        "wobble_scale_mm": recipe.wobble_scale_mm,
        "wobble_max_slope": max_slope,
    }
    silica: list[dict[str, object]] = []
    if growth.quartz_at is not None:
        model |= {
            "quartz_threshold": QUARTZ_THRESHOLD,
            "quartz_grain_mm": QUARTZ_GRAIN_MM,
            "quartz_elongation": 4.0,
            "quartz_axis_scatter_deg": 40.0,
        }
        silica.append({"type": "step", "at": growth.quartz_at, "value": QUARTZ_SUPPLY})
    schedules: dict[str, object] = {"silica": {"base": 1.0, "events": silica}}
    if recipe.floor:
        # A settling pulse first: floor first [@moxon2020, p. 16]. Width scaled
        # from one probe (40 mm round vesicle, 30 mm growth: 0.12 gave a floor
        # at about 0.55 of the height); `effective`, judged on the sheet.
        width = recipe.floor * 0.22 * (30.0 / g) * (r / 16.0)
        schedules["settling"] = {
            "base": 0.0,
            "events": [
                {
                    "type": "pulse",
                    "at": 0.0,
                    "width": float(np.clip(width, 0.01, 0.6)),
                    "value": 1.0,
                }
            ],
        }
    palette, iron = _palette(recipe)
    schedules["iron"] = {"base": iron}
    model["palette"] = palette
    model["schedules"] = schedules
    if recipe.colour == "carnelian":
        model |= {"alternation": 0.3, "opacity_contrast": 0.15}
    if recipe.outline == "pebble":  # the small unbanded pebble (s.7)
        model |= {"opacity_contrast": 0.05, "contrast_variability": 0.0, "iron_variability": 0.0}
    config = VolumeConfig.model_validate(
        {
            "name": f"variety-{recipe.index}",
            "seed": recipe.seed,
            "solid": recipe.solid.model_dump(),
            "model": model,
        }
    )
    return config, growth


def _palette(recipe: StoneRecipe) -> tuple[dict[str, float], float]:
    """Palette and iron supply per stone, not from host [@mrozik2023, p. 23;
    @moxon2020, p. 18]. Onyx black is carbon in fibrous bands [@goetze2012];
    agatesim's manganese black stands in for it here (an assumption)."""
    rng = np.random.default_rng(recipe.seed)
    match recipe.colour:
        case "neutral" | "stained":
            return {"iron": 1.0, "white": float(rng.uniform(0, 0.4))}, float(
                rng.uniform(0.03, 0.12)
            )
        case "iron":
            options = [
                {"hematite": 1.0, "white": 0.3},
                {"goethite": 1.0, "iron": 0.5},
                {"iron": 1.0},
            ]
            return options[int(rng.integers(0, 3))], float(rng.uniform(0.25, 0.45))
        case "onyx":
            return {"iron": 0.0, "manganese": 0.5, "white": 0.5}, 0.4
        case "carnelian":
            # Unbanded carnelian holds about 1.1 wt% Fe2O3 [@volkert2023, p. 12]: the
            # strongest iron of the sheet (the `physical` anchor; the value is effective).
            return {"iron": 0.0, "hematite": 1.0}, 0.6


@dataclass(frozen=True, eq=False)
class VarietyLayers:
    """The quick layers of one stone (`nodule.specimen.Layers`).

    Couplings, each an assumption unless cited:
    - quartz takes little iron [@mrozik2023, Table 5; @goetze2009] (`physical` ratio);
    - white bands fewer inward [@mrozik2023, pp. 13, 22]: translucency rises with
      growth time (the form is an assumption);
    - mottle patches tint only chalcedony, red, with darker rims (from figures);
    - a rim stain crosses bands, strongest at the wall [@goetze2020, p. 37]; patchy
      by world noise (assumption);
    - fracture stain spreads along the trace and pools where the crack meets the
      cavity or a porous (milky) band (from figures; which bands are porous is an
      assumption); no offset (none seen in any figure).
    """

    recipe: StoneRecipe
    solid_centre: FloatArray
    chalcedony_end: float
    mottle_noise: SmoothNoise3 | None
    stain_noise: SmoothNoise3 | None

    @classmethod
    def of(cls, recipe: StoneRecipe, solid: Solid, growth: Growth) -> VarietyLayers:
        seqs = np.random.SeedSequence(recipe.seed).spawn(2)
        mottle = None
        if recipe.mottle is not None:
            mottle = SmoothNoise3(solid, recipe.mottle.scale_mm, np.random.default_rng(seqs[0]))
        stain = None
        if recipe.rim_stain > 0:
            stain = SmoothNoise3(solid, 8.0, np.random.default_rng(seqs[1]))
        return cls(
            recipe,
            np.asarray(solid.centre(), dtype=np.float64),
            growth.chalcedony_end / 1.3 if recipe.centre == "solid" else growth.chalcedony_end,
            mottle,
            stain,
        )

    def fracture_distance(self, points: FloatArray) -> FloatArray:
        out = np.full(len(points), np.inf)
        for f in self.recipe.fractures:
            out = np.minimum(out, f.distance(points, self.solid_centre))
        return out

    def __call__(self, points: FloatArray, fields: Fields, inside: BoolArray) -> Fields:
        rec = self.recipe
        phase = fields.phase.copy()
        imp = np.nan_to_num(fields.impurity, nan=0.0).astype(np.float32)
        trans = fields.translucency.copy()
        pig = (
            np.full(len(points), Pigment.IRON, dtype=np.uint8)
            if fields.pigment is None
            else fields.pigment.copy()
        )
        t_src = fields.t_smooth if fields.t_smooth is not None else fields.t_dep
        t = np.nan_to_num(t_src, nan=0.0).astype(np.float64)
        chal = inside & (phase == Phase.CHALCEDONY)
        quartz = inside & (phase == Phase.QUARTZ)

        imp[quartz] *= QUARTZ_IRON_RATIO
        depth = np.clip(t / self.chalcedony_end, 0, 1)  # 0 at the wall, 1 at the centre
        trans[chal] += (rec.white_fade * depth[chal] * (1 - trans[chal])).astype(np.float32)

        def stain(mask: NDArray[np.bool_], amount: NDArray[np.float64], pigment: Pigment) -> None:
            a = np.clip(amount, 0, 1)
            imp[mask] += (a[mask] * (1 - imp[mask])).astype(np.float32)
            pig[mask & (a > 0.05)] = pigment

        if self.mottle_noise is not None and rec.mottle is not None:
            m = rec.mottle
            n = self.mottle_noise(points)
            thr = NormalDist().inv_cdf(1 - m.cover)
            band = m.rim_mm * 1.22 / m.scale_mm  # rim width in noise units (|grad n| ~ 1.22/scale)
            inner = np.where(n > thr, m.strength, 0.0)
            amount: NDArray[np.float64] = np.where(n > thr + band, 0.6 * m.strength, inner)
            # Milky (porous) bands take more than clear ones, so bands show through
            # a patch (assumption).
            stain(chal, amount * (0.4 + 0.6 * (1 - trans)), Pigment.HEMATITE)

        if self.stain_noise is not None:
            patch = np.clip(0.7 + 0.5 * self.stain_noise(points), 0, 1)
            stain(chal, rec.rim_stain * np.exp(-depth / 0.25) * patch, Pigment.IRON)

        for f in rec.fractures:
            d = f.distance(points, self.solid_centre)
            near = inside & (d < f.halo_mm + 4 * f.pool_mm + f.aperture_mm)
            if not near.any():
                continue
            milky = np.clip(1 - trans / 0.35, 0, 1)
            cavity = np.clip((depth - 0.9) / 0.1, 0, 1)
            porous = np.maximum(milky, cavity)
            reach = f.halo_mm + (f.pool_mm - f.halo_mm) * porous + 1e-3
            stain(
                near & chal,
                f.stain * np.exp(-np.maximum(d - f.aperture_mm / 2, 0) / reach),
                Pigment.IRON,
            )
            trace = near & (d < f.aperture_mm / 2) & (chal | quartz)
            match f.fill:
                case "open":
                    phase[trace] = Phase.VOID
                case "clear":
                    trans[trace] = 1.0
                    imp[trace] = 0.0
                case "iron":
                    imp[trace] = 0.9
                    pig[trace] = Pigment.IRON
                    phase[trace] = Phase.CHALCEDONY

        return replace(fields, phase=phase, impurity=imp, translucency=trans, pigment=pig)


RIDGE_MEAN = math.sqrt(2 / math.pi)  # mean of |n| for unit normal noise


@dataclass(frozen=True, eq=False)
class BandRoughness:
    """Reads the growth model at points shifted along the local band normal
    (`nodule.specimen.Warp`), so band edges roughen while the outline stays.

    The shift is A(d) R(n(w + d (s/r) e)), where d is the depth from the wall
    (growth time times growth_mm), w = p - d g the point's foot on the wall (g the
    unit inward normal, from growth time by central differences), n smooth world
    noise of scale s, r the renew depth and e a fixed unit vector, so the pattern
    holds along each normal and drifts into a new one over r. A(d) = A (1 - exp(-d/f))
    grows inward; R is the noise itself, or |n| - its mean for sharp creases. The
    shift varies slowly along the normal, so bands do not fold (`fold_share` checks
    it); along a band it may change fast, which gives the chevrons. Near the
    centre line, where bands from opposite walls meet and g flips, the shift fades
    out: there growth time rises more slowly than one growth_mm per mm over the
    0.5 mm difference step (and on floors, which grow faster than the walls). A form
    chosen for the look, not a growth mechanism (ADR 0002): an assumption. It also
    fades out over the last bands before a quartz centre, keeping that contact sharp."""

    roughness: Roughness
    growth_mm: float
    noise: SmoothNoise3
    drift: FloatArray
    quartz_at: float | None = None  # growth time at which quartz starts, if it does
    step_mm: float = 0.5

    @classmethod
    def of(cls, roughness: Roughness, solid: Solid, growth: Growth, seed: int) -> BandRoughness:
        rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[2])
        noise = SmoothNoise3(solid, roughness.scale_mm, rng)
        return cls(roughness, growth.growth_mm, noise, _unit(rng), growth.quartz_at)

    def __call__(
        self, points: FloatArray, growth_time: Callable[[FloatArray], FloatArray]
    ) -> FloatArray:
        r = self.roughness
        g, depth, weight = self.frame(points, growth_time)
        foot = points - depth[:, None] * g
        n = self.noise(foot + (depth * r.scale_mm / r.renew_mm)[:, None] * self.drift)
        shape = np.abs(n) - RIDGE_MEAN if r.ridged else n
        shift = r.amplitude_mm * (1 - np.exp(-depth / r.fade_mm)) * shape * weight
        return points + shift[:, None] * g

    def frame(
        self, points: FloatArray, growth_time: Callable[[FloatArray], FloatArray]
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Unit inward normal g, depth from the wall (mm) and the shift's weight
        (0 where growth time is undefined or near the centre line, else up to 1).
        Differences are one-sided where one side has no growth time (quartz, voids)."""
        h = self.step_mm
        t = growth_time(points)
        grad = np.full_like(points, np.nan)
        for k in range(3):
            e = np.zeros(3)
            e[k] = h
            up, down = growth_time(points + e), growth_time(points - e)
            both = np.isfinite(up) & np.isfinite(down)
            grad[:, k] = np.where(
                both, (up - down) / (2 * h), np.where(np.isfinite(up), up - t, t - down) / h
            )
        size = np.linalg.norm(grad, axis=1)
        ok = np.isfinite(t) & np.isfinite(size) & (size > 0)
        g = np.where(ok[:, None], grad / np.where(ok, size, 1.0)[:, None], 0.0)
        depth = np.where(ok, t * self.growth_mm, 0.0)
        rate = np.where(ok, size * self.growth_mm, 0.0)  # about 1 in wall bands
        weight = np.clip((rate - 0.5) / 0.4, 0.0, 1.0)
        if self.quartz_at is not None:
            # Fade out over the last bands before quartz, so the contact stays
            # sharp: reading across it would scatter quartz and chalcedony.
            left = (self.quartz_at - np.where(ok, t, 0.0)) * self.growth_mm
            span = max(1.5 * self.roughness.amplitude_mm, 0.1)
            weight *= np.clip((left - 0.05) / span, 0.0, 1.0)
        return g, depth, weight


def build(recipe: StoneRecipe, max_slope: float = MAX_SLOPE) -> tuple[Specimen, VarietyLayers]:
    if not 0 < max_slope < 1:
        raise ValueError("max_slope must lie in (0, 1): at 1 or above, bands can fold")
    config, growth = volume_config(recipe, max_slope)
    specimen = Specimen.build(SpecimenConfig(volume=config))
    layers = VarietyLayers.of(recipe, specimen.volume.solid, growth)
    warp = None
    if recipe.roughness is not None:
        warp = BandRoughness.of(recipe.roughness, specimen.volume.solid, growth, recipe.seed)
    return replace(specimen, layers=layers, warp=warp), layers


def cut_of(specimen: Specimen, recipe: StoneRecipe, spacing_mm: float) -> SpecimenCut:
    return specimen.cut(specimen.plane(recipe.cut), spacing_mm)


# --- checks, each with a negative control in the tests (ADR 0018 s.7) -----------------------


def has_cavity(cut: SpecimenCut, least: float = 0.002) -> bool:
    """Open space inside the stone, beyond a share `least` of the face."""
    inside = cut.inside
    return bool(inside.any() and (cut.fields.phase[inside] == Phase.VOID).mean() > least)


def _perimeter(mask: NDArray[np.bool_]) -> int:
    return int((mask[1:] != mask[:-1]).sum() + (mask[:, 1:] != mask[:, :-1]).sum())


def _box_blur(a: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    for axis in (0, 1):
        c = np.cumsum(
            np.pad(a, [(k + 1, k) if i == axis else (0, 0) for i in (0, 1)], mode="edge"), axis=axis
        )
        size = int(c.shape[axis])
        hi = np.take(c, np.arange(2 * k + 1, size, dtype=np.intp), axis=axis)
        lo = np.take(c, np.arange(0, size - 2 * k - 1, dtype=np.intp), axis=axis)
        a = (hi - lo) / (2 * k + 1)
    return a


def lining_roughness(cut: SpecimenCut, smooth_mm: float = 1.0) -> float:
    """Faceted-lining check: the cavity's boundary length over that of the same
    cavity smoothed over `smooth_mm`. A saw-tooth quartz lining scores well above 1;
    a smooth chalcedony wall near 1. NaN without a cavity."""
    void = cut.inside & (cut.fields.phase == Phase.VOID)
    if void.sum() < 50:
        return math.nan
    k = max(1, round(smooth_mm / cut.spacing_mm))
    smooth = _box_blur(_box_blur(void.astype(np.float64), k), k) > 0.5
    return _perimeter(void) / max(_perimeter(smooth), 1)


def flat_bands(cut: SpecimenCut, least_span: float = 0.4, most_spread_mm: float = 0.25) -> int:
    """Waterline check: bands whose top edge stays level (spread under
    `most_spread_mm`) across at least `least_span` of the stone's width. Height is
    measured along the plane's projection of world up."""
    up = cut.plane.in_plane(np.array([0.0, 0.0, 1.0]))
    if float(np.linalg.norm(up)) < 0.5:
        return 0
    up = up / float(np.linalg.norm(up))
    ss, tt = np.meshgrid(cut.s, cut.t)
    height = ss * up[0] + tt * up[1]
    across = ss * up[1] - tt * up[0]
    col = np.floor(across / cut.spacing_mm).astype(np.int64)
    inside = cut.inside & (cut.fields.phase == Phase.CHALCEDONY)
    if not inside.any():
        return 0
    width_cols = col[inside].max() - col[inside].min() + 1
    band = cut.fields.band
    count = 0
    for b in np.unique(band[inside]):
        m = inside & (band == b)
        c, h = col[m], height[m]
        order = np.lexsort((h, c))
        c, h = c[order], h[order]
        last = np.r_[c[1:] != c[:-1], True]
        tops, cols = h[last], c[last]
        if len(cols) < least_span * width_cols:
            continue
        if float(np.std(tops)) < most_spread_mm:
            count += 1
    return count


def fracture_stain(cut: SpecimenCut, layers: VarietyLayers, near_mm: float = 0.5) -> float:
    """Stained-fracture check: mean impurity of chalcedony within `near_mm` of a
    crack, less the median impurity beyond 3 mm. NaN without a crack in the face."""
    ss, tt = np.meshgrid(cut.s, cut.t)
    d = layers.fracture_distance(cut.plane.points(ss, tt)).reshape(ss.shape)
    chal = cut.inside & (cut.fields.phase == Phase.CHALCEDONY)
    imp = np.nan_to_num(cut.fields.impurity, nan=0.0)
    near, far = chal & (d < near_mm), chal & (d > 3.0)
    if near.sum() < 20 or far.sum() < 20:
        return math.nan
    return float(imp[near].mean() - np.median(imp[far]))


def shift_by_depth(
    cut: SpecimenCut, plain: SpecimenCut, growth_mm: float, depths_mm: tuple[float, float, float]
) -> tuple[float, float]:
    """Roughness-grows-inward check: the RMS shift of the bands (mm), from the same
    stone cut without roughness (`plain`, same plane and spacing), in chalcedony at
    depth `depths_mm[0]`-`[1]` from the wall and at `[1]`-`[2]`. Depth is read from
    the plain cut's growth time."""
    f, g = cut.fields, plain.fields
    tf = f.t_smooth if f.t_smooth is not None else f.t_dep
    tg = g.t_smooth if g.t_smooth is not None else g.t_dep
    both = cut.inside & (f.phase == Phase.CHALCEDONY) & (g.phase == Phase.CHALCEDONY)
    both &= np.isfinite(tf) & np.isfinite(tg)
    depth = np.asarray(tg, dtype=np.float64) * growth_mm
    shift = (np.asarray(tf, dtype=np.float64) - np.asarray(tg, dtype=np.float64)) * growth_mm
    out: list[float] = []
    for lo, hi in ((depths_mm[0], depths_mm[1]), (depths_mm[1], depths_mm[2])):
        m = both & (depth >= lo) & (depth < hi)
        out.append(float(np.sqrt(np.mean(shift[m] ** 2))) if m.sum() > 20 else math.nan)
    return out[0], out[1]


def fold_share(
    specimen: Specimen, samples: int = 4000, step_mm: float = 0.05, seed: int = 0
) -> float:
    """No-fold check: at random points in the stone, step `step_mm` inward along
    the unwarped band normal and read growth time through the warp at both ends.
    Returns the share of steps (among those where the unwarped bands rise inward)
    where the warped bands fall instead: bands met in reverse, a fold. Zero for an
    unwarped stone by construction."""
    warp = specimen.warp
    if not isinstance(warp, BandRoughness):
        raise ValueError("fold_share reads a stone warped by BandRoughness")
    solid = specimen.volume.solid
    rng = np.random.default_rng(seed)
    lo, hi = (np.asarray(b, dtype=np.float64) for b in solid.bounds())
    pts = lo + rng.uniform(size=(samples, 3)) * (hi - lo)
    pts = pts[solid.contains(pts)]
    gt = specimen.growth_time
    g, _, _ = warp.frame(pts, gt)
    ahead = pts + step_mm * g
    rising = gt(ahead) - gt(pts) > 0
    if not rising.any():
        return 0.0
    w0, w1 = gt(warp(pts, gt)), gt(warp(ahead, gt))
    folded = rising & np.isfinite(w0) & np.isfinite(w1) & (w1 < w0)
    return float(folded.sum() / rising.sum())


def stray_quartz(cut: SpecimenCut) -> float:
    """Sharp-contact check: the share of quartz pixels with fewer than two quartz
    neighbours (of four), i.e. quartz scattered into the chalcedony."""
    q = cut.inside & (cut.fields.phase == Phase.QUARTZ)
    if q.sum() < 50:
        return math.nan
    p = np.pad(q, 1)
    neighbours = p[:-2, 1:-1].astype(int) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    return float((q & (neighbours < 2)).sum() / q.sum())
