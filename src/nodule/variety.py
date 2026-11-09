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
from typing import Literal, NamedTuple

import numpy as np
from agatesim.fields import Fields, Phase, Pigment
from agatesim.grid import FloatArray
from agatesim.volume.config import SliceConfig, SolidConfig, VolumeConfig
from agatesim.volume.models import SmoothNoise3
from agatesim.volume.slicing import Plane
from agatesim.volume.solids import BoolArray, Rotated, Solid, SphereUnion
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from nodule._scipy import largest_patch_size
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
MOTTLE_SHARE = 1 / 3  # the first draw; kept: MOTTLE_STONES of these
# The owner on mottle (2026-11-08, ADR 0018 s.10; all `effective`): a third to a
# half as often; most patches small, a large one (at least LARGE_PATCH of the face)
# on 15-25 % of mottled stones; dark rims on about 1 patch in 4, lighter on the rest.
MOTTLE_STONES = (1, 2)  # per 12 stones (was 4)
LARGE_MOTTLE_CHANCE = 1 / 5
LARGE_PATCH = 0.25  # share of the face
LARGE_PATCH_MAX = 0.6  # beyond this a patch reads as a film over the face
# Noise scale as a share of the stone's size; patches come out about 3-4 x wider.
SMALL_MOTTLE = ((0.04, 0.06), (0.15, 0.3))  # (scale, cover)
LARGE_MOTTLE = ((0.2, 0.35), (0.35, 0.5))
MOTTLE_TRIES = 16  # noise draws until the face shows the drawn class
DARK_RIM_SHARE = 0.25  # of patches (by area of the rim-darkness noise)
LIGHT_RIM = 0.4  # a light rim's darkening, as a share of a dark rim's
# Edges (the owner, 2026-11-08): about half of mottled stones have soft, graded
# patches, the rest hard-edged ones, and a few (5 %) both. A soft patch has no rim
# and rises to SOFT_PEAK x strength over SOFT_WIDTH in noise units, centred on the
# edge (about 0.4-0.8 x the noise scale in mm). All `effective`.
EDGE_SHARES: dict[str, float] = {"hard": 0.475, "soft": 0.475, "mixed": 0.05}
SOFT_WIDTH = (0.5, 1.0)
SOFT_PEAK = 0.7
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
# Cracks (the owner, 2026-11-08: ruler-straight black cracks look fake). Open
# cracks are no longer drawn: the sources describe cracks healed by chalcedony,
# quartz or iron oxides [@goetze2020, §5.1; @mrozik2023, pp. 14, 27]. Each crack
# is a shallow bowl of one gentle curvature: it rises by a sagitta at its edge, a
# share of its radius (`effective`, for the eye; no source gives a form). The owner
# judged noise bends too strong (2026-11-08): only slight curves, rarely if ever
# switching direction. A bowl's trace in any face is one arc, never an S.
SAG_SHARE = (0.03, 0.1)  # steepest slope 2 x share, at most 0.2
# The owner on crack form and mix (2026-11-08, ADR 0018 s.10; all `effective`, for the
# eye: no source gives crack counts, lengths or fills for agates). Healed cracks in a
# stone usually share one shade; some are straight; most are short or partial, and
# long, central or crossing cracks are rare (cracking that deep usually splits the
# stone). Shares below are among cracked stones.
CRACK_FILLS: dict[str, float] = {"clear": 0.6, "iron": 0.4}  # red stones: RED_CRACK_FILLS
RED_CRACK_FILLS: dict[str, float] = {"clear": 0.9, "iron": 0.1}  # iron hardly shows on red
MIXED_FILL_CHANCE = 1 / 12  # one crack takes the other fill (a later crack)
STAIN_JITTER = 0.1  # each crack's iron within this of the stone's
STRAIGHT_SHARE = 1 / 3  # cracks with no bend
PARTIAL_CRACKS = (3, 8)  # per cracked stone: inward from the rim
PARTIAL_REACH = (0.2, 0.5)  # how far in, as a share of the inradius
LONG_CRACKS: dict[int, float] = {0: 24, 1: 11, 2: 1}  # per 36 cracked stones
# The owner on the strip of cracked stones (2026-11-08): two long cracks crossing
# through the centre very rare (2 long was 1 in 12 cracked stones); iron stain too
# intense (was 0.4-1.0, the iron fill's own trace 0.9); no long crack through a
# cavity; and the cutter avoids cutting along a crack (a slab would split there),
# since a crack near the face's plane shows as a wide smear. All `effective`.
STAIN = (0.25, 0.6)
IRON_TRACE = 0.2  # the iron fill's trace: impurity = stain + this, at most 0.8
CRACK_CUT_ANGLE = 30.0  # degrees: least angle between the cut and a crack it meets
CAVITY_MARGIN_MM = 0.5  # cracks keep this far from the open cavity (estimate)
CRACK_TRIES = 20  # redraws of a crack that would enter the cavity; then it is left out
LONG_RADIUS = (0.6, 1.2)  # share of the inradius
LONG_OFFSET = (0.2, 0.6)  # centre's distance from the stone's centre, share of inradius
# Cuts of overlapping-sphere outlines: the cutter's choice. A cut far off a
# sphere's centre shows its shells widening toward the middle, and one that only
# grazes a sphere leaves a nub on the outline; the owner judged both atypical.
CLEAN_CUT = 0.3  # largest |offset| / radius of any sphere the cut crosses
CUT_TRIES = 64
ODD_CUTS = (0, 1)  # per 12 stones: keep the first cut drawn, however oblique
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
    centre: Centre
    centre_width: float = Field(ge=0, le=1, description="Share of the inradius.")
    lining: float = Field(ge=0, le=1, description="Quartz-lined: share of the centre in quartz.")
    tint: Tint = "none"
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
    mottle_pool = plan.permutation(n)
    mottled = set(mottle_pool[: round(MOTTLE_SHARE * n)].tolist())
    odd_ok = [i for i, o in enumerate(outlines) if o in ("vesicle", "irregular")]
    odd = set(plan.permutation(odd_ok)[: _scaled(plan, ODD_CUTS, n)].tolist())
    # Drawn last, so the stones drawn before this choice keep their plan.
    mild = [i for i, w in enumerate(warps) if w == "mild"]
    full = set(plan.permutation(mild)[: _scaled(plan, FULL_WOBBLE_STONES, n)].tolist())
    # Mottle kept on fewer stones: the first of the first draw (whose stones still
    # draw their mottle, so their other draws stand).
    kept = set(mottle_pool[: _scaled(plan, MOTTLE_STONES, n)].tolist())
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
            keep_mottle=i in kept,
            odd_cut=i in odd,
            full_wobble=i in full,
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


def enters_cavity(solid: Solid, f: FractureLayer, cavity_mm: float) -> bool:
    """Whether crack f comes within CAVITY_MARGIN_MM (plus its largest bend) of the
    open cavity: the points more than `cavity_mm` from the nearest wall."""
    n = np.asarray(f.normal) / float(np.linalg.norm(f.normal))
    e1 = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= float(np.linalg.norm(e1))
    e2 = np.cross(n, e1)
    rho = f.radius_mm * np.sqrt(np.linspace(0.0, 1.0, 16))[:, None]
    phi = np.linspace(0.0, 2 * math.pi, 48, endpoint=False)[None, :]
    a = (rho * np.cos(phi)).reshape(-1, 1)
    b = (rho * np.sin(phi)).reshape(-1, 1)
    pts: FloatArray = solid.centre() + np.asarray(f.centre_mm) + a * e1 + b * e2
    margin = CAVITY_MARGIN_MM + SAG_SHARE[1] * f.radius_mm
    return bool(np.any(solid.wall_distance(pts) > cavity_mm - margin))


def draw_fractures(
    rng: np.random.Generator, solid: Solid, red: bool, cavity_mm: float | None = None
) -> tuple[FractureLayer, ...]:
    """A cracked stone's cracks (ADR 0018 s.7, s.10): 3-8 partial cracks running in
    from the rim, 0-3 nicks 1-4 mm long at the rim and, on 1 cracked stone in 3,
    one or (very rarely) two long cracks off the centre. All share one fill and about
    one iron strength, except that 1 cracked stone in 12 has one crack of the other
    fill. None is open; 1 in 3 is straight and the rest bend gently. The bends are
    drawn last, so the cracks lie where they were first drawn. With an open cavity
    (`cavity_mm` in from the wall), a crack that would reach it is redrawn, and
    left out after CRACK_TRIES."""
    r = inradius(solid)
    shares = RED_CRACK_FILLS if red else CRACK_FILLS
    fill: Fill = str(rng.choice(list(shares), p=_shares(shares)))  # type: ignore[assignment]
    stain = float(rng.uniform(*STAIN))

    def shade() -> float:
        return float(np.clip(stain + rng.uniform(-STAIN_JITTER, STAIN_JITTER), 0.0, 1.0))

    def from_rim(radius_mm: float, aperture: tuple[float, float]) -> FractureLayer:
        radial = _unit(rng)
        normal = np.cross(radial, _unit(rng))  # the crack runs inward from the rim
        return FractureLayer(
            centre_mm=tuple(rim_point(solid, radial).tolist()),
            normal=tuple((normal / float(np.linalg.norm(normal))).tolist()),
            radius_mm=radius_mm,
            aperture_mm=_log_uniform(rng, *aperture),
            fill=fill,
            stain=shade(),
            halo_mm=float(rng.uniform(0.0, 0.4)),
            pool_mm=float(rng.uniform(0.5, 2.0)),
        )

    def clear_of_cavity(draw: Callable[[], FractureLayer]) -> FractureLayer | None:
        for _ in range(CRACK_TRIES if cavity_mm is not None else 1):
            f = draw()
            if cavity_mm is None or not enters_cavity(solid, f, cavity_mm):
                return f
        return None

    def long_crack() -> FractureLayer:
        offset = float(rng.uniform(*LONG_OFFSET)) * r
        return FractureLayer(
            centre_mm=tuple((offset * _unit(rng)).tolist()),
            normal=tuple(_unit(rng).tolist()),
            radius_mm=float(rng.uniform(*LONG_RADIUS) * r),
            aperture_mm=_log_uniform(rng, 0.05, 0.4),
            fill=fill,
            stain=shade(),
            halo_mm=float(rng.uniform(0.0, 0.5)),
            pool_mm=float(rng.uniform(1.0, 3.0)),  # half the 2-6 mm pools
        )

    drawn: list[FractureLayer | None] = []
    n_long = int(
        rng.choice(
            list(LONG_CRACKS), p=[v / sum(LONG_CRACKS.values()) for v in LONG_CRACKS.values()]
        )
    )
    for _ in range(n_long):
        drawn.append(clear_of_cavity(long_crack))
    for _ in range(int(rng.integers(PARTIAL_CRACKS[0], PARTIAL_CRACKS[1] + 1))):
        reach = float(rng.uniform(*PARTIAL_REACH)) * r
        drawn.append(clear_of_cavity(lambda reach=reach: from_rim(reach, (0.05, 0.3))))
    main = sum(f is not None for f in drawn)  # long and partial; the nicks follow
    for _ in range(int(rng.integers(0, 4))):
        size = float(rng.uniform(1.0, 4.0))
        drawn.append(clear_of_cavity(lambda size=size: from_rim(size, (0.05, 0.2))))
    out = [f for f in drawn if f is not None]
    if not out:
        return ()
    if rng.uniform() < MIXED_FILL_CHANCE:
        k = int(rng.integers(max(main, 1)))  # a long or partial crack, so it shows
        other: Fill = "iron" if fill == "clear" else "clear"
        out[k] = out[k].model_copy(update={"fill": other})
    return tuple(
        f.model_copy(
            update={
                "sagitta_mm": 0.0
                if rng.uniform() < STRAIGHT_SHARE
                else float(rng.uniform(*SAG_SHARE)) * f.radius_mm
            }
        )
        for f in out
    )


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
    keep_mottle: bool = True,
    odd_cut: bool = False,
    full_wobble: bool = False,
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
            if not full_wobble:
                wobble *= MILD_WOBBLE_SHARE
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
    recipe = StoneRecipe(
        index=index,
        seed=int(rng.integers(0, 2**31)),
        outline=outline,  # type: ignore[arg-type]
        solid=solid_config,
        centre=centre,  # type: ignore[arg-type]
        centre_width=width,
        lining=float(rng.uniform(0.3, 0.7)),
        tint=tint,
        floor=float(rng.uniform(*FLOOR_HEIGHT)) if floor else 0.0,
        colour=colour,  # type: ignore[arg-type]
        band_mm=0.12 if outline == "pebble" else _log_uniform(rng, 0.2, 1.0),
        warp=warp,  # type: ignore[arg-type]
        wobble_mm=wobble,
        wobble_scale_mm=scale,
        full_wobble=full_wobble and warp == "mild",
        white_fade=float(rng.uniform(0.2, 0.5)),
        rim_stain=float(rng.uniform(0.5, 0.9)) if colour == "stained" else 0.0,
        mottle=mottle,
        cut=cut,
    )
    if cracked:  # drawn last, as before, so the stone's other draws stand
        cavity = None
        if centre == "quartz-lined":
            w = recipe.centre_width
            cavity = (1 - w) * r + recipe.lining * w * r
        red = colour in ("iron", "carnelian")
        recipe = recipe.model_copy(update={"fractures": draw_fractures(rng, solid, red, cavity)})
    return _refine(recipe, solid, r, odd_cut=odd_cut, keep_mottle=keep_mottle)


def mottle_edge_noise(recipe: StoneRecipe, solid: Solid) -> SmoothNoise3:
    """Which patches of a mixed stone are soft: noise at twice the patch scale."""
    m = recipe.mottle
    assert m is not None
    seq = np.random.SeedSequence(recipe.seed).spawn(4)[2].spawn(MOTTLE_TRIES + 2)[-1]
    return SmoothNoise3(solid, 2 * m.scale_mm, np.random.default_rng(seq))


def mottle_fields(
    recipe: StoneRecipe, solid: Solid, *, rim: bool = True
) -> tuple[SmoothNoise3, SmoothNoise3 | None]:
    """The mottle's patch noise (variant 0: the stone's first stream, as before) and
    the rim-darkness noise, at twice the patch scale so a patch's rim is mostly of
    one kind. Both in world coordinates (resolution-independent)."""
    m = recipe.mottle
    assert m is not None
    root = np.random.SeedSequence(recipe.seed).spawn(4)
    extra = root[2].spawn(MOTTLE_TRIES + 1)
    patch_seq = root[0] if m.variant == 0 else extra[m.variant]
    patch = SmoothNoise3(solid, m.scale_mm, np.random.default_rng(patch_seq))
    if not rim:
        return patch, None
    return patch, SmoothNoise3(solid, 2 * m.scale_mm, np.random.default_rng(extra[0]))


def largest_patch(recipe: StoneRecipe, solid: Solid, spacing_mm: float = 0.5) -> float:
    """Mottle-size check: the largest connected mottle patch on the cut face, as a
    share of the face (solid and plane only, before bands and centre)."""
    m = recipe.mottle
    if m is None:
        return 0.0
    plane = Plane.of(recipe.cut, solid.centre())
    half = 0.55 * recipe_size(solid)
    g = np.arange(-half, half, spacing_mm)
    ss, tt = np.meshgrid(g, g)
    pts = plane.points(ss, tt).reshape(-1, 3)
    inside = solid.contains(pts).reshape(ss.shape)
    if not inside.any():
        return 0.0
    noise, _ = mottle_fields(recipe, solid, rim=False)
    patch = inside & (noise(pts).reshape(ss.shape) > NormalDist().inv_cdf(1 - m.cover))
    return largest_patch_size(patch) / float(inside.sum())


def _mottle_size(rng: np.random.Generator, recipe: StoneRecipe, solid: Solid) -> Mottle:
    """Small or (1 in 5) large patches, then the first noise draw whose face shows
    that class, a large patch at most LARGE_PATCH_MAX (else the last); the face is
    the cutter's, so draw after the cut."""
    m = recipe.mottle
    assert m is not None
    large = bool(rng.uniform() < LARGE_MOTTLE_CHANCE)
    (scale, cover) = LARGE_MOTTLE if large else SMALL_MOTTLE
    size = recipe_size(solid)
    m = m.model_copy(
        update={
            "scale_mm": float(rng.uniform(*scale)) * size,
            "cover": float(rng.uniform(*cover)),
            "large": large,
        }
    )
    for v in range(MOTTLE_TRIES):
        m = m.model_copy(update={"variant": v})
        share = largest_patch(recipe.model_copy(update={"mottle": m}), solid)
        if (LARGE_PATCH <= share <= LARGE_PATCH_MAX) if large else share < LARGE_PATCH:
            break
    edge = str(rng.choice(list(EDGE_SHARES), p=_shares(EDGE_SHARES)))
    return m.model_copy(update={"edge": edge, "soft_width": float(rng.uniform(*SOFT_WIDTH))})


def _refine(
    recipe: StoneRecipe, solid: Solid, r: float, *, odd_cut: bool, keep_mottle: bool = True
) -> StoneRecipe:
    """Floor bands, the cutter's choice and band roughness (ADR 0018 s.9), drawn
    from the recipe's own stream so that the rest of the recipe stays as drawn."""
    rng = np.random.default_rng(np.random.SeedSequence(recipe.seed).spawn(4)[3])
    update: dict[str, object] = {}
    if recipe.floor:
        thick = rng.uniform() < THICK_FLOOR_CHANCE
        update["floor_supply"] = 1.0 if thick else _log_uniform(rng, *FLOOR_SUPPLY)
    if recipe.outline != "thunder-egg" and not odd_cut:
        update["cut"] = clean_cut(rng, solid, recipe.cut, r, recipe.fractures)
    elif recipe.outline == "thunder-egg" and recipe.fractures:
        update["cut"] = thunder_egg_cut(rng, solid, recipe.cut, r, recipe.fractures)
    if recipe.fractures:
        # A stone cracked along the saw's path would split; the cutter picks another
        # stone. So cracks the chosen cut still runs along are left out.
        cut = update.get("cut", recipe.cut)
        assert isinstance(cut, SliceConfig)
        kept = tuple(
            f for f in recipe.fractures if crack_cut_angle(solid, cut, (f,)) >= CRACK_CUT_ANGLE
        )
        if kept != recipe.fractures:
            update["fractures"] = kept
    level = str(rng.choice(list(ROUGHNESS_SHARES), p=_shares(ROUGHNESS_SHARES)))
    # Not on cracked stones (a warp takes no slipping fractures) or the pebble.
    if level != "none" and not recipe.fractures and recipe.outline != "pebble":
        update["roughness"] = Roughness(
            amplitude_mm=_log_uniform(rng, *ROUGHNESS_MM[level]),
            scale_mm=float(rng.uniform(1.5, 3.0)),
            ridged=False,
        )
    if recipe.mottle is not None:
        if keep_mottle:
            cut = update.get("cut", recipe.cut)
            update["mottle"] = _mottle_size(rng, recipe.model_copy(update={"cut": cut}), solid)
        else:
            update["mottle"] = None
    return recipe.model_copy(update=update)


def _shares(shares: dict[str, float]) -> list[float]:
    total = sum(shares.values())
    return [v / total for v in shares.values()]


def cut_obliqueness(solid: Solid, cut: SliceConfig) -> float:
    """Largest |offset| / radius over the spheres of an overlapping-sphere solid
    that the cut crosses: 0 through every centre, near 1 where it grazes one
    (a nub on the outline). 0 for other solids."""
    rotation, inner = (solid.rotation, solid.inner) if isinstance(solid, Rotated) else (None, solid)
    if not isinstance(inner, SphereUnion):
        return 0.0
    centres = inner.centres if rotation is None else inner.centres @ rotation.T
    plane = Plane.of(cut, solid.centre())
    z = np.abs((centres - plane.origin) @ plane.normal)
    return float(np.max(np.where(z < inner.radii, z / inner.radii, 0.0)))


def crack_cut_angle(solid: Solid, cut: SliceConfig, fractures: tuple[FractureLayer, ...]) -> float:
    """Smear check: the least angle (degrees) between the cut and any crack whose
    disc comes within its stain's reach of it, less the crack's largest bend; 90 if
    none does. A crack near
    the face's plane shows as a wide smear (trace width / sin angle)."""
    plane = Plane.of(cut, solid.centre())
    m = np.asarray(plane.normal)
    least = 90.0
    for f in fractures:
        n = np.asarray(f.normal) / float(np.linalg.norm(f.normal))
        cos = min(abs(float(n @ m)), 1.0)
        gap = abs(float(m @ (solid.centre() + np.asarray(f.centre_mm) - plane.origin)))
        stain_mm = f.halo_mm + 2 * f.pool_mm  # a crack just under the face stains it
        if gap > f.radius_mm * math.sqrt(1 - cos**2) + f.sagitta_mm + stain_mm:
            continue
        bend = math.degrees(math.atan(2 * f.sagitta_mm / f.radius_mm))
        least = min(least, max(math.degrees(math.acos(cos)) - bend, 0.0))
    return least


def _cut_fault(solid: Solid, cut: SliceConfig, fractures: tuple[FractureLayer, ...]) -> float:
    """0 for a clean cut; otherwise how far it misses, over both rules."""
    oblique = max(cut_obliqueness(solid, cut) - CLEAN_CUT, 0.0) / (1 - CLEAN_CUT)
    angle = crack_cut_angle(solid, cut, fractures)
    return oblique + max(CRACK_CUT_ANGLE - angle, 0.0) / CRACK_CUT_ANGLE


def clean_cut(
    rng: np.random.Generator,
    solid: Solid,
    first: SliceConfig,
    r: float,
    fractures: tuple[FractureLayer, ...] = (),
) -> SliceConfig:
    """The first of up to CUT_TRIES cuts (the drawn one first) that crosses every
    sphere it meets within CLEAN_CUT of its centre and meets no crack at under
    CRACK_CUT_ANGLE, else the one that misses least."""
    best, least = first, _cut_fault(solid, first, fractures)
    for _ in range(CUT_TRIES - 1):
        if least == 0:
            break
        cut = SliceConfig(
            tilt_deg=float(rng.uniform(70, 90)),
            azimuth_deg=float(rng.uniform(0, 360)),
            offset_mm=float(rng.uniform(-0.15, 0.15) * r),
        )
        b = _cut_fault(solid, cut, fractures)
        if b < least:
            best, least = cut, b
    return best


def thunder_egg_cut(
    rng: np.random.Generator,
    solid: Solid,
    first: SliceConfig,
    r: float,
    fractures: tuple[FractureLayer, ...],
) -> SliceConfig:
    """A thunder egg is cut across its star (tilt 0); the cutter moves the cut along
    the axis, within 0.1 r, to miss cracks lying near its plane."""
    best, least = first, crack_cut_angle(solid, first, fractures)
    for _ in range(CUT_TRIES - 1):
        if least >= CRACK_CUT_ANGLE:
            break
        cut = first.model_copy(update={"offset_mm": float(rng.uniform(-0.1, 0.1) * r)})
        a = crack_cut_angle(solid, cut, fractures)
        if a > least:
            best, least = cut, a
    return best


# --- recipe to specimen ------------------------------------------------------------------


@dataclass(frozen=True)
class Growth:
    """The growth run that gives the recipe's centre: total growth and the time
    (share of the run) at which silica drops to quartz supply."""

    growth_mm: float
    quartz_at: float | None
    start: float = 0.0  # growth time at which the walls start (after a floor)

    @property
    def chalcedony_end(self) -> float:
        """Growth time at which the last chalcedony band lies down."""
        return self.quartz_at if self.quartz_at is not None else 1.0

    def after_floor(self, floor_mm: float, supply: float) -> Growth:
        """This run (the walls) with a floor first. The floor grows floor_mm at
        `supply` over time W (g' supply W = floor_mm); the walls do not grow while
        it settles (agatesim fills the open space from its lowest point), then grow
        as this run does (g' (1 - W) = g). The caller sizes this run to the cavity
        left above the floor (ADR 0018 s.10), so the floor keeps its centre."""
        pre = floor_mm / supply
        g = pre + self.growth_mm
        w = pre / g
        quartz_at = None if self.quartz_at is None else w + (1 - w) * self.quartz_at
        return Growth(g, quartz_at, w)


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


def floor_growth_mm(share: float, r: float) -> float:
    """Growth (mm) whose volume settles into a floor at `share` of the height of a
    sphere of radius r: agatesim's settling conserves volume, so the floor's cap
    (pi h^2 (3r - h) / 3, h = 2 r share) equals the shell of that growth. Exact for
    a sphere; for other outlines an approximation (ADR 0018 s.10). At 0.5 it gives
    0.206 r, as the old one-probe scaling did (6.6 / 16 per unit share)."""
    return r * (1.0 - (1.0 - share**2 * (3.0 - 2.0 * share)) ** (1 / 3))


def volume_config(recipe: StoneRecipe, max_slope: float = MAX_SLOPE) -> tuple[VolumeConfig, Growth]:
    solid = recipe.solid.build()
    r = inradius(solid)
    growth = growth_for(recipe, r)
    if recipe.floor:
        # The walls are sized to the cavity left above the floor: in a sphere of
        # radius r, a floor at a share f of the height leaves room for a sphere of
        # radius r (1 - f) exactly; for other outlines an approximation.
        left = r * max(1.0 - recipe.floor, 0.1)
        growth = growth_for(recipe, left).after_floor(
            floor_growth_mm(recipe.floor, r), recipe.floor_supply
        )
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
        # A settling pulse first: floor first [@moxon2020, p. 16]. Settling takes
        # precedence over quartz in agatesim, so supply may drop below the quartz
        # threshold here; low supply gives thin floor bands (THICK_FLOOR_CHANCE).
        pulse: dict[str, object] = {"type": "pulse", "at": 0.0, "width": growth.start}
        schedules["settling"] = {"base": 0.0, "events": [pulse | {"value": 1.0}]}
        if recipe.floor_supply < 1:
            silica.append(pulse | {"value": recipe.floor_supply})
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
    start: float
    mottle_noise: SmoothNoise3 | None
    stain_noise: SmoothNoise3 | None
    rim_noise: SmoothNoise3 | None = None
    edge_noise: SmoothNoise3 | None = None

    @classmethod
    def of(cls, recipe: StoneRecipe, solid: Solid, growth: Growth) -> VarietyLayers:
        seqs = np.random.SeedSequence(recipe.seed).spawn(2)
        mottle = rim = edge = None
        if recipe.mottle is not None:
            mottle, rim = mottle_fields(recipe, solid)
            if recipe.mottle.edge == "mixed":
                edge = mottle_edge_noise(recipe, solid)
        stain = None
        if recipe.rim_stain > 0:
            stain = SmoothNoise3(solid, 8.0, np.random.default_rng(seqs[1]))
        end = growth.chalcedony_end
        if recipe.centre == "solid":  # the last band lies down at about 1/1.3 of the walls' run
            end = growth.start + (end - growth.start) / 1.3
        return cls(
            recipe,
            np.asarray(solid.centre(), dtype=np.float64),
            end,
            growth.start,
            mottle,
            stain,
            rim,
            edge,
        )

    def mottle_softness(self, points: FloatArray) -> FloatArray:
        """1 where mottle patches are soft, 0 where hard; on a mixed stone about
        half its patches are soft."""
        m = self.recipe.mottle
        if m is None or m.edge == "hard":
            return np.zeros(len(points))
        if m.edge == "soft" or self.edge_noise is None:
            return np.ones(len(points))
        return np.clip(0.5 + self.edge_noise(points) / 0.3, 0.0, 1.0)

    def mottle_amount(self, points: FloatArray) -> FloatArray:
        """Mottle stain before the band's porosity: hard patches step up at the
        edge to a rim, then fall to 0.6 x strength inside; soft ones rise
        smoothly across the edge (centred on it, so the patch keeps its size), with
        no rim."""
        m = self.recipe.mottle
        if m is None or self.mottle_noise is None:
            return np.zeros(len(points))
        n = self.mottle_noise(points)
        thr = NormalDist().inv_cdf(1 - m.cover)
        band = m.rim_mm * 1.22 / m.scale_mm  # rim width in noise units (|grad n| ~ 1.22/scale)
        # Rims: dark (as first drawn) on about DARK_RIM_SHARE of patches,
        # lighter (LIGHT_RIM of the darkening) on the rest.
        rim = 0.6 + 0.4 * (LIGHT_RIM + (1 - LIGHT_RIM) * self.rim_darkness(points))
        inner = np.where(n > thr, rim * m.strength, 0.0)
        hard: FloatArray = np.where(n > thr + band, 0.6 * m.strength, inner)
        ramp = np.clip((n - thr) / m.soft_width + 0.5, 0.0, 1.0)
        soft: FloatArray = SOFT_PEAK * m.strength * ramp
        b = self.mottle_softness(points)
        return (1 - b) * hard + b * soft

    def rim_darkness(self, points: FloatArray) -> FloatArray:
        """Mottle rim darkness, 1 dark to 0 light: dark on about DARK_RIM_SHARE of
        the stone (all dark without a rim noise, as first drawn)."""
        if self.rim_noise is None:
            return np.ones(len(points))
        q = NormalDist().inv_cdf(1 - DARK_RIM_SHARE)
        return np.clip((self.rim_noise(points) - q) / 0.3, 0.0, 1.0)

    def crack_distance(self, i: int, points: FloatArray) -> FloatArray:
        """Distance (mm) from points to crack i, bent as drawn."""
        return self.recipe.fractures[i].distance(points, self.solid_centre)

    def fracture_distance(self, points: FloatArray) -> FloatArray:
        out = np.full(len(points), np.inf)
        for i in range(len(self.recipe.fractures)):
            out = np.minimum(out, self.crack_distance(i, points))
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
        # 0 at the wall (and on the floor), 1 at the centre
        depth = np.clip((t - self.start) / (self.chalcedony_end - self.start), 0, 1)
        trans[chal] += (rec.white_fade * depth[chal] * (1 - trans[chal])).astype(np.float32)

        def stain(mask: NDArray[np.bool_], amount: NDArray[np.float64], pigment: Pigment) -> None:
            a = np.clip(amount, 0, 1)
            imp[mask] += (a[mask] * (1 - imp[mask])).astype(np.float32)
            pig[mask & (a > 0.05)] = pigment

        if self.mottle_noise is not None and rec.mottle is not None:
            amount = self.mottle_amount(points)
            # Milky (porous) bands take more than clear ones, so bands show through
            # a patch (assumption).
            stain(chal, amount * (0.4 + 0.6 * (1 - trans)), Pigment.HEMATITE)

        if self.stain_noise is not None:
            patch = np.clip(0.7 + 0.5 * self.stain_noise(points), 0, 1)
            stain(chal, rec.rim_stain * np.exp(-depth / 0.25) * patch, Pigment.IRON)

        for i, f in enumerate(rec.fractures):
            d = self.crack_distance(i, points)
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
                    imp[trace] = min(f.stain + IRON_TRACE, 0.8)
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
    start: float = 0.0  # growth time at which the walls start; the floor stays flat
    step_mm: float = 0.5

    @classmethod
    def of(cls, roughness: Roughness, solid: Solid, growth: Growth, seed: int) -> BandRoughness:
        rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[2])
        noise = SmoothNoise3(solid, roughness.scale_mm, rng)
        return cls(roughness, growth.growth_mm, noise, _unit(rng), growth.quartz_at, growth.start)

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
        depth = np.where(ok, np.maximum(t - self.start, 0.0) * self.growth_mm, 0.0)
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


def floor_bands(cut: SpecimenCut, layers: VarietyLayers) -> tuple[float, float]:
    """Floor check: the floor's height and its mean band width (mm), from the
    chalcedony laid down before the walls start. Height is measured along the
    plane's projection of world up; (0, 0) without a floor."""
    up = cut.plane.in_plane(np.array([0.0, 0.0, 1.0]))
    t = cut.fields.t_dep
    floor = cut.inside & (cut.fields.phase == Phase.CHALCEDONY) & (t < layers.start)
    if not floor.any() or float(np.linalg.norm(up)) < 0.5:
        return 0.0, 0.0
    up = up / float(np.linalg.norm(up))
    ss, tt = np.meshgrid(cut.s, cut.t)
    height = (ss * up[0] + tt * up[1])[floor]
    span = float(height.max() - height.min())
    return span, span / len(np.unique(cut.fields.band[floor]))


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


def _trace_centre(
    cut: SpecimenCut, layers: VarietyLayers, i: int, bin_mm: float
) -> tuple[FloatArray, FloatArray] | None:
    """Crack i's trace in the face as a centre line: the mean offset from the
    trace's main axis in bins of `bin_mm` along it. None if under 3 bins."""
    f = layers.recipe.fractures[i]
    ss, tt = np.meshgrid(cut.s, cut.t)
    d = layers.crack_distance(i, cut.plane.points(ss, tt)).reshape(ss.shape)
    trace = cut.inside & (d < max(f.aperture_mm / 2, cut.spacing_mm))
    xy = np.stack([ss[trace], tt[trace]], axis=1)
    if len(xy) < 10:
        return None
    xy = xy - xy.mean(axis=0)
    axes = np.linalg.svd(xy, full_matrices=False)[2]
    along, off = xy @ axes[0], xy @ axes[1]
    k = np.floor((along - along.min()) / bin_mm).astype(np.int64)
    counts = np.bincount(k)
    keep = counts >= 3
    if keep.sum() < 3:
        return None
    sums = np.zeros(len(counts))
    np.add.at(sums, k, off)
    x: FloatArray = (np.arange(len(counts))[keep] + 0.5) * bin_mm
    return x, sums[keep] / counts[keep]


def _fit_rms(x: FloatArray, y: FloatArray, degree: int) -> float:
    resid: FloatArray = y - np.polyval(np.polyfit(x, y, degree), x)
    return float(np.sqrt(np.mean(resid**2)))


def crack_bend(cut: SpecimenCut, layers: VarietyLayers, i: int, bin_mm: float = 1.0) -> float:
    """Bent-crack check: the RMS offset (mm) of crack i's trace in the face from
    its best straight line, so a straight trace of any width gives about 0. NaN
    if the trace spans less than 3 bins."""
    line = _trace_centre(cut, layers, i, bin_mm)
    return math.nan if line is None else _fit_rms(*line, 1)


def graded_share(layers: VarietyLayers, points: FloatArray) -> float:
    """Soft-edge check (ADR 0018 s.10): among points inside mottle patches, the share
    whose stain lies between 0.05 and 0.55 x strength, which a hard patch (0, then a
    rim of at least 0.76 x, then 0.6 x) never shows."""
    m = layers.recipe.mottle
    if m is None:
        return math.nan
    a = layers.mottle_amount(points) / m.strength
    patch = a > 0
    return float(np.mean((a[patch] > 0.05) & (a[patch] < 0.55))) if patch.any() else math.nan


def crack_traces(cut: SpecimenCut, layers: VarietyLayers, least_mm: float = 1.0) -> list[float]:
    """Crack-length check (ADR 0018 s.10): each crack's trace in the face, its
    length along its main axis as a share of the face's widest extent. Traces
    shorter than `least_mm` (or missing) are left out."""
    ss, tt = np.meshgrid(cut.s, cut.t)
    pts = cut.plane.points(ss, tt)
    inside = cut.inside
    s_in, t_in = ss[inside], tt[inside]
    width = max(float(s_in.max() - s_in.min()), float(t_in.max() - t_in.min()))
    out: list[float] = []
    for i, f in enumerate(layers.recipe.fractures):
        d = layers.crack_distance(i, pts).reshape(ss.shape)
        trace = inside & (d < max(f.aperture_mm / 2, cut.spacing_mm))
        if trace.sum() < 4:
            continue
        xy = np.stack([ss[trace], tt[trace]], axis=1)
        xy = xy - xy.mean(axis=0)
        along: FloatArray = xy @ np.linalg.svd(xy, full_matrices=False)[2][0]
        length = float(along.max() - along.min())
        if length >= least_mm:
            out.append(length / width)
    return out


class CrackMix(NamedTuple):
    mixed: float  # share of cracked stones whose cracks take more than one fill
    stain_spread: float  # largest spread of iron strength within a one-fill stone
    straight: float  # share of cracks with no bend
    long: float  # share of cracked stones with a long crack
    two_long: float  # share with two or more


def crack_mix(stones: list[tuple[tuple[FractureLayer, ...], float]]) -> CrackMix:
    """Crack-mix check (ADR 0018 s.10) over cracked stones, each given as its
    cracks and its inradius (mm). A long crack reaches LONG_RADIUS[0] x inradius."""
    cracks = [f for fs, _ in stones for f in fs]
    one_fill = [fs for fs, _ in stones if len({f.fill for f in fs}) == 1]
    longs = [sum(f.radius_mm >= LONG_RADIUS[0] * r for f in fs) for fs, r in stones]
    n = len(stones)
    return CrackMix(
        mixed=1 - len(one_fill) / n,
        stain_spread=max(
            (max(f.stain for f in fs) - min(f.stain for f in fs) for fs in one_fill), default=0.0
        ),
        straight=sum(f.sagitta_mm == 0 for f in cracks) / len(cracks),
        long=sum(k >= 1 for k in longs) / n,
        two_long=sum(k >= 2 for k in longs) / n,
    )


def crack_wiggle(cut: SpecimenCut, layers: VarietyLayers, i: int, bin_mm: float = 1.0) -> float:
    """No-S-turn check: the RMS offset (mm) of crack i's trace from its best
    parabola, the part of the bend that one curve cannot follow. An arc gives
    about 0; a trace whose curvature switches sign (an S-turn, a wiggle) does not.
    NaN if the trace spans less than 4 bins."""
    line = _trace_centre(cut, layers, i, bin_mm)
    return math.nan if line is None or len(line[0]) < 4 else _fit_rms(*line, 2)


def band_swelling(
    cut: SpecimenCut, layers: VarietyLayers, growth_mm: float, plain: SpecimenCut
) -> float:
    """Wobble swelling check (ADR 0018 s.10): the 99th percentile of band width over
    the same stone cut without wobble (`plain`, same plane and spacing), pixel by
    pixel, in chalcedony from 1 mm below the wall to 1 mm short of its end. Band
    width is 1 / |grad depth|, so the stone's outline and the cut's obliqueness
    cancel. Not for floor stones: a floor's flat bands give near-zero gradients."""
    widths: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    for c in (cut, plain):
        f = c.fields
        t = np.asarray(f.t_smooth if f.t_smooth is not None else f.t_dep, dtype=np.float64)
        ok = c.inside & np.isfinite(t) & (t < layers.chalcedony_end)
        depth = np.where(ok, t * growth_mm, np.nan)
        gy, gx = np.gradient(depth, c.spacing_mm)
        widths.append((depth, np.hypot(gx, gy)))
    (_, g), (d0, g0) = widths
    end = np.nanmax(d0)
    m = np.isfinite(d0) & np.isfinite(g) & (d0 > 1.0) & (d0 < end - 1.0) & (g > 0) & (g0 > 0)
    return float(np.percentile(g0[m] / g[m], 99)) if m.sum() > 100 else math.nan


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
