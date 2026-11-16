"""Recipe to specimen: growth, the volume configuration, the quick feature layers
(`VarietyLayers`), band roughness and `build` (ADR 0018).

Appearance only: never evidence for a formation mechanism (ADR 0002).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from statistics import NormalDist

import numpy as np
from agatesim.fields import Fields, Phase, Pigment
from agatesim.grid import FloatArray
from agatesim.models.growth import TIME_STEPS, ProceduralGrowth
from agatesim.volume.config import VolumeConfig
from agatesim.volume.models import SmoothNoise3
from agatesim.volume.solids import BoolArray, Solid
from numpy.typing import NDArray

from nodule.crystals import stage_mask, white_translucency
from nodule.dust import (
    BLOB_STRENGTH,
    PIGMENT_AT,
    SPECK_STRENGTH,
    DustLayer,
)
from nodule.plumes import CENTRE_SHARE as PLUME_CENTRE_SHARE
from nodule.plumes import STAGE as PLUME_STAGE
from nodule.plumes import reach_mm as plume_reach_mm
from nodule.specimen import Specimen, SpecimenConfig, SpecimenCut
from nodule.variety.cracks import IRON_TRACE
from nodule.variety.mottle import (
    DARK_RIM_SHARE,
    LIGHT_RIM,
    SOFT_PEAK,
    mottle_edge_noise,
    mottle_fields,
)
from nodule.variety.recipe import (
    MAX_SLOPE,
    QUARTZ_GRAIN_MM,
    QUARTZ_IRON_RATIO,
    QUARTZ_SUPPLY,
    QUARTZ_THRESHOLD,
    WOBBLE_PER_SCALE_PER_CAP,
    Roughness,
    StoneRecipe,
    inradius,
    unit_vector,
)

# How far early crystals move the bands inward, per mm of crystal length, on average
# over the wall: the centre (quartz or open) shrinks by about this much without the
# compensation in `growth_for`. Measured per habit on sheet 0 stone 0 (quartz-filled,
# 0.06 mm cut; equivalent radius 9.05 mm without crystals, less 1.68, 0.90 and 3.39 mm
# with them; ADR 0018 s.15). Near the crystals' reach, length plus tip radius
# (1 + 1 / elongation): the bands' front runs along the crystal tips. `effective`.
CENTRE_SHIFT: dict[str, float] = {"small blades": 1.12, "fine needles": 1.0, "large blades": 1.13}
# A floor settling after the crystal stage keeps its drawn volume: agatesim sizes it by
# the wall's surface alone (agatesim ADR 0087), not the crystals' too, so the earlier
# per-habit FLOOR_VOLUME factor is gone (ADR 0018 s.15).
# Above a floor, the floor's top bounds much of the cavity and carries no crystals, so
# the bands there are not shifted: a floor stone takes this share of the shift.
FLOOR_SHIFT_SHARE = 0.5


@dataclass(frozen=True)
class Growth:
    """The growth run that gives the recipe's centre: total growth and the time
    (share of the run) at which silica drops to quartz supply."""

    growth_mm: float
    quartz_at: float | None
    start: float = 0.0  # growth time at which the walls start (after a floor)
    plumes_end: float = 0.0  # growth time at which the plume stage ends (none: 0)

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

    def with_plumes(self, span: float) -> Growth:
        """This run with a plume stage of `span` first: no band grows during it
        (agatesim's dendrite stage lays no silica), so the run lengthens by the same
        share and its times shift, leaving floor and walls as they were."""
        if self.plumes_end:
            raise ValueError("a run takes one plume stage")
        g = self.growth_mm / (1 - span)
        quartz_at = None if self.quartz_at is None else span + (1 - span) * self.quartz_at
        return Growth(g, quartz_at, span + (1 - span) * self.start, span)


def growth_for(recipe: StoneRecipe, r: float, shift_mm: float = 0.0) -> Growth:
    """Chalcedony to (1 - w) r, then quartz (assumes the agatesim growth rate
    scales with supply, as its quartz-geode preset does). `shift_mm` is how far
    early crystals move the bands inward (`centre_shift_mm`): the chalcedony grows
    that much less, so the centre keeps its width (at most half the chalcedony)."""
    w = recipe.centre_width
    chalcedony = (1 - w) * r - min(shift_mm, 0.5 * (1 - w) * r)
    match recipe.centre:
        case "solid":
            # 1.3 r fills any cavity whose inradius is at most r; the last band
            # lies down at about 1/1.3 of the run.
            return Growth(1.3 * r, None)
        case "quartz-filled":
            g = 3.0 * r  # quartz at 0.3 supply then outgrows the centre for w <= 0.6
            return Growth(g, chalcedony / g)
        case "quartz-lined":
            # The run ends as the lining reaches its thickness, leaving the rest
            # open (agatesim allows no stage after quartz, so supply cannot stop).
            quartz = recipe.lining * w * r
            g = chalcedony + quartz / QUARTZ_SUPPLY
            return Growth(g, chalcedony / g)


def centre_shift_mm(recipe: StoneRecipe) -> float:
    """How far a stone's early crystals move its bands inward, on average over the
    wall (measured per habit, `CENTRE_SHIFT`); 0 without crystals."""
    c = recipe.crystals
    if c is None:
        return 0.0
    share = FLOOR_SHIFT_SHARE if recipe.floor else 1.0
    return share * CENTRE_SHIFT[c.habit] * c.length_mm


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
    shift = centre_shift_mm(recipe)
    growth = growth_for(recipe, r, shift)
    if recipe.floor:
        # The walls are sized to the cavity left above the floor: in a sphere of
        # radius r, a floor at a share f of the height leaves room for a sphere of
        # radius r (1 - f) exactly; for other outlines an approximation.
        left = r * max(1.0 - recipe.floor, 0.1)
        growth = growth_for(recipe, left, shift).after_floor(
            floor_growth_mm(recipe.floor, r), recipe.floor_supply
        )
    inclusions = recipe.inclusions  # plumes or moss: one dendrite stage
    if inclusions is not None or recipe.crystals is not None:
        # One early stage (plumes, moss or crystals) over the first PLUME_STAGE of the
        # run; no silica lies down in it.
        growth = growth.with_plumes(PLUME_STAGE)
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
        end = growth.plumes_end  # the floor settles after any plume stage
        pulse: dict[str, object] = {"type": "pulse", "at": end, "width": growth.start - end}
        schedules["settling"] = {"base": 0.0, "events": [pulse | {"value": 1.0}]}
        if recipe.floor_supply < 1:
            silica.append(pulse | {"value": recipe.floor_supply})
    if inclusions is not None:
        # The dendrite stage over the first PLUME_STAGE of the run (it overrides
        # every other mode there); the bands that follow wrap the plumes or moss (A3).
        model |= inclusions.model_params()
        # Plumes stop short of the centre: wall growth to where quartz starts or, in a
        # solid stone, to where the last band lies down (about 1/1.3 of the walls' run,
        # as `VarietyLayers` takes it).
        if growth.quartz_at is None:
            centre_mm = g * (1.0 - growth.start) / 1.3
        else:
            centre_mm = g * (growth.quartz_at - growth.start)
        wobble = float(model["wobble_mm"])  # type: ignore[arg-type]
        share = PLUME_CENTRE_SHARE if recipe.moss is None else recipe.moss.centre_share
        model["dendrite_reach_mm"] = plume_reach_mm(
            centre_mm, inclusions.wrap_rate, wobble, growth.quartz_at is None, share
        )
        stage = {"type": "step", "at": growth.plumes_end, "value": 0.0}
        schedules["dendrite"] = {"base": 1.0, "events": [stage]}
    if recipe.crystals is not None:
        # agatesim's stick stage in the same slot; bands wrap the crystals at the
        # wall's full rate (agatesim's fixed rate, kept by the owner, 2026-11-16).
        model |= recipe.crystals.model_params()
        stage = {"type": "step", "at": growth.plumes_end, "value": 0.0}
        schedules["sticks"] = {"base": 1.0, "events": [stage]}
    palette, iron = palette_of(recipe)
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


def palette_of(recipe: StoneRecipe) -> tuple[dict[str, float], float]:
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
    dust: DustLayer | None = None

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

    def depth(self, t: FloatArray) -> FloatArray:
        """0 at the wall (and on the floor), 1 where the last chalcedony lies down."""
        return np.clip((t - self.start) / (self.chalcedony_end - self.start), 0, 1)

    def per_step(self, fields: Fields) -> Fields:
        """What depends on the growth step alone: quartz low in iron, white bands
        thinning inward. Applied to the step's own values too (`band_*`), so they
        stay one per step."""
        # the step's own time (t_dep, its centre), so each step keeps one translucency
        depth = self.depth(np.nan_to_num(fields.t_dep, nan=0.0).astype(np.float64))
        quartz = fields.phase == Phase.QUARTZ
        chal = fields.phase == Phase.CHALCEDONY
        fade = self.recipe.white_fade * depth

        def iron(imp: NDArray[np.float32]) -> NDArray[np.float32]:
            out = np.nan_to_num(imp, nan=0.0).astype(np.float32)
            out[quartz] *= QUARTZ_IRON_RATIO
            return out

        def white(trans: NDArray[np.float32]) -> NDArray[np.float32]:
            out = trans.copy()
            out[chal] += (fade[chal] * (1 - out[chal])).astype(np.float32)
            return out

        bi, bt = fields.band_impurity, fields.band_translucency
        return replace(
            fields,
            impurity=iron(fields.impurity),
            translucency=white(fields.translucency),
            band_impurity=None if bi is None else iron(bi),
            band_translucency=None if bt is None else white(bt),
        )

    def __call__(
        self, points: FloatArray, fields: Fields, inside: BoolArray, pixel_mm: float = 0.0
    ) -> Fields:
        """The place-to-place part (after `per_step`): mottle, rim stain, cracks, dust."""
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
        early = np.zeros(len(points), dtype=bool)
        if rec.crystals is not None and rec.crystals.fill == "white":
            # White remnants: milky, unstained, a little varied per crystal (the white
            # is an assumption, set by the owner's eye; ADR 0018 s.15). They take no
            # later stain below, but a crack may still cross them.
            early = stage_mask(fields, inside) & (phase == Phase.QUARTZ)
            phase[early] = Phase.CHALCEDONY
            trans[early] = white_translucency(fields.grain[early])
            imp[early] = 0.0
            pig[early] = Pigment.IRON
        chal = inside & (phase == Phase.CHALCEDONY) & ~early
        quartz = inside & ((phase == Phase.QUARTZ) | early)

        depth = self.depth(t)

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

        if self.dust is not None:
            # Dust lies down with its band, before any later stain; drawn last here
            # only because each layer adds to what lies at the point. Not in quartz,
            # nor where a crack's trace replaced the chalcedony (an assumption).
            line, speck, blob = self.dust.amount(
                points, t, chal & (phase == Phase.CHALCEDONY), pixel_mm
            )
            d = self.dust.dust
            dust = np.maximum.reduce(
                [line * d.strength, speck * SPECK_STRENGTH, blob * BLOB_STRENGTH]
            )
            imp += (dust * (1 - imp)).astype(np.float32)
            pig[dust > PIGMENT_AT] = self.dust.pigment

        return replace(fields, phase=phase, impurity=imp, translucency=trans, pigment=pig)


RIDGE_MEAN = math.sqrt(2 / math.pi)  # mean of |n| for unit normal noise


FRAME_LATTICE_MM = 0.5  # spacing of the cached frame lattice; world-fixed


FRAME_TURN = 0.2  # largest change of the gradient within a cell, as a share of it


_CORNERS = np.array([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)])


@dataclass(eq=False)
class FrameLattice:
    """Growth time and its gradient (`BandRoughness.raw`) kept on a world-fixed
    lattice (nodes at whole multiples of FRAME_LATTICE_MM, so a preview and a full
    run read the same values), filled only where points ask and read trilinearly.
    Both vary slowly beside the roughness itself, and the gradient is already a
    difference over 0.5 mm; reading them here costs one look-up where the direct way
    reads the growth model seven times a point (measured 2026-11-15: 8.6 x the time
    of an unwarped read). Points whose cell has a node without a value (outside the
    stone, in quartz or a void, or beyond the lattice) are read the direct way."""

    lo: NDArray[np.int64]  # lattice index of the first node, per axis
    shape: tuple[int, int, int]
    values: NDArray[np.float32]  # (nodes, 4): growth time, gradient
    filled: NDArray[np.bool_]
    owner: object = None  # whose growth time the values are

    @classmethod
    def over(cls, solid: Solid) -> FrameLattice:
        lo, hi = (np.asarray(b, dtype=np.float64) for b in solid.bounds())
        first = np.floor(lo / FRAME_LATTICE_MM).astype(np.int64) - 1
        last = np.ceil(hi / FRAME_LATTICE_MM).astype(np.int64) + 1
        shape = tuple(int(n) for n in last - first + 1)
        n = math.prod(shape)
        return cls(first, (shape[0], shape[1], shape[2]),
                   np.full((n, 4), np.nan, np.float32), np.zeros(n, bool))  # fmt: skip

    def read(
        self, points: FloatArray, raw: Callable[[FloatArray], tuple[FloatArray, FloatArray]]
    ) -> tuple[FloatArray, FloatArray]:
        """(growth time, gradient) at points; `raw` gives them the direct way."""
        u = points / FRAME_LATTICE_MM
        base = np.floor(u)
        frac = u - base
        cell = base.astype(np.int64) - self.lo
        within = ((cell >= 0) & (cell < np.asarray(self.shape) - 1)).all(axis=1)
        c = cell[within][:, None, :] + _CORNERS[None, :, :]  # (m, 8, 3)
        flat = (c[..., 0] * self.shape[1] + c[..., 1]) * self.shape[2] + c[..., 2]
        new = np.unique(flat)
        new = new[~self.filled[new]]
        if len(new):
            ijk = np.stack(np.unravel_index(new, self.shape), axis=1) + self.lo
            t, grad = raw(ijk * FRAME_LATTICE_MM)
            self.values[new, 0] = t
            self.values[new, 1:] = grad
            self.filled[new] = True
        f = frac[within][:, None, :]
        w = np.where(_CORNERS[None, :, :] == 1, f, 1 - f).prod(axis=2)  # (m, 8)
        out = np.full((len(points), 4), np.nan)
        corners = self.values[flat].astype(np.float64)
        mean = (corners * w[..., None]).sum(axis=1)
        # where the gradient turns within the cell (beside a plume, at a centre line)
        # the lattice is too coarse: those points are read the direct way
        turn = np.linalg.norm(corners[..., 1:] - mean[:, None, 1:], axis=2).max(axis=1)
        steady = turn <= FRAME_TURN * np.linalg.norm(mean[:, 1:], axis=1)
        out[within] = np.where(steady[:, None], mean, np.nan)
        direct = ~np.isfinite(out).all(axis=1)
        if direct.any():
            t, grad = raw(points[direct])
            out[direct, 0] = t
            out[direct, 1:] = grad
        return out[:, 0], out[:, 1:]


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
    # The frame's cache (`FrameLattice`); None reads the growth model at every point
    lattice: FrameLattice | None = field(default=None, repr=False)

    @classmethod
    def of(cls, roughness: Roughness, solid: Solid, growth: Growth, seed: int) -> BandRoughness:
        rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[2])
        noise = SmoothNoise3(solid, roughness.scale_mm, rng)
        return cls(roughness, growth.growth_mm, noise, unit_vector(rng), growth.quartz_at,
                   growth.start, lattice=FrameLattice.over(solid))  # fmt: skip

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
        Growth time and its gradient come from the lattice when there is one."""
        lattice = self.lattice
        if lattice is None:
            t, grad = self.raw(points, growth_time)
        else:
            owner = getattr(growth_time, "__self__", growth_time)
            if lattice.owner is not owner:  # another stone's growth time: start again
                lattice.values[:] = np.nan
                lattice.filled[:] = False
                lattice.owner = owner
            t, grad = lattice.read(points, lambda q: self.raw(q, growth_time))
        return self.finish(t, grad)

    def raw(
        self, points: FloatArray, growth_time: Callable[[FloatArray], FloatArray]
    ) -> tuple[FloatArray, FloatArray]:
        """Growth time and its gradient by differences over step_mm, one-sided where
        one side has no growth time (quartz, voids); NaN where there is none."""
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
        return t, grad

    def finish(self, t: FloatArray, grad: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
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
    if recipe.dust is not None:
        tau = (np.arange(TIME_STEPS) + 0.5) / TIME_STEPS
        assert isinstance(config.model, ProceduralGrowth)
        supply = np.clip(config.model.schedules.silica.sample(tau), 0.0, 1.0)
        dust = DustLayer.of(
            recipe.dust,
            specimen.volume,
            growth.growth_mm * supply / TIME_STEPS,
            layers.start,
            layers.chalcedony_end,
            recipe.seed,
        )
        layers = replace(layers, dust=dust)
    warp = None
    if recipe.roughness is not None:
        warp = BandRoughness.of(recipe.roughness, specimen.volume.solid, growth, recipe.seed)
    return replace(specimen, layers=layers, warp=warp), layers


def cut_of(specimen: Specimen, recipe: StoneRecipe, spacing_mm: float) -> SpecimenCut:
    return specimen.cut(specimen.plane(recipe.cut), spacing_mm)
