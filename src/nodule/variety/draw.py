"""Drawing a sheet of stone recipes from one seed, and the cutter's choice of cut
(ADR 0018 s.7).

Appearance only: never evidence for a formation mechanism (ADR 0002).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from agatesim.models.growth import ProceduralGrowth
from agatesim.volume.config import SliceConfig
from agatesim.volume.features import wall_features
from agatesim.volume.models import Setting
from agatesim.volume.slicing import Plane
from agatesim.volume.solids import Rotated, Solid, SphereUnion

from nodule.crystals import NEEDLES_ONLY, NO_COMBOS, combo_plan, crystal_plan, draw_crystals
from nodule.dendrites import draw_dendrites
from nodule.dust import (
    BLOB_SHARE,
    LINE_SHARE,
    SPECK_SHARE,
    draw_dust,
)
from nodule.eyes import draw_eyes
from nodule.eyes import eligible as eyes_eligible
from nodule.moss import draw_moss, moss_plan
from nodule.plumes import draw_plumes, plume_plan
from nodule.rind import draw_rind
from nodule.tubes import draw_channel
from nodule.variety.cracks import (
    CRACK_CUT_ANGLE,
    CRACK_TRIES,
    PARTIAL_REACH,
    SAG_SHARE,
    STRAIGHT_SHARE,
    draw_fractures,
    enters_cavity,
)
from nodule.variety.dendrites import cut_site, meets_wall_on_cut
from nodule.variety.layers import palette_of, volume_config
from nodule.variety.mottle import MOTTLE_SHARE, MOTTLE_STONES, mottle_size
from nodule.variety.recipe import (
    CENTRE_SHARES,
    COLOUR_SHARES,
    FLOOR_HEIGHT,
    FLOOR_STONES,
    FLOOR_SUPPLY,
    FRACTURED_STONES,
    FULL_WOBBLE_STONES,
    MILD_WOBBLE_SHARE,
    OUTLINE_SHARES,
    ROUGHNESS_MM,
    ROUGHNESS_SHARES,
    THICK_FLOOR_CHANCE,
    TINT_CHANCE,
    WARP_SHARES,
    DustForms,
    FractureLayer,
    Mottle,
    Roughness,
    StoneRecipe,
    Tint,
    inradius,
    log_uniform,
    quota,
    recipe_size,
    rim_point,
    scaled_count,
    share_weights,
    solid_for,
    unit_vector,
)

# Cuts of overlapping-sphere outlines: the cutter's choice. A cut far off a
# sphere's centre shows its shells widening toward the middle, and one that only
# grazes a sphere leaves a nub on the outline; the owner judged both atypical.
CLEAN_CUT = 0.3  # largest |offset| / radius of any sphere the cut crosses


CUT_TRIES = 64


ODD_CUTS = (0, 1)  # per 12 stones: keep the first cut drawn, however oblique


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
    floors = set(plan.permutation(floor_ok)[: scaled_count(plan, FLOOR_STONES, n)].tolist())
    crack_ok = [i for i, o in enumerate(outlines) if o != "pebble"]
    cracked = set(plan.permutation(crack_ok)[: scaled_count(plan, FRACTURED_STONES, n)].tolist())
    mottle_pool = plan.permutation(n)
    mottled = set(mottle_pool[: round(MOTTLE_SHARE * n)].tolist())
    odd_ok = [i for i, o in enumerate(outlines) if o in ("vesicle", "irregular")]
    odd = set(plan.permutation(odd_ok)[: scaled_count(plan, ODD_CUTS, n)].tolist())
    # Drawn last, so the stones drawn before this choice keep their plan.
    mild = [i for i, w in enumerate(warps) if w == "mild"]
    full = set(plan.permutation(mild)[: scaled_count(plan, FULL_WOBBLE_STONES, n)].tolist())
    # Mottle kept on fewer stones: the first of the first draw (whose stones still
    # draw their mottle, so their other draws stand).
    kept = set(mottle_pool[: scaled_count(plan, MOTTLE_STONES, n)].tolist())
    dust = dust_plan(plan, outlines)
    stones = [
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
            dust=dust[i],
        )
        for i in range(n)
    ]
    # Plumes from their own stream, spawned after the stones' streams, so every
    # earlier draw stands (ADR 0018 s.13).
    plume_rng = np.random.default_rng(root.spawn(1)[0])
    # Moss and crystals are dealt around the whole plume pool, as before the plume
    # share fell, so their draws stand (ADR 0018 s.15).
    plume_pool, plumed = plume_plan(plume_rng, outlines)
    plume_streams = plume_rng.spawn(n)
    stones = [
        r.model_copy(update={"plumes": draw_plumes(plume_streams[i])}) if i in plumed else r
        for i, r in enumerate(stones)
    ]
    # Moss from a stream spawned after the plumes', on stones without plumes (ADR 0018
    # s.14).
    moss_rng = np.random.default_rng(root.spawn(1)[0])
    mossy = moss_plan(moss_rng, outlines, plume_pool)
    moss_streams = moss_rng.spawn(n)
    stones = [
        r.model_copy(
            update={
                "moss": draw_moss(
                    moss_streams[i],
                    r.centre == "solid",
                    r.centre_width,
                    r.floor,
                    thunder_egg=r.outline == "thunder-egg",
                )
            }
        )
        if i in mossy
        else r
        for i, r in enumerate(stones)
    ]
    # Early crystals from a stream spawned after the moss's, on stones with neither
    # plumes nor moss (ADR 0018 s.15). Thunder eggs turn their draw into needles.
    crystal_rng = np.random.default_rng(root.spawn(1)[0])
    crystal = crystal_plan(crystal_rng, outlines, plume_pool | mossy)
    crystal_streams = crystal_rng.spawn(n)
    out: list[StoneRecipe] = []
    for i, r in enumerate(stones):
        if i in crystal:
            c = draw_crystals(crystal_streams[i])
            c = c.as_needles() if r.outline in NEEDLES_ONLY else c
            r = r.model_copy(update={"crystals": c.capped(chalcedony_mm(r))})
        out.append(r)
    # Combinations from a stream spawned after the crystals', so every earlier draw
    # stands (ADR 0018 s.15): a few crystal stones also carry plumes or moss.
    combo_rng = np.random.default_rng(root.spawn(1)[0])
    eligible = [r.crystals is not None and not r.floor and r.outline not in NO_COMBOS for r in out]
    combos = combo_plan(combo_rng, eligible)
    combo_streams = combo_rng.spawn(n)
    for i, kind in combos.items():
        r, s = out[i], combo_streams[i]
        if kind == "plumes":
            out[i] = r.model_copy(update={"plumes": draw_plumes(s)})
        else:
            moss = draw_moss(s, r.centre == "solid", r.centre_width, r.floor)
            out[i] = r.model_copy(update={"moss": moss})
    # Host rinds from a stream spawned after the combinations', so every earlier draw
    # stands (ADR 0018 s.16); every stone draws its numbers, eligible or not.
    # Validated (model_copy would skip the recipe's check that the rind fits the host).
    rind_streams = np.random.default_rng(root.spawn(1)[0]).spawn(n)
    # Escape tubes from a stream spawned after the rinds' (ADR 0018 s.18), on the same terms.
    tube_streams = np.random.default_rng(root.spawn(1)[0]).spawn(n)
    # Mn dendrites from a stream spawned after the tubes' (ADR 0018 s.17), on the same terms.
    dendrite_streams = np.random.default_rng(root.spawn(1)[0]).spawn(n)
    # Eyes from a stream spawned after the dendrites' (ADR 0018 s.18), on the same terms.
    eye_streams = np.random.default_rng(root.spawn(1)[0]).spawn(n)
    final: list[StoneRecipe] = []
    for i, r in enumerate(out):
        rec = StoneRecipe.model_validate(
            r.model_dump()
            | {
                "rind": draw_rind(rind_streams[i], r.outline),
                "channel": draw_channel(tube_streams[i], r.outline, chalcedony_mm(r)),
            }
        )
        if rec.channel is not None and not channel_fits(rec):
            rec = StoneRecipe.model_validate(rec.model_dump() | {"channel": None})
        rec = with_dendrites(rec, dendrite_streams[i])
        early = rec.crystals is not None or rec.inclusions is not None
        ok = eyes_eligible(rec.outline, early)
        eyes = draw_eyes(eye_streams[i], rec.outline, rec.band_mm, ok, chalcedony_mm(rec))
        final.append(rec if eyes is None else rec.model_copy(update={"eyes": eyes}))
    return final


HAIRLINE_MM = 0.05  # the thinnest crack trace drawn (FractureLayer's range)


def is_hairline(f: FractureLayer) -> bool:
    """A dendrite carrier's hairline (`hairline`): clear, unstained, no pool. Drawn
    cracks always pool (0.5 mm or more), so none reads as one. Unlike drawn cracks it
    may lie on a stone with band roughness: the warp moves where the bands are read,
    and a layer crack, which never slips, is drawn where it lies (2026-11-18)."""
    return f.fill == "clear" and f.stain == 0 and f.pool_mm == 0


SITE_TRIES = 20  # draws for a wall site on the cut, per site


def with_dendrites(recipe: StoneRecipe, rng: np.random.Generator) -> StoneRecipe:
    """The stone with its Mn dendrites, if it draws any (`nodule.dendrites`), steered
    to its cut (the owner, 2026-11-18). A crack stone takes 1-2 of its cracks that the
    cut crosses where they meet the wall; with none, a healed hairline crack is added
    to carry them (clear, unstained, from the rim on the cut, clear of an open cavity
    and at a clean angle to the cut, so the cut stands); if none fits, its dendrites
    grow from the wall instead. Every wall site is moved onto the cut's outline (kept
    where no draw lands there). Validated (the recipe checks the carrier cracks exist)."""
    drawn = draw_dendrites(rng)
    if drawn is None:
        return recipe
    solid = recipe.solid.build()
    plane = Plane.of(recipe.cut, solid.centre())
    if drawn.carrier == "crack":
        on_cut = [i for i, f in enumerate(recipe.fractures) if meets_wall_on_cut(solid, f, plane)]
        if on_cut:
            picked = rng.permutation(on_cut)[: drawn.n_cracks]
            update = {"dendrites": drawn.on_cracks(tuple(sorted(int(i) for i in picked)))}
            return StoneRecipe.model_validate(recipe.model_dump() | update)
        if (hair := hairline(rng, recipe, solid, plane)) is not None:
            update = {
                "fractures": (*recipe.fractures, hair),
                "dendrites": drawn.on_cracks((len(recipe.fractures),)),
            }
            return StoneRecipe.model_validate(recipe.model_dump() | update)
    sites = tuple(_site_on_cut(rng, solid, plane, s) for s in drawn.sites)
    wall = drawn._replace(sites=sites).as_wall()
    return StoneRecipe.model_validate(recipe.model_dump() | {"dendrites": wall})


def _site_on_cut(
    rng: np.random.Generator, solid: Solid, plane: Plane, drawn: tuple[float, float, float]
) -> tuple[float, float, float]:
    """The drawn wall direction moved onto the cut's outline (`cut_site`), else up to
    SITE_TRIES fresh directions; the drawn one where none lands there."""
    v = np.asarray(drawn)
    for _ in range(SITE_TRIES + 1):
        if (d := cut_site(solid, plane, v)) is not None:
            return d
        v = unit_vector(rng)
    return drawn


def hairline(
    rng: np.random.Generator, recipe: StoneRecipe, solid: Solid, plane: Plane
) -> FractureLayer | None:
    """A healed hairline crack running in from the rim where the cut meets it (as a
    partial crack centred on the cut's outline, so the cut crosses it at the wall), or
    None if CRACK_TRIES draws all reach an open cavity, meet the cut at under
    CRACK_CUT_ANGLE, or find no place on the cut's outline."""
    r = inradius(solid)
    cavity = None
    if recipe.centre == "quartz-lined":
        w = recipe.centre_width
        cavity = (1 - w) * r + recipe.lining * w * r
    for _ in range(CRACK_TRIES):
        site = cut_site(solid, plane, unit_vector(rng))
        if site is None:
            continue
        radial = np.asarray(site)
        normal = np.cross(radial, unit_vector(rng))
        radius = float(rng.uniform(*PARTIAL_REACH)) * r
        bent = rng.uniform() >= STRAIGHT_SHARE
        f = FractureLayer(
            centre_mm=tuple(rim_point(solid, radial).tolist()),
            normal=tuple((normal / float(np.linalg.norm(normal))).tolist()),
            radius_mm=radius,
            aperture_mm=HAIRLINE_MM,
            fill="clear",
            stain=0.0,
            halo_mm=0.0,
            pool_mm=0.0,
            sagitta_mm=float(rng.uniform(*SAG_SHARE)) * radius if bent else 0.0,
        )
        if cavity is not None and enters_cavity(solid, f, cavity):
            continue
        if crack_cut_angle(solid, recipe.cut, (f,)) < CRACK_CUT_ANGLE:
            continue
        return f
    return None


def channel_fits(recipe: StoneRecipe) -> bool:
    """Whether agatesim places the stone's channel soundly (placement only, about 0.8 s;
    no growth). False when it finds no place on the wall, or when a piece's side
    direction is not finite. Before agatesim ADR 0088 a piece cutting a chord across a
    concave stretch of wall had a NaN side and the stone opened a void (18 of 174 tube
    stones on sheets 0-99; ADR 0018 s.18); kept as a guard, it passes every tube stone
    of sheets 0-11 since."""
    cfg = volume_config(recipe)[0]
    assert isinstance(cfg.model, ProceduralGrowth)
    setting = Setting.build(cfg.solid.build(), cfg.gravity.vector(), cfg.solid.filament_array())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # agatesim's 0/0 on the bad piece
        try:
            found = wall_features(cfg.model, setting, np.random.default_rng(cfg.seed))
        except ValueError:  # "no place on this cavity's wall can hold a channel"
            return False
    return all(bool(np.isfinite(c.sides).all()) for c in found.channels)


def chalcedony_mm(recipe: StoneRecipe) -> float:
    """Rough thickness of a stone's chalcedony from the wall (mm): the inradius left
    above any floor, less the centre."""
    r = inradius(recipe.solid.build()) * max(1.0 - recipe.floor, 0.1)
    return (1.0 - recipe.centre_width) * r


def dust_plan(plan: np.random.Generator, outlines: list[str]) -> list[DustForms]:
    """Which dust forms each stone takes (ADR 0018 s.11): lines on 1 stone in 2,
    speck strings and coarse blobs on 1 in 4 each, never all three on one stone
    (with the tinted band, that would be every form). Lines and specks follow bands,
    so not on the nearly unbanded pebble. Drawn after the rest of the plan."""
    n = len(outlines)
    banded = [i for i, o in enumerate(outlines) if o != "pebble"]
    lines = set(plan.permutation(banded)[: round(LINE_SHARE * n)].tolist())
    specks = set(plan.permutation(banded)[: round(SPECK_SHARE * n)].tolist())
    blobs = set(plan.permutation(n)[: round(BLOB_SHARE * n)].tolist())
    for i in sorted(lines & specks & blobs):
        free = [
            j
            for j in plan.permutation(n).tolist()
            if j not in blobs and not (j in lines and j in specks)
        ]
        if free:
            blobs.discard(i)
            blobs.add(free[0])
    return [(i in lines, i in specks, i in blobs) for i in range(n)]


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
    dust: DustForms = (False, False, False),
) -> StoneRecipe:
    solid_config, arms, arm_shape = solid_for(rng, outline)  # type: ignore[arg-type]
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
            scale_mm=log_uniform(rng, min(5.0, top), top),
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
        arms=arms,
        arm_shape=arm_shape,
        centre=centre,  # type: ignore[arg-type]
        centre_width=width,
        lining=float(rng.uniform(0.3, 0.7)),
        tint=tint,
        floor=float(rng.uniform(*FLOOR_HEIGHT)) if floor else 0.0,
        colour=colour,  # type: ignore[arg-type]
        band_mm=0.12 if outline == "pebble" else log_uniform(rng, 0.2, 1.0),
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
    refined = _refine(recipe, solid, r, odd_cut=odd_cut, keep_mottle=keep_mottle)
    if any(dust):  # drawn last, so the stone's other draws stand
        lines, specks, blobs = dust
        red = palette_of(refined)[0].get("hematite", 0.0) > 0  # the stone's own palette
        forms = draw_dust(rng, red, lines=lines, specks=specks, blobs=blobs)
        refined = refined.model_copy(update={"dust": forms})
    return refined


def _refine(
    recipe: StoneRecipe, solid: Solid, r: float, *, odd_cut: bool, keep_mottle: bool = True
) -> StoneRecipe:
    """Floor bands, the cutter's choice and band roughness (ADR 0018 s.9), drawn
    from the recipe's own stream so that the rest of the recipe stays as drawn."""
    rng = np.random.default_rng(np.random.SeedSequence(recipe.seed).spawn(4)[3])
    update: dict[str, object] = {}
    if recipe.floor:
        thick = rng.uniform() < THICK_FLOOR_CHANCE
        update["floor_supply"] = 1.0 if thick else log_uniform(rng, *FLOOR_SUPPLY)
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
    level = str(rng.choice(list(ROUGHNESS_SHARES), p=share_weights(ROUGHNESS_SHARES)))
    # Not on cracked stones (a warp takes no slipping fractures) or the pebble.
    if level != "none" and not recipe.fractures and recipe.outline != "pebble":
        update["roughness"] = Roughness(
            amplitude_mm=log_uniform(rng, *ROUGHNESS_MM[level]),
            scale_mm=float(rng.uniform(1.5, 3.0)),
            ridged=False,
        )
    if recipe.mottle is not None:
        if keep_mottle:
            cut = update.get("cut", recipe.cut)
            update["mottle"] = mottle_size(rng, recipe.model_copy(update={"cut": cut}), solid)
        else:
            update["mottle"] = None
    return recipe.model_copy(update=update)


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
