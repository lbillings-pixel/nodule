"""Variety recipes and quick layers (ADR 0018 s.7): quotas, resolution independence,
and each check with its negative control (CLAUDE.md)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from dataclasses import replace as dc_replace

import numpy as np
import pytest
from agatesim.fields import Phase
from agatesim.volume.config import SliceConfig, SolidConfig, VolumeConfig
from agatesim.volume.slicing import Plane

from nodule.specimen import Specimen, SpecimenConfig, SpecimenCut
from nodule.variety import (
    CLEAN_CUT,
    CRACK_CUT_ANGLE,
    FLOOR_HEIGHT,
    FULL_WOBBLE_STONES,
    LARGE_PATCH,
    LARGE_PATCH_MAX,
    LONG_RADIUS,
    MILD_WOBBLE_SHARE,
    MIXED_FILL_CHANCE,
    ROUGHNESS_LIMIT_MM,
    SAG_SHARE,
    STAIN,
    STAIN_JITTER,
    STRAIGHT_SHARE,
    BandRoughness,
    FractureLayer,
    Growth,
    Mottle,
    Roughness,
    StoneRecipe,
    VarietyLayers,
    band_swelling,
    build,
    clean_cut,
    crack_bend,
    crack_cut_angle,
    crack_mix,
    crack_traces,
    crack_wiggle,
    cut_obliqueness,
    cut_of,
    draw_fractures,
    draw_sheet,
    enters_cavity,
    flat_bands,
    floor_bands,
    fold_share,
    fracture_stain,
    graded_share,
    has_cavity,
    inradius,
    largest_patch,
    lining_roughness,
    quota,
    rim_point,
    shift_by_depth,
    stray_quartz,
    volume_config,
)

SPACING = 0.12  # mm


def recipe(**update: object) -> StoneRecipe:
    base = StoneRecipe(
        index=0,
        seed=7,
        outline="vesicle",
        solid=SolidConfig(preset="vesicle-round", size_mm=26),
        centre="solid",
        centre_width=0.0,
        lining=0.5,
        floor=0.0,
        colour="neutral",
        band_mm=0.5,
        warp="mild",
        wobble_mm=0.1,
        wobble_scale_mm=4.0,
        white_fade=0.3,
        rim_stain=0.0,
        cut=SliceConfig(tilt_deg=90, azimuth_deg=30, offset_mm=0.0),
    )
    return base.model_copy(update=update)


_CUTS: dict[str, tuple[SpecimenCut, VarietyLayers]] = {}


def cut_for(r: StoneRecipe) -> tuple[SpecimenCut, VarietyLayers]:
    key = r.model_dump_json()
    if key not in _CUTS:
        specimen, layers = build(r)
        _CUTS[key] = cut_of(specimen, r, SPACING), layers
    return _CUTS[key]


LINED = recipe(centre="quartz-lined", centre_width=0.45, lining=0.4)
FLOOR = recipe(floor=0.5)
CRACK = FractureLayer(
    centre_mm=(0, 0, 0),
    normal=(0.0, 1.0, 0.2),
    radius_mm=20,
    aperture_mm=0.3,
    fill="clear",
    stain=0.9,
    halo_mm=0.4,
    pool_mm=2.0,
)


def test_quota_counts_and_order() -> None:
    labels = quota(np.random.default_rng(0), {"a": 6, "b": 3, "c": 1, "d": 1, "e": 1}, 12)
    assert Counter(labels) == {"a": 6, "b": 3, "c": 1, "d": 1, "e": 1}
    assert quota(np.random.default_rng(0), {"x": 1, "y": 1, "z": 1}, 4).count("x") >= 1


def test_sheet_follows_the_recipe_and_no_stone_has_everything() -> None:
    sheet = draw_sheet(1)
    assert sheet == draw_sheet(1)  # one seed, one sheet
    assert Counter(r.centre for r in sheet) == {"solid": 4, "quartz-filled": 4, "quartz-lined": 4}
    assert Counter(r.colour for r in sheet) == {
        "neutral": 6,
        "iron": 3,
        "onyx": 1,
        "stained": 1,
        "carnelian": 1,
    }
    assert Counter(r.outline for r in sheet) == {
        "vesicle": 7,
        "thunder-egg": 2,
        "prolate": 1,
        "irregular": 1,
        "pebble": 1,
    }
    assert 2 <= sum(r.floor > 0 for r in sheet) <= 3
    assert 1 <= sum(bool(r.fractures) for r in sheet) <= 2
    for r in sheet:
        assert not (
            r.floor and r.fractures and r.mottle and r.centre != "solid" and r.warp != "mild"
        )
        if r.outline == "pebble":
            assert r.centre == "solid"
        if r.outline in ("thunder-egg", "pebble"):
            assert r.floor == 0
        volume_config(r)  # every recipe is a valid agatesim run


def test_preview_matches_full_resolution() -> None:
    r = recipe(
        colour="stained",
        rim_stain=0.8,
        mottle=Mottle(scale_mm=6, cover=0.3, rim_mm=0.8, strength=0.6),
        fractures=(CRACK,),
    )
    specimen, _ = build(r)
    plane = specimen.plane(r.cut)
    fine = specimen.cut(plane, 0.1)
    coarse = specimen.cut(plane, 0.2)
    i = np.rint((coarse.s - fine.s[0]) / 0.1).astype(int)
    j = np.rint((coarse.t - fine.t[0]) / 0.1).astype(int)
    ki, kj = (i >= 0) & (i < len(fine.s)), (j >= 0) & (j < len(fine.t))
    sub: tuple[np.ndarray, ...] = np.ix_(j[kj], i[ki])  # pyright: ignore[reportUnknownVariableType]
    np.testing.assert_allclose(fine.s[i[ki]], coarse.s[ki], atol=1e-9)
    assert ki.mean() > 0.9
    assert kj.mean() > 0.9
    np.testing.assert_array_equal(fine.fields.phase[sub], coarse.fields.phase[np.ix_(kj, ki)])
    np.testing.assert_allclose(
        fine.fields.impurity[sub], coarse.fields.impurity[np.ix_(kj, ki)], atol=1e-5
    )


def test_layers_keep_quartz_low_in_iron() -> None:
    cut, _ = cut_for(LINED)
    f = cut.fields
    quartz = cut.inside & (f.phase == Phase.QUARTZ)
    chal = cut.inside & (f.phase == Phase.CHALCEDONY)
    assert quartz.sum() > 50
    assert f.impurity[quartz].mean() < 0.5 * f.impurity[chal].mean()


def test_cavity_check_and_its_negative_control() -> None:
    assert has_cavity(cut_for(LINED)[0])
    assert not has_cavity(cut_for(recipe())[0])  # a solid stone reports no cavity


def test_faceted_lining_check_and_its_negative_control() -> None:
    faceted = lining_roughness(cut_for(LINED)[0])
    # The same open centre lined by smooth chalcedony: no quartz stage.
    config, _ = volume_config(LINED)
    model = config.model.model_copy(update={"quartz_threshold": 0.0})
    smooth_specimen = Specimen.build(
        SpecimenConfig(volume=config.model_copy(update={"model": model}))
    )
    smooth_cut = smooth_specimen.cut(smooth_specimen.plane(LINED.cut), SPACING)
    assert has_cavity(smooth_cut)
    smooth = lining_roughness(smooth_cut)
    assert faceted > 1.15
    assert smooth < 1.08
    assert np.isnan(lining_roughness(cut_for(recipe())[0]))


def test_waterline_check_and_its_negative_control() -> None:
    assert flat_bands(cut_for(FLOOR)[0]) >= 3
    assert flat_bands(cut_for(recipe())[0]) == 0  # curved, concentric bands


def test_stained_fracture_check_and_its_negative_control() -> None:
    stained = fracture_stain(*cut_for(recipe(fractures=(CRACK,))))
    clean = CRACK.model_copy(update={"stain": 0.0})
    unstained = fracture_stain(*cut_for(recipe(fractures=(clean,))))
    assert stained > 0.1
    assert unstained < 0.03


def test_open_crack_is_void_and_has_no_offset() -> None:
    open_crack = CRACK.model_copy(update={"fill": "open", "stain": 0.0})
    cut, layers = cut_for(recipe(fractures=(open_crack,)))
    plain, _ = cut_for(recipe())
    ss, tt = np.meshgrid(cut.s, cut.t)
    d = layers.fracture_distance(cut.plane.points(ss, tt)).reshape(ss.shape)
    trace = cut.inside & (d < 0.1)
    assert (cut.fields.phase[trace] == Phase.VOID).mean() > 0.9
    away = cut.inside & (d > 0.5)
    np.testing.assert_array_equal(cut.fields.phase[away], plain.fields.phase[away])
    np.testing.assert_array_equal(cut.fields.impurity[away], plain.fields.impurity[away])


@pytest.mark.parametrize("outline", ["thunder-egg", "pebble", "prolate", "irregular"])
def test_every_outline_builds(outline: str) -> None:
    r = next(r for r in draw_sheet(1) if r.outline == outline)
    config, _ = volume_config(r)
    assert isinstance(config, VolumeConfig)


def test_slope_cap_is_a_setting_below_one() -> None:
    crenulated = next(r for r in draw_sheet(1) if r.warp == "crenulated")
    low, _ = volume_config(crenulated, 0.5)
    high, _ = volume_config(crenulated, 0.9)
    wobble = [float(getattr(c.model, "wobble_mm")) for c in (low, high)]  # noqa: B009
    assert wobble[1] > wobble[0]
    with pytest.raises(ValueError, match="fold"):
        build(crenulated, 1.0)  # at 1 or above, bands could fold (ADR 0005 s.3)


ROUGH = Roughness(amplitude_mm=ROUGHNESS_LIMIT_MM, scale_mm=2.0)


def test_roughness_grows_inward_and_its_negative_control() -> None:
    plain, _ = cut_for(recipe())
    growth = volume_config(recipe())[1].growth_mm
    depths = (0.2, 1.5, 8.0)
    rim, inner = shift_by_depth(cut_for(recipe(roughness=ROUGH))[0], plain, growth, depths)
    assert inner > 2.5 * rim
    flat = ROUGH.model_copy(update={"fade_mm": 0.01})  # full strength from the wall
    rim0, inner0 = shift_by_depth(cut_for(recipe(roughness=flat))[0], plain, growth, depths)
    assert inner0 < 1.5 * rim0


def test_roughness_does_not_fold_and_its_negative_control() -> None:
    specimen, _ = build(recipe(roughness=ROUGH))
    assert fold_share(specimen, samples=1500) < 0.02
    # Past the limit, so built without validation.
    extreme = Roughness.model_construct(amplitude_mm=3.0, scale_mm=1.0, renew_mm=0.3, fade_mm=0.5)
    folded, _ = build(recipe(roughness=extreme))
    assert fold_share(folded, samples=1500) > 0.2


def test_roughness_keeps_the_outline() -> None:
    rough, _ = cut_for(recipe(roughness=ROUGH))
    plain, _ = cut_for(recipe())
    np.testing.assert_array_equal(rough.inside, plain.inside)


def test_quartz_contact_stays_sharp_and_its_negative_control() -> None:
    lined = LINED.model_copy(update={"roughness": ROUGH})
    specimen, _ = build(lined)
    assert stray_quartz(cut_of(specimen, lined, SPACING)) < 0.005
    assert isinstance(specimen.warp, BandRoughness)
    unfaded = dc_replace(specimen, warp=dc_replace(specimen.warp, quartz_at=None))
    assert stray_quartz(cut_of(unfaded, lined, SPACING)) > 0.015  # scattered contact


# --- the owner's notes on the plain stones and the roughness trial (ADR 0018 s.9) ---


def test_floor_comes_first_and_leaves_the_walls_their_run() -> None:
    full = Growth(20.0, 0.6).after_floor(5.0, 1.0)
    assert (full.growth_mm, full.start, full.quartz_at) == pytest.approx((25.0, 0.2, 0.68))
    run = Growth(20.0, 0.6).after_floor(5.0, 0.25)
    assert run.growth_mm * 0.25 * run.start == pytest.approx(5.0)  # the floor's growth
    assert run.growth_mm * (1 - run.start) == pytest.approx(20.0)  # the walls' run, whole
    # Quartz starts after the walls' own 12 mm, whatever the floor.
    assert run.quartz_at is not None
    assert run.growth_mm * (run.quartz_at - run.start) == pytest.approx(12.0)


def _phase_share(r: StoneRecipe, *phases: Phase) -> float:
    cut, _ = cut_for(r)
    return float(np.isin(cut.fields.phase[cut.inside], phases).mean())


@pytest.mark.parametrize("centre", ["quartz-filled", "quartz-lined"])
def test_a_floor_keeps_its_centre_and_its_negative_control(centre: str) -> None:
    """ADR 0018 s.10: walls sized to the cavity above the floor leave the centre.
    Before, every floor stone of sheets 1-6 lost its centre."""
    open_centre = recipe(centre=centre, centre_width=0.3, lining=0.5)
    floored = open_centre.model_copy(update={"floor": 0.5, "floor_supply": 0.2})
    alone = _phase_share(open_centre, Phase.QUARTZ, Phase.VOID)
    kept = _phase_share(floored, Phase.QUARTZ, Phase.VOID)
    assert kept > 0.2 * alone > 0
    # Control: a solid centre with the same floor shows no centre at all.
    solid = floored.model_copy(update={"centre": "solid", "centre_width": 0.0})
    assert _phase_share(solid, Phase.QUARTZ, Phase.VOID) == 0


BENT = CRACK.model_copy(update={"sagitta_mm": 1.5})  # share 0.075, mid-range


@dataclass(frozen=True, eq=False)
class _Wavy(VarietyLayers):
    """Negative control for the no-S-turn check: crack 0 shifted along its normal by
    sine waves in two in-plane directions (1 mm, 16 mm period), as the old noise
    bends did, so its trace in any face switches its curve's direction."""

    def crack_distance(self, i: int, points: np.ndarray) -> np.ndarray:
        f = self.recipe.fractures[i]
        n = np.asarray(f.normal) / np.linalg.norm(f.normal)
        e1 = np.cross(n, [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        rel = points - (self.solid_centre + np.asarray(f.centre_mm))
        wave = sum(np.sin(2 * np.pi * (rel @ e) / 16.0) for e in (e1, e2))
        z = rel @ n - 1.0 * wave
        r = np.linalg.norm(rel - (rel @ n)[:, None] * n, axis=1)
        return np.hypot(z, np.maximum(r - f.radius_mm, 0.0))


def test_bent_crack_check_and_its_negative_control() -> None:
    bent = crack_bend(*cut_for(recipe(fractures=(BENT,))), 0)
    straight = crack_bend(*cut_for(recipe(fractures=(CRACK,))), 0)  # control: a flat disc
    assert 0.08 < bent < 0.6  # a gentle bow: about 0.3 x the sag over the face
    assert straight < 0.03


def test_no_s_turn_check_and_its_negative_control() -> None:
    """ADR 0018 s.10 (owner, 2026-11-08): cracks curve slightly, rarely if ever
    switching direction. A bowl's trace is one arc; the wavy control is not."""
    cut, layers = cut_for(recipe(fractures=(BENT,)))
    arc = crack_wiggle(cut, layers, 0)
    wavy = crack_wiggle(cut, _Wavy(**vars(layers)), 0)
    assert arc < 0.03
    assert wavy > 0.2


def test_crack_length_check_and_its_negative_control() -> None:
    """ADR 0018 s.10 (owner, 2026-11-08): most cracks are short or partial. A crack
    running in from the rim a third of the way gives a short trace; the control, a
    crack through the centre wider than the stone (the old drawing), spans the face."""
    base = recipe()
    specimen, _ = build(base)
    plane = specimen.plane(base.cut)
    solid = specimen.volume.solid
    partial = CRACK.model_copy(
        update={
            "centre_mm": tuple(rim_point(solid, np.asarray(plane.u)).tolist()),
            "normal": tuple(np.asarray(plane.v).tolist()),
            "radius_mm": 0.35 * inradius(solid),
        }
    )
    (short,) = crack_traces(*cut_for(recipe(fractures=(partial,))))
    (long,) = crack_traces(*cut_for(recipe(fractures=(CRACK,))))
    assert 0.1 < short < 0.3
    assert long > 0.7  # across most of the face


def _old_style(rng: np.random.Generator, r: float) -> tuple[FractureLayer, ...]:
    """Negative control for the crack-mix check: the drawing before ADR 0018 s.10's
    crack fix, 1-3 long cracks near the centre, each with its own fill, all bent."""
    return tuple(
        CRACK.model_copy(
            update={
                "radius_mm": float(rng.uniform(0.6, 1.5) * r),
                "fill": str(rng.choice(["clear", "iron"], p=[0.6, 0.4])),
                "stain": float(rng.uniform(0.4, 1.0)),
                "sagitta_mm": float(rng.uniform(*SAG_SHARE)) * r,
            }
        )
        for _ in range(int(rng.integers(1, 4)))
    )


def test_smear_check_and_cutter_avoid_cutting_along_a_crack() -> None:
    """ADR 0018 s.10 (owner, 2026-11-08): a crack near the face's plane shows as a
    wide smear, and a cutter would not saw along a crack. Control: a crack lying in
    the cut's plane."""
    base = recipe()
    solid = base.solid.build()
    plane = Plane.of(base.cut, solid.centre())
    along = CRACK.model_copy(update={"normal": tuple(np.asarray(plane.normal).tolist())})
    across = CRACK.model_copy(update={"normal": tuple(np.asarray(plane.u).tolist())})
    assert crack_cut_angle(solid, base.cut, (along,)) < 1  # control: a smear
    assert crack_cut_angle(solid, base.cut, (across,)) > 89
    assert crack_cut_angle(solid, base.cut, ()) == 90
    chosen = clean_cut(np.random.default_rng(0), solid, base.cut, 10.0, (along,))
    assert crack_cut_angle(solid, chosen, (along,)) >= CRACK_CUT_ANGLE
    assert clean_cut(np.random.default_rng(0), solid, base.cut, 10.0) == base.cut  # no crack


def test_cavity_check_and_cracks_keep_out_of_open_cavities() -> None:
    """ADR 0018 s.10 (owner, 2026-11-08): no long crack through a cavity. A crack
    through the centre of a quartz-lined stone enters it (the control); a rim nick
    does not; and cracks drawn for that stone keep out."""
    solid = LINED.solid.build()
    r = inradius(solid)
    cavity = (1 - LINED.centre_width) * r + LINED.lining * LINED.centre_width * r
    nick = CRACK.model_copy(
        update={
            "centre_mm": tuple(rim_point(solid, np.array([1.0, 0.0, 0.0])).tolist()),
            "normal": (0.0, 1.0, 0.0),
            "radius_mm": 2.0,
        }
    )
    assert enters_cavity(solid, CRACK, cavity)  # control
    assert not enters_cavity(solid, nick, cavity)
    rng = np.random.default_rng(1)
    drawn = [f for _ in range(40) for f in draw_fractures(rng, solid, False, cavity)]
    assert drawn
    assert not any(enters_cavity(solid, f, cavity) for f in drawn)


def test_crack_stain_is_capped() -> None:
    """The owner (2026-11-08): iron stain on cracks was too intense (up to 1.0).
    Control: the old fixture's 0.9 lies above the cap."""
    rng = np.random.default_rng(2)
    solid = recipe().solid.build()
    stains = [f.stain for _ in range(300) for f in draw_fractures(rng, solid, False)]
    cap = STAIN[1] + STAIN_JITTER
    assert max(stains) <= cap
    assert CRACK.stain > cap


def test_mottle_size_check_and_its_negative_control() -> None:
    """ADR 0018 s.10 (owner, 2026-11-08): most mottle patches small, a large one
    (a quarter of the face or more) on 15-25 % of mottled stones. Control: a large
    pattern's patch, which the small class must reject."""
    small = recipe(mottle=Mottle(scale_mm=1.3, cover=0.2, rim_mm=0.8, strength=0.6, variant=1))
    large = recipe(mottle=Mottle(scale_mm=7.8, cover=0.5, rim_mm=0.8, strength=0.6, variant=1))
    solid = small.solid.build()
    assert largest_patch(small, solid) < LARGE_PATCH
    assert largest_patch(large, solid) >= LARGE_PATCH  # control
    assert largest_patch(recipe(), solid) == 0


def test_sheets_mottle_fewer_stones_and_show_the_drawn_size() -> None:
    for k in range(0, len(SHEETS), 12):
        assert 1 <= sum(r.mottle is not None for r in SHEETS[k : k + 12]) <= 2  # was 4
    for r in SHEETS:
        if r.mottle is not None:
            share = largest_patch(r, r.solid.build())
            assert (share >= LARGE_PATCH) == r.mottle.large
            assert share <= LARGE_PATCH_MAX


def test_mottle_rims_vary_and_its_negative_control() -> None:
    """Dark rims on about 1 patch in 4, lighter on the rest (owner, 2026-11-08).
    Control: the first drawing, every rim dark."""
    _, layers = cut_for(recipe(mottle=Mottle(scale_mm=4, cover=0.3, rim_mm=0.8, strength=0.6)))
    pts = np.random.default_rng(0).uniform(-12, 12, size=(20000, 3)) + layers.solid_centre
    share = float(np.mean(layers.rim_darkness(pts) > 0.5))
    assert 0.1 < share < 0.4
    assert np.mean(dc_replace(layers, rim_noise=None).rim_darkness(pts) > 0.5) == 1.0


def test_soft_mottle_check_and_its_negative_control() -> None:
    """ADR 0018 s.10 (owner, 2026-11-08): about half of mottled stones have soft,
    graded patches; 5 % have both kinds. Control: a hard-edged stone."""
    base = Mottle(scale_mm=4, cover=0.3, rim_mm=0.8, strength=0.6)
    pts = np.random.default_rng(0).uniform(-12, 12, size=(40000, 3))
    shares = {}
    for edge in ("hard", "soft", "mixed"):
        _, layers = cut_for(recipe(mottle=base.model_copy(update={"edge": edge})))
        shares[edge] = graded_share(layers, pts + layers.solid_centre)
        soft = float(np.mean(layers.mottle_softness(pts + layers.solid_centre) > 0.5))
        assert soft == {"hard": 0.0, "soft": 1.0}.get(edge, soft)
        if edge == "mixed":
            assert 0.25 < soft < 0.75  # about half its patches
    assert shares["hard"] == 0  # control
    assert shares["soft"] > 0.4
    assert 0.05 < shares["mixed"] < shares["soft"]


def test_crack_mix_check_and_its_negative_control() -> None:
    """One shade per stone (rare exceptions), a third straight, long cracks on a
    third of cracked stones and two on few (owner, 2026-11-08; ADR 0018 s.10)."""
    rng = np.random.default_rng(0)
    solid = recipe().solid.build()
    r = inradius(solid)
    drawn = crack_mix([(draw_fractures(rng, solid, bool(k % 4 == 0)), r) for k in range(600)])
    assert abs(drawn.mixed - MIXED_FILL_CHANCE) < 0.04
    assert drawn.stain_spread <= 0.2 + 1e-9
    assert abs(drawn.straight - STRAIGHT_SHARE) < 0.05
    assert 0.25 < drawn.long < 0.42  # 4 in 12
    assert 0.01 < drawn.two_long < 0.06  # 1 in 36
    old = crack_mix([(_old_style(rng, r), r) for _ in range(600)])
    assert old.mixed > 0.3
    assert old.stain_spread > 0.2
    assert old.straight == 0
    assert old.long > 0.9
    assert LONG_RADIUS[0] == 0.6


def test_sheets_draw_low_floors_and_bent_healed_cracks() -> None:
    floors = [r.floor for r in SHEETS if r.floor]
    assert floors
    assert max(floors) <= FLOOR_HEIGHT[1]
    cracks = [f for r in SHEETS for f in r.fractures]
    assert cracks
    assert all(f.fill != "open" for f in cracks)
    assert any(f.sagitta_mm == 0 for f in cracks)
    assert any(f.sagitta_mm > 0 for f in cracks)
    assert all(f.sagitta_mm <= SAG_SHARE[1] * f.radius_mm for f in cracks)


def test_low_supply_gives_thin_floor_bands() -> None:
    thin = floor_bands(*cut_for(FLOOR.model_copy(update={"floor_supply": 0.2})))
    thick = floor_bands(*cut_for(FLOOR))  # control: supply 1, the old thick bands
    assert thin[0] == pytest.approx(thick[0], rel=0.25)  # about the same floor
    assert thin[1] < 0.4 * thick[1]
    assert thin[1] < 2.5 * FLOOR.band_mm < thick[1]
    assert floor_bands(*cut_for(recipe())) == (0.0, 0.0)  # no floor, nothing measured


LOBATE = SolidConfig(preset="vesicle-lobate", size_mm=40).build()


def test_clean_cut_crosses_spheres_near_their_centres() -> None:
    grazing = SliceConfig(tilt_deg=90, azimuth_deg=0, offset_mm=0.0)
    for offset in np.linspace(-20, 20, 81):  # control: a cut that grazes a sphere
        grazing = grazing.model_copy(update={"offset_mm": float(offset)})
        if cut_obliqueness(LOBATE, grazing) > 0.9:
            break
    assert cut_obliqueness(LOBATE, grazing) > 0.9
    chosen = clean_cut(np.random.default_rng(3), LOBATE, grazing, 10.0)
    assert cut_obliqueness(LOBATE, chosen) <= CLEAN_CUT
    round_ = SolidConfig(preset="vesicle-round", size_mm=40).build()
    assert cut_obliqueness(round_, grazing) == 0.0


SHEETS = [r for seed in range(1, 7) for r in draw_sheet(seed)]


def test_sheets_keep_odd_cuts_rare() -> None:
    lobed = [r for r in SHEETS if cut_obliqueness(r.solid.build(), r.cut) > 0]
    odd = [r for r in lobed if cut_obliqueness(r.solid.build(), r.cut) > CLEAN_CUT]
    assert lobed
    assert len(odd) <= len(SHEETS) // 12  # at most one per sheet


def test_sheets_draw_gentle_smooth_roughness() -> None:
    rough = [r.roughness for r in SHEETS if r.roughness is not None]
    assert 0.25 * len(SHEETS) <= len(rough) <= 0.75 * len(SHEETS)
    assert all(not g.ridged and g.amplitude_mm <= ROUGHNESS_LIMIT_MM for g in rough)
    assert not any(r.roughness and (r.fractures or r.outline == "pebble") for r in SHEETS)
    with pytest.raises(ValueError, match="less than or equal"):  # control: the trial's 0.4 mm
        Roughness(amplitude_mm=0.4, scale_mm=2.0)


def test_sheets_keep_thick_floors_rare() -> None:
    floors = [r.floor_supply for r in SHEETS if r.floor]
    assert floors
    assert sum(f >= 1 for f in floors) <= 0.3 * len(floors)


def test_wobble_swelling_check_and_its_negative_control() -> None:
    """ADR 0018 s.10 (owner, 2026-11-08: a wide band droops). A mild stone at half
    wobble keeps its bands even; the control, the full-strength wobble of the
    drooping stone (0.05 mm per mm of scale), swells them."""
    plain = cut_for(recipe(wobble_mm=0.0))[0]
    growth = volume_config(recipe())[1].growth_mm
    half = band_swelling(*cut_for(recipe(wobble_mm=0.1)), growth, plain=plain)
    full = band_swelling(*cut_for(recipe(wobble_mm=0.2)), growth, plain=plain)
    assert half < 1.08
    assert full > 1.12


def test_sheets_halve_mild_wobble_but_one_stone_in_twelve() -> None:
    mild = [r for r in SHEETS if r.warp == "mild"]
    full = [r for r in mild if r.full_wobble]
    assert len(full) == FULL_WOBBLE_STONES[0] * len(SHEETS) // 12
    assert all(r.wobble_mm <= 0.25 * MILD_WOBBLE_SHARE for r in mild if not r.full_wobble)
    assert not any(r.full_wobble for r in SHEETS if r.warp != "mild")
