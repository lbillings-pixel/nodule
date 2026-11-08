"""Variety recipes and quick layers (ADR 0018 s.7): quotas, resolution independence,
and each check with its negative control (CLAUDE.md)."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace as dc_replace

import numpy as np
import pytest
from agatesim.fields import Phase
from agatesim.volume.config import SliceConfig, SolidConfig, VolumeConfig

from nodule.specimen import Specimen, SpecimenConfig, SpecimenCut
from nodule.variety import (
    CLEAN_CUT,
    ROUGHNESS_LIMIT_MM,
    BandRoughness,
    FractureLayer,
    Growth,
    Mottle,
    Roughness,
    StoneRecipe,
    VarietyLayers,
    build,
    clean_cut,
    cut_obliqueness,
    cut_of,
    draw_sheet,
    flat_bands,
    floor_bands,
    fold_share,
    fracture_stain,
    has_cavity,
    lining_roughness,
    quota,
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


def test_floor_stretches_only_its_own_phase() -> None:
    full = Growth(20.0, 0.6).after_floor(5.0, 1.0)  # control: the old timing
    assert (full.growth_mm, full.start, full.quartz_at) == pytest.approx((20.0, 0.25, 0.6))
    run = Growth(20.0, 0.6).after_floor(5.0, 0.25)
    assert run.growth_mm * 0.25 * run.start == pytest.approx(5.0)  # the floor's growth
    assert run.growth_mm * (1 - run.start) == pytest.approx(15.0)  # the rest as before
    # Quartz starts after the same wall growth as at full supply: 7 mm.
    assert run.quartz_at is not None
    assert run.growth_mm * (run.quartz_at - run.start) == pytest.approx(7.0)


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
