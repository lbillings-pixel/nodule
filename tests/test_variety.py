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
    BandRoughness,
    FractureLayer,
    Mottle,
    Roughness,
    StoneRecipe,
    VarietyLayers,
    build,
    cut_of,
    draw_sheet,
    flat_bands,
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


ROUGH = Roughness(amplitude_mm=0.4, scale_mm=2.0)


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
    extreme = Roughness(amplitude_mm=3.0, scale_mm=1.0, renew_mm=0.3, fade_mm=0.5)
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
