"""Eyes in the recipes (ADR 0018 s.18): the owner's rates and outlines, eyes showing few
rings kept rare, eyes never with a floor or an early stage, the growth budget, and the
domes themselves (negative control: the same stone without eyes has none)."""

from __future__ import annotations

import numpy as np
import pytest
from agatesim.models.growth import TIME_STEPS, Mode, ProceduralGrowth, growth_modes
from pydantic import ValidationError

from nodule.eyes import (
    EYE_CHANCE,
    EYE_OUTLINES,
    FEW_RINGS_WEIGHT,
    RADIUS_CAP_SHARE,
    Eyes,
    draw_eyes,
    eligible,
)
from nodule.specimen import Specimen
from nodule.variety import StoneRecipe, build, volume_config
from nodule.variety.layers import Growth


def test_eyes_on_their_outlines_at_the_owners_rates(sheets: list[StoneRecipe]) -> None:
    eyed = [r for r in sheets if r.eyes is not None]
    volcanic = [r for r in sheets if r.outline in ("vesicle", "prolate")]
    assert len(eyed) == 9  # all on vesicle or prolate stones in sheets 0-11, two with floors
    assert 0.04 < sum(r.outline in ("vesicle", "prolate") for r in eyed) / len(volcanic) < 0.2
    assert sum(bool(r.floor) for r in eyed) == 2
    for r in eyed:
        assert r.outline in EYE_OUTLINES
        assert r.crystals is None
        assert r.inclusions is None


def test_eyes_with_few_rings_are_rare() -> None:
    def rate(band_mm: float) -> float:
        rng = np.random.default_rng
        drawn = [draw_eyes(rng(s), "vesicle", band_mm, True, 100.0) for s in range(4000)]
        return float(np.mean([e is not None for e in drawn]))

    # every radius (2-4 mm) gives >= MIN_RINGS bands across it at 0.2 mm, none at 1 mm
    fine, coarse = rate(0.2), rate(1.0)
    assert fine == pytest.approx(EYE_CHANCE["vesicle"], abs=0.03)
    assert coarse == pytest.approx(FEW_RINGS_WEIGHT * EYE_CHANCE["vesicle"], abs=0.015)


def test_the_radius_is_capped_by_the_chalcedony() -> None:
    rng = np.random.default_rng
    for s in range(300):
        free = draw_eyes(rng(s), "vesicle", 0.1, True, 100.0)
        tight = draw_eyes(rng(s), "vesicle", 0.1, True, 6.0)
        assert (free is None) == (tight is None)
        if free is not None and tight is not None:
            assert tight.radius_mm == pytest.approx(min(free.radius_mm, RADIUS_CAP_SHARE * 6.0))
    # a wall too thin for the smallest eye (4 mm across) takes none
    assert all(draw_eyes(rng(s), "vesicle", 0.1, True, 3.5) is None for s in range(300))


def test_no_eyes_where_not_eligible(sheets: list[StoneRecipe]) -> None:
    rng = np.random.default_rng
    assert all(draw_eyes(rng(s), "vesicle", 0.3, False, 20.0) is None for s in range(300))
    for outline, early in [("irregular", False), ("pebble", False), ("vesicle", True)]:
        assert not eligible(outline, early)
    assert eligible("thunder-egg", False)
    eyes = {"radius_mm": 3.0, "spacing_mm": 9.0}
    crystal = next(r for r in sheets if r.crystals is not None and r.outline == "vesicle")
    sediment = next(r for r in sheets if r.outline == "irregular" and not r.floor)
    for r in (crystal, sediment):
        with pytest.raises(ValidationError):
            StoneRecipe.model_validate(r.model_dump() | {"eyes": eyes})
    floored = next(r for r in sheets if r.floor and r.outline == "vesicle" and r.inclusions is None
                   and r.crystals is None)  # fmt: skip
    assert StoneRecipe.model_validate(floored.model_dump() | {"eyes": eyes}).eyes is not None


def test_the_eye_stage_comes_out_of_the_run() -> None:
    g = Growth(30.0, 0.5).with_eyes(3.0, 3.6)
    assert g.growth_mm == pytest.approx(29.4)
    assert g.eyes_end == pytest.approx(3.0 / 29.4)
    # the walls lay 15 mm of chalcedony before; the domes moved them 3.6 mm of it
    assert g.quartz_at is not None
    assert (g.quartz_at * g.growth_mm - 3.0) == pytest.approx(15.0 - 3.6)
    # after a floor (6 mm of the 30 mm run): the floor keeps its growth, the eyes follow
    f = Growth(30.0, 0.5, start=0.2).with_eyes(3.0, 1.35)
    assert f.growth_mm == pytest.approx(31.65)
    assert f.start * f.growth_mm == pytest.approx(6.0)
    assert f.eyes_end * f.growth_mm == pytest.approx(9.0)
    assert f.quartz_at is not None
    assert f.quartz_at * f.growth_mm == pytest.approx(15.0 + 3.0 - 1.35)
    with pytest.raises(ValueError, match="no plume stage"):
        Growth(30.0, None, start=0.2, plumes_end=0.1).with_eyes(3.0, 3.0)


def _eyed(sheets: list[StoneRecipe]) -> StoneRecipe:
    return next(r for r in sheets if r.eyes is not None and r.centre == "solid" and not r.floor)


def test_the_eye_stage_reaches_the_model(sheets: list[StoneRecipe]) -> None:
    rec = _eyed(sheets)
    assert rec.eyes is not None
    config, growth = volume_config(rec)
    model = config.model
    assert isinstance(model, ProceduralGrowth)
    assert model.spherulite_spacing_mm == rec.eyes.spacing_mm
    tau = (np.arange(TIME_STEPS) + 0.5) / TIME_STEPS
    n = int(np.count_nonzero(growth_modes(model, tau) == int(Mode.SPHERULITIC)))
    assert n == pytest.approx((growth.eyes_end - growth.start) * TIME_STEPS, abs=1)
    plain = volume_config(rec.model_copy(update={"eyes": None}))[0].model
    assert isinstance(plain, ProceduralGrowth)
    assert not np.any(growth_modes(plain, tau) == Mode.SPHERULITIC)


def test_on_a_floor_stone_the_floor_settles_before_the_eyes(sheets: list[StoneRecipe]) -> None:
    rec = next(r for r in sheets if r.eyes is not None and r.floor)
    model = volume_config(rec)[0].model
    assert isinstance(model, ProceduralGrowth)
    tau = (np.arange(TIME_STEPS) + 0.5) / TIME_STEPS
    modes = growth_modes(model, tau)
    settle = np.flatnonzero(modes == int(Mode.SETTLING))
    eyes = np.flatnonzero(modes == int(Mode.SPHERULITIC))
    assert len(settle)
    assert len(eyes)
    assert settle.max() < eyes.min()


def _dome_share(sp: Specimen, depth_mm: float, eyes_end: float) -> float:
    """Share of the points at depth_mm from the wall laid down in the eye stage."""
    solid = sp.volume.solid
    lo, hi = solid.bounds()
    pts = np.random.default_rng(2).uniform(lo, hi, (300_000, 3))
    pts = pts[solid.contains(pts)]
    shell = pts[np.abs(solid.wall_distance(pts) - depth_mm) < 0.1]
    t = sp.growth_time(shell)
    return float(np.mean(t[np.isfinite(t)] < eyes_end))


def test_eyes_are_domes_on_the_wall(sheets: list[StoneRecipe]) -> None:
    """Under the skin (0.4 R in) part of the face lies in domes and part between them;
    the same stone without eyes is one band there (all or nothing)."""
    rec = _eyed(sheets)  # a solid vesicle, about 3 s each to build
    assert rec.eyes is not None
    _, growth = volume_config(rec)
    depth = 0.4 * rec.eyes.radius_mm
    eyed = _dome_share(build(rec)[0], depth, growth.eyes_end)
    plain = _dome_share(build(rec.model_copy(update={"eyes": None}))[0], depth, growth.eyes_end)
    assert 0.03 < eyed < 0.97  # 4-10: domes 8 mm across, 22 mm apart, cover 8 %
    assert plain in (0.0, 1.0)


def test_eyes_summary_and_push() -> None:
    eyes = Eyes(radius_mm=3.0, spacing_mm=6.6)  # touching: 2.2 radii
    assert eyes.summary() == "eyes 6 mm"
    assert eyes.advance_mm("vesicle") == pytest.approx(3.6)
    assert eyes.advance_mm("thunder-egg") < eyes.advance_mm("vesicle")
    assert eyes.advance_mm("vesicle", floor=0.4) < eyes.advance_mm("vesicle")
    # 3D thunder eggs (ADR 0018 s.23): domes standing apart push the centre far less
    assert eyes.advance_mm("thunder-egg") == pytest.approx(0.75 * 3.0)
    apart = Eyes(radius_mm=3.0, spacing_mm=18.0)  # 6 radii
    assert apart.advance_mm("thunder-egg") == pytest.approx(0.15 * 3.0)
    middle = Eyes(radius_mm=3.0, spacing_mm=13.5)  # 4.5 radii: between the measured 3.5 and 6
    assert 0.15 * 3.0 < middle.advance_mm("thunder-egg") < 0.55 * 3.0
