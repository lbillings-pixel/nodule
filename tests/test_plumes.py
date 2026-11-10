"""Plumes in the recipes (ADR 0018 s.13): the owner's shares, the plume stage's timing,
the colour ramp, and joining cut chunks where only some carry a step's own values."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from agatesim.fields import Fields, Phase
from agatesim.models.growth import ProceduralGrowth

from nodule.plumes import (
    CENTRE_SHARE,
    COLOUR_SHARES,
    INCLUSION_RAMP,
    NO_PLUMES,
    REACH_MARGIN_MM,
    STAGE,
    draw_plumes,
    inclusion_rgb,
    reach_mm,
)
from nodule.specimen import _concat  # pyright: ignore[reportPrivateUsage]
from nodule.variety import Growth, StoneRecipe, draw_sheet, volume_config


@pytest.fixture(scope="module")
def sheets() -> list[StoneRecipe]:
    return [r for seed in range(12) for r in draw_sheet(seed)]


def test_about_one_stone_in_four_has_plumes_never_a_pebble(sheets: list[StoneRecipe]) -> None:
    plumed = [r for r in sheets if r.plumes is not None]
    assert len(plumed) == 12 * 3  # 3 of 12 per sheet
    assert not any(r.outline in NO_PLUMES for r in plumed)


def test_draws_follow_the_owners_shares() -> None:
    rng = np.random.default_rng(7)
    draws = [draw_plumes(s) for s in rng.spawn(4000)]
    colours = Counter(p.colour for p in draws)
    for name, share in COLOUR_SHARES.items():
        assert abs(colours[name] / len(draws) - share / sum(COLOUR_SHARES.values())) < 0.025
    assert abs(np.mean([p.tilt_deg == 90 for p in draws]) - 0.25) < 0.025
    assert abs(np.mean([p.look == "sparse" for p in draws]) - 1 / 3) < 0.03
    halos = [p.halo_mm for p in draws if p.halo_mm > 0]
    assert abs(len(halos) / len(draws) - 0.75) < 0.025
    assert min(halos) >= 0.3
    assert max(halos) <= 1.0
    assert all(p.tilt_deg in (0, 90) for p in draws)  # never down: that is for moss
    assert {p.azimuth_deg for p in draws if p.tilt_deg == 0} == {0.0}
    wraps = np.array([p.wrap_rate for p in draws])
    assert wraps.min() >= 0.15
    assert wraps.max() <= 0.3
    assert 0.19 < float(np.median(wraps)) < 0.215  # leaning to the low end


def test_plume_stage_leaves_the_walls_run_alone() -> None:
    g = Growth(10.0, 0.6).after_floor(2.0, 0.25)
    p = g.with_plumes(STAGE)
    assert p.growth_mm * (1 - STAGE) == pytest.approx(g.growth_mm)
    assert p.plumes_end == STAGE
    assert p.start == pytest.approx(STAGE + (1 - STAGE) * g.start)
    assert g.quartz_at is not None
    assert p.quartz_at is not None
    assert p.quartz_at == pytest.approx(STAGE + (1 - STAGE) * g.quartz_at)
    # growth laid down after the plume stage is the same in mm at each wall time
    assert (p.quartz_at - p.start) * p.growth_mm == pytest.approx(
        (g.quartz_at - g.start) * g.growth_mm
    )
    with pytest.raises(ValueError, match="one plume stage"):
        p.with_plumes(STAGE)


def _on(model: object, name: str) -> np.ndarray:
    assert isinstance(model, ProceduralGrowth)
    tau = (np.arange(400) + 0.5) / 400
    return np.asarray(getattr(model.schedules, name).sample(tau)) >= 0.5


def test_plume_stones_get_the_stage_first_and_floors_after_it(
    sheets: list[StoneRecipe],
) -> None:
    floored = next(r for r in sheets if r.plumes is not None and r.floor)
    assert floored.plumes is not None
    config, _ = volume_config(floored)
    on, settle = _on(config.model, "dendrite"), _on(config.model, "settling")
    assert on.sum() == 2
    assert on[:2].all()
    assert not (settle & on).any()
    assert settle[2]
    assert isinstance(config.model, ProceduralGrowth)
    assert config.model.dendrite_tint == floored.plumes.tint
    # without plumes, the same stone has no stage and its floor starts at 0
    bare, _ = volume_config(floored.model_copy(update={"plumes": None}))
    assert not _on(bare.model, "dendrite").any()
    assert _on(bare.model, "settling")[0]


def test_ramp_ends_and_middle() -> None:
    rgb = inclusion_rgb(np.array([0.0, 0.5, 1.0, np.nan]))
    assert np.allclose(rgb[0], INCLUSION_RAMP[0][1])
    assert np.allclose(rgb[1], dict(INCLUSION_RAMP)[0.5])
    assert np.allclose(rgb[2], INCLUSION_RAMP[-1][1])
    assert np.allclose(rgb[3], rgb[2])  # no tint: read as black
    assert rgb[0, 1] > rgb[0, 0]  # green at 0
    assert rgb[1, 0] > 2 * rgb[1, 2]  # red at 0.5


def _fields(n: int, band: bool) -> Fields:
    imp = np.full(n, 0.2, np.float32)
    return Fields(
        phase=np.full(n, Phase.CHALCEDONY, np.uint8),
        band=np.zeros(n, np.int32),
        impurity=imp,
        translucency=np.full(n, 0.4, np.float32),
        t_dep=np.full(n, 0.5, np.float32),
        grain=np.full(n, -1, np.int32),
        micro_spacing_um=np.full(n, np.nan, np.float32),
        band_impurity=np.full(n, 0.6, np.float32) if band else None,
        band_translucency=np.full(n, 0.3, np.float32) if band else None,
    )


def test_chunks_join_when_only_some_carry_step_values() -> None:
    """A halo gives a chunk its step's own values; a chunk without one has none."""
    for parts in ([_fields(3, False), _fields(2, True)], [_fields(2, True), _fields(3, False)]):
        joined = _concat(parts)
        assert joined.band_impurity is not None
        assert joined.band_translucency is not None
        assert len(joined.band_impurity) == 5
        own = [p.band_impurity if p.band_impurity is not None else p.impurity for p in parts]
        assert np.array_equal(joined.band_impurity, np.concatenate(own))
    assert _concat([_fields(2, False), _fields(2, False)]).band_impurity is None


def test_plumes_stop_short_of_the_centre(sheets: list[StoneRecipe]) -> None:
    """Every plume stone has a reach, at most CENTRE_SHARE of the way to its centre; with
    a quartz or open centre, also short enough that the coats stay out of it."""
    for r in sheets:
        if r.plumes is None:
            continue
        config, growth = volume_config(r)
        assert isinstance(config.model, ProceduralGrowth)
        reach = config.model.dendrite_reach_mm
        walls = 1.0 - growth.start
        if growth.quartz_at is None:
            centre = growth.growth_mm * walls / 1.3
        else:
            centre = growth.growth_mm * (growth.quartz_at - growth.start)
            # the coat at the tip stays inside the chalcedony
            assert reach + r.plumes.wrap_rate * centre < centre or reach == 0.5
        assert 0.5 <= reach <= max(CENTRE_SHARE * centre, 0.5) + 1e-9
    assert reach_mm(12.0, 0.2, 0.3, solid=True) == pytest.approx(8.0)
    assert reach_mm(12.0, 0.3, 0.5, solid=False) == pytest.approx(8.4 - REACH_MARGIN_MM - 0.5)
    assert reach_mm(1.0, 0.3, 0.5, solid=False) == 0.5  # never below half a millimetre
