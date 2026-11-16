"""Early crystals in the recipes (ADR 0018 s.15): the owner's looks, one early stage per
stone, the run's timing and centre compensation, white remnants, and the wrap check
with its negative control (crystals pasted on finished bands)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from agatesim.fields import Phase
from agatesim.models.growth import ProceduralGrowth
from agatesim.volume.config import SliceConfig, SolidConfig

from nodule._scipy import distance_inside
from nodule.crystals import (
    FILLS,
    HABIT_SHARES,
    HABITS,
    MOST_REACH_SHARE,
    NO_CRYSTALS,
    SIZE_SPREAD,
    Crystals,
    draw_crystals,
    stage_mask,
)
from nodule.plumes import STAGE, draw_plumes
from nodule.variety import (
    CENTRE_SHIFT,
    StoneRecipe,
    build,
    chalcedony_mm,
    crystal_wrap,
    cut_of,
    draw_sheet,
    floor_growth_mm,
    inradius,
    volume_config,
    wrap_share,
)
from nodule.variety.layers import FLOOR_SHIFT_SHARE, FLOOR_VOLUME


@pytest.fixture(scope="module")
def sheets() -> list[StoneRecipe]:
    return [r for seed in range(12) for r in draw_sheet(seed)]


def test_crystals_on_a_sixth_of_stones_alone_in_their_stage(sheets: list[StoneRecipe]) -> None:
    found = [r for r in sheets if r.crystals is not None]
    assert len(found) == 12 * 2
    assert not any(r.outline in NO_CRYSTALS for r in found)
    assert not any(r.inclusions is not None for r in found)
    # Drawn last, from their own stream: plumes and moss as before.
    assert sum(r.plumes is not None for r in sheets) == 12 * 3
    assert sum(r.moss is not None for r in sheets) == 12 * 2
    for r in found:
        assert r.crystals is not None
        assert r.crystals.length_mm <= MOST_REACH_SHARE * chalcedony_mm(r) + 1e-9


def test_draws_keep_to_the_looks_the_owner_saw() -> None:
    draws = [draw_crystals(s) for s in np.random.default_rng(5).spawn(3000)]
    habits = Counter(c.habit for c in draws)
    for name, share in HABIT_SHARES.items():
        assert abs(habits[name] / 3000 - share) < 0.03
    for c in draws:
        assert c.fill in FILLS[c.habit]  # only the fills shown on look sheet 1
        spacing, length, elongation, scatter = HABITS[c.habit]
        k = c.length_mm / length
        assert SIZE_SPREAD[0] - 1e-9 <= k <= SIZE_SPREAD[1] + 1e-9
        assert c.spacing_mm == pytest.approx(spacing * k)
        assert (c.elongation, c.scatter_deg) == (elongation, scatter)
    assert {c.fill for c in draws if c.habit == "large blades"} == {"white"}


def test_large_blades_shrink_to_fit_a_thin_chalcedony() -> None:
    c = Crystals(habit="large blades", fill="white", spacing_mm=4.0, length_mm=3.0,
                 elongation=8.0, scatter_deg=30.0)  # fmt: skip
    small = c.capped(4.0)
    assert small.length_mm == pytest.approx(MOST_REACH_SHARE * 4.0)
    assert small.spacing_mm / small.length_mm == pytest.approx(4.0 / 3.0)
    assert c.capped(40.0) == c


def _stone(**update: object) -> StoneRecipe:
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
        cut=SliceConfig(tilt_deg=90, azimuth_deg=0, offset_mm=0.0),
    )
    return base.model_copy(update=update)


BLADES = Crystals(habit="small blades", fill="white", spacing_mm=2.5, length_mm=1.5,
                  elongation=6.0, scatter_deg=30.0)  # fmt: skip


def test_one_early_stage_per_stone() -> None:
    with pytest.raises(ValueError, match="one early stage"):
        StoneRecipe.model_validate(
            _stone().model_dump()
            | {"plumes": draw_plumes(np.random.default_rng(1)), "crystals": BLADES}
        )


def test_the_stick_stage_comes_first_and_the_centre_keeps_its_width() -> None:
    plain = _stone(centre="quartz-filled", centre_width=0.3)
    config, growth = volume_config(plain.model_copy(update={"crystals": BLADES}))
    _, before = volume_config(plain)
    model = config.model
    assert isinstance(model, ProceduralGrowth)
    assert model.stick_length_mm == 1.5
    assert model.stick_fill == "quartz"
    assert model.schedules.sticks.sample(np.array([0.5 * STAGE, 2 * STAGE])).tolist() == [1, 0]
    assert growth.plumes_end == pytest.approx(STAGE)
    # The chalcedony grows less by the crystals' measured shift, so the centre keeps
    # its width (CENTRE_SHIFT, ADR 0018 s.15).
    walls = growth.growth_mm * (growth.chalcedony_end - growth.plumes_end)
    want = before.growth_mm * before.chalcedony_end - CENTRE_SHIFT["small blades"] * 1.5
    assert walls == pytest.approx(want)
    # Above a floor, half the shift; the floor's growth is divided by the crystals'
    # extra surface (FLOOR_VOLUME), so it keeps its drawn height.
    floored = _stone(centre="quartz-filled", centre_width=0.3, floor=0.4, floor_supply=0.2)
    _, g2 = volume_config(floored.model_copy(update={"crystals": BLADES}))
    _, g1 = volume_config(floored)
    r = inradius(floored.solid.build())
    floor_mm = g2.growth_mm * (g2.start - g2.plumes_end) * 0.2
    assert floor_mm == pytest.approx(floor_growth_mm(0.4, r) / FLOOR_VOLUME["small blades"])
    assert g1.growth_mm * g1.start * 0.2 == pytest.approx(floor_growth_mm(0.4, r))
    walls = g2.growth_mm * (g2.chalcedony_end - g2.start)
    want = g1.growth_mm * (g1.chalcedony_end - g1.start) - FLOOR_SHIFT_SHARE * 1.12 * 1.5
    assert walls == pytest.approx(want)
    hollow = BLADES.model_copy(update={"fill": "hollow"})
    m = volume_config(plain.model_copy(update={"crystals": hollow}))[0].model
    assert isinstance(m, ProceduralGrowth)
    assert m.stick_fill == "hollow"


# The wrap check: bands touch a crystal with their first layers; a crystal pasted on
# finished bands cuts across later ones.

_N = 241
_RR, _CC = np.mgrid[0:_N, 0:_N].astype(np.float64)
_DISC = np.hypot(_RR - 120, _CC - 120) < 110
_PX = 0.05  # mm


def _blades() -> np.ndarray:
    """Blades 1.5 mm long, 0.25 mm wide, standing on the wall all round."""
    m = np.zeros((_N, _N), dtype=bool)
    for ang in np.radians(np.arange(0, 360, 30)):
        for t in np.arange(0, 30, 0.5):
            for w in (-2, -1, 0, 1, 2):
                r = 120 + (109 - t) * np.cos(ang) - w * np.sin(ang)
                c = 120 + (109 - t) * np.sin(ang) + w * np.cos(ang)
                m[round(r), round(c)] = True
    return m & _DISC


def test_wrapped_crystals_pass_and_pasted_ones_are_rejected() -> None:
    crystal = _blades()
    band = _DISC & ~crystal
    wall = distance_inside(_DISC) * _PX
    wrapped = np.minimum(wall, distance_inside(~crystal) * _PX)  # grown round them
    assert wrap_share(wrapped, crystal, band, _PX) > 0.9
    assert wrap_share(wall, crystal, band, _PX) < 0.5  # pasted on finished bands


def test_a_crystal_stone_wraps_its_white_blades() -> None:
    """Built (a small solid stone): the bands wrap the blades, which are white and
    unstained; the same stone without them, with their outline pasted on, fails."""
    r = _stone(crystals=BLADES)
    specimen, _ = build(r)
    growth = volume_config(r)[1]
    cut = cut_of(specimen, r, 0.08)
    crystal = stage_mask(cut.fields, cut.inside)
    assert crystal.sum() > 200
    assert (cut.fields.phase[crystal] == Phase.CHALCEDONY).all()  # white remnants
    assert cut.fields.translucency[crystal].max() <= 0.1
    assert crystal_wrap(cut, growth, 2 * 1.5 / 6.0) > 0.8
    # Too narrow for this cut: not judged.
    assert np.isnan(crystal_wrap(cut, growth, 0.1))
    plain = _stone()
    bare = cut_of(build(plain)[0], plain, 0.08)
    f = bare.fields
    grown = (np.asarray(f.t_dep, dtype=np.float64)) * volume_config(plain)[1].growth_mm
    band = bare.inside & (f.phase == Phase.CHALCEDONY) & ~crystal
    assert wrap_share(grown, crystal, band, 0.08) < 0.5
    assert inradius(plain.solid.build()) > 10  # the blades sit well inside the walls
    # Hollow moulds: the same crystals, left open.
    h = _stone(crystals=BLADES.model_copy(update={"fill": "hollow"}))
    mould = cut_of(build(h)[0], h, 0.08).fields.phase[crystal]
    assert (mould == Phase.VOID).all()
