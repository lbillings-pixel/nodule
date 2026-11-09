"""Thunder-egg stars: sinuous arms and the check that tells them from straight ones."""

from __future__ import annotations

import numpy as np
import pytest
from agatesim.geometry import outline_problem

from nodule.thunderegg import (
    BEND_RANGE,
    BUDS_RANGE,
    SINUOUS_MIN,
    WIDTH_RANGE,
    ArmShape,
    arm_meander,
    outline_mask,
    thunder_egg_outline,
)
from nodule.variety import draw_sheet


def _meander(style: str, k: int) -> float:
    _, _, outline = thunder_egg_outline(np.random.default_rng([29, k]), style)  # type: ignore[arg-type]
    return arm_meander(outline_mask(np.array(outline)))


@pytest.mark.parametrize("k", range(6))
def test_sinuous_check_and_its_negative_control(k: int) -> None:
    """Sinuous arms pass; the angular star (straight arms) is the negative control."""
    assert _meander("sinuous", k) > SINUOUS_MIN
    assert _meander("angular", k) < SINUOUS_MIN


def test_sinuous_outlines_are_valid_and_repeatable() -> None:
    for k in range(4):
        style, _, a = thunder_egg_outline(np.random.default_rng([31, k]), "sinuous")
        _, _, b = thunder_egg_outline(np.random.default_rng([31, k]), "sinuous")
        assert style == "sinuous"
        assert a == b
        assert outline_problem(np.array(a)) is None
        assert 100 <= len(a) <= 2000


def test_sheets_draw_every_thunder_egg_sinuous_with_varied_shapes() -> None:
    eggs = [r for seed in range(1, 9) for r in draw_sheet(seed) if r.outline == "thunder-egg"]
    assert len(eggs) == 16
    assert {r.arms for r in eggs} == {"sinuous"}
    shapes = [r.arm_shape for r in eggs if r.arm_shape is not None]
    assert len(shapes) == 16
    assert all(BEND_RANGE[0] <= s.bend <= BEND_RANGE[1] for s in shapes)
    assert all(WIDTH_RANGE[0] <= s.width <= WIDTH_RANGE[1] for s in shapes)
    assert {s.buds for s in shapes} <= set(range(BUDS_RANGE[0], BUDS_RANGE[1] + 1))
    assert len({s.buds for s in shapes}) >= 3
    others = [r for r in draw_sheet(1) if r.outline != "thunder-egg"]
    assert {(r.arms, r.arm_shape) for r in others} == {(None, None)}


@pytest.mark.parametrize(
    "shape",
    [ArmShape(bend=0.7, width=1.3, buds=-1), ArmShape(bend=1.4, width=0.75, buds=2)],
)
def test_extreme_shapes_stay_sinuous_and_valid(shape: ArmShape) -> None:
    for k in range(3):
        _, _, outline = thunder_egg_outline(np.random.default_rng([37, k]), shape=shape)
        assert outline_problem(np.array(outline)) is None
        assert arm_meander(outline_mask(np.array(outline))) > SINUOUS_MIN


def test_more_lobes_add_area_and_leave_other_arms_alone() -> None:
    """Side lobes use their own streams: more lobes only add to the star."""
    from nodule.thunderegg import sinuous_mask

    fewer = sinuous_mask(np.random.default_rng(41), 6, ArmShape(buds=-1))
    more = sinuous_mask(np.random.default_rng(41), 6, ArmShape(buds=2))
    assert more.sum() > fewer.sum()
    assert (fewer & ~more).sum() < 0.01 * fewer.sum()
