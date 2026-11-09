"""Thunder-egg stars: sinuous and jagged arms, and the checks that tell them apart."""

from __future__ import annotations

import numpy as np
import pytest
from agatesim.geometry import outline_problem

from nodule.thunderegg import (
    BEND_RANGE,
    EXTRA_ARMS_RANGE,
    JAGGED_MIN,
    SINUOUS_MIN,
    WIDTH_RANGE,
    ArmShape,
    ArmStyle,
    arm_meander,
    outline_mask,
    thunder_egg_outline,
    wall_corners,
)
from nodule.variety import draw_sheet


def _outline(style: ArmStyle, k: int, seed: int = 29) -> np.ndarray:
    return np.array(thunder_egg_outline(np.random.default_rng([seed, k]), style)[2])


@pytest.mark.parametrize("k", range(6))
def test_sinuous_check_and_its_negative_control(k: int) -> None:
    """Sinuous arms pass; the angular star (straight arms) is the negative control."""
    assert arm_meander(outline_mask(_outline("sinuous", k))) > SINUOUS_MIN
    assert arm_meander(outline_mask(_outline("angular", k))) < SINUOUS_MIN


@pytest.mark.parametrize("k", range(6))
def test_jagged_check_and_its_negative_controls(k: int) -> None:
    """Stepped, zigzag walls pass; smooth winding and straight-sided walls fail."""
    assert wall_corners(_outline("jagged", k)) > JAGGED_MIN
    assert wall_corners(_outline("sinuous", k)) < JAGGED_MIN
    assert wall_corners(_outline("angular", k)) < JAGGED_MIN


@pytest.mark.parametrize("style", ["sinuous", "jagged"])
def test_outlines_are_valid_and_repeatable(style: ArmStyle) -> None:
    for k in range(4):
        got, _, a = thunder_egg_outline(np.random.default_rng([31, k]), style)
        _, _, b = thunder_egg_outline(np.random.default_rng([31, k]), style)
        assert got == style
        assert a == b
        assert outline_problem(np.array(a)) is None
        assert 30 <= len(a) <= 2000


def test_sheets_mix_sinuous_and_jagged_thunder_eggs() -> None:
    eggs = [r for seed in range(1, 13) for r in draw_sheet(seed) if r.outline == "thunder-egg"]
    assert len(eggs) == 24
    styles = [r.arms for r in eggs]
    assert set(styles) == {"sinuous", "jagged"}
    assert 0.15 <= styles.count("jagged") / len(styles) <= 0.5
    shapes = [r.arm_shape for r in eggs if r.arms == "sinuous"]
    assert all(s is not None for s in shapes)
    assert all(r.arm_shape is None for r in eggs if r.arms == "jagged")
    shapes = [s for s in shapes if s is not None]
    assert all(BEND_RANGE[0] <= s.bend <= BEND_RANGE[1] for s in shapes)
    assert all(WIDTH_RANGE[0] <= s.width <= WIDTH_RANGE[1] for s in shapes)
    extra = {s.extra_arms for s in shapes}
    assert extra <= set(range(EXTRA_ARMS_RANGE[0], EXTRA_ARMS_RANGE[1] + 1))
    assert len(extra) >= 3
    others = [r for r in draw_sheet(1) if r.outline != "thunder-egg"]
    assert {(r.arms, r.arm_shape) for r in others} == {(None, None)}


@pytest.mark.parametrize(
    "shape",
    [ArmShape(bend=0.7, width=1.3, extra_arms=-1), ArmShape(bend=1.4, width=0.75, extra_arms=2)],
)
def test_extreme_shapes_stay_sinuous_and_valid(shape: ArmShape) -> None:
    for k in range(3):
        rng = np.random.default_rng([37, k])
        _, _, outline = thunder_egg_outline(rng, "sinuous", shape)
        assert outline_problem(np.array(outline)) is None
        assert arm_meander(outline_mask(np.array(outline))) > SINUOUS_MIN


def test_extra_arms_change_the_arm_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sinuous star is drawn with the base count (5-8) plus its extra arms."""
    import nodule.thunderegg as te

    seen: list[int] = []

    def spy(
        rng: np.random.Generator, arms: int, shape: ArmShape | None = None
    ) -> list[tuple[float, float]]:
        seen.append(arms)
        return []

    monkeypatch.setattr(te, "sinuous_outline", spy)
    for extra in (-1, 0, 2):
        te.thunder_egg_outline(np.random.default_rng(43), "sinuous", ArmShape(extra_arms=extra))
    base = int(np.random.default_rng(43).integers(te.ARMS_BASE[0], te.ARMS_BASE[1] + 1))
    assert seen == [base - 1, base, base + 2]
    counts: set[int] = set()
    for k in range(200):
        seen.clear()
        te.thunder_egg_outline(np.random.default_rng([47, k]), "sinuous")
        counts.add(seen[0])
    assert counts == set(range(4, 11))
