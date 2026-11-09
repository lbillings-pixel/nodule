"""Thunder-egg stars: sinuous arms and the check that tells them from straight ones."""

from __future__ import annotations

import numpy as np
import pytest
from agatesim.geometry import outline_problem

from nodule.thunderegg import (
    SINUOUS_MIN,
    arm_meander,
    outline_mask,
    thunder_egg_outline,
)
from nodule.variety import draw_sheet


def _meander(style: str, k: int) -> float:
    _, outline = thunder_egg_outline(np.random.default_rng([29, k]), style)  # type: ignore[arg-type]
    return arm_meander(outline_mask(np.array(outline)))


@pytest.mark.parametrize("k", range(6))
def test_sinuous_check_and_its_negative_control(k: int) -> None:
    """Sinuous arms pass; the angular star (straight arms) is the negative control."""
    assert _meander("sinuous", k) > SINUOUS_MIN
    assert _meander("angular", k) < SINUOUS_MIN


def test_sinuous_outlines_are_valid_and_repeatable() -> None:
    for k in range(4):
        style, a = thunder_egg_outline(np.random.default_rng([31, k]), "sinuous")
        _, b = thunder_egg_outline(np.random.default_rng([31, k]), "sinuous")
        assert style == "sinuous"
        assert a == b
        assert outline_problem(np.array(a)) is None
        assert 100 <= len(a) <= 2000


def test_sheets_mix_arm_styles_mostly_sinuous() -> None:
    styles = [
        r.arms for seed in range(1, 9) for r in draw_sheet(seed) if r.outline == "thunder-egg"
    ]
    assert len(styles) == 16
    assert 0.5 <= styles.count("sinuous") / len(styles) < 1.0
    others = [r.arms for r in draw_sheet(1) if r.outline != "thunder-egg"]
    assert set(others) == {None}
