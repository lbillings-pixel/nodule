"""Wide open centres on some quartz-lined stones (ADR 0018 s.24)."""

from __future__ import annotations

import numpy as np
import pytest

from nodule.variety.draw import wide_centre, wide_plan
from nodule.variety.recipe import (
    WIDE_CENTRE_CHANCE,
    WIDE_OPEN_SHARE,
    WIDE_QUARTZ_MIN_MM,
    WIDE_QUARTZ_SHARE,
)

OUTLINES = ["vesicle", "thunder-egg", "irregular", "prolate"] * 3
CENTRES = ["quartz-lined"] * 8 + ["solid", "quartz-filled"] * 2


def plans(n: int) -> list[list[tuple[float, float] | None]]:
    return [wide_plan(np.random.default_rng(s), OUTLINES, CENTRES) for s in range(n)]


def test_about_one_in_three_lined_stones_wide_never_eggs_or_other_centres() -> None:
    wide = lined = 0
    for plan in plans(2000):
        for o, c, w in zip(OUTLINES, CENTRES, plan, strict=True):
            if c != "quartz-lined" or o == "thunder-egg":
                assert w is None
                continue
            lined += 1
            wide += w is not None
    assert wide / lined == pytest.approx(WIDE_CENTRE_CHANCE, abs=0.02)


def test_shares_inside_their_ranges() -> None:
    drawn = [w for plan in plans(300) for w in plan if w is not None]
    opens, quartz = np.array(drawn).T
    assert WIDE_OPEN_SHARE[0] <= opens.min()
    assert opens.max() <= WIDE_OPEN_SHARE[1]
    assert WIDE_QUARTZ_SHARE[0] <= quartz.min()
    assert quartz.max() <= WIDE_QUARTZ_SHARE[1]


def test_a_stones_draw_does_not_depend_on_the_others() -> None:
    """Every stone takes its draws, so changing another stone's centre leaves it be."""
    other = ["solid", *CENTRES[1:]]
    a = wide_plan(np.random.default_rng(7), OUTLINES, CENTRES)
    b = wide_plan(np.random.default_rng(7), OUTLINES, other)
    assert a[1:] == b[1:]


def test_wide_centre_gives_the_open_and_quartz_depths() -> None:
    depth = 20.0
    width, lining = wide_centre((0.35, 0.12), depth)
    assert (1 - lining) * width * depth == pytest.approx(0.35 * depth)  # open radius
    assert lining * width * depth == pytest.approx(0.12 * depth)  # quartz thickness


def test_negative_control_thin_quartz_is_raised_to_the_minimum() -> None:
    """A shallow cavity at the low quartz share would leave under three grains of
    quartz: the lining is raised to WIDE_QUARTZ_MIN_MM, the open radius kept."""
    depth = 8.0
    width, lining = wide_centre((0.3, 0.08), depth)
    assert 0.08 * depth < WIDE_QUARTZ_MIN_MM
    assert lining * width * depth == pytest.approx(WIDE_QUARTZ_MIN_MM)
    assert (1 - lining) * width * depth == pytest.approx(0.3 * depth)
