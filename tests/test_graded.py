"""Graded chalcedony-to-quartz transitions on some quartz centres (ADR 0018 s.24)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from nodule.variety import StoneRecipe, draw
from nodule.variety.draw import graded_mm
from nodule.variety.layers import VarietyLayers
from nodule.variety.recipe import GRADE_MM, GRADED_CHANCE


def first(sheets: list[StoneRecipe], centre: str) -> StoneRecipe:
    return next(r for r in sheets if r.centre == centre)


def fixed(mm: float) -> Callable[[StoneRecipe], float]:
    """A stand-in for `draw.centre_mm`: every centre `mm` thick."""

    def centre(_: StoneRecipe) -> float:
        return mm

    return centre


def test_one_in_three_quartz_centres_graded_never_solid(
    sheets: list[StoneRecipe], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(draw, "centre_mm", fixed(10.0))  # wide: the cap never binds
    quartz, solid = first(sheets, "quartz-lined"), first(sheets, "solid")
    rng = np.random.default_rng(0)
    drawn = [graded_mm(rng.uniform(size=2), quartz) for _ in range(6000)]
    assert not any(graded_mm(rng.uniform(size=2), solid) for _ in range(500))
    share = np.mean([g > 0 for g in drawn])
    assert share == pytest.approx(GRADED_CHANCE, abs=0.02)
    graded = [g for g in drawn if g]
    assert GRADE_MM[0] <= min(graded)
    assert max(graded) <= GRADE_MM[1]


def test_the_zone_is_at_most_the_centres_thickness(
    sheets: list[StoneRecipe], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: a wide centre keeps the drawn width; a narrow one caps it."""
    quartz = first(sheets, "quartz-filled")
    u = np.array([0.0, 1.0])  # graded, widest
    monkeypatch.setattr(draw, "centre_mm", fixed(10.0))
    assert graded_mm(u, quartz) == pytest.approx(GRADE_MM[1])
    monkeypatch.setattr(draw, "centre_mm", fixed(0.7))
    assert graded_mm(u, quartz) == pytest.approx(0.7)


def test_sheet_zones_fit_their_centres(sheets: list[StoneRecipe]) -> None:
    for r in sheets[:24]:
        if r.grade_mm:
            assert r.grade_mm <= draw.centre_mm(r) + 1e-9


def test_sheets_carry_graded_quartz_centres_only(sheets: list[StoneRecipe]) -> None:
    quartz = [r for r in sheets if r.centre != "solid"]
    graded = [r for r in quartz if r.grade_mm]
    assert all(r.centre != "solid" for r in sheets if r.grade_mm)
    assert 0.15 < len(graded) / len(quartz) < 0.5
    assert max(r.grade_mm for r in graded) <= GRADE_MM[1]


def test_negative_control_a_solid_stone_cannot_be_graded(sheets: list[StoneRecipe]) -> None:
    solid = next(r for r in sheets if r.centre == "solid")
    with pytest.raises(ValueError, match="quartz centre"):
        StoneRecipe.model_validate(solid.model_dump() | {"grade_mm": 3.0})


def layers(recipe: StoneRecipe, grade_t: float) -> VarietyLayers:
    return VarietyLayers(recipe, np.zeros(3), 0.8, 0.0, None, None, grade_t=grade_t)


def test_the_zone_ramps_to_the_quartz(sheets: list[StoneRecipe]) -> None:
    r = next(r for r in sheets if r.grade_mm)
    lay = layers(r, 0.1)
    t = np.array([0.5, 0.7, 0.75, 0.8])
    assert lay.grade(t) == pytest.approx([0.0, 0.0, 0.5, 1.0])
