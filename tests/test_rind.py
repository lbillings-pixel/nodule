"""Host rind and host specks (ADR 0018 s.16): the deal on the recipes, the distance by
the level's slope (negative control: the raw level), the sub-pixel drawing (negative
control: a thresholded draw), windows, world-coordinate specks and the padded sheet
picture."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from agatesim.volume.config import SolidConfig
from pydantic import ValidationError

from nodule.rind import (
    RHYOLITE_GRAINS,
    RHYOLITE_SHAPES,
    SEDIMENT,
    VALUE_STD,
    VOLCANIC,
    VOLCANIC_CAP_MM,
    Rind,
    coverage,
    draw_rind_rgb,
    grains,
    host_specks,
    rind_width,
    value_noise,
    wall_distance,
)
from nodule.sheet import framed_rgb, preview_rgb
from nodule.variety import StoneRecipe, build, cut_of, rim_point


def test_value_noise_scale_constant() -> None:
    pts = np.random.default_rng(9).uniform(-50, 50, (200_000, 3))
    assert float(np.std(value_noise(pts, 1.0, 1234))) == pytest.approx(VALUE_STD, rel=0.01)


def test_rinds_follow_the_host(sheets: list[StoneRecipe]) -> None:
    kinds = Counter(
        (r.outline in VOLCANIC, None if r.rind is None else r.rind.kind) for r in sheets
    )
    volcanic = sum(r.outline in VOLCANIC for r in sheets)
    sediment = [r for r in sheets if r.outline in SEDIMENT]
    assert kinds[(True, "volcanic")] == volcanic  # every volcanic stone
    assert all(r.rind is None for r in sheets if r.outline == "thunder-egg")
    pale = [r for r in sediment if r.rind is not None]
    assert all(r.rind is not None and r.rind.kind == "pale" for r in pale)
    assert len(pale) == 11  # "sometimes": about half
    assert len(sediment) == 24
    windows = [r for r in sheets if r.rind is not None and r.rind.window is not None]
    assert len(windows) == 6
    assert all(r.outline in VOLCANIC for r in windows)
    chars = [r.rind.character for r in sheets if r.rind is not None and r.rind.kind == "volcanic"]
    assert np.median(chars) < 0.35  # most stones towards the thin end
    for r in sheets:
        if r.rind is not None and r.rind.kind == "volcanic":
            assert 0.6 - 1e-9 <= r.rind.thickness_mm <= 4.0 + 1e-9
            assert r.rind.reach_mm() <= VOLCANIC_CAP_MM + 0.2 + 1e-9
        elif r.rind is not None:
            assert 0.05 - 1e-9 <= r.rind.thickness_mm <= 1.0 + 1e-9


def test_a_rind_must_fit_its_host(sheets: list[StoneRecipe]) -> None:
    egg = next(r for r in sheets if r.outline == "thunder-egg")
    vesicle = next(r for r in sheets if r.outline == "vesicle")
    volcanic = Rind(kind="volcanic", thickness_mm=1.0, tone="green")
    with pytest.raises(ValidationError):
        StoneRecipe.model_validate(egg.model_dump() | {"rind": volcanic.model_dump()})
    pale = Rind(kind="pale", thickness_mm=0.2, tone="tan")
    with pytest.raises(ValidationError):
        StoneRecipe.model_validate(vesicle.model_dump() | {"rind": pale.model_dump()})
    StoneRecipe.model_validate(vesicle.model_dump() | {"rind": volcanic.model_dump()})
    with pytest.raises(ValidationError):  # the pale rind is tan, and only it
        Rind(kind="pale", thickness_mm=0.2, tone="echo")
    with pytest.raises(ValidationError):
        Rind(kind="volcanic", thickness_mm=1.0, tone="green", window=(0.0, 0.0, 0.0))


def test_the_pale_rind_stays_inside_its_reach() -> None:
    """The drawn width never passes the reach, so the outer edge is the noise's at any
    pixel size (a cut-off would move with the pixel)."""
    rind = Rind(kind="pale", thickness_mm=1.0, tone="tan")
    foot = np.random.default_rng(2).uniform(-100, 100, (200_000, 3))
    w = rind_width(rind, foot, seed=4, window_at=None)
    assert w.max() <= rind.reach_mm() - 0.05 + 1e-9
    assert np.mean(w >= 1.6 - 1e-9) < 0.05  # clipped on a few per cent of the wall


def _wall_points(solid: object, n: int = 200) -> np.ndarray:
    rng = np.random.default_rng(3)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    c = solid.centre()  # type: ignore[attr-defined]
    return np.array([c + rim_point(solid, d) for d in dirs])  # type: ignore[arg-type]


def test_distance_by_slope_and_the_raw_level_rejected() -> None:
    """On an ellipsoid (convex: offsetting a wall point along its normal by D gives a
    point exactly D away) the slope-corrected level matches; the raw level (negative
    control) runs short by up to a fifth."""
    solid = SolidConfig(preset="vesicle-round", size_mm=45.0, yaw_deg=30.0, pitch_deg=10.0).build()
    wall = _wall_points(solid)
    _, g = wall_distance(solid, wall, 1.0)
    for dist in (0.5, 2.0, 6.0):
        p = wall - dist * g
        est, _ = wall_distance(solid, p, 1e9)
        assert np.max(np.abs(est - dist)) < 0.01 * dist + 0.01
        raw = -solid.level(p)
        assert np.max(np.abs(raw - dist)) > 0.15 * dist  # the uncorrected level fails


class HalfSpace:
    """Inside where x < 0; the level is not a distance (slope 2) unless corrected."""

    def level(self, points: np.ndarray) -> np.ndarray:
        return -2.0 * points[:, 0]


@pytest.mark.parametrize("px", [0.1, 0.03])
def test_a_rind_thinner_than_a_pixel_keeps_its_width(px: float) -> None:
    """Ink across the wall (rind share x pixel) equals the drawn width at any phase of
    the pixel grid; a thresholded draw (negative control) does not."""
    rind = Rind(kind="pale", thickness_mm=0.05, tone="tan")
    for phase in np.linspace(0, 1, 7, endpoint=False):
        x = (np.arange(-20, 40) + phase) * px
        pts = np.stack([x, np.full_like(x, 3.0), np.full_like(x, -2.0)], 1)
        rgb = np.zeros((len(x), 3))
        sel, w, cov = draw_rind_rgb(
            rgb, pts, x < 0, px, HalfSpace(), rind, (255, 255, 255), rgb.copy(), seed=5
        )
        assert np.ptp(w) < 1e-9  # one foot point along the row: one width
        assert float(np.sum(cov) * px) == pytest.approx(float(w[0]), abs=1e-6)
        # negative control: whole pixels where the share passes one half
        if px >= float(w[0]):
            thresholded = float(np.sum(coverage(x[sel], w, px) >= 0.5) * px)
            assert abs(thresholded - float(w[0])) > 0.5 * float(w[0])


def test_a_window_clears_the_rind() -> None:
    rind = Rind(kind="volcanic", thickness_mm=1.5, character=0.3, tone="green", window_mm=2.5)
    at = np.array([[0.0, 0.0, 0.0]])
    foot = np.array([[0.0, 0.5, 0.0], [0.0, 2.2, 0.0], [0.0, 5.0, 0.0]])  # ramp 1.9-2.5
    w = rind_width(rind, foot, seed=1, window_at=at[0])
    free = rind_width(rind, foot, seed=1, window_at=None)
    assert w[0] == 0.0
    assert 0 < w[1] < free[1]
    assert w[2] == free[2]


def test_specks_are_fixed_in_world_coordinates() -> None:
    """Rhyolite specks: the same share of the host at two spacings (world-coordinate
    grains); about 5 % of the host over a large sample (the owner's verdict, measured)."""
    cell, p, r0, r1 = RHYOLITE_GRAINS
    pts = np.random.default_rng(1).uniform(-200, 200, (300_000, 3))
    assert 0.04 < grains(pts, cell, 1, p, r0, r1, RHYOLITE_SHAPES)[0].mean() < 0.06
    shares: list[float] = []
    for px in (0.08, 0.04):
        ss, tt = np.meshgrid(np.arange(0, 60, px), np.arange(0, 6, px))
        pts = np.stack([ss.ravel(), tt.ravel(), np.full(ss.size, 1.7)], 1)
        hit, _, _ = grains(pts, cell, 77, p, r0, r1, RHYOLITE_SHAPES)
        shares.append(float(hit.mean()))
        rgb = np.full((len(pts), 3), 100.0)
        host_specks(rgb, pts, "thunder-egg", 3)
        assert 0.01 < np.mean(np.any(rgb != 100.0, axis=1)) < 0.10
    assert shares[0] == pytest.approx(shares[1], abs=0.003)


def test_sheet_picture_pads_round_the_cut_and_keeps_the_agate(sheets: list[StoneRecipe]) -> None:
    rec = next(r for r in sheets if r.outline == "pebble" and r.rind is not None)
    specimen, _ = build(rec)
    lo, hi = specimen.volume.solid.bounds()
    spacing = float(np.max(np.asarray(hi) - np.asarray(lo))) / 120
    cut = cut_of(specimen, rec, spacing)
    img = framed_rgb(specimen, cut, rec)
    base = preview_rgb(cut, rec)
    assert rec.rind is not None
    k = int(np.ceil((rec.rind.reach_mm() + spacing) / spacing))
    assert img.shape == (base.shape[0] + 2 * k, base.shape[1] + 2 * k, 3)
    core = img[k : k + base.shape[0], k : k + base.shape[1]]
    ss, tt = np.meshgrid(cut.s, cut.t)
    lev = specimen.volume.solid.level(cut.plane.points(ss, tt).reshape(-1, 3))
    deep = (lev.reshape(cut.inside.shape) > 2 * spacing)[::-1]
    assert np.array_equal(core[deep], base[deep])  # agate away from the wall untouched
    # the rind shows: host-side pixels next to the wall differ from the flat host
    assert np.any(np.any(core[~cut.inside[::-1]] != base[~cut.inside[::-1]], axis=1))
