"""Each irregular and lobate stone's own lumps (ADR 0018 s.25, `nodule.variety.lumps`)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from agatesim.volume.config import SliceConfig, SolidConfig

from nodule.variety import StoneRecipe, draw_sheet
from nodule.variety import lumps as lumps_module
from nodule.variety import recipe as recipe_module
from nodule.variety.draw import CLEAN_CUT, clean_cut, cut_obliqueness
from nodule.variety.lumps import (
    IRREGULAR,
    LOBATE,
    _remembered,  # pyright: ignore[reportPrivateUsage]
    ball_level,
    balls_of,
    draw_balls,
    lumps_solid,
)

# draws that do not depend on the solid's geometry
OWN_DRAWS = ("seed", "colour", "warp", "band_mm", "wobble_mm", "wobble_scale_mm",
             "white_fade", "rim_stain", "tint", "centre")  # fmt: skip


def lumpy(r: StoneRecipe) -> bool:
    return r.solid.sampled is not None and r.outline in ("irregular", "vesicle")


def test_sheets_give_irregular_and_lobate_stones_their_own_lumps(
    sheets: list[StoneRecipe],
) -> None:
    """Every irregular stone and every lobate vesicle has its own drawn solid, all
    different; round vesicles and the other outlines keep their solids. Control:
    with OWN_LUMPS off every irregular stone shares one preset outline."""
    irregular = [r for r in sheets if r.outline == "irregular"]
    vesicles = [r for r in sheets if r.outline == "vesicle"]
    assert len(irregular) >= 8
    assert all(r.solid.sampled is not None for r in irregular)
    lobate = [r for r in vesicles if r.solid.sampled is not None]
    assert len(lobate) >= 10
    assert all(r.solid.preset == "vesicle-round" for r in vesicles if r.solid.sampled is None)
    texts = [r.solid.sampled.level for r in sheets if lumpy(r)]  # type: ignore[union-attr]
    assert len(set(texts)) == len(texts)
    hosts = {r.solid.host for r in irregular}
    assert hosts == {"sedimentary"}
    assert {r.solid.host for r in lobate} == {"volcanic"}


def test_lumps_leave_every_other_draw_as_it_was(
    sheets: list[StoneRecipe], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stones without lumps are unchanged whole; lumpy stones keep every draw that
    does not depend on their solid (the lumps come from a spawned stream). Sheets
    0-2 only, to keep the test short."""
    monkeypatch.setattr(recipe_module, "OWN_LUMPS", False)
    before = [r for seed in range(3) for r in draw_sheet(seed)]
    after = sheets[: len(before)]
    assert any(lumpy(r) for r in after)
    for b, a in zip(before, after, strict=True):
        if not lumpy(a):
            assert a == b
            continue
        assert b.solid.preset in ("nodule-irregular", "vesicle-lobate")
        assert a.solid.size_mm == b.solid.size_mm
        for name in OWN_DRAWS:
            assert getattr(a, name) == getattr(b, name), name


def test_balls_follow_their_style() -> None:
    """Counts, radii and reaches within each style's ranges; lobes near one plane
    and spread round it; irregular lumps at least min_angle_deg apart."""
    rng = np.random.default_rng(4)
    for style in (IRREGULAR, LOBATE):
        for _ in range(200):
            centres, radii = draw_balls(rng, style)
            n = len(radii) - 1
            assert n in style.counts
            assert style.core[0] <= radii[0] <= style.core[1]
            assert np.all((radii[1:] >= style.radius[0]) & (radii[1:] <= style.radius[1]))
            reach = np.linalg.norm(centres[1:], axis=1)
            assert np.all((reach >= style.reach[0] - 1e-9) & (reach <= style.reach[1] + 1e-9))
            dirs = centres[1:] / reach[:, None]
            if style.plane_deg is not None:
                el = np.degrees(np.arcsin(np.abs(dirs[:, 2])))
                assert np.all(el <= style.plane_deg + 1e-9)
                az = np.sort(np.arctan2(dirs[:, 1], dirs[:, 0]))
                gaps = np.diff(np.r_[az, az[0] + 2 * math.pi])
                assert gaps.min() > 0.5 * 2 * math.pi / n
    # irregular lumps keep their angle (as a share: the last try may be taken)
    cos: list[float] = []
    for _ in range(300):
        centres, _ = draw_balls(rng, IRREGULAR)
        d = centres[1:] / np.linalg.norm(centres[1:], axis=1)[:, None]
        g = d @ d.T
        cos += [float(v) for v in g[np.triu_indices(len(d), 1)]]
    assert np.mean(np.array(cos) < math.cos(math.radians(IRREGULAR.min_angle_deg))) > 0.99


def test_ball_level_is_the_union() -> None:
    centres = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    radii = np.array([0.6, 0.4])
    level = ball_level(centres, radii, grid=64)
    c = (np.arange(64) + 0.5) * (2 * 1.25 / 64) - 1.25
    i0 = int(np.argmin(np.abs(c - 0.0)))
    assert level[i0, i0, i0] == pytest.approx(0.6 - abs(c[i0]) * math.sqrt(3), abs=0.03)
    # a point inside the second ball only
    i1 = int(np.argmin(np.abs(c - 0.85)))
    assert level[i1, i0, i0] > 0
    j = int(np.argmin(np.abs(c + 0.85)))
    assert level[j, i0, i0] < 0  # control: the mirror point lies outside


def test_lumpy_solids_are_one_piece_of_the_drawn_size() -> None:
    """A lumpy solid's longest side is the drawn size, it contains its centre and
    has room inside; the same stream draws the same solid again (cache cleared)."""
    for kind in ("irregular", "lobate"):
        cfg = lumps_solid(np.random.default_rng(np.random.SeedSequence(77)), kind, 44.0)
        _remembered.cache_clear()
        again = lumps_solid(np.random.default_rng(np.random.SeedSequence(77)), kind, 44.0)
        assert again is not cfg
        assert again == cfg
        solid = cfg.build()
        lo, hi = (np.asarray(b) for b in solid.bounds())
        assert float(np.max(hi - lo)) == pytest.approx(44.0, rel=0.02)
        assert solid.contains(np.zeros((1, 3)))[0]
        assert float(solid.wall_distance(np.zeros((1, 3)))[0]) > 8.0


def test_a_lumpy_solids_balls_sit_on_its_wall() -> None:
    """The balls registered for the cutter's rule (`balls_of`) are the stored solid's, in
    its own mm: points on a ball's surface outside every other ball lie on the solid's wall
    (within a few tenths of a cell), points at the balls' centres inside. Controls: balls
    shifted by 2 mm miss the wall; another kind of solid has no balls."""
    for kind in ("irregular", "lobate"):
        solid = lumps_solid(np.random.default_rng(np.random.SeedSequence(5)), kind, 40.0).build()
        balls = balls_of(solid)
        assert balls is not None
        centres, radii = balls
        assert solid.contains(centres).all()
        rng = np.random.default_rng(1)
        d = rng.normal(size=(4000, 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        k = rng.integers(0, len(radii), 4000)
        p = centres[k] + radii[k, None] * d
        exposed = np.all(np.linalg.norm(p[:, None] - centres[None], axis=2) >= radii - 1e-9,
                         axis=1)  # fmt: skip
        assert exposed.sum() > 1000
        cell = float(np.max(radii)) / 40  # a rough cell (the core spans ~40 grid cells)
        assert np.median(np.abs(solid.level(p[exposed]))) < 0.3 * cell
        shifted = np.abs(solid.level(p[exposed] + 2.0))
        assert np.median(shifted) > 0.5
    preset = SolidConfig(preset="vesicle-round", size_mm=40).build()
    assert balls_of(preset) is None


def test_the_cutter_avoids_grazing_a_lump(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cutter's rule (`cut_obliqueness`, `clean_cut`) reads a lumpy solid's balls: a
    cut grazing a lump scores near 1 and is redrawn to a clean one. Control: without its
    balls (another process's solid) the rule cannot see the lump."""
    solid = lumps_solid(np.random.default_rng(np.random.SeedSequence(9)), "lobate", 40.0).build()
    rng = np.random.default_rng(2)
    grazing = SliceConfig(tilt_deg=90.0, azimuth_deg=0.0, offset_mm=0.0)
    for _ in range(400):
        grazing = SliceConfig(tilt_deg=float(rng.uniform(70, 90)),
                              azimuth_deg=float(rng.uniform(0, 360)),
                              offset_mm=float(rng.uniform(-8.0, 8.0)))  # fmt: skip
        if cut_obliqueness(solid, grazing) > 0.9:
            break
    assert cut_obliqueness(solid, grazing) > 0.9
    chosen = clean_cut(np.random.default_rng(3), solid, grazing, 10.0)
    assert cut_obliqueness(solid, chosen) <= CLEAN_CUT
    monkeypatch.setattr(lumps_module, "_BALLS", {})
    assert cut_obliqueness(solid, grazing) == 0.0
