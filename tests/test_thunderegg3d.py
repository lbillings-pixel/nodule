"""Thunder-egg cavities drawn in 3D (ADR 0018 s.20; agatesim ADR 0089)."""

from __future__ import annotations

import numpy as np
import pytest
from agatesim._scipy import linear_at
from agatesim.volume.sampled import Sampled
from agatesim.volume.slicing import Plane

import nodule.variety.recipe as recipe_module
from nodule import thunderegg3d as t3
from nodule.thunderegg import ArmShape, sinuous_mask, thunder_egg_outline
from nodule.variety import StoneRecipe
from nodule.variety.recipe import inradius, solid_for

HEIGHTS = (-0.3, -0.15, 0.0, 0.15, 0.3)  # shape units: about +-8 mm on a 55 mm egg


def _field(k: int) -> np.ndarray:
    rng = np.random.default_rng([53, k])
    return t3.sinuous_field(rng, ArmShape.draw(rng))


def test_cuts_gain_and_lose_arms_and_the_prism_does_not() -> None:
    changed = 0
    for k in range(4):
        counts = {t3.arms_crossing(m) for m in t3.sections(_field(k), HEIGHTS)}
        changed += len(counts) > 1
    assert changed >= 3
    # Negative control: the old prism, the 2D star extruded, shows one star throughout.
    flat = sinuous_mask(np.random.default_rng(5), 7).reshape(192, 2, 192, 2).mean(axis=(1, 3)).T
    slab = (np.abs(t3._C) <= 0.36).astype(np.float64)  # pyright: ignore[reportPrivateUsage]
    prism = (flat[:, :, None] * slab[None, None, :]).astype(np.float32)
    assert len({t3.arms_crossing(m) for m in t3.sections(prism, HEIGHTS)}) == 1


def test_agatesim_solid_matches_the_drawing_in_3d() -> None:
    """Checked by sampling the whole cavity, not one cut: agatesim's solid and
    the drawn field agree except within a small part of a cell of the wall."""
    shape, cfg = t3.sinuous_solid(np.random.default_rng(71), 55.0)
    rng = np.random.default_rng(71)
    assert ArmShape.draw(rng) == shape
    field = t3.sinuous_field(rng, shape)  # the drawing agatesim accepted first time
    solid = cfg.build()
    assert isinstance(solid, Sampled)
    lo, hi = (np.asarray(b, dtype=np.float64) for b in solid.bounds())
    pts = np.random.default_rng(3).uniform(lo, hi, (100_000, 3))
    got = solid.contains(pts)
    idx = (pts - solid.origin) / solid.cell_mm
    crop = np.maximum(np.argwhere(field > 0.5).min(axis=0) - t3.PAD, 0)
    at = idx + crop
    want = linear_at(field.astype(np.float64), [at[:, 0], at[:, 1], at[:, 2]]) > 0.5
    bad = got != want
    assert bad.mean() < 0.01
    assert np.abs(solid.level(pts[bad])).max() < 0.5 * solid.cell_mm
    assert solid.contains(np.asarray(solid.centre())[None, :])[0]


def test_drawn_cavities_are_remembered_only_for_fresh_streams() -> None:
    a = t3.sinuous_solid(np.random.default_rng(73), 50.0)
    assert t3.sinuous_solid(np.random.default_rng(73), 50.0) is a
    used = np.random.default_rng(73)
    used.random()
    assert t3.sinuous_solid(used, 50.0)[1] != a[1]
    # A spawned stream of a list seed (as in tests and sheets) is remembered too.
    b = t3.sinuous_solid(np.random.default_rng([61, 0]).spawn(1)[0], 50.0)
    assert t3.sinuous_solid(np.random.default_rng([61, 0]).spawn(1)[0], 50.0) is b


def test_one_node_bridges_do_not_split_the_stored_cavity() -> None:
    """Draw 1 of the build check (root 1717) broke into pieces in agatesim's 8-bit
    store until the level kept only the largest stored piece."""
    rng = np.random.default_rng(1717).spawn(20)[1]
    size = float(rng.uniform(40, 70))
    level = t3.level_cells(t3.sinuous_field(rng, ArmShape.draw(rng)))
    t3.SolidConfig(sampled=t3.SampledLevel.of(level), host="volcanic", size_mm=size)  # type: ignore[arg-type]


def test_jagged_thunder_eggs_draw_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 3D build changed only winding stars: a jagged stone's draws are those of
    the 2D `thunder_egg_outline` (a control on the stream bookkeeping)."""

    def no_3d(rng: np.random.Generator, size_mm: float) -> tuple[ArmShape, None]:
        return ArmShape(), None  # winding stars are not compared here

    monkeypatch.setattr(recipe_module, "sinuous_solid", no_3d)
    jagged = 0
    for k in range(30):
        cfg, style, _ = solid_for(np.random.default_rng([61, k]), "thunder-egg")
        old = np.random.default_rng([61, k])
        old.uniform(0, 360)  # `solid_for` draws a yaw first, for every outline
        old_style, _, outline = thunder_egg_outline(old)
        assert style == old_style
        if style == "jagged":
            jagged += 1
            assert cfg.outline == outline
            assert cfg.size_mm == float(old.uniform(40, 70))
    assert jagged >= 5


def test_the_cut_passes_near_the_deepest_point(sheets: list[StoneRecipe]) -> None:
    """Checked on the plane agatesim cuts, against a deepest point found apart from
    `body_height` (random points, not its grid)."""
    eggs = [r for r in sheets if r.arms == "sinuous"]
    assert len(eggs) >= 8
    # The cuts furthest from the box's middle: a wrong sign or frame shows most there.
    for r in sorted(eggs, key=lambda e: -abs(e.cut.offset_mm))[:3]:
        solid = r.solid.build()
        assert r.solid.sampled is not None
        lo, hi = (np.asarray(b, dtype=np.float64) for b in solid.bounds())
        pts = np.random.default_rng(5).uniform(lo, hi, (200_000, 3))
        pts = pts[solid.contains(pts)]
        deep = pts[int(np.argmax(solid.wall_distance(pts)))]
        plane = Plane.of(r.cut, np.asarray(solid.centre()))
        assert abs(float((deep - plane.origin) @ plane.normal)) <= 0.1 * inradius(solid) + 1.0
