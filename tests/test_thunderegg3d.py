"""Thunder-egg cavities drawn in 3D (ADR 0018 s.20; agatesim ADR 0089)."""

from __future__ import annotations

import numpy as np
import pytest
from agatesim._scipy import linear_at
from agatesim.volume.sampled import Sampled
from agatesim.volume.slicing import Plane

import nodule.variety.recipe as recipe_module
from nodule import thunderegg3d as t3
from nodule.thunderegg import ARMS_BASE, JAGGED_SHARE, ArmShape, sinuous_mask
from nodule.variety import StoneRecipe
from nodule.variety.recipe import body_height, inradius, solid_for

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


def test_thunder_egg_streams_are_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    """A control on the stream bookkeeping: the style is drawn as in the 2D
    `thunder_egg_outline`, each 3D cavity from the stone's first spawned stream,
    and the size next from the stone's own stream (so a winding stone draws as
    before the jagged build)."""
    seen: list[tuple[str, tuple[int, ...], float]] = []

    def fake_sinuous(rng: np.random.Generator, size_mm: float) -> tuple[ArmShape, str]:
        seq = rng.bit_generator.seed_seq
        assert isinstance(seq, np.random.SeedSequence)
        seen.append(("sinuous", tuple(seq.spawn_key), size_mm))
        return ArmShape(), "solid"

    def fake_jagged(rng: np.random.Generator, size_mm: float) -> str:
        seq = rng.bit_generator.seed_seq
        assert isinstance(seq, np.random.SeedSequence)
        seen.append(("jagged", tuple(seq.spawn_key), size_mm))
        return "solid"

    def fake_lens(rng: np.random.Generator, size_mm: float) -> str:
        seq = rng.bit_generator.seed_seq
        assert isinstance(seq, np.random.SeedSequence)
        seen.append(("lens", tuple(seq.spawn_key), size_mm))
        return "solid"

    monkeypatch.setattr(recipe_module, "sinuous_solid", fake_sinuous)
    monkeypatch.setattr(recipe_module, "jagged_solid", fake_jagged)
    monkeypatch.setattr(recipe_module, "lens_solid", fake_lens)
    styles: list[str | None] = []
    for k in range(3000):
        _, style, _ = solid_for(np.random.default_rng([61, k]), "thunder-egg")
        ref = np.random.default_rng([61, k])
        ref.uniform(0, 360)  # `solid_for` draws a yaw first, for every outline
        ref.integers(ARMS_BASE[0], ARMS_BASE[1] + 1)  # as `thunder_egg_outline`
        old_style = "jagged" if ref.random() < JAGGED_SHARE else "sinuous"
        assert seen[-1][2] == float(ref.uniform(40, 70))
        assert style == seen[-1][0]
        if style == "lens":  # chosen and drawn on the second spawned stream (s.22)
            assert seen[-1][1] == (1, 0)
        else:
            assert style == old_style
            assert seen[-1][1] == (0,)  # the stone's first spawned stream
        styles.append(style)
    assert 800 <= styles.count("jagged") <= 1200
    # 1 in 60: 50 expected of 3000 (binomial sd 7).
    assert 28 <= styles.count("lens") <= 72


def test_lens_cuts_are_long_and_star_cuts_are_not() -> None:
    """The lens check on middle cuts; negative controls: winding and jagged stars."""
    lenses = [t3.lens_field(r) for r in np.random.default_rng(1724).spawn(8)]
    assert all(
        t3.cut_elongation(t3.sections(f, (0.0,))[0]) > t3.LENS_ELONGATION_MIN for f in lenses
    )
    stars = [_field(k) for k in range(3)] + [_jagged_inside(k) for k in range(3)]
    assert all(t3.cut_elongation(t3.sections(f, (0.0,))[0]) < t3.LENS_ELONGATION_MIN for f in stars)


def test_lens_sizes_lean_fuller_and_a_quarter_are_bent() -> None:
    """The owner's pick (2026-11-18): thin to fuller, leaning fuller; ~1 in 4 bent."""
    sizes = np.array([t3.lens_size(np.random.default_rng([67, s])) for s in range(2000)])
    ratio = sizes[:, 2] / sizes[:, 0]
    lo, hi = t3.LENS.thick
    assert lo <= ratio.min()
    assert ratio.max() <= hi
    assert np.median(ratio) > (lo + hi) / 2 + 0.03
    assert 0.21 < float((sizes[:, 3] != 0).mean()) < 0.29


def test_agatesim_lens_solid_matches_the_drawing_in_3d() -> None:
    """Checked by sampling the whole cavity, as the stars."""
    cfg = t3.lens_solid(np.random.default_rng(83), 55.0)
    assert t3.lens_solid(np.random.default_rng(83), 55.0) is cfg  # remembered
    field = t3.lens_field(np.random.default_rng(83))
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


def _jagged_inside(k: int) -> np.ndarray:
    return (t3.jagged_union(np.random.default_rng([59, k])) > 0).astype(np.float32)


def test_jagged_cuts_gain_and_lose_arms() -> None:
    changed = sum(
        len({t3.arms_crossing(m) for m in t3.sections(_jagged_inside(k), HEIGHTS)}) > 1
        for k in range(4)
    )
    assert changed >= 3


def test_agatesim_jagged_solid_matches_the_drawing_in_3d() -> None:
    """Checked by sampling the whole cavity: agatesim's solid and the drawn union
    agree except within a small part of a cell of the wall."""
    cfg = t3.jagged_solid(np.random.default_rng(79), 55.0)
    assert t3.jagged_solid(np.random.default_rng(79), 55.0) is cfg  # remembered
    level = t3.jagged_union(np.random.default_rng(79))
    solid = cfg.build()
    assert isinstance(solid, Sampled)
    lo, hi = (np.asarray(b, dtype=np.float64) for b in solid.bounds())
    pts = np.random.default_rng(3).uniform(lo, hi, (100_000, 3))
    got = solid.contains(pts)
    idx = (pts - solid.origin) / solid.cell_mm
    crop = np.maximum(np.argwhere(level > 0).min(axis=0) - t3.PAD, 0)
    at = idx + crop
    want = linear_at(level.astype(np.float64), [at[:, 0], at[:, 1], at[:, 2]]) > 0
    bad = got != want
    assert bad.mean() < 0.01
    assert np.abs(solid.level(pts[bad])).max() < 0.5 * solid.cell_mm
    assert solid.contains(np.asarray(solid.centre())[None, :])[0]


def test_jagged_walls_are_cornered_and_winding_ones_are_not(sheets: list[StoneRecipe]) -> None:
    """The corner check, as a group (it is not reliable stone by stone: on fresh
    seeds 1 jagged cavity in 20 fell below). Negative control: winding cavities,
    every one below. Four of each from the shared sheets."""

    def score(r: StoneRecipe) -> float:
        solid = r.solid.build()
        assert isinstance(solid, Sampled)
        return t3.cut_corners(solid, float(np.asarray(solid.centre())[2]) + body_height(solid))

    eggs = [r for r in sheets if r.outline == "thunder-egg"]
    jag = [score(r) for r in [r for r in eggs if r.arms == "jagged"][:4]]
    wind = [score(r) for r in [r for r in eggs if r.arms == "sinuous"][:4]]
    assert len(jag) == len(wind) == 4
    assert float(np.median(jag)) > t3.CORNER_MIN
    assert sum(j > t3.CORNER_MIN for j in jag) >= 3
    assert max(wind) < t3.CORNER_MIN


def test_the_cut_passes_near_the_deepest_point(sheets: list[StoneRecipe]) -> None:
    """Checked on the plane agatesim cuts, against a deepest point found apart from
    `body_height` (random points, not its grid)."""
    eggs = [r for r in sheets if r.outline == "thunder-egg"]
    assert {r.arms for r in eggs} >= {"sinuous", "jagged"}  # and a lens (sheet 0, stone 6)
    assert len(eggs) >= 12
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
