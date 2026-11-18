"""Mn dendrites in the recipes (ADR 0018 s.17): the owner's rate and carriers, the
hairline crack carrier, growth that does not depend on which points ask first, the
largest-trees thinning, and the layer, which must draw Mn near a dendrite and nowhere
else (negative control: a stone without dendrites, and points far away, unchanged)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from agatesim.fields import Phase, Pigment
from agatesim.volume.dendrites import Aggregate3, Lattice3
from agatesim.volume.slicing import Plane
from numpy.typing import NDArray
from pydantic import ValidationError

from nodule.dendrites import (
    DENDRITE_CHANCE,
    WALL_SHARE,
    Dendrites,
    box_dimension,
    dendritic,
    draw_dendrites,
)
from nodule.specimen import Specimen
from nodule.variety import StoneRecipe, build, draw_sheet
from nodule.variety.dendrites import (
    CAVITY_DEPTH,
    SITE_TOL_MM,
    CrackGrowth,
    DendriteLayer,
    WallGrowth,
    largest_trees,
    meets_wall_on_cut,
)
from nodule.variety.draw import CRACK_CUT_ANGLE, HAIRLINE_MM, crack_cut_angle, is_hairline
from nodule.variety.layers import VarietyLayers, chalcedony_depth
from nodule.variety.recipe import rim_point

SMALL = WallGrowth(reach_mm=2.0, h_mm=0.08)  # a quick wall growth for tests


def test_dendrites_on_few_stones_of_any_kind(sheets: list[StoneRecipe]) -> None:
    carrying = [r for r in sheets if r.dendrites is not None]
    assert [(r.outline, r.dendrites.carrier) for r in carrying if r.dendrites] == [
        ("vesicle", "wall"),
        ("vesicle", "wall"),
        ("thunder-egg", "crack"),
        ("vesicle", "crack"),
        ("irregular", "crack"),
    ]  # 5 of 144, under the owner's 5 %


def _cut_gap(r: StoneRecipe, direction: NDArray[np.float64]) -> float:
    """How far (mm) the rim point in a direction from the centre lies off the cut."""
    solid = r.solid.build()
    plane = Plane.of(r.cut, solid.centre())
    p = np.asarray(solid.centre()) + rim_point(solid, direction)
    return abs(float((p - plane.origin) @ np.asarray(plane.normal)))


def test_dendrites_are_steered_to_the_cut(sheets: list[StoneRecipe]) -> None:
    """The owner (2026-11-18): every wall site lies on the cut's outline, and every
    carrier crack is crossed by the cut where it meets the wall. Negative control:
    random wall directions on the same stones mostly miss the cut, and a crack moved
    off the cut fails the crack check."""
    carrying = [r for r in sheets if r.dendrites is not None]
    rng = np.random.default_rng(3)
    for r in carrying:
        assert r.dendrites is not None
        solid = r.solid.build()
        plane = Plane.of(r.cut, solid.centre())
        for s in r.dendrites.sites:
            assert _cut_gap(r, np.asarray(s)) < SITE_TOL_MM
        for i in r.dendrites.cracks:
            f = r.fractures[i]
            assert meets_wall_on_cut(solid, f, plane)
            m = np.asarray(plane.normal)
            moved = f.model_copy(update={"centre_mm": tuple(np.asarray(f.centre_mm) + 5.0 * m)})
            assert not meets_wall_on_cut(solid, moved, plane)
        v = rng.standard_normal((50, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        assert np.mean([_cut_gap(r, x) < SITE_TOL_MM for x in v]) < 0.1


def test_the_draw_rate_and_carrier_split() -> None:
    draws = [draw_dendrites(np.random.default_rng(s)) for s in range(4000)]
    got = [d for d in draws if d is not None]
    assert abs(len(got) / len(draws) - DENDRITE_CHANCE) < 0.01
    walls = sum(d.carrier == "wall" for d in got) / len(got)
    assert abs(walls - WALL_SHARE) < 0.12
    assert all(1 <= len(d.sites) <= 3 for d in got)


def test_the_hairline_carrier_is_healed_and_at_a_clean_angle(sheets: list[StoneRecipe]) -> None:
    hairs = [(r, r.fractures[-1]) for r in sheets if r.fractures and is_hairline(r.fractures[-1])]
    assert len(hairs) == 3
    for r, f in hairs:
        assert r.dendrites is not None
        assert r.dendrites.cracks == (len(r.fractures) - 1,)
        assert f.aperture_mm == HAIRLINE_MM
        assert f.halo_mm == 0
        assert crack_cut_angle(r.solid.build(), r.cut, (f,)) >= CRACK_CUT_ANGLE
    # negative control: no other crack on these sheets looks like a hairline
    assert sum(is_hairline(f) for r in sheets for f in r.fractures) == 3


def test_the_recipe_checks_the_carrier() -> None:
    with pytest.raises(ValidationError):
        Dendrites(carrier="crack", key=1)
    with pytest.raises(ValidationError):
        Dendrites(carrier="wall", key=1, sites=((1.0, 0.0, 0.0),), cracks=(0,))
    rec = next(r for r in draw_sheet(0) if not r.fractures)
    bad = {"dendrites": {"carrier": "crack", "key": 1, "cracks": [0]}}
    with pytest.raises(ValidationError):
        StoneRecipe.model_validate(rec.model_dump() | bad)


def test_box_dimension_controls() -> None:
    n = 512
    line = np.zeros((n, n), bool)
    line[n // 2, 20:490] = True
    yy, xx = np.mgrid[:n, :n]
    disc = (yy - n / 2) ** 2 + (xx - n / 2) ** 2 < (0.45 * n) ** 2
    assert box_dimension(line) == pytest.approx(1.0, abs=0.05)
    assert box_dimension(disc) == pytest.approx(2.0, abs=0.1)  # 1.92: edge bias at big boxes
    assert np.isnan(box_dimension(np.zeros((n, n), bool)))


def test_crack_dendrites_read_as_dendrites_face_on(sheets: list[StoneRecipe]) -> None:
    """Face-on (on the crack's own bowl, 0.02 mm pixels) the 9-1 crack dendrite has the
    published face-on box dimension (measured 1.70) with its pixels in branches (0.98).
    Negative controls: a line, a filled disc and random specks (box dimension 1.66 at
    5 % cover, but no branches) all fail."""
    rec = sheets[12 * 9 + 1]
    _, layers = build(rec)  # about 7 s with the growth
    assert layers.dendrites is not None
    g = layers.dendrites.grown(0)
    x = np.arange(-15.0, 15.0, 0.02)
    ss, tt = np.meshgrid(x, x)
    q = np.stack([ss.ravel(), tt.ravel(), np.zeros(ss.size)], 1)
    assert dendritic((g.agg.distance(q) <= 0).reshape(ss.shape))
    # Shown thicker across the crack only (the owner's "darker", 2026-11-18): within
    # its plane the distance is the grown film's, so the face-on pattern stands.
    ins = q[g.agg.distance(q) <= 0]
    np.testing.assert_allclose(g.distance(g.frame.to_world(ins)), g.agg.distance(ins), atol=1e-6)
    # Moved 0.08 mm across, no point lies in the grown film (two jittered cell layers)
    # but some in the shown one (0.2 mm; 28 % measured); 0.12 mm across, in neither.
    for dz, grown, shown in ((0.08, 0.0, 0.15), (0.12, 0.0, 0.0)):
        for sign in (1, -1):
            off = ins + np.array([0.0, 0.0, sign * dz])
            assert np.mean(g.agg.distance(off) <= 0) == grown
            share = np.mean(g.distance(g.frame.to_world(off)) <= 0)
            assert share > shown if shown else share == 0
    n = 1000
    line = np.zeros((n, n), bool)
    line[n // 2, 50:950] = True
    yy, xx = np.mgrid[:n, :n]
    disc = (yy - n / 2) ** 2 + (xx - n / 2) ** 2 < (0.45 * n) ** 2
    rng = np.random.default_rng(0)
    for cover in (0.01, 0.05, 0.1):
        assert not dendritic(rng.uniform(size=(n, n)) < cover)
    assert not dendritic(line)
    assert not dendritic(disc)


def test_largest_trees() -> None:
    lat = Lattice3.around(np.zeros(3), np.ones(3), 0.1)
    plume = np.repeat(np.arange(5), [9, 1, 30, 6, 12])  # tree 1 never grew (< 5 cells)
    agg = Aggregate3(lat, np.arange(len(plume)), np.zeros(len(plume)), plume, None)
    assert set(largest_trees(agg, 0.25).plume.tolist()) == {2}  # type: ignore[union-attr]
    assert set(largest_trees(agg, 0.5).plume.tolist()) == {2, 4}  # type: ignore[union-attr]
    assert len(largest_trees(agg, 1.0).cells) == len(plume) - 1


Stone = tuple[StoneRecipe, Dendrites, Specimen, VarietyLayers]


@pytest.fixture(scope="module")
def wall_stone(sheets: list[StoneRecipe]) -> Stone:
    rec = sheets[2]  # 0-2, a vesicle with one wall site
    assert rec.dendrites is not None
    assert rec.dendrites.carrier == "wall"
    sp, layers = build(rec)
    return rec, rec.dendrites, sp, layers


def test_growth_does_not_depend_on_who_asks_first(wall_stone: Stone) -> None:
    rec, den, sp, layers = wall_stone
    solid = sp.volume.solid
    a = DendriteLayer.of(den, rec.fractures, solid, chalcedony_depth(sp, layers), SMALL)
    b = DendriteLayer.of(den, rec.fractures, solid, chalcedony_depth(sp, layers), SMALL)
    site = a.sites[0]
    rng = np.random.default_rng(5)
    near = rng.uniform(site.lo, site.hi, (20_000, 3))
    far = near + 100.0
    assert np.isinf(b.distance(far)).all()
    assert not b.grown_sites()  # far points grow nothing
    da = a.distance(near)
    db = b.distance(near[::-1])[::-1]
    np.testing.assert_array_equal(da, db)
    assert np.any(da <= 0)  # negative control: something grew there
    # resolution: a point sample is covered exactly where the distance is <= 0
    np.testing.assert_array_equal(a.coverage(near, 0.0), (da <= 0).astype(float))


def test_wall_dendrites_stay_in_the_chalcedony(wall_stone: Stone) -> None:
    rec, den, sp, layers = wall_stone
    lay = DendriteLayer.of(den, rec.fractures, sp.volume.solid,
                           chalcedony_depth(sp, layers), SMALL)  # fmt: skip
    g = lay.grown(0)
    pts = g.frame.to_world(g.agg.centres())
    assert len(pts) > 0
    inside = sp.volume.solid.contains(pts)
    assert inside.mean() > 0.99  # balls jitter by 0.2 cell
    depth = chalcedony_depth(sp, layers)(pts[inside])  # outside reads as depth 1
    assert np.all(depth < CAVITY_DEPTH + 0.02)


def test_crack_dendrites_lie_on_their_crack() -> None:
    rec = draw_sheet(9)[1]  # an irregular stone with a hairline carrier
    assert rec.dendrites is not None
    assert rec.dendrites.carrier == "crack"
    sp, layers = build(rec)
    crack = rec.fractures[rec.dendrites.cracks[0]]
    g = CrackGrowth(h_mm=0.08)
    lay = DendriteLayer.of(rec.dendrites, rec.fractures, sp.volume.solid,
                           chalcedony_depth(sp, layers), crack=g)  # fmt: skip
    grown = lay.grown(0)
    pts = grown.frame.to_world(grown.agg.centres())
    assert len(pts) > 20
    centre = np.asarray(sp.volume.solid.centre(), float)
    assert np.all(crack.distance(pts, centre) < g.h_mm)


def test_dendrites_on_a_cracked_stones_own_cracks(sheets: list[StoneRecipe]) -> None:
    """Every crack carrier on sheets 0-99 is a hairline (none lands on a cracked stone:
    it needs both rare draws), so this one is forced: stone 1-9's bent 8.4 mm crack
    (sagitta 0.47) and its 3.4 mm one, two carriers at once. Cells stay on their own
    crack (0.035 mm measured)."""
    rec = sheets[12 + 9]
    assert rec.dendrites is not None
    den = rec.dendrites.model_copy(update={"carrier": "crack", "sites": (), "cracks": (3, 7)})
    rec = StoneRecipe.model_validate(rec.model_dump() | {"dendrites": den.model_dump()})
    sp, layers = build(rec)
    assert layers.dendrites is not None
    centre = np.asarray(sp.volume.solid.centre(), float)
    for j, k in enumerate((3, 7)):
        g = layers.dendrites.grown(j)
        pts = g.frame.to_world(g.agg.centres())
        assert len(pts) > 1000
        assert np.all(rec.fractures[k].distance(pts, centre) < 0.05)
    assert rec.fractures[3].sagitta_mm > 0.4  # the bent one


def test_the_layer_draws_mn_only_near_a_dendrite(wall_stone: Stone) -> None:
    rec, den, sp, layers = wall_stone
    lay = DendriteLayer.of(den, rec.fractures, sp.volume.solid,
                           chalcedony_depth(sp, layers), SMALL)  # fmt: skip
    g = lay.grown(0)
    on = g.frame.to_world(g.agg.centres()[:400])
    inward: NDArray[np.float64] = np.asarray(sp.volume.solid.centre(), np.float64) - on[0]
    away = on + 30.0 * inward / float(np.linalg.norm(inward))
    pts = np.concatenate([on, away])
    with_mn = replace(sp, layers=replace(layers, dendrites=lay))
    plain = replace(sp, layers=replace(layers, dendrites=None))
    f, ins = with_mn.evaluate(pts)
    p, _ = plain.evaluate(pts)
    chal = ins & (f.phase == Phase.CHALCEDONY)
    assert f.pigment is not None
    assert p.pigment is not None
    mn = f.pigment == Pigment.MANGANESE
    assert mn[: len(on)][chal[: len(on)]].mean() > 0.9  # negative control: it draws
    np.testing.assert_array_equal(f.impurity[len(on) :], p.impurity[len(on) :])
    assert not np.any(mn[len(on) :] & (p.pigment[len(on) :] != Pigment.MANGANESE))


def test_no_dendrites_no_layer(sheets: list[StoneRecipe]) -> None:
    rec = next(r for r in sheets if r.dendrites is None and r.outline == "pebble")
    _, layers = build(rec)
    assert layers.dendrites is None
