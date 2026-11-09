"""Band-following dust (ADR 0018 s.11): checks, each with a negative control."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from agatesim.fields import Phase
from agatesim.models.growth import TIME_STEPS
from agatesim.volume.config import SliceConfig, SolidConfig

from nodule.dust import (
    INNER,
    Dust,
    cell_uniforms,
    dust_excess,
    dust_on_bands,
    strip_cover,
)
from nodule.specimen import Specimen, SpecimenCut
from nodule.variety import StoneRecipe, VarietyLayers, build, cut_of, draw_sheet

SPACING = 0.05  # mm


def recipe(**update: object) -> StoneRecipe:
    base = StoneRecipe(
        index=0,
        seed=7,
        outline="vesicle",
        solid=SolidConfig(preset="vesicle-round", size_mm=26),
        centre="solid",
        centre_width=0.0,
        lining=0.5,
        floor=0.0,
        colour="neutral",
        band_mm=0.5,
        warp="mild",
        wobble_mm=0.1,
        wobble_scale_mm=4.0,
        white_fade=0.3,
        rim_stain=0.0,
        cut=SliceConfig(tilt_deg=90, azimuth_deg=30, offset_mm=0.0),
    )
    return base.model_copy(update=update)


LINES = Dust(lines=True, boundary_share=0.5, strength=0.7)
ALL = Dust(lines=True, specks=True, blobs=True, boundary_share=0.5, blob_per_cm2=20.0)
_BUILT: dict[str, tuple[Specimen, VarietyLayers]] = {}
_CUTS: dict[tuple[str, float, bool], SpecimenCut] = {}


def built(r: StoneRecipe) -> tuple[Specimen, VarietyLayers]:
    key = r.model_dump_json()
    if key not in _BUILT:
        _BUILT[key] = build(r)
    return _BUILT[key]


def cut_for(r: StoneRecipe, spacing: float = SPACING, dust: bool = True) -> SpecimenCut:
    key = (r.model_dump_json(), spacing, dust)
    if key not in _CUTS:
        specimen, layers = built(r)
        if not dust:
            specimen = replace(specimen, layers=replace(layers, dust=None))
        _CUTS[key] = cut_of(specimen, r, spacing)
    return _CUTS[key]


def excess_of(cut: SpecimenCut) -> tuple[np.ndarray, np.ndarray]:
    f = cut.fields
    banded = cut.inside & (f.phase == Phase.CHALCEDONY) & (f.band >= 0)
    return dust_excess(f.impurity, f.band, banded), banded


def test_band_table_gives_each_points_band() -> None:
    r = recipe(dust=LINES)
    _, layers = built(r)
    assert layers.dust is not None
    f = cut_for(r).fields
    ok = (f.phase == Phase.CHALCEDONY) & (f.band >= 0)
    k = np.floor(f.t_dep[ok] * TIME_STEPS).astype(int)
    assert (layers.dust.table.band[k] == f.band[ok]).mean() > 0.999
    assert len(layers.dust.dusty) >= 3


def test_dust_follows_band_boundaries_and_random_dust_fails() -> None:
    """Negative control 1: the same dust at random positions."""
    cut = cut_for(recipe(dust=LINES))
    excess, banded = excess_of(cut)
    found = dust_on_bands(excess, cut.fields.band, banded, SPACING)
    assert found.dusty
    assert found.share > 0.9
    shuffled = excess.copy()
    shuffled[banded] = np.random.default_rng(0).permutation(excess[banded])
    control = dust_on_bands(shuffled, cut.fields.band, banded, SPACING)
    assert not control.dusty
    assert 0.8 < control.enrichment < 1.2


def test_banded_stone_without_dust_is_not_flagged() -> None:
    """Negative control 2: band contrast alone leaves no dust."""
    for r in (recipe(), recipe(colour="iron", band_mm=0.3)):
        cut = cut_for(r)
        excess, banded = excess_of(cut)
        assert not dust_on_bands(excess, cut.fields.band, banded, SPACING).dusty


def test_no_dust_in_quartz_floor_or_white_bands() -> None:
    r = recipe(centre="quartz-filled", centre_width=0.4, floor=0.4, floor_supply=0.3, dust=ALL)
    _, layers = built(r)
    assert layers.dust is not None
    with_dust, without = cut_for(r), cut_for(r, dust=False)
    added = with_dust.fields.impurity - without.fields.impurity
    f = with_dust.fields
    assert (added > 0.05).sum() > 100  # the stone has dust
    quartz = f.phase == Phase.QUARTZ
    assert quartz.sum() > 1000
    assert np.abs(added[quartz]).max() == 0
    ts = np.asarray(f.t_smooth, dtype=np.float64)
    floor = (f.phase == Phase.CHALCEDONY) & (ts < layers.start)
    assert floor.sum() > 1000
    assert np.abs(added[floor]).max() == 0
    chal = (f.phase == Phase.CHALCEDONY) & np.isfinite(f.t_dep)
    k = np.floor(f.t_dep[chal] * TIME_STEPS).astype(int)
    white = layers.dust.table.white[k]
    assert white.any()
    assert (~white).any()
    # Only pixels straddling a boundary take a neighbour's strip: few, and fewer
    # than on coloured bands (the control).
    on_white = (added[chal][white] > 0.05).mean()
    on_coloured = (added[chal][~white] > 0.05).mean()
    assert on_white < 0.01
    assert on_white < 0.5 * on_coloured


def test_area_coverage_keeps_dust_per_area_across_resolutions() -> None:
    r = recipe(dust=LINES)
    specimen, layers = built(r)
    assert layers.dust is not None
    plane = specimen.plane(r.cut)
    totals: list[float] = []
    for h in (0.02, 0.04, 0.08):
        s = np.arange(-13, 13, h)
        ss, tt = np.meshgrid(s, s)
        points = plane.points(ss, tt)
        f, inside = specimen.evaluate(points)
        t = np.asarray(f.t_smooth, dtype=np.float64)
        line, _, _ = layers.dust.amount(points, t, inside & (f.phase == Phase.CHALCEDONY), h)
        totals.append(float(line.sum()) * h * h)
    assert max(totals) - min(totals) < 0.05 * totals[0]
    # Control: point samples at the coarse spacing miss lines thinner than a pixel.
    assert float(strip_cover(np.array([0.02]), 0.045, 0.0)[0]) == 1.0
    assert float(strip_cover(np.array([0.06]), 0.045, 0.0)[0]) == 0.0
    assert abs(float(strip_cover(np.array([0.0225]), 0.045, 0.04)[0]) - 0.045 / 0.08) < 1e-9


def test_specks_and_blobs_are_the_same_in_any_batch() -> None:
    """Resolution independence: point samples of the random forms do not depend on
    which other points share the call."""
    specimen, _ = built(recipe(dust=ALL))
    plane = specimen.plane(recipe().cut)
    s = np.arange(-12, 12, 0.1)
    ss, tt = np.meshgrid(s, s)
    points = plane.points(ss, tt)
    whole, _ = specimen.evaluate(points)
    part, _ = specimen.evaluate(points[::7])
    np.testing.assert_array_equal(whole.impurity[::7], part.impurity)


def test_blobs_only_in_inner_chalcedony() -> None:
    r = recipe(dust=Dust(blobs=True, blob_per_cm2=40.0))
    specimen, layers = built(r)
    dust = layers.dust
    assert dust is not None
    plane = specimen.plane(r.cut)
    s = np.arange(-13, 13, 0.03)
    ss, tt = np.meshgrid(s, s)
    points = plane.points(ss, tt)
    f, inside = specimen.evaluate(points)
    t = np.nan_to_num(np.asarray(f.t_smooth, dtype=np.float64), nan=np.nan)
    chal = inside & (f.phase == Phase.CHALCEDONY)
    _, _, blob = dust.amount(points, t, chal, 0.03)
    depth = (t - dust.start) / (dust.chalcedony_end - dust.start)
    assert (blob > 0.5).sum() > 50
    assert np.all(depth[blob > 0] > INNER)
    # Control: the same blobs drawn without the inner rule would reach the outer bands.
    outer = chal & (depth < INNER)
    assert outer.sum() > 1000
    assert dust._blobs(points[outer], 0.03).max() > 0.5  # pyright: ignore[reportPrivateUsage]


def test_cell_draws_are_fixed_per_cell() -> None:
    cells = np.array([[0, 0, 0], [1, 0, 0], [-3, 5, 2], [0, 0, 0]])
    u = cell_uniforms(11, cells, 3)
    np.testing.assert_array_equal(u[0], u[3])
    assert not np.allclose(u[0], u[1])
    assert not np.allclose(cell_uniforms(12, cells, 3)[0], u[0])
    big = cell_uniforms(5, np.random.default_rng(0).integers(-500, 500, (20000, 3)), 2)
    assert abs(big.mean() - 0.5) < 0.01
    assert abs(np.corrcoef(big[:, 0], big[:, 1])[0, 1]) < 0.03


def test_sheets_draw_dust_shares_and_never_every_form() -> None:
    for seed in range(1, 9):
        sheet = draw_sheet(seed)
        dusts = [r.dust for r in sheet]
        assert sum(bool(d and d.lines) for d in dusts) == 6
        assert sum(bool(d and d.specks) for d in dusts) == 3
        assert sum(bool(d and d.blobs) for d in dusts) == 3
        for r in sheet:
            d = r.dust
            if d is None:
                continue
            assert not (d.lines and d.specks and d.blobs)
            if r.outline == "pebble":
                assert not d.lines
                assert not d.specks
            if r.colour == "carnelian":
                assert d.pigment == "hematite"


def test_dust_layer_draws_fewer_dusty_boundaries_inward() -> None:
    """Control: at the wall the dusty share is the drawn share (1 here)."""
    outer = inner = outer_n = inner_n = 0
    for seed in range(6):
        _, vl = built(recipe(seed=seed, dust=Dust(lines=True, boundary_share=1.0)))
        d = vl.dust
        assert d is not None
        e = d.table.boundaries()
        e = e[~d.table.white[e - 1]]
        mid = np.median(e)
        outer += int(np.isin(e[e < mid], d.dusty).sum())
        inner += int(np.isin(e[e >= mid], d.dusty).sum())
        outer_n += int((e < mid).sum())
        inner_n += int((e >= mid).sum())
    assert inner / inner_n < outer / outer_n
    assert outer / outer_n > 0.75


def test_a_dusty_stone_shows_some_dust() -> None:
    """Even when the draw picks no boundary; control: blobs alone keep none."""
    _, vl = built(recipe(dust=Dust(lines=True, boundary_share=0.0)))
    assert vl.dust is not None
    assert len(vl.dust.dusty) == 1
    _, vl = built(recipe(dust=Dust(blobs=True, boundary_share=0.0)))
    assert vl.dust is not None
    assert len(vl.dust.dusty) == 0
