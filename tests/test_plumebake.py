"""Plumes in the renderer (ADR 0018 s.13, step 4): left out of the band record and baked
as absorption, their halos as lost colour. Each check has its negative control."""

from __future__ import annotations

from dataclasses import replace
from functools import cache

import numpy as np
import pytest
from agatesim.fields import Fields, Phase, Pigment
from agatesim.volume.config import SliceConfig, SolidConfig

from nodule import plumes
from nodule._scipy import distance_inside
from nodule.plumes import Plumes
from nodule.render import tgrid
from nodule.render.fakes import stone_record
from nodule.render.layerbake import (
    GREEN_STRENGTH,
    PLUME_MUA,
    Baker,
    plume_absorption,
    plume_stops,
)
from nodule.render.materials import OpticsTable
from nodule.render.record import BandRecord, plume_points
from nodule.render.whiteopal import RIM_MM
from nodule.sheet import WHITE_FAINT_RGB, WHITE_SOLID_RGB, preview_rgb
from nodule.specimen import Specimen
from nodule.variety import StoneRecipe, build, cut_of

N = 200_000  # sample points through the stone


def recipe(**update: object) -> StoneRecipe:
    base = StoneRecipe(
        index=0,
        seed=11,
        outline="prolate",
        solid=SolidConfig(preset="vesicle-round", size_mm=20),
        centre="solid",
        centre_width=0.0,
        lining=0.5,
        floor=0.0,
        colour="iron",
        band_mm=0.5,
        warp="mild",
        wobble_mm=0.1,
        wobble_scale_mm=4.0,
        white_fade=0.3,
        rim_stain=0.0,
        cut=SliceConfig(tilt_deg=90, azimuth_deg=0, offset_mm=0.0),
    )
    return base.model_copy(update=update)


def plume(**update: object) -> Plumes:
    base = Plumes(
        look="dense",
        colour="red",
        tint=0.5,
        tint_change=0.0,
        tint_spread=0.0,
        tilt_deg=0.0,
        azimuth_deg=0.0,
        halo_mm=0.8,
        halo_share=1.0,
        wrap_rate=0.2,
    )
    return base.model_copy(update=update)


@cache
def stone(kind: str) -> Specimen:
    """'plumes', 'bare' (control: no plumes) or 'no-halo' (control: plumes, no halo);
    plumes grown on a small lattice so the test stays quick."""
    p = {"plumes": plume(), "bare": None, "no-halo": plume(halo_mm=0.0, halo_share=1.0)}[kind]
    mp = pytest.MonkeyPatch()
    mp.setattr(plumes, "CELLS_M", 0.5)
    try:
        specimen, _ = build(recipe(plumes=p))
    finally:
        mp.undo()
    return specimen


def points(specimen: Specimen, n: int = N) -> np.ndarray:
    lo, hi = specimen.volume.solid.bounds()
    return np.random.default_rng(0).uniform(lo, hi, (n, 3))


def gains(specimen: Specimen) -> tuple[np.ndarray, Fields, np.ndarray]:
    p = points(specimen)
    raw, inside = specimen.restored_fields(p)
    baker = Baker.of(specimen, OpticsTable(), 0.2, 1)
    return baker.gain(p, raw, inside, 0.0), raw, inside


def test_the_record_takes_a_plume_stone_and_leaves_the_plumes_out() -> None:
    """A plume stone's record builds (before: refused), and no step holds an
    inclusion. Control: the plume points are there to leave out."""
    sp = stone("plumes")
    f, inside = sp.step_fields(points(sp))
    assert plume_points(f, inside).sum() > 50
    rec = stone_record(sp, np.random.default_rng(0), N)
    assert not (rec.phase[rec.seen] == Phase.INCLUSION).any()
    assert rec.seen.sum() > 50


def test_plumes_gain_their_absorption_and_a_bare_stone_none() -> None:
    sp = stone("plumes")
    g, raw, inside = gains(sp)
    at = plume_points(sp.placed(points(sp), raw, inside), inside)
    assert g[at].max() == pytest.approx(PLUME_MUA, rel=1e-6)
    assert (g[at].max(1) > 0.5 * PLUME_MUA).mean() > 0.95
    # red plume: absorbs blue (450) well above red (650)
    assert np.median(g[at][:, 0]) > 5 * np.median(g[at][:, 2])
    gb, _, _ = gains(stone("bare"))
    assert gb.max() < 1.0  # control: no plumes, no plume gain (nor mottle or dust here)


def test_halos_lose_band_colour_and_no_halo_loses_none() -> None:
    g, _, _ = gains(stone("plumes"))
    assert (g.min(1) < -0.05).sum() > 50
    gn, _, _ = gains(stone("no-halo"))
    assert gn.min() > -1e-6  # control


def test_the_grid_bakes_the_plumes() -> None:
    """End to end on a cut's grid: the fill accepts the plume stone and bakes it.
    Control: the bare stone's fill bakes no plume."""
    out: list[int] = []
    for kind in ("plumes", "bare"):
        sp = stone(kind)
        grid = tgrid.GrowthGrid.fill(sp, sp.plane(recipe().cut), 0.2, 0.4, optics=OpticsTable())
        assert grid.absorb is not None
        assert grid.bands is not None
        out.append(int((grid.absorb.max(0) > 0.5 * PLUME_MUA).sum()))
    assert out[0] > 5  # this cut crosses a few branches (13 nodes at seed 11)
    assert out[1] == 0


def test_orange_gold_passes_red_better_than_green_and_its_negative_control() -> None:
    optics = OpticsTable()
    gold = plume_stops(optics)[1]  # the 0.3 stop; channels blue, green, red
    assert gold[0] == pytest.approx(PLUME_MUA)  # blue absorbed most
    assert gold[2] < 0.65 * gold[1]  # red passes well below green: orange, not olive
    # Control: goethite alone absorbs red nearly as much as green (it rendered olive).
    goethite = np.asarray(optics.pigment_mua(int(Pigment.GOETHITE), 1.0))
    assert goethite[2] > 0.65 * goethite[1]


def test_the_ramp_runs_green_red_black_and_skips_silica() -> None:
    stops = plume_stops(OpticsTable())
    n = 4
    z = np.zeros(n, np.float32)
    f = Fields(
        phase=np.array([Phase.INCLUSION] * 3 + [Phase.CHALCEDONY], np.uint8),
        band=np.full(n, -1, np.int32),
        impurity=np.array([0.0, 0.5, 1.0, 0.9], np.float32),
        translucency=z,
        t_dep=np.full(n, 0.01, np.float32),
        grain=np.full(n, -1, np.int32),
        micro_spacing_um=z,
    )
    a = plume_absorption(stops, f, np.ones(n, bool))
    green, red, black, silica = a
    assert green[1] < min(green[0], green[2])  # passes green
    assert green.max() == pytest.approx(GREEN_STRENGTH * PLUME_MUA)  # the owner's nudge
    assert red[0] > 5 * red[2]  # passes red
    assert np.allclose(black, black[0])  # neutral
    assert black[0] == pytest.approx(PLUME_MUA)  # and dark
    assert not silica.any()  # control: chalcedony takes no plume absorption
    tube = plume_absorption(stops, replace(f, t_dep=z * np.nan), np.ones(n, bool))
    assert not tube.any()  # control: tubes (no growth time) are not plumes


WHITE = OpticsTable(white="plumes", white_transport=8.0)
RIMS = OpticsTable(white="rims", white_transport=16.0)


def test_white_plumes_gain_nothing_where_the_white_row_carries_them() -> None:
    """White plumes (ADR 0019 s.13): a plume point gains nothing, its step's own pigment
    included (the white row carries it), while halos still lose their colour. Control:
    the plain bake gives the same points their pigment."""
    sp = stone("plumes")
    p = points(sp)
    raw, inside = sp.restored_fields(p)
    at = plume_points(sp.placed(p, raw, inside), inside)
    white = Baker.of(sp, WHITE, 0.2, 1).gain(p, raw, inside, 0.0)
    assert at.sum() > 50
    assert not white[at].any()
    assert (white.min(1) < -0.05).sum() > 50  # halos kept
    plain, _, _ = gains(sp)
    assert (plain[at].max(1) > 0.5 * PLUME_MUA).mean() > 0.95  # control
    assert np.array_equal(white[~at], plain[~at])


def test_the_fill_keeps_the_cover_for_white_opal() -> None:
    """With white opal in the optics the fill keeps each node's share of sub-points in a
    plume: about 1 inside a branch, between 0 and 1 at its edge. White plumes drop the
    pigment there; rims keep the plain bake bit for bit (the layers read once, passed
    on). Controls: the plain fill keeps no cover; the bare stone's cover is 0."""
    sp = stone("plumes")
    plane = sp.plane(recipe().cut)
    plain = tgrid.GrowthGrid.fill(sp, plane, 0.2, 0.4, optics=OpticsTable())
    white = tgrid.GrowthGrid.fill(sp, plane, 0.2, 0.4, optics=WHITE)
    rims = tgrid.GrowthGrid.fill(sp, plane, 0.2, 0.4, optics=RIMS)
    assert plain.cover is None  # control
    assert white.cover is not None
    assert rims.cover is not None
    assert plain.absorb is not None
    assert white.absorb is not None
    assert rims.absorb is not None
    pigment = plain.absorb.max(0) > 0.5 * PLUME_MUA
    assert pigment.sum() > 5
    assert (white.cover[pigment] >= 0.5).all()
    edge = (white.cover > 0) & (white.cover < 1)
    assert edge.sum() > pigment.sum() / 2  # edges and thin fronds keep a part
    assert white.absorb.max(0)[pigment].max() < 0.1 * PLUME_MUA  # no pigment
    assert np.array_equal(rims.absorb, plain.absorb)
    assert np.array_equal(rims.cover, white.cover)
    bare = tgrid.GrowthGrid.fill(stone("bare"), plane, 0.2, 0.4, optics=WHITE)
    assert bare.cover is not None
    assert not bare.cover.any()  # control


def test_previews_draw_white_plumes_and_white_rims() -> None:
    """The sheet preview (ADR 0019 s.13): white plumes take the faint-to-solid white of
    their level; white rims a solid white ring within RIM_MM of the strands, on the stone.
    Control: without white the strands keep their tint and nothing is white."""
    sp = stone("plumes")
    base = recipe(plumes=plume(), cut=SliceConfig(tilt_deg=90, azimuth_deg=0, offset_mm=-1.0))
    cut = cut_of(sp, base, 0.1)
    strands = plume_points(cut.fields, cut.inside)[::-1]  # preview rows run top down
    assert strands.sum() > 100  # 178 pixels at seed 11
    plain = preview_rgb(cut, base)
    solid = np.rint(WHITE_SOLID_RGB).astype(np.uint8)
    faint = np.rint(WHITE_FAINT_RGB).astype(np.uint8)

    def white(kind: str, level: float) -> np.ndarray:
        return preview_rgb(cut, base.model_copy(update={"white": kind, "white_level": level}))

    assert (white("plumes", 1.0)[strands] == solid).all()
    assert (white("plumes", 0.0)[strands] == faint).all()
    rims = white("rims", 1.0)
    ring = (rims == solid).all(-1) & ~(plain == solid).all(-1)
    assert ring.sum() > strands.sum() / 2
    assert not ring[strands].any()
    near = distance_inside(~strands) * 0.1 <= RIM_MM + 1e-9
    assert not ring[~near].any()
    assert not (plain[strands] == solid).all(-1).any()  # control


def test_per_step_lifts_halos_only() -> None:
    """A halo's place values go back to its step's; fields without halos pass as they
    are (control)."""
    sp = stone("plumes")
    f, inside = sp.restored_fields(points(sp))
    assert f.band_impurity is not None
    halo = (f.impurity != f.band_impurity) & inside
    assert halo.sum() > 50
    step = sp.per_step(f)
    assert step.band_impurity is None
    assert not (step.impurity[halo] < f.impurity[halo]).any()
    plain, _ = stone("bare").restored_fields(points(sp, 1000))
    assert plain.band_impurity is None
    assert BandRecord.of(sp.per_step(f), inside).seen.any()
