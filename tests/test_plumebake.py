"""Plumes in the renderer (ADR 0018 s.13, step 4): left out of the band record and baked
as absorption, their halos as lost colour. Each check has its negative control."""

from __future__ import annotations

from dataclasses import replace
from functools import cache

import numpy as np
import pytest
from agatesim.fields import Fields, Phase
from agatesim.volume.config import SliceConfig, SolidConfig

from nodule import plumes
from nodule.plumes import Plumes
from nodule.render import tgrid
from nodule.render.fakes import stone_record
from nodule.render.layerbake import PLUME_MUA, Baker, plume_absorption, plume_stops
from nodule.render.materials import OpticsTable
from nodule.render.record import BandRecord, plume_points
from nodule.specimen import Specimen
from nodule.variety import StoneRecipe, build

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
        out.append(int((grid.absorb.max(0) > 5).sum()))
    assert out[0] > 5  # this cut crosses a few branches (13 nodes at seed 11)
    assert out[1] == 0


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
    assert red[0] > 5 * red[2]  # passes red
    assert np.allclose(black, black[0])  # neutral
    assert black[0] == pytest.approx(PLUME_MUA)  # and dark
    assert not silica.any()  # control: chalcedony takes no plume absorption
    tube = plume_absorption(stops, replace(f, t_dep=z * np.nan), np.ones(n, bool))
    assert not tube.any()  # control: tubes (no growth time) are not plumes


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
