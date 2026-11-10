"""Open cavities (ADR 0019 A1): the void renders as its lining's material plus a stand-in
absorption, dark under the dome and passing the back light. Each check has a negative
control."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from agatesim import _scipy as agatesim_scipy  # pyright: ignore[reportPrivateUsage]
from agatesim.fields import Phase
from agatesim.specimen.quality import luminance

from nodule.render import _gpu
from nodule.render.cutface import VOID_MUA, CutRenderer, Rendering
from nodule.render.fakes import stone_record
from nodule.render.record import BandRecord, void_points
from nodule.render.scene import with_void
from nodule.render.tgrid import GrowthGrid, fill_nan, fill_void
from nodule.sheet import window_cut
from nodule.specimen import Specimen, SpecimenCut
from nodule.variety import build, cut_of, draw_sheet

gpu = pytest.mark.skipif(not _gpu.available(), reason="needs a CUDA GPU and CuPy")
GRID_MM = 0.25
WINDOW_MM = 14.0


# --- the grid's void nodes -------------------------------------------------------------------


def void_block() -> tuple[np.ndarray, np.ndarray]:
    """A grid with t rising along the last axis, a void block 16 nodes wide in its middle
    and an unfilled border (outside the stone)."""
    g = np.tile(np.linspace(0.0, 1.0, 24, dtype=np.float32), (24, 24, 1))
    void = np.zeros(g.shape, bool)
    void[4:20, 4:20, 4:20] = True
    g[void] = np.nan
    g[:, :, 0] = np.nan  # outside
    return g, void


def test_fill_void_gives_each_void_node_its_nearest_lining_and_its_control() -> None:
    """Each void node takes its nearest filled node's t, so its centre takes the lining's
    and stays within the lining's range. Control: the edge fill alone (`fill_nan`, three
    layers deep) leaves the void's middle at 0, the stone's first band."""
    g, void = void_block()
    out = fill_void(g, void)
    lining = g[4:20, 4:20, [3, 20]]
    assert not np.isnan(out[void]).any()
    assert lining.min() <= out[void].min()
    assert out[void].max() <= lining.max()
    assert np.isnan(out[:, :, 0]).all()  # outside the stone: left for fill_nan
    assert out[12, 12, 5] == g[12, 12, 3]
    assert out[12, 12, 18] == g[12, 12, 20]
    plain = fill_nan(g)
    assert plain[12, 12, 11] == 0.0
    assert fill_nan(out)[12, 12, 11] > 0.1


def test_void_absorption_lands_on_void_nodes_only_and_its_control() -> None:
    g, void = void_block()
    grid = GrowthGrid(plane=None, s=np.arange(24.0), t=np.arange(24.0), w=np.arange(24.0),  # type: ignore[arg-type]
                      values=g, void=void)  # fmt: skip
    layers = np.full((3, *g.shape), 0.5, np.float32)
    out = with_void(layers, grid, 10.0)
    assert out is not None
    assert np.allclose(out[:, void], 10.5)
    assert np.allclose(out[:, ~void], 0.5)
    assert np.all(layers == 0.5)  # a new array
    assert with_void(None, grid, 10.0) is not None
    # Control: no void stand-in, or a grid without a void, adds nothing
    assert with_void(layers, grid, 0.0) is layers
    no_void = GrowthGrid(plane=None, s=grid.s, t=grid.t, w=grid.w, values=g)  # type: ignore[arg-type]
    assert with_void(layers, no_void, 10.0) is layers
    with pytest.raises(ValueError, match="void_mua"):
        with_void(layers, grid, -1.0)


def test_the_back_light_passes_the_void_and_the_dome_darkens_it() -> None:
    rec = BandRecord.empty()
    assert Rendering.look(rec).void_mua == VOID_MUA > 0
    assert Rendering.reflected(rec).void_mua == VOID_MUA
    assert Rendering.transmitted(rec).void_mua == 0.0
    with pytest.raises(ValueError, match="void_mua"):
        Rendering.look(rec, void_mua=-1.0)


# --- a variety stone with an open cavity (sheet 1, stone 2) ----------------------------------


@cache
def cavity_stone() -> tuple[Specimen, SpecimenCut, BandRecord]:
    """Sheet 1's stone 2 (quartz-lined, its cavity open on the cut), a WINDOW_MM window
    on the cavity at GRID_MM pixels. Without the plumes it has drawn since ADR 0018 s.13:
    they fill its centre."""
    r = draw_sheet(1)[1].model_copy(update={"plumes": None})
    assert r.centre == "quartz-lined"
    specimen, _ = build(r)
    whole = cut_of(specimen, r, 0.5)
    v = void_points(whole.fields, whole.inside)
    jj, ii = np.nonzero(v)
    centre = (float(whole.s[ii].mean()), float(whole.t[jj].mean()))
    cut = window_cut(specimen, whole, centre, WINDOW_MM, GRID_MM)
    return specimen, cut, stone_record(specimen, np.random.default_rng(r.seed))


def rings(cut: SpecimenCut) -> tuple[np.ndarray, np.ndarray]:
    """The cut's void pixels away from its wall, and a ring of stone 1-3 mm outside it."""
    v = void_points(cut.fields, cut.inside)
    core = agatesim_scipy.erode(v, 2)
    near = agatesim_scipy.dilate(v, round(3 / GRID_MM)) & ~agatesim_scipy.dilate(
        v, round(1 / GRID_MM)
    )
    return core, near & cut.inside


def test_a_cavity_stone_fills_with_its_lining_in_the_void() -> None:
    """The fill marks the void's nodes and gives them growth steps the record has seen,
    mostly the quartz lining's. Control: the stone's cut does hold a void (else the
    check would pass on any stone)."""
    specimen, cut, record = cavity_stone()
    assert void_points(cut.fields, cut.inside).sum() > 100
    grid = GrowthGrid.fill(specimen, cut.plane, GRID_MM, 1.0, ((cut.s[0], cut.s[-1]),
                                                               (cut.t[0], cut.t[-1])))  # fmt: skip
    assert grid.void is not None
    assert grid.void.sum() > 100
    t = grid.filled()[grid.void]
    k = np.clip((t * len(record.seen)).astype(int), 0, len(record.seen) - 1)
    assert record.seen[k].mean() > 0.9
    share = float((grid.bands.phase[k] == Phase.QUARTZ).mean())  # type: ignore[union-attr]
    print(f"void nodes {grid.void.sum()}, quartz share {share:.2f}")
    assert share > 0.5


@gpu
def test_the_look_shows_the_cavity_dark_and_the_slice_shows_a_hole() -> None:
    """Under the dome the cavity is far darker than the stone around it; under the back
    light it is at least as bright. Control: the look without the stand-in absorption
    (void_mua 0) shows the cavity as lit as its lining's clear quartz, not dark."""
    specimen, cut, record = cavity_stone()
    core, near = rings(cut)
    look = CutRenderer(specimen, cut, Rendering.look(record, grid_mm=GRID_MM),
                       np.random.default_rng(0), fill_depth_mm=3.0)  # fmt: skip
    back = look.on(Rendering.transmitted(record, grid_mm=GRID_MM, depth_mm=3.0))
    bare = CutRenderer(specimen, cut, Rendering.look(record, grid_mm=GRID_MM, void_mua=0.0),
                       np.random.default_rng(0), fill_depth_mm=3.0)  # fmt: skip
    out: dict[str, tuple[float, float]] = {}
    for name, cr in (("look", look), ("back", back), ("bare", bare)):
        cr.more(32)
        y = luminance(cr.rgb) / 255.0
        out[name] = (float(y[core].mean()), float(y[near].mean()))
    print({k: (round(a, 3), round(b, 3)) for k, (a, b) in out.items()})
    assert out["look"][0] < 0.25 * out["look"][1]
    assert out["back"][0] >= out["back"][1]
    assert out["bare"][0] > 0.5 * out["bare"][1]


@gpu
def test_a_backlit_slice_on_the_looks_fill_equals_its_own_with_a_void() -> None:
    """The second view swaps only the void's absorption: equal, sample for sample, to a
    slice on its own fill. Control: another seed differs."""
    specimen, cut, record = cavity_stone()
    back = Rendering.transmitted(record, grid_mm=GRID_MM, depth_mm=3.0)
    via = CutRenderer(specimen, cut, Rendering.look(record, grid_mm=GRID_MM),
                      np.random.default_rng(5), fill_depth_mm=3.0).on(back)  # fmt: skip
    own = CutRenderer(specimen, cut, back, np.random.default_rng(5))
    other = CutRenderer(specimen, cut, back, np.random.default_rng(6))
    for cr in (via, own, other):
        cr.more(16)
    assert np.array_equal(via.rgb, own.rgb)
    assert not np.array_equal(via.rgb, other.rgb)
