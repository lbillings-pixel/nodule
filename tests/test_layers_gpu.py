"""Variety stones on the GPU (ADR 0019): any outline through the baked level, the
layers baked as absorption, and band roughness read through the grid. Each check has
a negative control."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from agatesim.fields import Phase
from agatesim.volume.config import SliceConfig, SolidConfig

from nodule.dust import Dust
from nodule.render import _gpu, scene, tgrid
from nodule.render.camera import Light, render
from nodule.render.cutface import Rendering, view_of
from nodule.render.fakes import stone_record
from nodule.render.layerbake import absorption, absorption_table, bake_absorption
from nodule.render.materials import OpticsTable
from nodule.variety import Mottle, Roughness, StoneRecipe, build, cut_of

pytestmark = pytest.mark.skipif(not _gpu.available(), reason="needs a CUDA GPU and CuPy")

H, DEPTH = 0.1, 1.0


def recipe(**update: object) -> StoneRecipe:
    base = StoneRecipe(
        index=0,
        seed=7,
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
        cut=SliceConfig(tilt_deg=80, azimuth_deg=30, offset_mm=0.5),
    )
    return base.model_copy(update=update)


def classes(dev: scene.DeviceScene, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cp = _gpu.cupy()
    n = len(pts)
    cls, tt = cp.empty(n, cp.int32), cp.empty(n, cp.float32)
    k = _gpu.kernel("scene", "classify_points")
    k(((n + 255) // 256,), (256,),
      (np.int64(n), cp.asarray(pts, cp.float32), dev.struct, np.int32(0), cls, tt))  # fmt: skip
    return cls.get(), tt.get().astype(np.float64)


def slab_points(grid: tgrid.GrowthGrid, n: int = 100_000) -> np.ndarray:
    rng = np.random.default_rng(3)
    ps, pt = rng.uniform(grid.s[0], grid.s[-1], n), rng.uniform(grid.t[0], grid.t[-1], n)
    return tgrid.to_world(grid.plane, ps, pt, rng.uniform(0, DEPTH, n))


def test_any_outline_through_the_level_grid_and_its_control() -> None:
    """A rotated sphere union (no longer refused): the GPU's inside test matches the
    solid except within a fraction of a cell of the wall. Control: the same scene with
    its level grid left out puts every point outside."""
    r = recipe()
    specimen, _ = build(r)
    solid = specimen.volume.solid
    assert type(solid).__name__ != "SphereUnion"
    grid = tgrid.GrowthGrid.fill(specimen, specimen.plane(r.cut), H, DEPTH)
    packed = scene.pack(specimen, grid)
    pts = slab_points(grid)
    got, _ = classes(scene.upload(packed), pts)
    want = np.where(solid.contains(pts), 2, 0)
    near_wall = np.abs(solid.level(pts)) < 0.3 * H
    assert (got == want)[~near_wall].mean() > 0.999
    assert (want == 2).mean() > 0.2
    blind, _ = classes(scene.upload(replace(packed, level=None)), pts)
    assert (blind == 2).sum() == 0


def test_band_roughness_reaches_the_grid_and_its_control() -> None:
    """The grid's growth time is the warped one. Control: the unwarped model differs."""
    r = recipe(roughness=Roughness(amplitude_mm=0.2, scale_mm=2.0, ridged=False))
    specimen, _ = build(r)
    grid = tgrid.GrowthGrid.fill(specimen, specimen.plane(r.cut), H, DEPTH)
    rng = np.random.default_rng(5)
    m = 20_000
    k, jj, ii = (rng.integers(0, n, m) for n in grid.values.shape)
    nodes = tgrid.to_world(grid.plane, grid.s[ii], grid.t[jj], grid.w[k])
    on_grid = grid.values[k, jj, ii].astype(np.float64)
    inside = specimen.volume.solid.contains(nodes)
    warped = np.asarray(specimen.restored_step_fields(nodes)[0].t_smooth, np.float64)
    plain = np.asarray(specimen.volume.evaluate(nodes, inside).t_smooth, np.float64)
    ok = inside & np.isfinite(on_grid) & np.isfinite(plain)
    assert ok.sum() > 1000
    np.testing.assert_allclose(on_grid[ok], warped[ok], atol=1e-6)
    assert np.abs(on_grid[ok] - plain[ok]).max() > 1e-3


def test_absorption_is_averaged_after_the_floor_and_its_control() -> None:
    """A thin dust line under the iron floor when its impurity is averaged over a cell
    still absorbs when absorption is averaged. Control: averaging impurity loses it."""
    optics = OpticsTable()
    table = absorption_table(optics)
    n = 4
    base = np.full(n, 0.1, np.float32)
    line = np.array([0.1, 0.1, 0.1, 0.8], np.float32)  # one sub-sample of four on the line

    def fields(imp: np.ndarray):
        from agatesim.fields import Fields, Pigment

        z = np.zeros(len(imp), np.float32)
        return Fields(
            phase=np.full(len(imp), Phase.CHALCEDONY, np.uint8),
            band=np.zeros(len(imp), np.int32),
            impurity=imp,
            translucency=z,
            t_dep=z,
            grain=np.full(len(imp), -1, np.int32),
            micro_spacing_um=z,
            pigment=np.full(len(imp), Pigment.HEMATITE, np.uint8),
        )

    after = (absorption(table, fields(line)) - absorption(table, fields(base))).mean(0)
    before = absorption(table, fields(np.array([line.mean()], np.float32))) - absorption(
        table, fields(np.array([0.1], np.float32))
    )
    assert after.max() > 0
    assert np.all(before == 0)


def test_layers_bake_where_they_act_and_its_control() -> None:
    """Mottle and dust gain absorption at nodes in the stone; a stone without the
    place-to-place layers bakes nothing (control)."""
    optics = OpticsTable()
    plain = recipe()
    r = recipe(
        mottle=Mottle(scale_mm=4, cover=0.3, rim_mm=0.8, strength=0.7),
        dust=Dust(lines=True, specks=True, boundary_share=0.6),
    )
    sizes: list[float] = []
    for rr in (r, plain):
        specimen, _ = build(rr)
        grid = tgrid.GrowthGrid.fill(specimen, specimen.plane(rr.cut), H, 0.4)
        baked = bake_absorption(specimen, grid, optics)
        assert baked is not None
        sizes.append(float((baked.max(0) > 0.05).mean()))
    assert sizes[0] > 0.02
    assert sizes[1] == 0.0


def test_a_layered_variety_stone_renders_and_mottle_darkens_it() -> None:
    """End to end: the record from the per-step fields, the bake, the camera. Mottle
    lowers the reflectance where it lies; control: the same stone without layers."""
    r = recipe(mottle=Mottle(scale_mm=4, cover=0.35, rim_mm=0.8, strength=0.8))
    specimen, layers = build(r)
    cut = cut_of(specimen, r, 0.4)
    record = stone_record(specimen, np.random.default_rng(0), 200_000)
    rendering = Rendering(Light("dome"), record, 1.0, grid_mm=0.2)
    optics = rendering.optics
    mats = optics.materials(record)
    grid = tgrid.GrowthGrid.fill(specimen, cut.plane, 0.2, 1.0)
    images: list[np.ndarray] = []
    for absorb in (bake_absorption(specimen, grid, optics), None):
        dev = scene.upload(scene.pack(specimen, grid, absorb=absorb))
        out = render(dev, view_of(cut), mats, rendering.light, spp=64, depth_mm=1.0, seed=3)
        images.append(out.image.mean(-1)[::-1])
    patch = layers.mottle_amount(cut.plane.points(*np.meshgrid(cut.s, cut.t)).reshape(-1, 3))
    patch = patch.reshape(cut.inside.shape)[::-1]
    chal = (cut.fields.phase == Phase.CHALCEDONY)[::-1] & cut.inside[::-1]
    mottled = chal & (patch > 0.3)
    clear = chal & (patch == 0)
    assert mottled.sum() > 30
    ratio_m = images[0][mottled].mean() / images[1][mottled].mean()
    ratio_c = images[0][clear].mean() / images[1][clear].mean()
    assert ratio_m < 0.95
    assert abs(ratio_c - 1) < 0.03


def test_the_fill_bakes_as_the_separate_pass_does() -> None:
    """One pass reads each node once; it must give the separate bake's values.
    Control: a fill without optics bakes nothing."""
    optics = OpticsTable()
    r = recipe(mottle=Mottle(scale_mm=4, cover=0.3, rim_mm=0.8, strength=0.7))
    specimen, _ = build(r)
    plane = specimen.plane(r.cut)
    fused = tgrid.GrowthGrid.fill(specimen, plane, 0.2, 0.4, optics=optics)
    plain = tgrid.GrowthGrid.fill(specimen, plane, 0.2, 0.4)
    separate = bake_absorption(specimen, plain, optics)
    assert fused.absorb is not None
    assert separate is not None
    assert fused.absorb.max() > 0.05
    np.testing.assert_array_equal(fused.absorb, separate)
    assert plain.absorb is None
