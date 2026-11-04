"""Materials that vary along the bands on the GPU (ADR 0016 section 4), each with its control.

Skipped without a CUDA GPU (CI). Run locally with `uv run --no-sync pytest`.
"""

# pyright: basic

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest
from agatesim.fields import Phase
from agatesim.volume.config import SliceConfig

from nodule.render import _gpu, scene, tgrid
from nodule.render.camera import Flag, Light, View, render
from nodule.render.materials import OpticsTable
from nodule.render.shift import ShiftArrays, upload, variation_of

pytestmark = pytest.mark.skipif(not _gpu.available(), reason="needs a CUDA GPU and CuPy")

CUT = SliceConfig(tilt_deg=90.0, azimuth_deg=270.0, offset_mm=0.0)
DEPTH = 2.0
DOME = Light("dome")


@pytest.fixture(scope="module")
def host(look):
    v = variation_of(look)
    assert v is not None
    return ShiftArrays.of(v)


@pytest.fixture(scope="module")
def banded_points(look):
    """~1e5 points in banded chalcedony, with agatesim's own growth time (float64)."""
    rng = np.random.default_rng(4)
    lo, hi = look.volume.solid.bounds()
    p = rng.uniform(lo, hi, (260_000, 3))
    p = p[look.volume.solid.contains(p)]
    f = look.volume.evaluate(p, np.ones(len(p), bool))
    b = (f.phase == Phase.CHALCEDONY) & (f.band >= 0) & np.isfinite(f.t_smooth)
    return p[b], f.t_smooth[b].astype(np.float64)


def gpu_shift(host: ShiftArrays, pts: np.ndarray, ts: np.ndarray):
    """The kernel's gain and change at points and growth times."""
    cp = _gpu.cupy()
    dev = upload(host, None)
    n = len(pts)
    gain, change = cp.empty(n, cp.float32), cp.empty(n, cp.float32)
    k = _gpu.kernel("shift", "shift_points")
    k(((n + 255) // 256,), (256,),
      (np.int64(n), cp.asarray(pts, cp.float32), cp.asarray(ts, cp.float32), dev.struct,
       gain, change))  # fmt: skip
    return gain.get().astype(np.float64), change.get().astype(np.float64)


def test_gpu_copy_matches_agatesim(look, host, banded_points) -> None:
    pts, ts = banded_points
    assert len(pts) > 80_000
    v = variation_of(look)
    assert v is not None
    want_g, want_c = v.shift(pts, ts)
    gain, change = gpu_shift(host, pts, ts)
    assert (want_g > 1.01).mean() > 0.1  # lenses are sampled
    eg, ec = np.abs(gain - want_g), np.abs(change - want_c)
    # measured: gain max 3.1e-4, p99 2.3e-5; change max 2.1e-6 (float32, fast math)
    assert eg.max() < 1e-3
    assert np.percentile(eg, 99) < 1e-4
    assert ec.max() < 1e-5


def test_control_a_wrong_copy_fails(look, host, banded_points) -> None:
    pts, ts = banded_points
    v = variation_of(look)
    assert v is not None
    want_g, want_c = v.shift(pts, ts)
    # a transposed rotation must fail the guard's bounds by far (measured: lens 1.5, drift 0.075)
    for name, bound in (("lens", 1e-3), ("drift", 1e-5)):
        sheet = getattr(host, name)
        bad = dataclasses.replace(host, **{name: dataclasses.replace(sheet, M=sheet.M.T.copy())})
        gain, change = gpu_shift(bad, pts, ts)
        err = np.abs(gain - want_g) if name == "lens" else np.abs(change - want_c)
        assert err.max() > 100 * bound


def test_drift_has_no_structure_finer_than_a_band(host, banded_points) -> None:
    """At fixed places, the drift half a band deeper is nearly the same; the 0.1 mm
    lenses (the control) are not."""
    pts, ts = banded_points
    band_mm = 30.0 * 0.015  # n2-slice: growth_mm x band_period, the mean band's width
    steps = len(host.depth) - 1
    depth = np.interp(ts * steps, np.arange(steps + 1), host.depth)
    edges = np.arange(steps + 1) / steps
    ok = depth + band_mm / 2 < host.depth[-1]
    later = np.interp(depth[ok] + band_mm / 2, host.depth, edges)
    g0, c0 = gpu_shift(host, pts[ok], ts[ok])
    g1, c1 = gpu_shift(host, pts[ok], later)
    drift_r = np.corrcoef(c0, c1)[0, 1]
    lens_r = np.corrcoef(g0, g1)[0, 1]
    print(f"half-band correlation: drift {drift_r:.3f}, lens {lens_r:.3f}")
    assert drift_r > 0.7
    assert lens_r < 0.3


@pytest.fixture(scope="module")
def look_scene(look, host):
    grid = tgrid.GrowthGrid.fill(look, look.plane(CUT), 0.2, DEPTH)
    assert grid.bands is not None
    dev = scene.upload(scene.pack(look, grid))
    full = View.covering(grid, 0.2)
    view = View(full.s0 + 15, full.t0 + 15, 0.2, 100, 100)  # 20 x 20 mm in the middle
    return grid.bands, dev, view


def lum(img: np.ndarray) -> np.ndarray:
    return img @ np.array([0.0722, 0.7152, 0.2126])  # channels 450, 550, 650 nm


def test_shift_with_no_takers_renders_identically(host, look_scene) -> None:
    rec, dev, view = look_scene
    none = dataclasses.replace(rec, shifts=np.zeros_like(rec.shifts))
    tab = OpticsTable()
    plain, with_cols = tab.materials(none), tab.materials(none, host)
    kw: dict[str, Any] = dict(spp=8, seed=11, depth_mm=DEPTH, flags=Flag.BLACK_BOTTOM)
    a = render(dev, view, plain, DOME, **kw).image
    b = render(dev, view, with_cols, DOME, shift=upload(host, with_cols), **kw).image
    assert np.array_equal(a, b)


def test_constant_lens_renders_as_the_per_step_pigment(host, look_scene) -> None:
    """Gain 2.5 everywhere (threshold far below the noise), no drift: the per-sample path
    must render as the per-step table with 2.5 x each band's pigment (16b)."""
    rec, dev, view = look_scene
    tab = OpticsTable()
    const = dataclasses.replace(host, threshold=-1e9, drift=None)
    mats = tab.materials(rec, const)
    assert mats.along is not None
    takes = mats.along[:, 0] >= 0
    assert (mats.along[takes, 1:] > 0).any()  # some band in the view is above the iron floor
    step = dataclasses.replace(mats, mua=(mats.mua + 1.5 * mats.along[:, 1:]).astype(np.float32),
                               along=None)  # fmt: skip
    mb = float((step.mua + step.kept()[0].astype(np.float32)).max()) * 1.001
    kw: dict[str, Any] = dict(spp=16, seed=12, depth_mm=DEPTH, flags=Flag.BLACK_BOTTOM, mubar=mb)
    per_sample = lum(render(dev, view, mats, DOME, shift=upload(const, mats), **kw).image)
    per_step = lum(render(dev, view, step, DOME, **kw).image)
    plain = lum(render(dev, view, tab.materials(rec), DOME, **kw).image)
    same = np.abs(per_sample - per_step).mean()
    lensless = np.abs(plain - per_step).mean()
    print(f"mean |diff|: per-sample vs per-step {same:.2e}, no lens vs per-step {lensless:.2e}")
    assert same < 0.02 * lensless  # the same paths, but for float rounding
    assert lensless > 1e-3  # control: without the lens the image differs


def test_shift_render_cost(host, look_scene) -> None:
    """Measured, not asserted beyond a sanity bound: ADR 0016 flags a slowdown over 2x."""
    rec, dev, view = look_scene
    tab = OpticsTable()
    mats = tab.materials(rec, host)
    sh = upload(host, mats)
    kw: dict[str, Any] = dict(spp=64, seed=13, depth_mm=DEPTH, flags=Flag.BLACK_BOTTOM)
    render(dev, view, mats, DOME, shift=sh, **{**kw, "spp": 1})  # compile
    on = render(dev, view, mats, DOME, shift=sh, **kw)
    off = render(dev, view, tab.materials(rec), DOME, **kw)
    ratio = on.seconds / off.seconds
    print(f"shift on {on.seconds:.2f} s ({on.events:.0f} events/path), "
          f"off {off.seconds:.2f} s ({off.events:.0f}); x{ratio:.2f}")  # fmt: skip
    assert math.isfinite(on.image.sum())
    assert ratio < 20
