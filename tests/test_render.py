"""nodule.render, the parts without a GPU (these run in CI)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from agatesim.volume.config import SliceConfig
from agatesim.volume.fractures import FLAT

from nodule.render import reference, scene, tgrid
from nodule.render.camera import Light, Materials, View, srgb
from nodule.specimen import Specimen, load_preset

CUT = SliceConfig(tilt_deg=90.0, azimuth_deg=270.0, offset_mm=0.0)


@pytest.fixture(scope="module")
def specimen() -> Specimen:
    return Specimen.build(load_preset("n2-slice"))


def linear_grid(shape: tuple[int, int, int]) -> np.ndarray:
    k, j, i = np.indices(shape)
    return (0.3 * k - 0.2 * j + 0.1 * i + 1.0).astype(np.float32)


def test_fill_nan_is_exact_for_linear_t() -> None:
    exact = linear_grid((12, 12, 12))
    holed = exact.copy()
    holed[4:8, 4:8, 4:8] = np.nan  # a hole 4 cells across: 2 passes reach its centre
    holed[:, :, :2] = np.nan  # and an outside edge
    out = tgrid.fill_nan(holed, passes=3)
    assert np.abs(out - exact).max() < 1e-5


def test_plain_averaging_fill_is_not_exact() -> None:
    """Negative control for the linear fill."""
    exact = linear_grid((12, 12, 12))
    holed = exact.copy()
    holed[:, :, :2] = np.nan
    out = tgrid.fill_nan(holed, passes=3, linear=False)
    assert np.abs(out - exact).max() > 0.05


def test_fill_nan_leaves_zero_beyond_its_passes() -> None:
    holed = np.full((9, 9, 9), np.nan, np.float32)
    holed[4, 4, 4] = 1.0
    out = tgrid.fill_nan(holed, passes=1)
    assert out[4, 4, 5] == 1.0
    assert out[0, 0, 0] == 0.0


def test_grid_lookup_is_trilinear_and_nan_off_grid(specimen: Specimen) -> None:
    plane = specimen.plane(CUT)
    s, t, w = tgrid.grid_axes(specimen, plane, 1.0, 2.0)
    vals = np.zeros((len(w), len(t), len(s)), np.float32)
    ww, tt, ss = np.meshgrid(w, t, s, indexing="ij")
    vals[:] = 0.5 * ss - 0.25 * tt + 2.0 * ww
    grid = tgrid.GrowthGrid(plane=plane, s=s, t=t, w=w, values=vals)
    rng = np.random.default_rng(3)
    ps = rng.uniform(s[0], s[-2], 100)
    pt = rng.uniform(t[0], t[-2], 100)
    pw = rng.uniform(w[0], w[-2], 100)
    got = grid.lookup(tgrid.to_world(plane, ps, pt, pw))
    assert np.allclose(got, 0.5 * ps - 0.25 * pt + 2.0 * pw, atol=1e-4)
    off = tgrid.to_world(plane, np.array([s[-1] + 5]), np.array([t[0]]), np.array([0.0]))
    assert np.isnan(grid.lookup(off)).all()


def test_to_plane_inverts_to_world(specimen: Specimen) -> None:
    plane = specimen.plane(CUT)
    a = np.array([1.0, -2.0]), np.array([3.0, 0.5]), np.array([0.2, 4.0])
    back = tgrid.to_plane(plane, tgrid.to_world(plane, *a))
    for x, y in zip(a, back, strict=True):
        assert np.allclose(x, y)


def test_grid_axes_share_nodes_across_spacings(specimen: Specimen) -> None:
    plane = specimen.plane(CUT)
    fine = tgrid.grid_axes(specimen, plane, 0.2, 3.0)
    coarse = tgrid.grid_axes(specimen, plane, 0.4, 3.0)
    for f, c in zip(fine, coarse, strict=True):
        overlap = c[(c >= f[0]) & (c <= f[-1])]  # the coarse pad may reach one node further
        assert len(overlap) >= len(c) - 2
        assert np.isin(np.round(overlap, 6), np.round(f, 6)).all()


def test_fill_cost_flags_the_fine_grid(specimen: Specimen) -> None:
    plane = specimen.plane(CUT)
    n, sec, mb = tgrid.fill_cost(specimen, plane, 0.2, 10.0)
    assert 4e6 < n < 5e6  # RESULTS.md part 2a: 4.55 M
    assert 5 < sec < 15
    assert 15 < mb < 20
    assert tgrid.fill_cost(specimen, plane, 0.1, 10.0)[1] > 30  # over the flag


def test_pack_layout(specimen: Specimen) -> None:
    plane = specimen.plane(CUT)
    s, t, w = tgrid.grid_axes(specimen, plane, 1.0, 2.0)
    grid = tgrid.GrowthGrid(plane, s, t, w, np.zeros((len(w), len(t), len(s)), np.float32))
    packed = scene.pack(specimen, grid)
    h = packed.header
    assert (int(h["ns"]), int(h["nt"]), int(h["nw"])) == (len(s), len(t), len(w))
    assert int(h["nsph"]) == len(packed.spheres) == 7
    assert int(h["ntube"]) == len(packed.tubes) == 2
    assert int(h["nfrac"]) == len(packed.fractures) == 1
    f = specimen.fractures[0]
    assert packed.fractures[0, 16] == f.step_count
    assert np.allclose(packed.fractures[0, 3:12], f.frame.ravel())


def test_pack_refuses_profiles_the_kernel_lacks(specimen: Specimen) -> None:
    plane = specimen.plane(CUT)
    s, t, w = tgrid.grid_axes(specimen, plane, 1.0, 2.0)
    grid = tgrid.GrowthGrid(plane, s, t, w, np.zeros((len(w), len(t), len(s)), np.float32))
    flat = dataclasses.replace(specimen.fractures[0], along=FLAT)
    other = dataclasses.replace(specimen, fractures=(flat,))
    with pytest.raises(ValueError, match="BELL"):
        scene.pack(other, grid)


def test_view_coarser_keeps_origin() -> None:
    v = View(-3.0, 2.0, 0.1, 101, 50)
    c = v.coarser(4)
    assert (c.s0, c.t0, c.nx, c.ny) == (-3.0, 2.0, 25, 12)
    assert math.isclose(c.pixel_mm, 0.4)


def test_white_standard() -> None:
    assert Light("dome", radiance=2.0).white() == 2.0
    assert math.isclose(Light("directional", theta_deg=60.0).white(), 0.5 / math.pi)


def test_materials_validation() -> None:
    m = Materials.of([[0.1, 0.1, 0.1]], [[1, 1, 1]], [0.9])
    assert m.mua.shape == (1, 3)
    assert m.profile.tolist() == [0]
    with pytest.raises(ValueError, match=r"\(m, 3\)"):
        Materials.of([[0.1, 0.1]], [[1, 1]], [0.9])
    with pytest.raises(ValueError, match="non-negative"):
        Materials.of([[-0.1, 0.1, 0.1]], [[1, 1, 1]], [0.9])


def test_srgb_endpoints_and_orientation() -> None:
    x = np.zeros((2, 1, 3))
    x[1] = 1.0  # top row (+v) is the last row
    out = srgb(x)
    assert out[0, 0].tolist() == [255, 255, 255]
    assert out[1, 0].tolist() == [0, 0, 0]
    assert srgb(np.full((1, 1, 3), 0.5))[0, 0, 0] == 188  # sRGB(0.5) = 0.7354


def test_fresnel_reference() -> None:
    assert math.isclose(reference.fresnel(1.0, 1.0, 1.5), 0.04)
    assert reference.fresnel(0.1, 1.54, 1.0) == 1.0  # total internal reflection


@pytest.mark.parametrize("w", [0.5, 0.9, 0.99])
def test_h_function_integral(w: float) -> None:
    h = reference.ChandrasekharH(w)
    assert math.isclose(h.integral, 2 * (1 - math.sqrt(1 - w)) / w, rel_tol=1e-9)


def test_t2_diffuse_part_from_h() -> None:
    """MCML Table 2's 0.2600 includes the 0.04 specular (RESULTS.md part 1): the
    matched-iso exact value is far from 0.26, so only a mismatched-face
    comparison with the specular added can test it (in the GPU tests)."""
    assert reference.T2["rd_with_specular"] - 0.04 == pytest.approx(0.22)
    assert reference.halfspace_albedo_iso(0.9) == pytest.approx(0.41495, abs=1e-5)
