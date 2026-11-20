"""The hollow drawn as air in the camera (ADR 0019 s.6.12 step 4), each check with its
negative control: the wall's turn against Snell, a flat air layer against the Fresnel stack,
a white furnace with air (open at the face or buried), no air giving the plain kernel's
image, and the GPU's air rule against `VoidBox.air` on a stone.

Skipped without a CUDA GPU (CI)."""

# pyright: basic

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from nodule.render import _gpu, reference, scene, voidgpu
from nodule.render.camera import Flag, Light, Materials, View, render

pytestmark = pytest.mark.skipif(not _gpu.available(), reason="needs a CUDA GPU and CuPy")

DOME = Light("dome")
VIEW = View(0.0, 0.0, 0.1, 32, 32)
N = 1.54
F0 = reference.fresnel(1.0, 1.0, N)


@pytest.fixture(scope="module")
def flat():
    return scene.upload(scene.homogeneous())


def snell(d: np.ndarray, nn: np.ndarray, n1: float, n2: float) -> np.ndarray:
    ci = -np.sum(d * nn, axis=1, keepdims=True)
    eta = n1 / n2
    k = 1 - eta**2 * (1 - ci**2)
    return eta * d + (eta * ci - np.sqrt(np.maximum(k, 0))) * nn


def unit(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_wall_turn_is_snell_and_the_mirror() -> None:
    rng = np.random.default_rng(1)
    nn = unit(rng.normal(size=(20_000, 3)))
    d = unit(rng.normal(size=(20_000, 3)))
    d = np.where(np.sum(d * nn, axis=1, keepdims=True) < 0, d, -d)  # toward the wall
    for n1, n2 in ((1.0, N), (N, 1.0)):
        out = voidgpu.turn_points(d, nn, n1, n2, 1.0 - 1e-7)  # refract unless TIR or grazing
        ci = -np.sum(d * nn, axis=1)
        tir = (n1 / n2) ** 2 * (1 - ci**2) >= 1
        refr = out[:, 3] == 1
        assert not np.any(refr & tir)
        assert np.mean(refr[~tir]) > 0.999
        want = snell(d, nn, n1, n2)
        assert np.abs(out[refr, :3] - want[refr]).max() < 1e-5
        mirror = d - 2 * np.sum(d * nn, axis=1, keepdims=True) * nn
        if (~refr).any():
            assert np.abs(out[~refr, :3] - mirror[~refr]).max() < 1e-5
        # control: the indices swapped give another direction
        wrong = snell(d, nn, n2, n1)
        ok = refr & ((n2 / n1) ** 2 * (1 - ci**2) < 1) & (ci < 0.95)
        assert np.median(np.abs(out[ok, :3] - wrong[ok]).max(axis=1)) > 0.05
    out = voidgpu.turn_points(d, nn, N, 1.0, 1e-7)  # xi below every reflectance: the mirror
    mirror = d - 2 * np.sum(d * nn, axis=1, keepdims=True) * nn
    assert out[:, 3].sum() == 0
    assert np.abs(out[:, :3] - mirror).max() < 1e-5


def mean_se(im: np.ndarray) -> tuple[float, float]:
    px = im[..., 0].ravel()
    return float(px.mean()), float(px.std(ddof=1) / math.sqrt(px.size))


def stack(mua: float, top: float) -> float:
    """A pure absorber over an air layer, seen along the normal, the face's own glare left
    out: incoherent multiple reflections between the face (R) and the layer's two walls."""
    r, t, e2 = F0, 1 - F0, math.exp(-2 * mua * top)
    r23 = r + t * t * r / (1 - r * r)  # stone/air/stone: nothing returns from below
    return t * t * e2 * r23 / (1 - r * r23 * e2)


def test_flat_air_layer_matches_the_fresnel_stack(flat) -> None:
    m = Materials.of([[0.1] * 3], [[0.0] * 3], [0.0])
    kw: dict[str, Any] = dict(
        spp=256, depth_mm=2.0, flags=Flag.HOMOGENEOUS | Flag.BLACK_BOTTOM, glare=False
    )
    v = voidgpu.upload(voidgpu.layer(0.5, 1.0))
    out = render(flat, VIEW, m, DOME, seed=3, void=v, **kw)
    mean, se = mean_se(out.image)
    want = stack(0.1, 0.5)
    assert abs(mean - want) < 3 * se
    assert out.crossings > 1.0
    assert out.guessed == 0.0
    # control: without the layer a pure absorber over a black bottom returns nothing
    none = render(flat, VIEW, m, DOME, seed=3, void=voidgpu.upload(voidgpu.none()), **kw)
    assert float(none.image.max()) == 0.0
    assert want > 50 * se
    # open at the face: the camera starts in air and sees the stone's reflection alone
    mouth = render(flat, VIEW, m, DOME, seed=4, void=voidgpu.upload(voidgpu.layer(0.0, 1.0)), **kw)
    mean, se = mean_se(mouth.image)
    assert abs(mean - F0) < 3 * se


FURNACE = Materials.of([[0.0] * 3], [[2.0] * 3], [0.9])
FURNACE_KW: dict[str, Any] = dict(spp=64, depth_mm=2.0, flags=Flag.HOMOGENEOUS | Flag.BLACK_BOTTOM,
                  bottom_albedo=(1.0, 1.0, 1.0), glare=False)  # fmt: skip


@pytest.mark.parametrize(("top", "bottom", "want"), [(0.5, 1.0, 1 - F0), (0.0, 0.5, 1.0)])
def test_white_furnace_with_an_air_layer(flat, top, bottom, want) -> None:
    """No absorption over a white bottom: every path leaves through the face (1 - F0 of the
    dome) or the open mouth (all of it), whatever the air does on the way."""
    v = voidgpu.upload(voidgpu.layer(top, bottom))
    im = render(flat, VIEW, FURNACE, DOME, seed=5, void=v, **FURNACE_KW).image
    assert np.allclose(im, want, rtol=1e-4)


SPHERE = ((1.6, 1.6, 0.4), 1.0)  # centre above the face: open over a disc


def sphere_pixels() -> tuple[np.ndarray, np.ndarray]:
    """Pixels wholly off the sphere's mouth (radius sqrt(r^2 - c_z^2)) and wholly in it."""
    (cx, cy, cz), r = SPHERE
    s = 0.1 * (np.arange(32) + 0.5)
    rr = np.hypot(*np.meshgrid(s - cx, s - cy))
    mouth = math.sqrt(r * r - cz**2)
    return rr > mouth + 0.08, rr < mouth - 0.08


def test_white_furnace_with_an_air_sphere(flat) -> None:
    """A curved wall: off the mouth 1 - F0 of the dome, in it all of it."""
    v = voidgpu.upload(voidgpu.sphere(*SPHERE))
    out = render(flat, VIEW, FURNACE, DOME, seed=6, void=v, **FURNACE_KW)
    far, open_ = sphere_pixels()
    assert open_.sum() > 100
    assert far.sum() > 100
    assert np.allclose(out.image[far], 1 - F0, rtol=1e-4)
    assert np.allclose(out.image[open_], 1.0, rtol=1e-4)
    assert out.crossings > 0.2


def test_control_wall_without_n2_breaks_the_furnace(flat) -> None:
    """Paths entering through the mouth and leaving through the face (or the reverse) carry
    the n^2; left out, the same furnace is rejected (an air layer alone would not show it:
    its n^2 cancels on the way in and out)."""
    v = voidgpu.upload(voidgpu.sphere(*SPHERE, flags=voidgpu.NO_WALL_N2))
    im = render(flat, VIEW, FURNACE, DOME, seed=6, void=v, **FURNACE_KW).image
    far, open_ = sphere_pixels()
    assert float(im[open_].mean()) > 1.2  # n^2 too bright
    assert not np.allclose(im[far], 1 - F0, rtol=1e-2)


def test_no_air_gives_the_plain_kernel_image(flat) -> None:
    m = Materials.of([[0.05] * 3], [[3.0] * 3], [0.9])
    kw: dict[str, Any] = dict(spp=64, seed=9, depth_mm=2.0, glare=False,
                              flags=Flag.HOMOGENEOUS | Flag.BLACK_BOTTOM,
                              bottom_albedo=(0.15, 0.15, 0.15))  # fmt: skip
    plain = render(flat, VIEW, m, DOME, **kw).image
    with_void = render(flat, VIEW, m, DOME, void=voidgpu.upload(voidgpu.none()), **kw).image
    assert np.allclose(with_void, plain, rtol=1e-6, atol=1e-9)
    # control: an air layer changes it
    layer = render(flat, VIEW, m, DOME, void=voidgpu.upload(voidgpu.layer(0.5, 1.0)), **kw).image
    assert not np.allclose(layer, plain, rtol=1e-3)


def test_void_needs_the_dome_without_glare(flat) -> None:
    m = Materials.of([[0.1] * 3], [[1.0] * 3], [0.9])
    v = voidgpu.upload(voidgpu.layer(0.5, 1.0))
    with pytest.raises(ValueError, match="dome"):
        render(flat, VIEW, m, Light("disc"), spp=1, flags=Flag.HOMOGENEOUS, void=v, glare=False)
    with pytest.raises(ValueError, match="glare"):
        render(flat, VIEW, m, DOME, spp=1, flags=Flag.HOMOGENEOUS, void=v)


# --- a stone's box on the GPU ------------------------------------------------------------


@pytest.fixture(scope="module")
def lined():
    from nodule.render.voidbox import build_void_box
    from nodule.variety import build
    from test_voidbox import recipe  # pyright: ignore[reportMissingImports]

    vol = build(recipe())[0].volume
    box = build_void_box(vol)
    assert box is not None
    return vol, box


def test_index_matched_box_walls_all_pass(lined) -> None:
    """With n = 1 nothing reflects, so every wall meeting passes and no path circles at a
    wall (it did before VOID_FACING: one path met the same wall until the event cap).
    Control: at n = 1.54 walls reflect."""
    from nodule.specimen import Specimen
    from test_voidbox import recipe  # pyright: ignore[reportMissingImports]

    vol, box = lined
    plane = Specimen(volume=vol, fractures=()).plane(recipe().cut)
    sa = scene.homogeneous()
    h = sa.header.copy()
    h["o"], h["u"], h["v"], h["n"] = plane.origin, plane.u, plane.v, plane.normal
    dev = scene.upload(scene.SceneArrays(h, sa.spheres, sa.tubes, sa.fractures, sa.grid))
    v = voidgpu.upload(voidgpu.pack_box(box))
    view = View(-6.0, -6.0, 0.12, 100, 100)
    m = Materials.of([[0.01] * 3], [[0.05] * 3], [0.9])
    kw: dict[str, Any] = dict(spp=32, depth_mm=5.0, glare=False, void=v,
                              flags=Flag.HOMOGENEOUS | Flag.BLACK_BOTTOM,
                              bottom_albedo=(0.15,) * 3)  # fmt: skip
    one = render(dev, view, m, DOME, n_in=1.0, **kw)
    assert one.crossings > 0.3
    assert one.passed == 1.0
    assert one.events < 10
    assert one.guessed < 0.01
    glass = render(dev, view, m, DOME, spp=256, **{k: v for k, v in kw.items() if k != "spp"})
    assert glass.passed < 0.95
    assert glass.crossings < 10  # 116 a path on the probe stone while paths circled
    assert glass.events < 20


def test_gpu_air_matches_the_box(flat, lined) -> None:
    _, box = lined
    rng = np.random.default_rng(2)
    lo, hi = box.a * box.h, (box.a + np.array(box.shape) - 1) * box.h
    p = lo + (hi - lo) * rng.random((400_000, 3))
    lev = box.level(p)
    phi = box.chalcedony_level(p)
    near = (np.abs(lev) < 0.05) | (np.abs(phi) < 0.05)
    p = np.concatenate([p[:100_000], p[near]])  # random points and near-wall ones
    p = np.concatenate([p, lo - 1.0 + rng.random((1000, 3))])  # off the box: never air
    want = box.air(p)
    dev = voidgpu.upload(voidgpu.pack_box(box))
    got, _vals = voidgpu.air_points(flat, dev, p)
    assert near.sum() > 10_000
    assert want.sum() > 5_000
    assert not got[-1000:].any()
    diff = got != want
    # disagreements only where float32 cannot tell the side: within a hair of either surface
    lv, ph = box.level(p), box.chalcedony_level(p)
    hair = (np.abs(lv) < 1e-4) | (np.abs(ph) < 1e-4)
    assert not np.any(diff & ~hair), f"{np.sum(diff & ~hair)} points disagree away from a wall"
    assert diff.mean() < 1e-3
    # control: a box with no gate is all solid, rejected by the same comparison
    packed = voidgpu.pack_box(box)
    assert packed.nf is not None
    bare = voidgpu.VoidArrays(packed.header, packed.ids, packed.nf & ~np.uint8(3), packed.phi,
                              packed.rows)  # fmt: skip
    got0, _ = voidgpu.air_points(flat, voidgpu.upload(bare), p)
    assert not got0.any()
    assert np.mean(got0 != want) > 0.05
