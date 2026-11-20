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


def test_void_needs_the_dome_or_the_lamp_without_glare(flat) -> None:
    m = Materials.of([[0.1] * 3], [[1.0] * 3], [0.9])
    v = voidgpu.upload(voidgpu.layer(0.5, 1.0))
    with pytest.raises(ValueError, match="dome"):
        render(flat, VIEW, m, Light("directional"), spp=1, flags=Flag.HOMOGENEOUS, void=v,
               glare=False)  # fmt: skip
    with pytest.raises(ValueError, match="glare"):
        render(flat, VIEW, m, DOME, spp=1, flags=Flag.HOMOGENEOUS, void=v)
    with pytest.raises(ValueError, match="glare"):
        render(flat, VIEW, m, LAMP, spp=1, flags=Flag.HOMOGENEOUS, void=v)


# --- the small lamp (ADR 0019 s.6.12 step 5) ---------------------------------------------

LAMP = Light("disc", theta_deg=45.0, phi_deg=90.0, radius_deg=2.5)
LAMP_KW: dict[str, Any] = dict(depth_mm=2.0, flags=Flag.HOMOGENEOUS | Flag.BLACK_BOTTOM,
                               glare=False)  # fmt: skip
CLEAR = Materials.of([[0.05] * 3], [[0.0] * 3], [0.0])


def facet_toward(tilt_deg: float, phi_deg: float = 90.0) -> Any:
    """A facet through the middle of the view at 1 mm deep, its normal tilted from the face's
    normal (+z: the homogeneous scene's face) toward azimuth phi (from u = x toward v = y)."""
    t, p = math.radians(tilt_deg), math.radians(phi_deg)
    normal = (math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t))
    return voidgpu.upload(voidgpu.facet((1.6, 1.6, -1.0), normal))


def test_a_facet_glints_at_the_mirror_angle_only(flat) -> None:
    """A clear stone over a black bottom, open above a flat facet: a camera ray in the air
    meets the facet and reflects (Fresnel R at the facet's tilt) or refracts into the stone
    and is lost below. Tilted half the lamp's polar angle toward it, the mirror sends it into
    the lamp: every open pixel reads R(22.5 deg) Lr exactly. Controls: an untilted facet, and
    one tilted away from the lamp, stay dark."""
    out = render(flat, VIEW, CLEAR, LAMP, spp=256, seed=11, void=facet_toward(22.5), **LAMP_KW)
    want = reference.fresnel(math.cos(math.radians(22.5)), 1.0, N) * LAMP.disc_radiance
    mean, se = mean_se(out.image)  # every pixel is open: the facet lies below them all
    assert abs(mean - want) < 4 * se
    assert se < 0.02 * want
    assert np.mean(out.image[..., 0] > 0) > 0.99
    assert abs(out.crossings - 1.0) < 1e-6  # one wall meeting a path
    for tilt, phi in ((0.0, 90.0), (22.5, 270.0)):
        dark = render(flat, VIEW, CLEAR, LAMP, spp=64, seed=12, void=facet_toward(tilt, phi),
                      **LAMP_KW).image  # fmt: skip
        assert float(dark.max()) == 0.0


def test_the_lamp_weights_stay_unbiased_with_air(flat) -> None:
    """A lightly scattering stone under a wide lamp, with buried air (a flat layer under a
    thin cap, so most shadow legs cross it; then a sphere, a curved wall): the combined
    estimate (shadow rays plus phase sampling) equals phase sampling alone, so a shadow leg
    that meets air carries nothing and the routes it cannot make keep weight 1. Controls:
    shadow rays alone miss the routes through the air (darker); shadow legs that pass the
    air as stone count light the phase routes also count (biased)."""
    m = Materials.of([[0.02] * 3], [[0.5] * 3], [0.6])
    lamp = Light("disc", theta_deg=40.0, phi_deg=90.0, radius_deg=10.0)

    def lit(flags: int, seed: int, vflags: int = 0, shape: str = "layer") -> tuple[float, float]:
        """The image's mean, and its standard error from 8 independent batches (pixels
        differ in brightness, so their spread is no noise estimate)."""
        air = (voidgpu.layer(0.3, 1.0, flags=vflags) if shape == "layer"
               else voidgpu.sphere((1.6, 1.6, -1.0), 0.8, flags=vflags))  # fmt: skip
        v = voidgpu.upload(air)
        means = [float(render(flat, VIEW, m, lamp, seed=seed + 100 * k, void=v, spp=2048,
                              depth_mm=2.0, flags=Flag.HOMOGENEOUS | Flag.BLACK_BOTTOM | flags,
                              glare=False).image[..., 0].mean())
                 for k in range(8)]  # fmt: skip
        return float(np.mean(means)), float(np.std(means, ddof=1) / math.sqrt(len(means)))

    def apart(a: tuple[float, float], b: tuple[float, float]) -> float:
        return abs(a[0] - b[0]) / math.hypot(a[1], b[1])

    both, phase = lit(0, 13), lit(256, 14)
    assert apart(both, phase) < 4
    shadow = lit(128, 15)  # shadow rays alone: darker
    assert shadow[0] < both[0]
    assert apart(both, shadow) > 4
    assert apart(lit(0, 16, vflags=voidgpu.LEGS_IGNORE_AIR), phase) > 4  # legs leak: biased
    assert apart(lit(0, 17, shape="sphere"), lit(256, 18, shape="sphere")) < 4


def test_no_air_gives_the_plain_kernel_image_under_the_lamp(flat) -> None:
    m = Materials.of([[0.05] * 3], [[3.0] * 3], [0.9])
    kw: dict[str, Any] = dict(LAMP_KW, spp=64, seed=16)
    plain = render(flat, VIEW, m, LAMP, **kw).image
    with_void = render(flat, VIEW, m, LAMP, void=voidgpu.upload(voidgpu.none()), **kw).image
    assert np.allclose(with_void, plain, rtol=1e-6, atol=1e-9)
    facet = render(flat, VIEW, m, LAMP, void=facet_toward(22.5), **kw).image
    assert not np.allclose(facet, plain, rtol=1e-3)  # control: air changes it


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


def test_shell_exit_turns_once(lined) -> None:
    """A path leaving the shell below the slab meets the matte bottom's rule once: the turn
    is made from inside the shell (ADR 0019 s.6.12, 2026-11-19). In a scattering stone (short
    flights, as chalcedony; on geode 4/4 1.12 turns a run) a run of turns with nothing
    between is then rare (a floor: one step inside ends a run); in clear stone a turned path
    can fly out again at once (1.7 here, runs/hollow-air/shell_test_probe.py). Control:
    turned in place, outside the shell, the next short flight meets the rule again (2.3 a
    run on 4/4)."""
    from nodule.specimen import Specimen
    from test_voidbox import recipe  # pyright: ignore[reportMissingImports]

    vol, box = lined
    plane = Specimen(volume=vol, fractures=()).plane(recipe().cut)
    sa = scene.homogeneous()
    h = sa.header.copy()
    h["o"], h["u"], h["v"], h["n"] = plane.origin, plane.u, plane.v, plane.normal
    dev = scene.upload(scene.SceneArrays(h, sa.spheres, sa.tubes, sa.fractures, sa.grid))
    view = View(-6.0, -6.0, 0.12, 100, 100)
    m = Materials.of([[0.01] * 3], [[20.0] * 3], [0.9])  # scattering like chalcedony
    kw: dict[str, Any] = dict(spp=64, depth_mm=1.0, glare=False, seed=3,
                              flags=Flag.HOMOGENEOUS | Flag.BLACK_BOTTOM,
                              bottom_albedo=(0.15,) * 3)  # fmt: skip
    fixed = render(dev, view, m, DOME, void=voidgpu.upload(voidgpu.pack_box(box)), **kw)
    stay = render(dev, view, m, DOME, **kw,
                  void=voidgpu.upload(voidgpu.pack_box(box, flags=voidgpu.SHELL_STAY)))  # fmt: skip
    assert fixed.shell_exits > 0.1  # the check sees enough exits (0.7 a path)
    assert fixed.shell_turns / fixed.shell_exits < 1.3  # 1.11
    assert stay.shell_turns / stay.shell_exits > 1.8  # 2.29


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
    got, _vals, shell = voidgpu.air_points(flat, dev, p)
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
    got0, _, _ = voidgpu.air_points(flat, voidgpu.upload(bare), p)
    assert not got0.any()
    assert np.mean(got0 != want) > 0.05
    # the shell (nearest node's flag) as the CPU reads it; control: the shell bit cleared
    in_shell = box.in_shell(p)
    assert in_shell.mean() > 0.2
    assert np.mean(shell != in_shell) < 1e-4
    noshell = voidgpu.VoidArrays(packed.header, packed.ids, packed.nf & ~np.uint8(voidgpu.SHELL),
                                 packed.phi, packed.rows)  # fmt: skip
    _, _, shell0 = voidgpu.air_points(flat, voidgpu.upload(noshell), p)
    assert not shell0.any()


def test_cut_renderer_draws_the_hollow_as_air_and_shares_its_fill(lined) -> None:
    """`CutRenderer(box=)` with hollow="air": the grid continues under the hollow, the
    look meets the wall, and `.on()` a stand-in rendering on the same fill does not (it
    keeps the dark stand-in in the hollow); control: the stand-in alone never meets a wall."""
    from nodule.render.cutface import CutRenderer, Rendering
    from nodule.render.fakes import stone_record
    from nodule.render.record import void_points
    from nodule.specimen import window_cut
    from nodule.variety import build, cut_of
    from test_voidbox import recipe  # pyright: ignore[reportMissingImports]

    _, box = lined
    spec, _ = build(recipe())
    cut = cut_of(spec, recipe(), 0.2)
    jj, ii = np.nonzero(void_points(cut.fields, cut.inside))
    near = window_cut(spec, cut, (float(cut.s[ii].mean()), float(cut.t[jj].min())), 6.0, 0.1)
    record = stone_record(spec, np.random.default_rng(1))
    look = Rendering.look(record, grid_mm=0.2, depth_mm=1.0, hollow="air")
    air = CutRenderer(spec, near, look, np.random.default_rng(2), box=box)
    assert air._grid.full_depth_mm == 1.0
    assert air._grid.w[-1] > 3.0  # continued under the hollow
    air.more(16)
    assert air.crossings > 0.05
    stand = air.on(Rendering.look(record, grid_mm=0.2, depth_mm=1.0))
    stand.more(16)
    assert stand.crossings == 0.0
    hole = void_points(near.fields, near.inside)
    assert hole.sum() > 100
    lum = [cr.rgb[hole].astype(float).mean() for cr in (air, stand)]
    assert lum[0] > lum[1] + 10  # the stand-in reads dark, the air shows the far wall
    with pytest.raises(ValueError, match="void box"):
        CutRenderer(spec, near, look, np.random.default_rng(2)).on(look)


def test_the_sparkle_view_adds_the_lamp_and_keeps_the_face(lined) -> None:
    """The sparkle view (`sparkle_rgb`): a lamp pass (`Rendering.lamp`) on the look's fill,
    added and exposed so the face without its hollow keeps the look's mean lightness, the
    hollow lit by the lamp as well. Controls: only a disc makes a lamp; a lamp pass of zero
    irradiance gives the look back."""
    from nodule.render.cutface import CutRenderer, Rendering, sparkle_rgb
    from nodule.render.fakes import stone_record
    from nodule.render.record import void_points
    from nodule.specimen import window_cut
    from nodule.variety import build, cut_of
    from test_voidbox import recipe  # pyright: ignore[reportMissingImports]

    _, box = lined
    spec, _ = build(recipe())
    cut = cut_of(spec, recipe(), 0.2)
    jj, ii = np.nonzero(void_points(cut.fields, cut.inside))
    near = window_cut(spec, cut, (float(cut.s[ii].mean()), float(cut.t[jj].min())), 6.0, 0.1)
    record = stone_record(spec, np.random.default_rng(1))
    look = CutRenderer(spec, near, Rendering.look(record, grid_mm=0.2, depth_mm=1.0,
                                                  hollow="air"), np.random.default_rng(2),
                       box=box)  # fmt: skip
    look.more(32)
    lamp_light = Light("disc", theta_deg=45.0, phi_deg=135.0, irradiance=math.pi / 2)
    lamp = look.on(Rendering.lamp(record, lamp_light, grid_mm=0.2, depth_mm=1.0, hollow="air"))
    lamp.more(32)
    assert lamp.crossings > 0.05  # the lamp pass draws the hollow as air too
    rgb = sparkle_rgb(look, lamp)
    face = look._face

    def lightness(im: np.ndarray) -> float:  # mean sRGB-encoded luminance, as the look's
        y = im.astype(float) / 255
        return float((0.2126 * y[..., 0] + 0.7152 * y[..., 1] + 0.0722 * y[..., 2])[face].mean())

    assert face.sum() > 1000
    assert abs(lightness(rgb) / lightness(look.rgb) - 1.0) < 0.005
    hole = void_points(near.fields, near.inside)
    assert not np.array_equal(rgb[hole], look.rgb[hole])
    dark = look.on(Rendering.lamp(record, Light("disc", irradiance=0.0), grid_mm=0.2,
                                  depth_mm=1.0, hollow="air"))  # fmt: skip
    with np.errstate(divide="ignore", invalid="ignore"):  # its own preview has no white
        dark.more(2)
    assert np.array_equal(sparkle_rgb(look, dark), look.rgb)
    with pytest.raises(ValueError, match="disc"):
        Rendering.lamp(record, DOME)


def test_the_lamp_weights_stay_unbiased_over_a_stone_box(lined) -> None:
    """As over the analytic air, on a stone's own crystal-lined hollow (its box clips the
    shadow legs before testing them for air): the combined estimate equals phase sampling
    alone under a wide lamp. Control: shadow legs let through the air read biased."""
    from nodule.specimen import Specimen
    from test_voidbox import recipe  # pyright: ignore[reportMissingImports]

    vol, box = lined
    plane = Specimen(volume=vol, fractures=()).plane(recipe().cut)
    sa = scene.homogeneous()
    h = sa.header.copy()
    h["o"], h["u"], h["v"], h["n"] = plane.origin, plane.u, plane.v, plane.normal
    dev = scene.upload(scene.SceneArrays(h, sa.spheres, sa.tubes, sa.fractures, sa.grid))
    view = View(-6.0, -6.0, 0.24, 50, 50)
    m = Materials.of([[0.02] * 3], [[0.5] * 3], [0.6])
    lamp = Light("disc", theta_deg=40.0, phi_deg=90.0, radius_deg=10.0)

    def lit(flags: int, seed: int, vflags: int = 0) -> tuple[float, float]:
        v = voidgpu.upload(voidgpu.pack_box(box, flags=vflags))
        means = [float(render(dev, view, m, lamp, seed=seed + 100 * k, void=v, spp=512,
                              depth_mm=3.0, flags=Flag.HOMOGENEOUS | Flag.BLACK_BOTTOM | flags,
                              glare=False).image[..., 0].mean())
                 for k in range(8)]  # fmt: skip
        return float(np.mean(means)), float(np.std(means, ddof=1) / math.sqrt(len(means)))

    def apart(a: tuple[float, float], b: tuple[float, float]) -> float:
        return abs(a[0] - b[0]) / math.hypot(a[1], b[1])

    phase = lit(256, 21)
    assert apart(lit(0, 22), phase) < 4
    assert apart(lit(0, 23, vflags=voidgpu.LEGS_IGNORE_AIR), phase) > 4


def test_a_milky_linings_stand_in_hole_stays_clear(lined, monkeypatch) -> None:
    """A stand-in hollow inside a milky lining (backlit slice): where the void mask says
    hollow, the camera takes the lining without the milk (`Scene.voidf`), so the hole
    passes the light box as with a clear lining; the growth time stays the lining's, so
    nothing leaks into the cells along the wall. Control: without the mask the milky
    lining fills the hole and darkens it."""
    from dataclasses import replace

    from nodule.render.cutface import CutRenderer, Rendering
    from nodule.render.fakes import stone_record
    from nodule.render.materials import OpticsTable
    from nodule.render.record import void_points
    from nodule.specimen import window_cut
    from nodule.variety import build, cut_of
    from test_voidbox import recipe  # pyright: ignore[reportMissingImports]

    spec, _ = build(recipe())
    cut = cut_of(spec, recipe(), 0.2)
    jj, ii = np.nonzero(void_points(cut.fields, cut.inside))
    near = window_cut(spec, cut, (float(cut.s[ii].mean()), float(cut.t[jj].mean())), 6.0, 0.1)
    hole = void_points(near.fields, near.inside)
    assert hole.sum() > 200
    record = stone_record(spec, np.random.default_rng(1))
    milky = replace(OpticsTable(), milk_mus=30.0, milk_from=1)

    def hole_light(optics: OpticsTable) -> float:
        back = Rendering.transmitted(record, grid_mm=0.2, depth_mm=1.0, optics=optics,
                                     exposure=3.0)  # fmt: skip
        cr = CutRenderer(spec, near, back, np.random.default_rng(2))
        cr.more(64)
        return float(cr.linear()[hole].mean())

    clear, masked = hole_light(OpticsTable()), hole_light(milky)
    assert abs(masked / clear - 1.0) < 0.1
    monkeypatch.setattr(CutRenderer, "_void_mask", lambda self, rendering: None)
    assert hole_light(milky) < 0.7 * clear


def test_glints_only_keeps_the_mirror_and_drops_the_scattered_glow(flat) -> None:
    """`Flag.MOUTH_GLINTS_ONLY` (the sparkle view's lamp): over a scattering stone under the
    tilted facet, every open pixel reads exactly the mirror's R(22.5 deg) Lr: the facet's
    reflection counts (no scattering on the way), light scattered in the stone that leaves
    through the mouth into the lamp does not, and shadow legs from below cross the air, so
    carry nothing. Control: without the flag the scattered glow adds to it."""
    m = Materials.of([[0.02] * 3], [[2.0] * 3], [0.6])
    # a wide lamp: the glint's radiance falls as 1/sin^2(radius), the scattered glow does not,
    # so the control can see the glow (at 2.5 deg it is ~1 % of the glint here)
    lamp = Light("disc", theta_deg=45.0, phi_deg=90.0, radius_deg=10.0)
    want = reference.fresnel(math.cos(math.radians(22.5)), 1.0, N) * lamp.disc_radiance
    t = math.radians(22.5)  # the facet 1.5 mm deep: it meets the face 2 mm off the view
    facet = voidgpu.upload(voidgpu.facet((1.6, 1.6, -1.5), (0.0, math.sin(t), math.cos(t))))
    kw: dict[str, Any] = dict(depth_mm=3.0, glare=False, void=facet)
    flags = Flag.HOMOGENEOUS | Flag.BLACK_BOTTOM
    glints = render(flat, VIEW, m, lamp, spp=1024, seed=21, flags=flags | Flag.MOUTH_GLINTS_ONLY,
                    **kw)  # fmt: skip
    mean, se = mean_se(glints.image)
    assert abs(mean - want) < 4 * se
    every = render(flat, VIEW, m, lamp, spp=1024, seed=22, flags=flags, **kw)
    more, se2 = mean_se(every.image)
    assert more > mean + 4 * math.hypot(se, se2)
