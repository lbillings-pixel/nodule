"""White opal by place in the camera (ADR 0019 s.13), each check with its negative
control: a node's white share picks the white row with that probability at each
tentative collision, which must equal the exact mixture of the two rows; a share of 0
must leave the image bit for bit as without the share.

Skipped without a CUDA GPU (CI)."""

# pyright: basic

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from agatesim.volume.config import SliceConfig

from nodule.render import _gpu, scene, tgrid, voidgpu
from nodule.render.camera import Light, Materials, View, render
from nodule.specimen import Specimen, load_preset

pytestmark = pytest.mark.skipif(not _gpu.available(), reason="needs a CUDA GPU and CuPy")

CUT = SliceConfig(tilt_deg=90.0, azimuth_deg=270.0, offset_mm=0.0)
H, DEPTH = 0.2, 3.0
# host and tube core (the GPU tests' stand-ins), a clear band A, and a white row W like
# the build's (sub-micron pores in opal at transport 16 per mm): `effective`, plumbing only
HOST = ([0.6, 1.0, 1.6], [15.0] * 3, [0.8] * 3)
TUBE = ([2.5, 0.8, 1.8], [10.0] * 3, [0.8] * 3)
A = ([0.05, 0.08, 0.12], [3.0] * 3, [0.9] * 3)
W = ([0.13] * 3, [47.2, 30.3, 20.3], [0.57, 0.47, 0.39])
DOME, BACK = Light("dome"), Light("back")
LIGHTS: dict[str, tuple[Light, dict[str, Any]]] = {
    "dome": (DOME, dict(depth_mm=DEPTH)),
    "directional": (Light("directional", theta_deg=45.0), dict(depth_mm=DEPTH)),
    "back": (BACK, dict(depth_mm=DEPTH)),
    "void kernel": (DOME, dict(depth_mm=DEPTH, glare=False)),
}


@pytest.fixture(scope="module")
def stone():
    """n2-slice's grid with three scenes: no share, a share of 0, a share of 1/2."""
    sp = Specimen.build(load_preset("n2-slice"))
    grid = tgrid.GrowthGrid.fill(sp, sp.plane(CUT), H, DEPTH)
    packed = scene.pack(sp, grid)
    shape = grid.values.shape
    devs = {
        "none": scene.upload(packed),
        "zero": scene.upload(replace(packed, white=np.zeros(shape, np.float32))),
        "half": scene.upload(replace(packed, white=np.full(shape, 0.5, np.float32))),
    }
    return View.covering(grid, 0.8), devs


def table(*band: tuple[list[float], list[float], list[float]], f: list[float] | None = None,
          gb: list[float] | None = None, white: int = -1) -> Materials:  # fmt: skip
    rows = [HOST, TUBE, *band]
    ff = [[0.0] * 3, [0.0] * 3] + [f or [0.0] * 3] + [[0.0] * 3] * (len(band) - 1)
    gg = [[0.0] * 3, [0.0] * 3] + [gb or [0.0] * 3] + [[0.0] * 3] * (len(band) - 1)
    return Materials.of(mua=[r[0] for r in rows], mus=[r[1] for r in rows],
                        g=[r[2] for r in rows], f=ff, gb=gg, white=white)  # fmt: skip


def mixture(s: float) -> Materials:
    """A and W mixed at white share s, exactly: absorption and scattering weighted by s;
    lobe a is A's phase function, lobe b W's, lobe b's share of scattering s mus_W / mus."""
    mua = [s * w + (1 - s) * a for a, w in zip(A[0], W[0], strict=True)]
    mus = [s * w + (1 - s) * a for a, w in zip(A[1], W[1], strict=True)]
    f = [s * w / m for w, m in zip(W[1], mus, strict=True)]
    return table((mua, mus, A[2]), f=f, gb=W[2])


PICK = table(A, W, white=3)  # band row 2 is A everywhere; W the white row


def mean_se(dev, view: View, m: Materials, light: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The image's mean per channel over two independent halves, and its standard error
    from their pixel differences."""
    lt, kw = LIGHTS[light]
    void = voidgpu.upload(voidgpu.none()) if light == "void kernel" else None
    ims = [render(dev, view, m, lt, spp=64, seed=seed + k, void=void, **kw).image.reshape(-1, 3)
           for k in (0, 1)]  # fmt: skip
    noise = np.std(ims[0] - ims[1], axis=0, ddof=1) / math.sqrt(2)
    return (ims[0] + ims[1]).mean(0) / 2, noise / math.sqrt(2 * len(ims[0]))


@pytest.mark.parametrize("light", list(LIGHTS))
def test_a_share_of_zero_renders_bit_for_bit(stone, light) -> None:
    """No number is drawn where the share is 0, so every path is the one it was: the
    image equals the one without a share up to the order of the pixels' float additions
    (atomic, so two identical renders differ by ~1e-7 relative, measured 2026-11-20).
    Control: a share of 1/2 changes the image."""
    view, devs = stone
    lt, kw = LIGHTS[light]
    void = voidgpu.upload(voidgpu.none()) if light == "void kernel" else None
    ims = {k: render(devs[k], view, PICK, lt, spp=4, seed=5, void=void, **kw).image
           for k in ("none", "zero", "half")}  # fmt: skip
    assert np.allclose(ims["zero"], ims["none"], rtol=1e-6, atol=1e-9)
    assert not np.allclose(ims["half"], ims["none"], rtol=1e-3)


@pytest.mark.parametrize("light", list(LIGHTS))
def test_a_half_share_is_the_mixture(stone, light) -> None:
    """The white row picked half the time equals the rows' exact mixture under each light
    and in the hollow-as-air build (main loop, shadow rays, the back light). Control: the
    mixture for a share of 1/4 is far off."""
    view, devs = stone
    got, se = mean_se(devs["half"], view, PICK, light, 11)
    want, se_w = mean_se(devs["none"], view, mixture(0.5), light, 21)
    z = (got - want) / np.hypot(se, se_w)
    print(f"{light}: got {np.round(got, 4)}, want {np.round(want, 4)}, z {np.round(z, 2)}")
    assert np.all(np.abs(z) < 3.5)
    low, se_l = mean_se(devs["none"], view, mixture(0.25), light, 31)
    zc = (got - low) / np.hypot(se, se_l)
    print(f"  control 1/4: z {np.round(zc, 1)}")
    assert np.all(np.abs(zc) > 10)


GREY = [
    ([1.0] * 3, [15.0] * 3, [0.8] * 3),
    ([1.0] * 3, [10.0] * 3, [0.8] * 3),
    ([0.08] * 3, [3.0] * 3, [0.9] * 3),
]  # host, tube core, band: alike in every channel
W_NEUTRAL = ([0.13] * 3, [30.3] * 3, [0.47] * 3)  # W's green channel in all three


def grey(white: tuple[list[float], list[float], list[float]]) -> Materials:
    rows = [*GREY, white]
    return Materials.of([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows],
                        white=3)  # fmt: skip


def test_white_opal_warms_the_backlit_slice(stone) -> None:
    """Under the back light the bluish white row takes more blue than red out of the
    light through the slice, so the same pixels read warmer (red over blue) with a share
    of 1/2 than with none: the probe's tan shadow (s.13), here in a stone otherwise grey,
    measured locally. Control: a white row scattering alike in every channel leaves red
    over blue at 1."""
    view, devs = stone
    lt, kw = LIGHTS["back"]

    def warmth(dev: Any, m: Materials) -> float:
        im = render(dev, view, m, lt, spp=64, seed=51, **kw).image.reshape(-1, 3)
        return float(im[:, 2].sum() / im[:, 0].sum())

    plain = warmth(devs["none"], grey(W))
    bluish = warmth(devs["half"], grey(W))
    neutral = warmth(devs["half"], grey(W_NEUTRAL))
    print(f"red over blue: none {plain:.4f}, bluish half {bluish:.4f}, neutral half {neutral:.4f}")
    assert plain == pytest.approx(1.0, abs=1e-5)
    assert bluish > 1.1
    assert neutral == pytest.approx(1.0, abs=1e-5)  # control


def test_a_white_share_needs_the_white_row(stone) -> None:
    view, devs = stone
    with pytest.raises(ValueError, match="white row"):
        render(devs["half"], view, table(A), DOME, spp=1, depth_mm=DEPTH)
    with pytest.raises(ValueError, match="white row"):
        table(A, W, white=1)  # the tube core's row
