"""Rendering a measured cut (ADR 0008 section 2 and amendment 5; `nodule.render.cutface`).

CPU tests run in CI; the GPU tests skip there. Run locally with
`uv run --no-sync pytest`.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from agatesim._scipy import gaussian
from agatesim.cut_shapes import Look
from agatesim.specimen.quality import luminance
from agatesim.volume.config import SliceConfig

from nodule._scipy import blur_noise_factor
from nodule.descriptors import PhotoCut, measure, noise_to_add, photo_cut
from nodule.render import _gpu, fakes
from nodule.render.camera import Flag, Light, srgb
from nodule.render.cutface import (
    REFLECTED_EXPOSURE,
    CutRender,
    CutRenderer,
    Rendering,
    half_noise,
    spp_for,
    to_rgb,
    view_of,
)
from nodule.render.record import BandRecord
from nodule.specimen import Specimen, load_preset

CUT = SliceConfig(tilt_deg=90.0, azimuth_deg=270.0, offset_mm=0.0)
LOOK = Look(photo="test", edge_per_spacing=0.38, noise=1.35, spacing_px=9.9)  # as test_descriptors
gpu = pytest.mark.skipif(not _gpu.available(), reason="needs a CUDA GPU and CuPy")


@pytest.fixture(scope="module")
def specimen() -> Specimen:
    return Specimen.build(load_preset("n2-slice"))


# --- CPU ------------------------------------------------------------------------------------


def test_view_pixel_centres_are_the_cut_points(specimen: Specimen) -> None:
    cut = specimen.cut(specimen.plane(CUT), 1.0)
    v = view_of(cut)
    assert (v.nx, v.ny) == (len(cut.s), len(cut.t))
    centres_s = v.s0 + (np.arange(v.nx) + 0.5) * v.pixel_mm
    centres_t = v.t0 + (np.arange(v.ny) + 0.5) * v.pixel_mm
    assert np.allclose(centres_s, cut.s)
    assert np.allclose(centres_t, cut.t)
    # control: pixel edges at the cut points would sit half a pixel off
    assert not np.allclose(v.s0 + np.arange(v.nx) * v.pixel_mm + v.pixel_mm, cut.s)


def test_rgb_keeps_the_cut_row_order() -> None:
    image = np.zeros((4, 3, 3))
    image[0] = 0.5  # row 0 = the cut's first t
    dome = Light("dome")
    assert (to_rgb(image, dome, 1.0)[0] > 0).all()
    assert (to_rgb(image, dome, 1.0)[1:] == 0).all()
    assert (srgb(image)[0] == 0).all()  # control: the display order is flipped


def test_samples_per_pixel_reach_the_wanted_noise() -> None:
    spp = spp_for(6.0, 256, 2.0, 100_000)
    assert spp == 2304  # 256 x (6 / 2)^2, even
    assert 6.0 * math.sqrt(256 / spp) <= 2.0
    assert spp_for(6.0, 256, 2.0, 1024) == 1024  # the cap
    assert spp_for(1.0, 256, 2.0, 1024) == 256  # never fewer than already rendered
    assert spp_for(6.0, 256, 0.0, 1024) == 1024  # a noiseless target: as many as allowed
    assert spp_for(6.0, 256, 5.9, 100_000) % 2 == 0  # two equal halves


def test_half_noise_is_the_noise_of_the_mean() -> None:
    g = np.random.default_rng(1)
    base = np.full((200, 200, 3), 120.0)
    sigma = 4.0  # per render; their mean has sigma / sqrt(2)
    a, b = (np.clip(np.rint(base + g.normal(0, sigma, (200, 200, 1))), 0, 255).astype(np.uint8)
            for _ in range(2))  # fmt: skip
    face = np.ones((200, 200), bool)
    assert half_noise(a, b, face) == pytest.approx(sigma / math.sqrt(2), rel=0.03)
    assert half_noise(a, a, face) == 0.0  # control: one render twice has no measurable noise


@pytest.mark.parametrize("sigma", [0.6, 0.8, 1.5, 3.0])
def test_blur_noise_factor_matches_blurred_white_noise(sigma: float) -> None:
    x = np.random.default_rng(2).normal(0, 1, (600, 600))
    measured = float(gaussian(x, sigma).std())
    assert blur_noise_factor(sigma) == pytest.approx(measured, rel=0.03)
    # control: a 1D blur's factor (the square root) is wrong in 2D
    assert abs(math.sqrt(blur_noise_factor(sigma)) - measured) > 0.1 * measured
    assert blur_noise_factor(0.0) == 1.0


def test_only_the_lacking_noise_is_added() -> None:
    def rendered(blurred: float | None) -> CutRender:
        return CutRender(256, 3.0, blurred, None if blurred is None else blurred <= 1.5, 1.0, 1.0)

    look = replace(LOOK, noise=1.5)
    added = noise_to_add(look, rendered(0.9)).noise
    assert added == pytest.approx(1.2)
    assert math.hypot(added, 0.9) == pytest.approx(1.5)  # in quadrature: the photo's
    assert noise_to_add(look, rendered(2.0)).noise == 0.0  # noisier than the photo: none
    assert noise_to_add(look, rendered(None)).noise == 1.5  # no blur found: the photo's
    assert noise_to_add(look, None).noise == 1.5  # the stand-in: the photo's


def test_rendering_takes_the_stratum_light() -> None:
    rec = BandRecord.empty()
    r1, r3 = Rendering.reflected(rec), Rendering.transmitted(rec, exposure=2.0)
    assert (r1.light.kind, r1.depth_mm) == ("dome", 2.0)  # ADR 0010 10b
    assert (r3.light.kind, r3.depth_mm, r3.exposure) == ("back", 3.0, 2.0)
    assert Rendering.reflected(rec, depth_mm=4.0).depth_mm == 4.0


def test_exposure_is_set_per_light() -> None:
    """ADR 0013: reflected faces at 2.2, slabs at 1.0 until calibrated; an explicit
    exposure still wins, so 1.0 gives back the step-2 renders (negative control)."""
    rec = BandRecord.empty()
    assert REFLECTED_EXPOSURE == 2.2
    assert Rendering.reflected(rec).exposure == REFLECTED_EXPOSURE
    assert Rendering.transmitted(rec).exposure == 1.0
    assert Rendering.reflected(rec, exposure=1.0).exposure == 1.0


def test_bottom_is_air_unless_asked() -> None:
    """ADR 0014: the black bottom is an option, off by default, so earlier renders
    are unchanged; the back light cannot have one."""
    rec = BandRecord.empty()
    assert Rendering.reflected(rec).bottom == "air"
    assert Rendering.reflected(rec).flags == 0
    assert Rendering.reflected(rec, bottom="black").flags == Flag.BLACK_BOTTOM
    with pytest.raises(ValueError, match="bottom must be"):
        Rendering.reflected(rec, bottom="grey")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="back light"):
        Rendering.transmitted(rec, bottom="black")


# --- GPU ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rendering(specimen: Specimen) -> Rendering:
    record = fakes.stone_record(specimen, np.random.default_rng(3), points=200_000)  # ~0.3 s
    # small, for test time: coarse grid, shallow, few samples
    return Rendering.reflected(record, depth_mm=2.0, grid_mm=0.5, first_spp=32, most_spp=128)


@gpu
def test_renderer_measures_its_own_noise(specimen: Specimen, rendering: Rendering) -> None:
    cut = specimen.cut(specimen.plane(CUT), 0.25)

    def made(seed: int, spp: int) -> CutRenderer:
        r = CutRenderer(specimen, cut, rendering, np.random.default_rng(seed))
        r.more(spp)
        return r

    a, b = made(1, 32), made(2, 32)
    assert a.rgb.shape == (*cut.inside.shape, 3)
    assert luminance(a.rgb)[cut.inside].std() > 5  # the bands show
    assert (made(1, 32).rgb == a.rgb).all()  # seeded
    diff = luminance(a.rgb).astype(np.float64) - luminance(b.rgb)
    measured = float(np.std(diff[cut.inside]) / math.sqrt(2))
    assert a.noise == pytest.approx(measured, rel=0.1)
    a.more(128)  # four times the samples: half the noise
    assert a.spp == 128
    assert a.noise == pytest.approx(measured / 2, rel=0.15)
    # control: the mean of the halves is less noisy than either half
    assert a.noise < measured


@gpu
def test_rendered_path_is_measured_and_f5_skips_the_blur(
    specimen: Specimen, rendering: Rendering
) -> None:
    pc = photo_cut(specimen, specimen.plane(CUT), LOOK, 384, rendering, np.random.default_rng(5))
    assert isinstance(pc, PhotoCut)
    r = pc.rendered
    assert r is not None
    assert pc.rgb.shape == (*pc.cut.inside.shape, 3)
    assert 32 <= r.spp <= 128
    assert r.blurred_noise is not None
    assert r.blurred_noise < r.noise  # the blur lowers it
    out = measure(pc, LOOK, np.random.default_rng(0))
    assert out["bands"] is True
    assert isinstance(out["C6_count"], float)
    f5 = measure(pc, LOOK, np.random.default_rng(0), degrade=False)
    assert f5["sigma_px"] == 0.0
    assert f5["matched"] is None
    blurry = measure(pc, replace(LOOK, edge_per_spacing=1.0), np.random.default_rng(0))
    sigma = blurry["sigma_px"]
    assert isinstance(sigma, float)
    assert sigma > 0  # control: the photo's look blurs
