"""Rendering a measured cut (ADR 0008 section 2; `nodule.render.cutface`).

CPU tests run in CI; the GPU tests skip there. Run locally with
`uv run --no-sync pytest`.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from agatesim.cut_shapes import Look
from agatesim.specimen.quality import luminance
from agatesim.volume.config import SliceConfig

from nodule.descriptors import PhotoCut, measure, photo_cut
from nodule.render import _gpu, fakes
from nodule.render.camera import Light, srgb
from nodule.render.cutface import (
    NOISE_SHARE,
    PILOT_SPP,
    Rendering,
    coarse_mask,
    render_cut,
    spp_for,
    to_rgb,
    view_of,
)
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


def test_samples_per_pixel_meet_the_noise_share() -> None:
    pilot, photo = 6.0, 1.5
    spp = spp_for(pilot, photo, 10_000)
    assert pilot * math.sqrt(PILOT_SPP / spp) <= NOISE_SHARE * photo
    assert pilot * math.sqrt(PILOT_SPP / (spp - 1)) > NOISE_SHARE * photo  # the fewest
    assert spp_for(pilot, photo, 64) == 64  # the cap
    assert spp_for(0.01, photo, 64) == 1
    assert spp_for(pilot, 0.0, 64) == 64  # a noiseless photo: as many as allowed


def test_coarse_mask_keeps_whole_pixels_only() -> None:
    inside = np.ones((6, 6), bool)
    inside[0, 0] = False
    m = coarse_mask(inside, 2, 3, 3)
    assert m.shape == (3, 3)
    assert not m[0, 0]
    assert m.sum() == 8


def test_rendering_takes_the_stratum_light() -> None:
    from nodule.render.record import BandRecord

    rec = BandRecord.empty()
    r1, r3 = Rendering.reflected(rec), Rendering.transmitted(rec, exposure=2.0)
    assert (r1.light.kind, r1.depth_mm) == ("dome", 10.0)
    assert (r3.light.kind, r3.depth_mm, r3.exposure) == ("back", 3.0, 2.0)
    assert Rendering.reflected(rec, depth_mm=4.0).depth_mm == 4.0


# --- GPU ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rendering(specimen: Specimen) -> Rendering:
    record = fakes.stone_record(specimen, np.random.default_rng(3), points=200_000)  # ~0.3 s
    return Rendering.reflected(record, depth_mm=2.0, grid_mm=0.5)  # small, for test time


@gpu
def test_rendered_cut_is_seeded_and_as_noisy_as_estimated(
    specimen: Specimen, rendering: Rendering
) -> None:
    cut = specimen.cut(specimen.plane(CUT), 0.25)
    a = render_cut(specimen, cut, rendering, LOOK.noise, seed=11)
    assert a.rgb.shape == (*cut.inside.shape, 3)
    assert luminance(a.rgb)[cut.inside].std() > 5  # the bands show
    assert (render_cut(specimen, cut, rendering, LOOK.noise, seed=11).rgb == a.rgb).all()
    b = render_cut(specimen, cut, rendering, LOOK.noise, seed=97)
    assert a.spp == b.spp
    diff = luminance(a.rgb).astype(np.float64) - luminance(b.rgb)
    measured = float(np.std(diff[cut.inside]) / math.sqrt(2))
    assert measured <= 1.2 * a.noise  # the pilot errs high (larger pixels)
    if not a.capped:
        assert measured <= 1.2 * NOISE_SHARE * LOOK.noise
    # control: one sample per pixel is flagged and noisier than asked
    one = render_cut(specimen, cut, Rendering.reflected(rendering.record, depth_mm=2.0,
                     grid_mm=0.5, most_spp=1), LOOK.noise, seed=11)  # fmt: skip
    assert one.spp == 1
    assert one.capped
    assert one.noise > NOISE_SHARE * LOOK.noise


@gpu
def test_rendered_path_is_measured_and_f5_skips_the_blur(
    specimen: Specimen, rendering: Rendering
) -> None:
    pc = photo_cut(specimen, specimen.plane(CUT), LOOK, 384, rendering, seed=5)
    assert isinstance(pc, PhotoCut)
    assert pc.rendered is not None
    assert pc.rgb.shape == (*pc.cut.inside.shape, 3)
    out = measure(pc, LOOK, np.random.default_rng(0))
    assert out["bands"] is True
    assert isinstance(out["C6_count"], float)
    f5 = measure(pc, LOOK, np.random.default_rng(0), degrade=False)
    assert f5["sigma_px"] == 0.0
    assert f5["matched"] is None
    blurry = measure(pc, replace(LOOK, edge_per_spacing=1.0), np.random.default_rng(0))
    sigma = blurry["sigma_px"]
    assert isinstance(sigma, float)
    assert sigma > 0
