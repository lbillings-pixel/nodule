"""Band descriptors of the specimen's cuts (N2 step 3b), with negative controls."""

from __future__ import annotations

import numpy as np
import pytest
from agatesim.cut_shapes import Look
from agatesim.render import VOID
from agatesim.volume.config import SliceConfig

from nodule.descriptors import PhotoCut, bandless, measure, photo_cut, stand_in_rgb
from nodule.specimen import Specimen, load_preset

# A median library look (2026-10-31: edge/spacing 0.38, noise 1.35, spacing 9.9 px),
# fixed here as CI has no library cache.
LOOK = Look(photo="test", edge_per_spacing=0.38, noise=1.35, spacing_px=9.9)
MOST = 768  # pixels across, kept small for test time


@pytest.fixture(scope="module")
def specimen() -> Specimen:
    return Specimen.build(load_preset("n2-slice"))


@pytest.fixture(scope="module")
def central(specimen: Specimen) -> PhotoCut:
    plane = specimen.plane(SliceConfig(tilt_deg=90, azimuth_deg=270, offset_mm=0))
    return photo_cut(specimen, plane, LOOK, MOST)


def test_stand_in_draws_the_stone_and_dark_surroundings(central: PhotoCut) -> None:
    rgb, inside = central.rgb, central.cut.inside
    assert rgb.shape == (*inside.shape, 3)
    assert (rgb[~inside] == VOID.astype(np.uint8)).all()
    grey = rgb[inside].mean(axis=1)
    assert grey.std() > 5  # the bands differ in translucency, so the face is not even


def test_cut_is_sized_like_the_photo(central: PhotoCut) -> None:
    assert 256 <= central.pixels <= MOST


def test_banded_cut_is_measured(central: PhotoCut) -> None:
    out = measure(central, LOOK, np.random.default_rng(0))
    assert out["bands"] is True
    for name in ("C7_spread", "C9_ratio", "C11_hurst", "C6_count"):
        assert isinstance(out[name], float), name
    count = out["C6_count"]
    assert isinstance(count, float)
    assert count >= 3
    assert out["C17_core"] == 0.0  # the preset has no quartz core
    for name in ("C16_coherence", "C16_order", "C16_corr", "C16_parallel", "C16_banded"):
        assert out[name] is None or isinstance(out[name], float), name
    assert isinstance(out["C16_coherence"], float)
    assert isinstance(out["C16_corr_censored"], bool)


def test_measurement_repeats_with_its_seed(central: PhotoCut) -> None:
    a = measure(central, LOOK, np.random.default_rng(1))
    b = measure(central, LOOK, np.random.default_rng(1))
    assert a == b


def test_negative_control_bandless_cut_gives_no_bands(central: PhotoCut) -> None:
    flat = bandless(central.cut)
    pc = PhotoCut(flat, stand_in_rgb(flat), central.pixels)
    out = measure(pc, LOOK, np.random.default_rng(0))
    assert out["level"] == "not fit"  # the gate rejects the face
    assert out["C6_count"] in (None, 0.0)
    for name in ("C7_spread", "C8_trend", "C9_ratio", "C10_persistence", "C11_hurst", "C16_order"):
        assert out[name] is None, name
