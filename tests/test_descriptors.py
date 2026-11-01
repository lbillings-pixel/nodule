"""Band descriptors of the specimen's cuts (N2 step 3b), with negative controls."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from agatesim.cut_shapes import Look
from agatesim.render import VOID
from agatesim.volume.config import SliceConfig

from nodule.compare import library_table
from nodule.descriptors import (
    PhotoCut,
    bandless,
    measure,
    photo_cut,
    rendering_for,
    stand_in_rgb,
    stratum_looks,
)
from nodule.render.cutface import Rendering
from nodule.render.record import FIELDS
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
    for name in ("C15_curv", "C15_curv_high"):
        assert out[name] is None or isinstance(out[name], float), name
    assert out["C15_corners"] is None or isinstance(out["C15_corners"], int)
    assert out["C14_density"] is None or isinstance(out["C14_density"], float)
    for name in ("C12_corr", "C12_near", "C13_drift", "C13_complete"):
        assert out[name] is None or isinstance(out[name], float), name
    for name in ("C13_sets", "C13_depth"):
        assert out[name] is None or isinstance(out[name], int), name
    for name in ("C14_ends", "C14_junctions"):
        assert out[name] is None or isinstance(out[name], int), name


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
    for name in ("C15_curv", "C15_curv_high", "C15_corners", "C14_density", "C14_ends"):
        assert out[name] is None, name
    assert out["C12_corr"] is None
    assert out["C13_sets"] is None


def test_f5_measures_the_image_as_it_is(central: PhotoCut) -> None:
    out = measure(central, LOOK, np.random.default_rng(0), degrade=False)
    assert out["sigma_px"] == 0.0
    assert out["matched"] is None
    assert out["bands"] is True
    blurry = replace(LOOK, edge_per_spacing=1.0)  # a soft photo, wider edges than the cut's
    blurred = measure(central, blurry, np.random.default_rng(0))
    sigma = blurred["sigma_px"]
    assert isinstance(sigma, float)
    assert sigma > 0
    assert isinstance(blurred["matched"], bool)


def test_stone_and_fakes_share_one_record(specimen: Specimen) -> None:
    def rendering(run: str, stratum: str = "reflected") -> Rendering:
        return rendering_for(specimen, run, stratum, np.random.default_rng(4), points=100_000)

    stone, f5 = rendering("stone"), rendering("F5")  # ~0.2 s each
    assert not stone.undegraded
    assert f5.undegraded
    for name in ("seen", *FIELDS):
        assert np.array_equal(getattr(stone.record, name), getattr(f5.record, name)), name
    for run in ("F1", "F3"):  # control: a fake changes the record
        fake = rendering(run).record
        assert not np.array_equal(fake.impurity, stone.record.impurity), run
    assert rendering("stone", "transmitted").light.kind == "back"
    with pytest.raises(ValueError, match="no rendered run"):
        rendering("F9")


def test_stratum_looks_keep_its_photos_only() -> None:
    slab = sorted(library_table("transmitted").ids)[0]
    face = sorted(library_table("reflected").ids)[0]
    looks = [Look(photo=p, edge_per_spacing=0.4, noise=1.0, spacing_px=9.0) for p in (slab, face)]
    assert [lk.photo for lk in stratum_looks(looks, "transmitted")] == [slab]
    assert [lk.photo for lk in stratum_looks(looks, "reflected")] == [face]
