"""The owner's look (ADR 0019 B3): the dome without the face's glare, at the photo
compare's lightness, and several lights on one filled grid. Each check has a negative
control."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import numpy as np
import pytest
from agatesim.fields import Pigment
from agatesim.volume.config import SliceConfig, SolidConfig

from nodule.render import _gpu
from nodule.render.camera import Light, render
from nodule.render.cutface import (
    F0,
    REFLECTED_EXPOSURE,
    CutRenderer,
    Rendering,
    look_exposure,
    mean_lightness,
)
from nodule.render.fakes import stone_record
from nodule.render.materials import OpticsTable
from nodule.render.record import BandRecord
from nodule.specimen import Specimen, SpecimenCut
from nodule.variety import StoneRecipe, build, cut_of
from nodule.varietyrender import CARNELIAN_LOADING, HEMATITE_RED_K, optics_for

gpu = pytest.mark.skipif(not _gpu.available(), reason="needs a CUDA GPU and CuPy")


def test_look_exposure_matches_the_compare_lightness_and_its_control() -> None:
    """A dark face without glare, exposed by `look_exposure`, has the mean lightness
    the glare-on face has at the compare's exposure. Control: the compare's exposure
    on the glare-free face leaves it darker, which the check rejects."""
    rng = np.random.default_rng(1)
    bare = rng.uniform(0.0, 0.03, (40, 50, 3))  # body reflectance of translucent bands
    face = np.zeros((40, 50), bool)
    face[5:35, 5:45] = True
    target = mean_lightness(bare + F0, face, REFLECTED_EXPOSURE)
    got = mean_lightness(bare, face, look_exposure(bare, face))
    assert abs(got - target) < 1e-3
    control = mean_lightness(bare, face, REFLECTED_EXPOSURE)
    assert abs(control - target) > 0.05


def test_hematite_red_k_changes_only_the_hematite_pigments_red_and_its_control() -> None:
    """The setting lowers the HEMATITE pigment's red absorption alone: its blue and green
    stay, and the IRON pigment (the photo compare's) is untouched. Control: without the
    setting, IRON at impurity 1 (all hematite) equals HEMATITE, which the setting breaks."""
    base, look = OpticsTable(), OpticsTable(hematite_red_k=0.05)
    hb, hl = base.pigment_mua(Pigment.HEMATITE, 0.6), look.pigment_mua(Pigment.HEMATITE, 0.6)
    assert hl[2] < 0.5 * hb[2]
    assert np.allclose(hl[:2], hb[:2])
    for x in (0.3, 0.6, 1.0):
        assert np.allclose(look.pigment_mua(Pigment.IRON, x), base.pigment_mua(Pigment.IRON, x))
    assert np.allclose(base.pigment_mua(Pigment.IRON, 1.0), base.pigment_mua(Pigment.HEMATITE, 1.0))
    assert not np.allclose(
        look.pigment_mua(Pigment.IRON, 1.0), look.pigment_mua(Pigment.HEMATITE, 1.0)
    )


def test_carnelian_alone_takes_the_lighter_loading() -> None:
    """Variety optics: every stone gets the hematite red setting; carnelian alone half the
    loading. Control: an iron stone keeps the base loading."""
    base = OpticsTable()
    carn = optics_for(small_recipe("carnelian"), base)
    iron = optics_for(small_recipe("iron"), base)
    assert carn.hematite_red_k == iron.hematite_red_k == HEMATITE_RED_K
    assert carn.phi_iron_oxide == pytest.approx(CARNELIAN_LOADING * base.phi_iron_oxide)
    assert iron.phi_iron_oxide == base.phi_iron_oxide


def test_look_needs_the_dome() -> None:
    record = object()
    with pytest.raises(ValueError, match="dome"):
        Rendering(Light("back"), record, 3.0, match_lightness=True)  # type: ignore[arg-type]


def small_recipe(colour: str = "carnelian") -> StoneRecipe:
    return StoneRecipe(
        index=0,
        seed=11,
        outline="vesicle",
        solid=SolidConfig(preset="vesicle-round", size_mm=14),
        centre="solid",
        centre_width=0.0,
        lining=0.5,
        floor=0.0,
        colour=colour,  # type: ignore[arg-type]
        band_mm=0.5,
        warp="mild",
        wobble_mm=0.1,
        wobble_scale_mm=4.0,
        white_fade=0.3,
        rim_stain=0.0,
        cut=SliceConfig(tilt_deg=80, azimuth_deg=30, offset_mm=0.5),
    )


def small_stone() -> tuple[Specimen, SpecimenCut, BandRecord]:
    r = small_recipe()
    specimen, _ = build(r)
    cut = cut_of(specimen, r, 0.25)
    return specimen, cut, stone_record(specimen, np.random.default_rng(r.seed))


@gpu
def test_glare_off_removes_exactly_the_face_reflection_and_its_control() -> None:
    """Same seed, glare off: the dome image is lower by F0 Ld everywhere, nothing else.
    Control: two glare-on renders differ by nothing, so the check (a gap of F0) fails."""
    specimen, cut, record = small_stone()
    base = CutRenderer(specimen, cut, Rendering.reflected(record, grid_mm=0.25),
                       np.random.default_rng(0))  # fmt: skip
    args = (base._dev, base._view, base._mats, Light("dome"))
    kw = dict(spp=8, seed=5, depth_mm=2.0, flags=base.rendering.flags, shift=base._shift)
    on = render(*args, **kw).image  # type: ignore[arg-type]
    off = render(*args, glare=False, **kw).image  # type: ignore[arg-type]
    assert np.allclose(on - off, F0, atol=1e-6)
    again = render(*args, **kw).image  # type: ignore[arg-type]
    assert not np.allclose(on - again, F0, atol=1e-6)


@gpu
def test_one_fill_serves_a_backlit_slice_and_its_controls() -> None:
    """A backlit slice rendered on the look's grid (filled 3 mm deep) equals one on its
    own fill, sample for sample. Controls: another seed differs (the check can fail);
    a slab deeper than the fill, or other optics, is refused."""
    specimen, cut, record = small_stone()
    look = Rendering.look(record, grid_mm=0.25)
    back = Rendering.transmitted(record, grid_mm=0.25, depth_mm=3.0)
    shared = CutRenderer(specimen, cut, look, np.random.default_rng(5), fill_depth_mm=3.0)
    via = shared.on(back)
    via.more(16)
    own = CutRenderer(specimen, cut, back, np.random.default_rng(5))
    own.more(16)
    assert np.array_equal(via.rgb, own.rgb)
    other = CutRenderer(specimen, cut, back, np.random.default_rng(6))
    other.more(16)
    assert not np.array_equal(via.rgb, other.rgb)
    with pytest.raises(ValueError, match="filled"):
        shared.on(Rendering.transmitted(record, grid_mm=0.25, depth_mm=4.0))
    with pytest.raises(ValueError, match="optics"):
        shared.on(Rendering.transmitted(record, grid_mm=0.25, optics=OpticsTable(iron_floor=0)))
    shared.more(16)  # the look itself: lightness matched, exposure set per render
    assert shared.exposure != REFLECTED_EXPOSURE
