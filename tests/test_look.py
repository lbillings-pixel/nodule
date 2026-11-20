"""The owner's look (ADR 0019 B3): the dome without the face's glare, at the photo
compare's lightness, and several lights on one filled grid. Each check has a negative
control."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import math

import numpy as np
import pytest
from agatesim.fields import Pigment
from agatesim.volume.config import SliceConfig, SolidConfig

import nodule.varietyrender as varietyrender
from nodule.crystals import Crystals
from nodule.plumes import STAGE
from nodule.render import _gpu
from nodule.render.camera import Light, render, rolloff
from nodule.render.cutface import (
    F0,
    LOOK_DEPTH_MM,
    REFLECTED_EXPOSURE,
    CutRenderer,
    Rendering,
    look_exposure,
    mean_lightness,
    to_rgb,
)
from nodule.render.fakes import stone_record
from nodule.render.materials import OpticsTable
from nodule.render.record import STEPS, BandRecord
from nodule.specimen import Specimen, SpecimenCut
from nodule.variety import StoneRecipe, build, cut_of
from nodule.varietyrender import (
    BACKLIT_EXPOSURE,
    BACKLIT_KNEE,
    CARNELIAN_LOADING,
    CLOSE_LOOK_MM,
    FILL_MM,
    HEMATITE_RED_K,
    NEEDLE_GRID_MM,
    NEEDLE_VOXELS,
    QUARTZ_TINT_MUA,
    SLICE_MM,
    TINT_STRENGTH,
    WHOLE_GRID_MM,
    optics_for,
    whole_grid_mm,
)

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


def test_tinted_centres_take_their_tint_after_the_early_stage() -> None:
    """Amethyst and smoky stones get their colour centres' absorption on quartz after the
    early stage (clear early crystals stay clear). Control: an untinted stone gets none."""
    first = math.ceil(STAGE * STEPS - 1e-9)
    assert first >= 1
    for tint in ("amethyst", "smoky"):
        o = optics_for(small_recipe().model_copy(update={"tint": tint}))
        want = [TINT_STRENGTH * v for v in QUARTZ_TINT_MUA[tint]]
        assert o.quartz_tint_mua == pytest.approx(want)
        assert o.quartz_tint_from == first
    amethyst = QUARTZ_TINT_MUA["amethyst"]
    assert amethyst[1] > max(amethyst[0], amethyst[2])  # absorbs green: purple
    plain = optics_for(small_recipe())
    assert plain.quartz_tint_mua == (0.0, 0.0, 0.0)


def test_white_fine_needles_alone_take_a_finer_grid_within_the_budget() -> None:
    """White fine needles render on NEEDLE_GRID_MM, coarsened for a big face so the
    grid stays within NEEDLE_VOXELS, never past WHOLE_GRID_MM. Controls: blades,
    hollow needles and stones without crystals keep WHOLE_GRID_MM."""

    class Face:
        def __init__(self, mm: float) -> None:
            self.s = np.linspace(0.0, mm, 50)
            self.t = np.linspace(0.0, mm, 50)

    needles = Crystals(habit="fine needles", fill="white", spacing_mm=1.0, length_mm=0.9,
                       elongation=12.0, scatter_deg=45.0)  # fmt: skip
    r = small_recipe().model_copy(update={"crystals": needles})
    assert whole_grid_mm(r, Face(30.0)) == NEEDLE_GRID_MM  # type: ignore[arg-type]
    big = whole_grid_mm(r, Face(80.0))  # type: ignore[arg-type]
    assert NEEDLE_GRID_MM < big <= WHOLE_GRID_MM
    assert max(SLICE_MM, LOOK_DEPTH_MM) == FILL_MM  # the grid serves the look and the slice
    assert (80.0 + 2 * FILL_MM) ** 2 * FILL_MM / big**3 == pytest.approx(NEEDLE_VOXELS)
    for c in (needles.model_copy(update={"fill": "hollow"}),
              needles.model_copy(update={"habit": "small blades"}), None):  # fmt: skip
        other = small_recipe().model_copy(update={"crystals": c})
        assert whole_grid_mm(other, Face(30.0)) == WHOLE_GRID_MM  # type: ignore[arg-type]


def test_close_ups_keep_the_shallow_look(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whole faces render the look LOOK_DEPTH_MM deep on a FILL_MM grid; close-ups keep
    CLOSE_LOOK_MM on the slice's depth (owner, ADR 0019 s.6.10). Control: the two calls
    differ, so the check can tell them apart. With the stone's void box the look draws
    the hollow as air and is handed the box (ADR 0019 s.6.12 step 4); control: without
    one, the stand-in and no box."""
    seen: list[tuple[float, float, str, object]] = []

    class StopError(Exception):
        pass

    def fake(_sp: object, _cut: object, rendering: Rendering, _rng: object,
             fill_depth_mm: float, box: object = None) -> None:  # fmt: skip
        seen.append((rendering.depth_mm, fill_depth_mm, rendering.hollow, box))
        raise StopError

    monkeypatch.setattr(varietyrender, "CutRenderer", fake)
    rec, rng = BandRecord.empty(), np.random.default_rng(0)
    args = (None, None, small_recipe(), rec, 0.25, OpticsTable(), 2, rng, None)
    box = object()  # stands in for a VoidBox; the fake never reads it
    for extra in (("whole",), ("close", CLOSE_LOOK_MM), ("whole", LOOK_DEPTH_MM, box)):
        with pytest.raises(StopError):
            varietyrender.views(*args, *extra)  # type: ignore[arg-type]
    assert seen[0] == (LOOK_DEPTH_MM, FILL_MM, "stand-in", None)
    assert seen[1] == (CLOSE_LOOK_MM, SLICE_MM, "stand-in", None)
    assert seen[0][:2] == (5.0, 5.0)
    assert seen[1][:2] == (2.0, 3.0)
    assert seen[0] != seen[1]
    assert seen[2] == (LOOK_DEPTH_MM, FILL_MM, "air", box)


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
    """A backlit slice rendered on the look's grid (filled FILL_MM deep, as
    `varietyrender.views`) equals one on its own fill of that depth, sample for sample.
    Controls: another seed differs (the check can fail); a slab deeper than the fill, or
    other optics, is refused."""
    specimen, cut, record = small_stone()
    look = Rendering.look(record, grid_mm=0.25)
    back = Rendering.transmitted(record, grid_mm=0.25, depth_mm=SLICE_MM)
    shared = CutRenderer(specimen, cut, look, np.random.default_rng(5), fill_depth_mm=FILL_MM)
    via = shared.on(back)
    via.more(16)
    own = CutRenderer(specimen, cut, back, np.random.default_rng(5), fill_depth_mm=FILL_MM)
    own.more(16)
    assert np.array_equal(via.rgb, own.rgb)
    other = CutRenderer(specimen, cut, back, np.random.default_rng(6))
    other.more(16)
    assert not np.array_equal(via.rgb, other.rgb)
    with pytest.raises(ValueError, match="filled"):
        shared.on(Rendering.transmitted(record, grid_mm=0.25, depth_mm=FILL_MM + 1.0))
    with pytest.raises(ValueError, match="optics"):
        shared.on(Rendering.transmitted(record, grid_mm=0.25, optics=OpticsTable(iron_floor=0)))
    shared.more(16)  # the look itself: lightness matched, exposure set per render
    assert shared.exposure != REFLECTED_EXPOSURE


# --- the backlit view's highlight roll-off (ADR 0019 s.6.6) ---------------------------------


def test_rolloff_keeps_values_below_the_knee_and_makes_the_light_box_white() -> None:
    """Below the knee nothing changes (carnelian's colour); the bare light box (factor 1,
    x = exposure) maps to white; the curve rises smoothly (slope 1 at the knee)."""
    x = np.linspace(0.0, 3.0, 3001)
    y = rolloff(x, BACKLIT_KNEE, 3.0)
    below = x <= BACKLIT_KNEE
    assert np.array_equal(y[below], x[below])
    assert y[-1] == pytest.approx(1.0)
    assert (np.diff(y) > 0).all()
    h = 1e-6
    k = BACKLIT_KNEE
    slope = (rolloff(np.array([k + h]), k, 3.0) - k) / h
    assert slope[0] == pytest.approx(1.0, abs=1e-4)
    with pytest.raises(ValueError, match="knee"):
        rolloff(x, 0.6, 1.0)  # no highlight above white to roll off


def test_rolloff_keeps_bright_bands_apart_and_clipping_does_not() -> None:
    """Clear bands passing 40-90 % of the light at exposure 3 differ on screen after the
    roll-off (distinct grey levels, 239-254: only a few apart near the top, none white).
    Control: plain clipping makes them all white, so the same check rejects it."""

    def apart(levels: np.ndarray) -> bool:
        return bool((np.diff(levels) >= 1).all()) and int(levels.max()) < 255

    light = Light("back")
    bands = np.repeat(np.linspace(0.4, 0.9, 6), 3).reshape(1, 6, 3) * light.white()
    assert apart(to_rgb(bands, light, BACKLIT_EXPOSURE, BACKLIT_KNEE)[0, :, 0].astype(int))
    assert not apart(to_rgb(bands, light, BACKLIT_EXPOSURE)[0, :, 0].astype(int))


def test_rolloff_is_for_the_backlit_view_only() -> None:
    """The variety backlit view takes it; the dome and the compare's L3 do not."""
    rec = BandRecord.empty()
    back = Rendering.transmitted(rec, exposure=BACKLIT_EXPOSURE, rolloff_knee=BACKLIT_KNEE)
    assert back.rolloff_knee == BACKLIT_KNEE
    assert Rendering.transmitted(rec).rolloff_knee is None
    with pytest.raises(ValueError, match="back light only"):
        Rendering.look(rec, rolloff_knee=0.6)
    with pytest.raises(ValueError, match="exposure"):
        Rendering.transmitted(rec, rolloff_knee=0.6)  # exposure 1: nothing to roll off
