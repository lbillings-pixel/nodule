"""White opal by place (ADR 0019 s.13) off the GPU: the white row's optics, the share
grid (white plumes take the cover; rims RIM_MM wide at any spacing, exact in chunks),
the recipe's draw and its mapping to the optics. Each check with its negative control."""

from __future__ import annotations

import numpy as np
import pytest
from agatesim.volume.config import SliceConfig, SolidConfig

from nodule._scipy import distance_inside
from nodule.moss import draw_moss
from nodule.plumes import draw_plumes
from nodule.render import whiteopal
from nodule.render.camera import Materials
from nodule.render.materials import WAVELENGTH_NM
from nodule.render.whiteopal import RIM_MM, rim_ramp, white_optics, white_share
from nodule.variety import StoneRecipe
from nodule.variety.draw import white_of
from nodule.variety.recipe import WHITE_PLUME_SHARE, WHITE_RIM_LEVEL, WHITE_RIM_SHARE
from nodule.varietyrender import WHITE_FAINT, WHITE_SOLID, optics_for, white_transport


def bare() -> StoneRecipe:
    return StoneRecipe(
        index=0,
        seed=11,
        outline="vesicle",
        solid=SolidConfig(preset="vesicle-round", size_mm=14),
        centre="solid",
        centre_width=0.0,
        lining=0.5,
        floor=0.0,
        colour="iron",
        band_mm=0.5,
        warp="mild",
        wobble_mm=0.1,
        wobble_scale_mm=4.0,
        white_fade=0.3,
        rim_stain=0.0,
        cut=SliceConfig(tilt_deg=80, azimuth_deg=30, offset_mm=0.5),
    )


def with_plumes() -> StoneRecipe:
    return bare().model_copy(update={"plumes": draw_plumes(np.random.default_rng(3))})


def with_moss() -> StoneRecipe:
    moss = draw_moss(np.random.default_rng(3), True, 0.0, 0.0)
    return bare().model_copy(update={"moss": moss})


# --- the white row ------------------------------------------------------------------------


def test_the_white_row_is_bluish_and_set_at_550(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sub-micron air pores in opal scatter blue most, so the shadow backlit is tan; the
    transport at 550 nm is the one asked for; the probe's rows (s.13, probe.log).
    Control: micron pores (the probe's other kind) scatter near neutral."""
    mus, g = white_optics(16.0, WAVELENGTH_NM)
    assert mus[1] * (1 - g[1]) == pytest.approx(16.0)
    assert np.allclose(mus, [47.16, 30.26, 20.30], atol=0.01)
    assert np.allclose(g, [0.574, 0.471, 0.385], atol=0.001)
    assert mus[0] > 2 * mus[2]
    four, _ = white_optics(4.0, WAVELENGTH_NM)
    assert np.allclose(four, mus / 4)
    with pytest.raises(ValueError, match="positive"):
        white_optics(0.0, WAVELENGTH_NM)
    monkeypatch.setattr(whiteopal, "PORE_RADII_UM", (1.0, 1.5, 2.0, 2.5))
    micron, _ = white_optics(16.0, WAVELENGTH_NM)
    assert micron[0] < 1.1 * micron[2]  # control


def test_the_white_row_takes_no_shift() -> None:
    mua, mus, g = [[0.1] * 3] * 3, [[1.0] * 3] * 3, [0.9] * 3
    assert Materials.of(mua, mus, g, white=2).white == 2
    with pytest.raises(ValueError, match="no shift"):
        Materials.of(mua, mus, g, along=[[-1, 0, 0, 0], [-1, 0, 0, 0], [0.5, 0, 0, 0]], white=2)
    with pytest.raises(ValueError, match="stone row"):
        Materials.of(mua, mus, g, white=1)


# --- the share ----------------------------------------------------------------------------


def test_white_plumes_take_the_cover_on_the_stone() -> None:
    """A white plume's share is each node's cover, on the stone's nodes only. Controls:
    no white, no cover, or no covered node give no share."""
    rng = np.random.default_rng(1)
    shape = (4, 6, 8)
    cover = np.where(rng.uniform(size=shape) < 0.3, rng.uniform(size=shape), 0.0)
    cover = cover.astype(np.float32)
    values = np.zeros(shape, np.float32)
    values[:, 0] = np.nan  # off the stone, a tube or a hollow
    share = white_share("plumes", cover, values, 0.1)
    assert share is not None
    assert np.array_equal(share[:, 1:], cover[:, 1:])
    assert not share[:, 0].any()
    assert white_share("none", cover, values, 0.1) is None
    assert white_share("plumes", None, values, 0.1) is None
    assert white_share("plumes", np.zeros_like(cover), values, 0.1) is None


def flat_face(h: float, face_mm: float) -> np.ndarray:
    """The cover of nodes at x = k h with a plume below a flat face at face_mm."""
    x = np.arange(64) * h
    return np.clip((face_mm - x) / h + 0.5, 0.0, 1.0).astype(np.float32)[None, None, :]


def test_a_rim_is_rim_mm_wide_at_any_spacing() -> None:
    """Over where a flat face falls between nodes, the rim beyond it is RIM_MM wide on
    average at 0.05, 0.1 and 0.2 mm, within a node each time. Control: the probe's hard
    rim (share 1 out to RIM_MM from the half-covered nodes) is 2/3 as wide at 0.2 mm."""
    for h in (0.05, 0.1, 0.2):
        widths: list[float] = []
        for k in range(200):
            cover = flat_face(h, 2.0 + (k + 0.5) / 200 * h)
            share = white_share("rims", cover, np.zeros_like(cover), h)
            assert share is not None
            widths.append(float(share.sum()) * h)
        assert np.mean(widths) == pytest.approx(RIM_MM, rel=0.01)
        assert np.ptp(widths) <= h + 1e-6
    hard: list[float] = []
    for k in range(200):
        seed = flat_face(0.2, 2.0 + (k + 0.5) / 200 * 0.2) >= 0.5
        d = distance_inside(~seed) * 0.2
        hard.append(float(((d > 0) & (d <= RIM_MM + 1e-9)).sum()) * 0.2)
    assert np.mean(hard) < 0.75 * RIM_MM  # control


def test_the_rim_ramp_is_exact_in_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Worked in padded chunks, the ramp equals the whole grid's. Control: the same
    chunks without their padding miss seeds across their edges."""
    seed = np.random.default_rng(2).uniform(size=(5, 40, 30)) < 0.01
    reach = 3.0
    want = np.clip(reach + 1.0 - distance_inside(~seed), 0.0, 1.0)
    monkeypatch.setattr(whiteopal, "CHUNK_ROWS", 4)
    assert np.allclose(rim_ramp(seed, reach), want, atol=1e-6)
    unpadded = np.zeros_like(want)
    for a in range(0, 40, 4):
        sub = seed[:, a : a + 4]
        if sub.any():
            unpadded[:, a : a + 4] = np.clip(reach + 1.0 - distance_inside(~sub), 0.0, 1.0)
    assert not np.allclose(unpadded, want, atol=1e-6)  # control


def test_rims_take_what_the_strand_leaves_and_stay_on_the_stone() -> None:
    """A rim takes 1 - cover: none inside a strand, the rest of a node its face crosses,
    all of the nodes within RIM_MM beyond, none farther or off the stone (a hollow
    beside it). Control: a stone without strands has no rim."""
    cover = np.zeros((3, 20, 20), np.float32)
    cover[:, 8:12, 8:12] = 1.0
    cover[:, 12, 8:12] = 0.4
    values = np.zeros(cover.shape, np.float32)
    values[:, 14:] = np.nan
    share = white_share("rims", cover, values, 0.1)
    assert share is not None
    assert not share[:, 8:12, 8:12].any()
    assert np.allclose(share[:, 12, 8:12], 0.6)
    assert (share[:, 13, 8:12] == 1).all()
    assert (share[:, 8:12, 5:8] == 1).all()
    assert not share[:, 8:12, :5].any()
    assert not share[:, 14:].any()
    assert white_share("rims", np.zeros_like(cover), values, 0.1) is None  # control


# --- the recipe ---------------------------------------------------------------------------


def test_white_opal_rates_on_the_draw() -> None:
    """Over many draws: 1 in 4 plume stones take white plumes, their levels spread over
    0..1; 1 in 4 plume or moss stones take white rims at WHITE_RIM_LEVEL; never both.
    Control: a stone without plumes or moss takes none."""
    u = np.random.default_rng(4).uniform(size=(20000, 2))
    plume = [white_of(x, with_plumes()) for x in u]
    kinds = [k for k, _ in plume]
    assert kinds.count("plumes") / len(u) == pytest.approx(WHITE_PLUME_SHARE, abs=0.01)
    assert kinds.count("rims") / len(u) == pytest.approx(WHITE_RIM_SHARE, abs=0.01)
    levels = np.array([v for k, v in plume if k == "plumes"])
    assert levels.min() < 0.02
    assert levels.max() > 0.98
    assert np.mean(levels) == pytest.approx(0.5, abs=0.02)
    assert all(v == WHITE_RIM_LEVEL for k, v in plume if k == "rims")
    moss = [k for k, _ in (white_of(x, with_moss()) for x in u)]
    assert "plumes" not in moss
    assert moss.count("rims") / len(u) == pytest.approx(WHITE_RIM_SHARE, abs=0.01)
    assert all(white_of(x, bare()) == ("none", 0.0) for x in u[:2000])  # control


def test_sheets_draw_white_on_plume_and_moss_stones_only(sheets: list[StoneRecipe]) -> None:
    whites = [r for r in sheets if r.white != "none"]
    assert len(whites) >= 3
    assert all(r.inclusions is not None for r in whites)
    assert all(r.plumes is not None for r in sheets if r.white == "plumes")
    assert all(r.white_level == WHITE_RIM_LEVEL for r in sheets if r.white == "rims")
    assert all(r.white_level == 0.0 for r in sheets if r.white == "none")


def test_the_recipe_refuses_white_without_its_inclusions() -> None:
    plume = StoneRecipe.model_validate(
        with_plumes().model_dump() | {"white": "plumes", "white_level": 0.5}
    )
    assert "white plumes 0.50" in plume.summary()
    rims = StoneRecipe.model_validate(
        with_moss().model_dump() | {"white": "rims", "white_level": 1.0}
    )
    assert "white rims" in rims.summary()
    with pytest.raises(ValueError, match="white plumes need plumes"):
        StoneRecipe.model_validate(with_moss().model_dump() | {"white": "plumes"})
    with pytest.raises(ValueError, match="plumes or moss"):
        StoneRecipe.model_validate(bare().model_dump() | {"white": "rims", "white_level": 1.0})
    with pytest.raises(ValueError, match="no white level"):
        StoneRecipe.model_validate(bare().model_dump() | {"white_level": 0.3})


def test_the_level_maps_to_the_strength() -> None:
    """Level 0 the faint tile, 1 the solid one, even in the logarithm between. Control:
    a stone without white keeps optics without it."""
    assert white_transport(0.0) == pytest.approx(WHITE_FAINT)
    assert white_transport(1.0) == pytest.approx(WHITE_SOLID)
    assert white_transport(0.5) == pytest.approx(np.sqrt(WHITE_FAINT * WHITE_SOLID))
    r = with_plumes().model_copy(update={"white": "plumes", "white_level": 0.5})
    o = optics_for(r)
    assert o.white == "plumes"
    assert o.white_transport == pytest.approx(8.0)
    plain = optics_for(with_plumes())
    assert plain.white == "none"
    assert plain.white_transport == 0.0  # control
