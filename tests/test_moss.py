"""Moss in the recipes (ADR 0018 s.14): the owner's draws, one dendrite stage per stone,
and the moss check with its negative control (a straight radial spray)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from agatesim.models.growth import ProceduralGrowth
from agatesim.volume.config import SliceConfig, SolidConfig

from nodule.moss import (
    COLOUR_SHARES,
    G_SHARE,
    MOSS_SHARE,
    NO_MOSS,
    PLAIN_C_SHARE,
    REACH_CONTRAST,
    THUNDER_EGG_CONTRAST,
    draw_moss,
)
from nodule.plumes import draw_plumes
from nodule.variety import (
    StoneRecipe,
    build,
    cut_of,
    draw_sheet,
    moss_vertical,
    vertical_share,
    volume_config,
)


@pytest.fixture(scope="module")
def sheets() -> list[StoneRecipe]:
    return [r for seed in range(12) for r in draw_sheet(seed)]


def test_moss_on_a_sixth_of_stones_never_a_pebble_or_with_plumes(
    sheets: list[StoneRecipe],
) -> None:
    mossy = [r for r in sheets if r.moss is not None]
    assert len(mossy) == 12 * round(12 * MOSS_SHARE)
    assert not any(r.outline in NO_MOSS for r in mossy)
    assert not any(r.plumes is not None for r in mossy)
    assert sum(r.plumes is not None for r in sheets) == 12 * 3  # plumes unchanged
    for r in mossy:
        assert r.moss is not None
        assert r.moss.reach_floor == r.floor
        # Lobe contrast, but not in a thunder egg's narrow arms (the owner, 2026-11-16).
        want = THUNDER_EGG_CONTRAST if r.outline == "thunder-egg" else REACH_CONTRAST
        assert r.moss.reach_contrast == want
    assert {r.outline == "thunder-egg" for r in mossy} == {True, False}


def test_draws_follow_the_owners_shares() -> None:
    rng = np.random.default_rng(3)
    solid = [draw_moss(s, True, 0.0, 0.0) for s in rng.spawn(3000)]
    short = [draw_moss(s, False, 0.1, 0.0) for s in rng.spawn(3000)]
    assert abs(np.mean([m.look == "G" for m in solid]) - G_SHARE) < 0.03
    assert {m.look for m in solid} == {"E", "G"}
    # At centre width 0.1 short moss grows fully toward E, but for plain C.
    assert abs(np.mean([m.look == "C>0.0" for m in short]) - PLAIN_C_SHARE) < 0.03
    assert {m.look for m in short} == {"C>0.0", "C>1.0"}
    colours = Counter(m.colour for m in solid + short)
    for name, share in COLOUR_SHARES.items():
        assert abs(colours[name] / 6000 - share) < 0.025
    clearing = np.array([m.clearing for m in solid + short])
    assert clearing.min() >= 0.4
    assert clearing.max() <= 0.6
    assert all(m.reach_share == 0.9 for m in solid)
    assert all(abs(m.reach_share - 0.6 * 0.9) < 1e-9 for m in short)
    starts = np.array([m.start_height for m in short])
    assert starts.min() >= 0.35


def test_moss_grows_down_with_the_owners_reach_settings() -> None:
    m = draw_moss(np.random.default_rng(1), True, 0.0, 0.3)
    p = m.model_params()
    assert p["dendrite_tilt_deg"] == 180.0
    assert p["dendrite_reach_contrast"] == REACH_CONTRAST == 2.0
    assert p["dendrite_reach_floor"] == 0.3
    assert p["dendrite_halo_share"] == 1.0


def _stone(**update: object) -> StoneRecipe:
    base = StoneRecipe(
        index=0,
        seed=7,
        outline="vesicle",
        solid=SolidConfig(preset="vesicle-round", size_mm=26),
        centre="solid",
        centre_width=0.0,
        lining=0.5,
        floor=0.0,
        colour="neutral",
        band_mm=0.5,
        warp="mild",
        wobble_mm=0.1,
        wobble_scale_mm=4.0,
        white_fade=0.3,
        rim_stain=0.0,
        cut=SliceConfig(tilt_deg=90, azimuth_deg=0, offset_mm=0.0),
    )
    return base.model_copy(update=update)


def test_a_stone_carries_plumes_or_moss_not_both() -> None:
    rng = np.random.default_rng(2)
    plumes, moss = draw_plumes(rng), draw_moss(rng, True, 0.0, 0.0)
    with pytest.raises(ValueError, match="not both"):
        StoneRecipe.model_validate(_stone().model_dump() | {"plumes": plumes, "moss": moss})


def test_moss_reaches_the_whole_way_in_a_solid_stone() -> None:
    """A solid stone's moss may reach the centre (centre share 1); plumes stop at 2/3."""
    rng = np.random.default_rng(4)
    reach_moss = volume_config(_stone(moss=draw_moss(rng, True, 0.0, 0.0)))[0].model
    reach_plumes = volume_config(_stone(plumes=draw_plumes(rng)))[0].model
    assert isinstance(reach_moss, ProceduralGrowth)
    assert isinstance(reach_plumes, ProceduralGrowth)
    assert reach_moss.dendrite_tilt_deg == 180.0
    assert reach_moss.dendrite_reach_mm == pytest.approx(1.5 * reach_plumes.dendrite_reach_mm)


# The moss check: strands on the side walls run vertically, not along the wall's normal.

_N = 201
_RR, _CC = np.mgrid[0:_N, 0:_N].astype(np.float64)
_DISC = np.hypot(_RR - 100, _CC - 100) < 90
_UP = (-1.0, 0.0)  # row 0 at the top


def _strand(mask: np.ndarray, r0: float, c0: float, dr: float, dc: float, n: int) -> None:
    for t in range(n):
        r, c = round(r0 + dr * t), round(c0 + dc * t)
        mask[r : r + 2, c : c + 2] = True


def _hanging() -> np.ndarray:
    """Strands hanging straight down from the upper side walls."""
    m = np.zeros((_N, _N), dtype=bool)
    for ang in np.radians([100, 120, 140, 160, -100, -120, -140, -160]):
        r0, c0 = 100 + 86 * np.cos(ang), 100 + 86 * np.sin(ang)
        _strand(m, r0, c0 - 1 if np.sin(ang) > 0 else c0, 1.0, 0.0, 25)
    return m & _DISC


def _radial() -> np.ndarray:
    """Negative control: straight strands along the inward normal, all round."""
    m = np.zeros((_N, _N), dtype=bool)
    for ang in np.radians(np.arange(0, 360, 20)):
        r0, c0 = 100 + 86 * np.cos(ang), 100 + 86 * np.sin(ang)
        _strand(m, r0, c0, -np.cos(ang), -np.sin(ang), 25)
    return m & _DISC


def test_hanging_strands_pass_and_a_radial_spray_is_rejected() -> None:
    assert vertical_share(_hanging(), _DISC, _UP) > 0.9
    assert vertical_share(_radial(), _DISC, _UP) < 0.1


def test_a_moss_stone_cut_shows_vertical_strands() -> None:
    """A built moss stone (solid) passes the check on a vertical cut."""
    r = _stone(moss=draw_moss(np.random.default_rng(5), True, 0.0, 0.0))
    specimen, _ = build(r)
    share = moss_vertical(cut_of(specimen, r, 0.12))
    assert share > 0.6
