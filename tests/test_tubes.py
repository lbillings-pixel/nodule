"""Escape tubes in the recipes (ADR 0018 s.18): the owner's share and depths, none on
pebbles, the depth cap, and a tube that changes the stone only near its line (negative
control: it must change something there)."""

from __future__ import annotations

import numpy as np
import pytest
from agatesim.fields import Phase
from agatesim.models.growth import ProceduralGrowth
from agatesim.volume.features import wall_features
from pydantic import ValidationError

from nodule.tubes import DEPTH_CAP_SHARE, DEPTH_MM, NO_TUBES, Channel, draw_channel
from nodule.variety import StoneRecipe, build, chalcedony_mm, draw_sheet, volume_config
from nodule.variety.draw import channel_fits


def test_tubes_on_a_sixth_of_the_stones_never_on_pebbles(sheets: list[StoneRecipe]) -> None:
    tubes = [r for r in sheets if r.channel is not None]
    eligible = [r for r in sheets if r.outline not in NO_TUBES]
    assert len(tubes) == 23  # about 1 in 6 of the 132 eligible stones
    assert 0.1 < len(tubes) / len(eligible) < 0.25
    for r in tubes:
        assert r.channel is not None
        assert r.outline not in NO_TUBES
        assert r.channel.depth_mm <= DEPTH_MM[1] + 1e-9
        assert r.channel.depth_mm <= DEPTH_CAP_SHARE * chalcedony_mm(r) + 1e-9


def test_the_depth_is_capped_by_the_chalcedony() -> None:
    rng = np.random.default_rng
    free = [draw_channel(rng(s), "vesicle", 100.0) for s in range(200)]
    tight = [draw_channel(rng(s), "vesicle", 8.0) for s in range(200)]
    for a, b in zip(free, tight, strict=True):
        assert (a is None) == (b is None)  # the cap (4 mm) stays above MIN_DEPTH_MM
        if a is not None and b is not None:
            assert b.depth_mm == pytest.approx(min(a.depth_mm, 4.0))
    assert all(draw_channel(rng(s), "pebble", 100.0) is None for s in range(200))


def test_no_tube_on_a_pebble(sheets: list[StoneRecipe]) -> None:
    pebble = next(r for r in sheets if r.outline == "pebble")
    with pytest.raises(ValidationError):
        StoneRecipe.model_validate(pebble.model_dump() | {"channel": {"depth_mm": 3.0}})


def test_a_tube_changes_the_stone_only_near_its_line() -> None:
    rec = draw_sheet(0)[0]  # a vesicle with a 6.6 mm tube, about 3 s to build
    assert rec.channel is not None
    sp, _ = build(rec)
    plain, _ = build(rec.model_copy(update={"channel": None}))
    cfg = volume_config(rec)[0]
    assert isinstance(cfg.model, ProceduralGrowth)
    line = (
        wall_features(cfg.model, sp.volume.setting, np.random.default_rng(cfg.seed))
        .channels[0]
        .line
    )
    lo, hi = sp.volume.solid.bounds()
    pts = np.random.default_rng(3).uniform(lo, hi, (60_000, 3))
    reach = rec.channel.lateral_mm + rec.channel.length_mm / 2 + rec.channel.depth_mm + 2.0
    near = np.min(np.linalg.norm(pts[:, None, :] - line[None, :, :], axis=2), axis=1)
    a, ins = sp.volume.evaluate(pts, sp.volume.solid.contains(pts)), sp.volume.solid.contains(pts)
    b = plain.volume.evaluate(pts, ins)
    differs = ins & (np.nan_to_num(a.t_dep, nan=-1) != np.nan_to_num(b.t_dep, nan=-1))
    assert not np.any(differs & (near > reach))  # nothing moves far from the line
    assert np.any(differs & (near < rec.channel.lateral_mm))  # negative control: it acts


def _void_share(rec: StoneRecipe) -> float:
    sp, _ = build(rec)
    lo, hi = sp.volume.solid.bounds()
    pts = np.random.default_rng(3).uniform(lo, hi, (40_000, 3))
    ins = sp.volume.solid.contains(pts)
    phase = np.asarray(sp.volume.evaluate(pts, ins).phase)[ins]
    return int(np.count_nonzero(phase == Phase.VOID)) / max(len(phase), 1)


def test_tubes_round_a_concave_wall_open_no_void(sheets: list[StoneRecipe]) -> None:
    """Stones 2-0 and 4-9 have a channel line that bends round a concave stretch of
    wall; before agatesim ADR 0088 a piece's side was NaN there and stone 2-0 opened a
    void of about 15 % (agatesim's test carries that negative control). Now both are
    dealt their tubes, placement passes, and the tube adds almost no void."""
    for s, i in ((2, 0), (4, 9)):
        rec = sheets[12 * s + i]
        assert rec.channel is not None
        assert channel_fits(rec)
    rec = sheets[24]  # 2-0, about 3 s for the pair
    plain = rec.model_copy(update={"channel": None})
    assert _void_share(rec) - _void_share(plain) < 0.001


def test_channel_settings_reach_the_model() -> None:
    ch = Channel(depth_mm=4.0)
    assert ch.model_params()["channel_count"] == 1
    rec = next(r for r in draw_sheet(0) if r.channel is None and r.outline != "pebble")
    model = volume_config(rec.model_copy(update={"channel": ch}))[0].model
    plain = volume_config(rec)[0].model
    assert isinstance(model, ProceduralGrowth)
    assert isinstance(plain, ProceduralGrowth)
    assert model.channel_count == 1
    assert plain.channel_count == 0
