"""The hollow's void box (ADR 0019 s.6.12 step 3): the gate, world anchoring and the stones
without a hollow, each check with its negative control (CLAUDE.md)."""

from __future__ import annotations

import numpy as np
import pytest
from agatesim.fields import Phase
from agatesim.volume.config import SliceConfig, SolidConfig
from agatesim.volume.simulate import VolumeSimulation

from nodule.render.voidbox import VoidBox, build_void_box
from nodule.variety import StoneRecipe, build


def recipe(**update: object) -> StoneRecipe:
    base = StoneRecipe(
        index=0,
        seed=7,
        outline="vesicle",
        solid=SolidConfig(preset="vesicle-round", size_mm=26),
        centre="quartz-lined",
        centre_width=0.45,
        lining=0.4,
        floor=0.0,
        colour="neutral",
        band_mm=0.5,
        warp="mild",
        wobble_mm=0.1,
        wobble_scale_mm=4.0,
        white_fade=0.3,
        rim_stain=0.0,
        cut=SliceConfig(tilt_deg=90, azimuth_deg=30, offset_mm=0.0),
    )
    return base.model_copy(update=update)


@pytest.fixture(scope="module")
def lined() -> tuple[VolumeSimulation, VoidBox]:
    vol = build(recipe())[0].volume
    box = build_void_box(vol)
    assert box is not None
    return vol, box


def in_box(box: VoidBox, n: int, seed: int) -> np.ndarray:
    lo = box.a * box.h
    hi = (box.a + np.array(box.shape) - 1) * box.h
    return lo + (hi - lo) * np.random.default_rng(seed).random((n, 3))


def test_gated_air_is_the_hollow_and_never_chalcedony(
    lined: tuple[VolumeSimulation, VoidBox],
) -> None:
    vol, box = lined
    p = in_box(box, 200_000, 0)  # points inside cells, not only nodes
    ins = vol.solid.contains(p)
    ph = np.where(ins, vol.evaluate(p, ins).phase, 254)
    air = box.air(p)
    chal, void, quartz = (ph == Phase.CHALCEDONY), (ph == Phase.VOID), (ph == Phase.QUARTZ)
    # Control: the level's sign alone finds a false hollow behind the crystals' sites.
    assert np.sum(chal & (box.level(p) > 0)) > 1000
    assert not np.any(air & chal)
    assert not np.any(air & (ph == 254))
    assert void.sum() > 1000
    assert np.mean(air[void]) > 0.995  # slivers aside
    assert np.mean(air[quartz]) < 0.001


def test_box_nodes_do_not_depend_on_its_bounds(lined: tuple[VolumeSimulation, VoidBox]) -> None:
    vol, box = lined
    big = build_void_box(vol, pad_mm=5.0)
    assert big is not None
    assert all(b > s for b, s in zip(big.shape, box.shape, strict=True))
    o = box.a - big.a
    sl = tuple(slice(int(o[i]), int(o[i]) + box.shape[i]) for i in range(3))
    assert np.array_equal(big.ids[sl], box.ids)
    assert np.array_equal(big.gate[sl], box.gate)
    # Control: nodes anchored at another corner (half a cell off the world lattice) differ.
    flat = np.arange(0, box.ids.size, 97)
    nodes = (np.stack(np.unravel_index(flat, box.shape), axis=-1) + box.a) * box.h
    _, moved = box.wall.arrival(nodes + 0.5 * box.h)
    _, same = box.wall.arrival(nodes)
    assert np.array_equal(same, box.ids.reshape(-1)[::97])
    assert np.mean(moved != same) > 0.05


def test_no_hollow_or_no_crystal_wall_gives_no_box() -> None:
    filled = build(recipe(centre="quartz-filled"))[0].volume
    assert filled.crystal_wall is not None  # crystals, but they close the centre
    assert build_void_box(filled) is None
    solid = build(recipe(centre="solid", centre_width=0.0, lining=0.5))[0].volume
    assert solid.crystal_wall is None
    assert build_void_box(solid) is None
