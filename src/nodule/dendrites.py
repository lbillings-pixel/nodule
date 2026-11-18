"""Black manganese dendrites in the recipes (work plan item 14; ADR 0018 s.17).

No peer-reviewed study of Mn dendrites in agate was found (ADR 0018 s.17), so this is
appearance only (ADR 0002) and every number here is `effective`. The owner's choices
(2026-11-17/18): dendrites on at most 5 % of stones (stones that carry them), any kind
of stone; half grow into the chalcedony from the wall, made sparse, as in the one agate
nodule cut through its middle [@bayirli2017] (*verified* for that stone); half grow on
cracks, from where a crack meets the wall, by analogy with other rocks
[@swartzlow1934; @ngteh2009] (*verified*, not agate). Sites are steered onto the
stone's cut (the owner, 2026-11-18, after unsteered sites showed on about 1 cut in
100): wall sites on the cut's outline, carrier cracks crossed by the cut where they
meet the wall (`nodule.variety.draw.with_dendrites`).

This module holds the recipe entry and its draw; the growth (agatesim's 3D
diffusion-limited aggregation on world-fixed lattices, so every cut at any spacing
shows the same dendrite) is `nodule.variety.dendrites`.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nodule._scipy import label_patches

Carrier = Literal["wall", "crack"]

# 4 % of stones: under the owner's cap of 5 % (no source gives a rate).
DENDRITE_CHANCE = 0.04
WALL_SHARE = 0.5  # the owner: half from the wall, half on cracks
WALL_SITES = (1, 3)  # wall patches a stone's dendrites grow from
CARRIER_CRACKS = (1, 2)  # cracks that carry them, where the stone has that many
KEEP_SHARE = 1 / 8  # the owner's pick "largest eighth" of the wall growth's trees


class Dendrites(BaseModel):
    """A stone's Mn dendrites. `key` seeds their growth (one spawned stream per site
    or crack). Wall sites are unit directions from the solid's centre; the dendrite
    grows from where that ray leaves the stone. Crack carriers are indices into the
    recipe's fractures."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    carrier: Carrier
    key: int = Field(ge=0)
    sites: tuple[tuple[float, float, float], ...] = ()
    cracks: tuple[int, ...] = ()
    keep_share: float = Field(default=KEEP_SHARE, gt=0, le=1)

    @model_validator(mode="after")
    def _carrier_has_places(self) -> Dendrites:
        if self.carrier == "wall" and (not self.sites or self.cracks):
            raise ValueError("wall dendrites take sites and no cracks")
        if self.carrier == "crack" and (not self.cracks or self.sites):
            raise ValueError("crack dendrites take cracks and no sites")
        return self

    def summary(self) -> str:
        if self.carrier == "wall":
            n = len(self.sites)
            return f"dendrites from the wall ({n} site{'s' * (n > 1)})"
        n = len(self.cracks)
        return f"dendrites on {n} crack{'s' * (n > 1)}"


class DendriteDraw(NamedTuple):
    """A stone's dendrite draw, before its cracks are known."""

    carrier: Carrier
    key: int
    n_cracks: int  # carrier cracks wanted (fewer if the stone has fewer)
    sites: tuple[tuple[float, float, float], ...]  # wall directions

    def as_wall(self) -> Dendrites:
        return Dendrites(carrier="wall", key=self.key, sites=self.sites)

    def on_cracks(self, cracks: tuple[int, ...]) -> Dendrites:
        return Dendrites(carrier="crack", key=self.key, cracks=cracks)


def draw_dendrites(rng: np.random.Generator) -> DendriteDraw | None:
    """Whether a stone carries dendrites, and if so its carrier, growth key, number of
    carrier cracks wanted and wall directions (as many as its sites; a crack stone
    that ends up with no carrier crack falls back to them). Every stone draws the same
    numbers, so a later change of eligibility moves no other stone's draw."""
    u = rng.random(2)
    key = int(rng.integers(0, 2**31))
    n_sites = int(rng.integers(WALL_SITES[0], WALL_SITES[1] + 1))
    n_cracks = int(rng.integers(CARRIER_CRACKS[0], CARRIER_CRACKS[1] + 1))
    v = rng.standard_normal((WALL_SITES[1], 3))
    sites = tuple(_direction(x) for x in v[:n_sites])
    if u[0] >= DENDRITE_CHANCE:
        return None
    carrier: Carrier = "wall" if u[1] < WALL_SHARE else "crack"
    return DendriteDraw(carrier, key, n_cracks, sites)


def box_dimension(mask: NDArray[np.bool_], smallest: int = 2, share: float = 0.25) -> float:
    """Box-counting dimension of a 2D mask: the slope of log(boxes holding any of it)
    against log(1 / box side), over sides from `smallest` pixels, doubling, up to
    `share` of the mask's extent (its bounding box's shorter side). The measure behind
    the published values for dendrites (1.57-1.88 face-on in vein quartz, [@ngteh2009,
    Table 2]; 1.63 on the one agate section, [@bayirli2017]). NaN if under three sides
    fit."""
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return float("nan")
    m = mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    sides: list[int] = []
    counts: list[int] = []
    s = smallest
    while s <= share * min(m.shape) or (s <= share * max(m.shape) and len(sides) < 3):
        h, w = -(-m.shape[0] // s) * s, -(-m.shape[1] // s) * s
        padded = np.zeros((h, w), bool)
        padded[: m.shape[0], : m.shape[1]] = m
        boxes = padded.reshape(h // s, s, w // s, s).any(axis=(1, 3))
        sides.append(s)
        counts.append(int(boxes.sum()))
        s *= 2
    if len(sides) < 3:
        return float("nan")
    slope = np.polyfit(np.log(1 / np.asarray(sides, float)), np.log(counts), 1)[0]
    return float(slope)


FACE_ON_DIMENSION = (1.57, 1.88)  # face-on dendrites in vein quartz [@ngteh2009, Table 2]
BRANCH_PX = 50  # a 4-connected patch this big counts as a branch (`effective`)
BRANCHED_SHARE = 0.8  # share of a dendrite's pixels in branches (`effective`)


def branched_share(mask: NDArray[np.bool_], min_px: int = BRANCH_PX) -> float:
    """Share of the mask's pixels in 4-connected patches of at least `min_px`."""
    found, n = label_patches(mask)
    if not n:
        return 0.0
    sizes = np.bincount(found.ravel())[1:]
    return float(sizes[sizes >= min_px].sum() / sizes.sum())


def dendritic(mask: NDArray[np.bool_]) -> bool:
    """A face-on view (a mask at a pixel well under the branch width) reads as a
    dendrite: box dimension in the published face-on range, and most pixels in
    connected branches. Box dimension alone passes random specks at a few per cent
    cover (1.66 at 5 %, measured 2026-11-18), hence the second part. Negative controls
    in the tests: a line, a filled disc, random specks."""
    d = box_dimension(mask)
    lo, hi = FACE_ON_DIMENSION
    return lo <= d <= hi and branched_share(mask) >= BRANCHED_SHARE


def _direction(v: np.ndarray) -> tuple[float, float, float]:
    x, y, z = (v / float(np.linalg.norm(v))).tolist()
    return (x, y, z)
