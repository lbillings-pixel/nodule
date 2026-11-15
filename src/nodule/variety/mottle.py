"""Mottle patches of a variety stone: their constants, noise fields and size draw
(ADR 0018 s.10).

Appearance only: never evidence for a formation mechanism (ADR 0002).
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
from agatesim.volume.models import SmoothNoise3
from agatesim.volume.slicing import Plane
from agatesim.volume.solids import Solid

from nodule._scipy import largest_patch_size
from nodule.variety.recipe import Mottle, StoneRecipe, recipe_size, share_weights

MOTTLE_SHARE = 1 / 3  # the first draw; kept: MOTTLE_STONES of these


# The owner on mottle (2026-11-08, ADR 0018 s.10; all `effective`): a third to a
# half as often; most patches small, a large one (at least LARGE_PATCH of the face)
# on 15-25 % of mottled stones; dark rims on about 1 patch in 4, lighter on the rest.
MOTTLE_STONES = (1, 2)  # per 12 stones (was 4)


LARGE_MOTTLE_CHANCE = 1 / 5


LARGE_PATCH = 0.25  # share of the face


LARGE_PATCH_MAX = 0.6  # beyond this a patch reads as a film over the face


# Noise scale as a share of the stone's size; patches come out about 3-4 x wider.
SMALL_MOTTLE = ((0.04, 0.06), (0.15, 0.3))  # (scale, cover)


LARGE_MOTTLE = ((0.2, 0.35), (0.35, 0.5))


MOTTLE_TRIES = 16  # noise draws until the face shows the drawn class


DARK_RIM_SHARE = 0.25  # of patches (by area of the rim-darkness noise)


LIGHT_RIM = 0.4  # a light rim's darkening, as a share of a dark rim's


# Edges (the owner, 2026-11-08): about half of mottled stones have soft, graded
# patches, the rest hard-edged ones, and a few (5 %) both. A soft patch has no rim
# and rises to SOFT_PEAK x strength over SOFT_WIDTH in noise units, centred on the
# edge (about 0.4-0.8 x the noise scale in mm). All `effective`.
EDGE_SHARES: dict[str, float] = {"hard": 0.475, "soft": 0.475, "mixed": 0.05}


SOFT_WIDTH = (0.5, 1.0)


SOFT_PEAK = 0.7


def mottle_edge_noise(recipe: StoneRecipe, solid: Solid) -> SmoothNoise3:
    """Which patches of a mixed stone are soft: noise at twice the patch scale."""
    m = recipe.mottle
    assert m is not None
    seq = np.random.SeedSequence(recipe.seed).spawn(4)[2].spawn(MOTTLE_TRIES + 2)[-1]
    return SmoothNoise3(solid, 2 * m.scale_mm, np.random.default_rng(seq))


def mottle_fields(
    recipe: StoneRecipe, solid: Solid, *, rim: bool = True
) -> tuple[SmoothNoise3, SmoothNoise3 | None]:
    """The mottle's patch noise (variant 0: the stone's first stream, as before) and
    the rim-darkness noise, at twice the patch scale so a patch's rim is mostly of
    one kind. Both in world coordinates (resolution-independent)."""
    m = recipe.mottle
    assert m is not None
    root = np.random.SeedSequence(recipe.seed).spawn(4)
    extra = root[2].spawn(MOTTLE_TRIES + 1)
    patch_seq = root[0] if m.variant == 0 else extra[m.variant]
    patch = SmoothNoise3(solid, m.scale_mm, np.random.default_rng(patch_seq))
    if not rim:
        return patch, None
    return patch, SmoothNoise3(solid, 2 * m.scale_mm, np.random.default_rng(extra[0]))


def largest_patch(recipe: StoneRecipe, solid: Solid, spacing_mm: float = 0.5) -> float:
    """Mottle-size check: the largest connected mottle patch on the cut face, as a
    share of the face (solid and plane only, before bands and centre)."""
    m = recipe.mottle
    if m is None:
        return 0.0
    plane = Plane.of(recipe.cut, solid.centre())
    half = 0.55 * recipe_size(solid)
    g = np.arange(-half, half, spacing_mm)
    ss, tt = np.meshgrid(g, g)
    pts = plane.points(ss, tt).reshape(-1, 3)
    inside = solid.contains(pts).reshape(ss.shape)
    if not inside.any():
        return 0.0
    noise, _ = mottle_fields(recipe, solid, rim=False)
    patch = inside & (noise(pts).reshape(ss.shape) > NormalDist().inv_cdf(1 - m.cover))
    return largest_patch_size(patch) / float(inside.sum())


def mottle_size(rng: np.random.Generator, recipe: StoneRecipe, solid: Solid) -> Mottle:
    """Small or (1 in 5) large patches, then the first noise draw whose face shows
    that class, a large patch at most LARGE_PATCH_MAX (else the last); the face is
    the cutter's, so draw after the cut."""
    m = recipe.mottle
    assert m is not None
    large = bool(rng.uniform() < LARGE_MOTTLE_CHANCE)
    (scale, cover) = LARGE_MOTTLE if large else SMALL_MOTTLE
    size = recipe_size(solid)
    m = m.model_copy(
        update={
            "scale_mm": float(rng.uniform(*scale)) * size,
            "cover": float(rng.uniform(*cover)),
            "large": large,
        }
    )
    for v in range(MOTTLE_TRIES):
        m = m.model_copy(update={"variant": v})
        share = largest_patch(recipe.model_copy(update={"mottle": m}), solid)
        if (LARGE_PATCH <= share <= LARGE_PATCH_MAX) if large else share < LARGE_PATCH:
            break
    edge = str(rng.choice(list(EDGE_SHARES), p=share_weights(EDGE_SHARES)))
    return m.model_copy(update={"edge": edge, "soft_width": float(rng.uniform(*SOFT_WIDTH))})
