"""Each irregular and lobate stone's own lumps (owner 2026-11-19 night; ADR 0018 s.25).

agatesim's `nodule-irregular` solid is one fixed union of seven balls and
`vesicle-lobate` one of four (agatesim `solids._LUMPS`, `_LOBES`), so every such stone
had the same outline up to size and turn, and the wall-distance bands turned its lobes
into the same corners inward (4/4's square hollow, ADR 0019 s.6.12 "Diagnosis"). Here
each stone draws its own union of balls about the presets' numbers and hands it to
agatesim as a drawn 3D cavity (agatesim ADR 0089), as thunder eggs do
(`nodule.thunderegg3d`, whose storage this follows).

The balls' level is exact outside the union and near its wall (the largest over balls
of radius less distance); agatesim measures the true distance inside from the wall, so
the bands follow the wall as with the presets' exact sphere unions. Shape units: the
stored cavity is scaled so its longest side is the drawn size.

Every number here is `effective`, an assumption: no source gives lump counts or sizes;
the presets' own are the anchor (core 0.62; lumps 0.30-0.42 at about 0.45 from the core's
centre in all directions; lobes 0.38-0.42 at about 0.62, near one plane, about 120 deg
apart). Lopsided lobes are rare in real agates (the owner, 2026-11-08), so the spreads
stay moderate. Appearance only: never evidence for how cavities form (ADR 0002).
"""

from __future__ import annotations

import functools
import hashlib
import math
from dataclasses import dataclass

import numpy as np
from agatesim.volume.config import SampledLevel, SolidConfig
from agatesim.volume.sampled import CLAMP_CELLS, QUANT, Sampled
from agatesim.volume.solids import Solid
from numpy.typing import NDArray

from nodule._scipy import largest_component_3d

FloatArray = NDArray[np.float64]

# The drawing grid, in shape units. Balls are smooth, so it is coarser than the thunder
# eggs' (192): agatesim's first distance query on a drawn cavity costs about its inside
# nodes (`runs/lumps/grid_cost.log`). Balls reach at most 1.14, so a wall stays at least
# 5 cells inside the box (agatesim needs 2).
GRID = 128
EXTENT = 1.25
CELL = 2 * EXTENT / GRID
CENTRES = (np.arange(GRID) + 0.5) * CELL - EXTENT
PAD = 6  # nodes kept beyond the wall's box
TRIES = 8  # redraws (same stream) if agatesim refuses a drawing
DIRECTION_TRIES = 200  # draws per lump direction before taking the last one


@dataclass(frozen=True)
class LumpStyle:
    """How one outline's balls are drawn (shape units; module notes)."""

    core: tuple[float, float]  # the central ball's radius
    counts: tuple[int, ...]  # lump counts, drawn with `weights`
    weights: tuple[float, ...]
    radius: tuple[float, float]  # each lump's radius
    reach: tuple[float, float]  # its centre's distance from the core's centre
    min_angle_deg: float  # least angle between two lumps' directions
    plane_deg: float | None  # lobes: within this of one plane, spread round it; None: any


# Irregular sediment nodules: the preset's six lumps at 0.43-0.47 (radii 0.30-0.42) on a
# 0.62 core, spread in all directions.
IRREGULAR = LumpStyle(
    core=(0.56, 0.68),
    counts=(4, 5, 6, 7, 8),
    weights=(0.15, 0.2, 0.3, 0.2, 0.15),
    radius=(0.28, 0.44),
    reach=(0.38, 0.52),
    min_angle_deg=40.0,
    plane_deg=None,
)
# Lobate vesicles: the preset's three lobes at 0.60-0.63 (radii 0.38-0.42), about 120 deg
# apart and within about 11 deg of one plane.
LOBATE = LumpStyle(
    core=(0.56, 0.66),
    counts=(2, 3, 4),
    weights=(0.25, 0.5, 0.25),
    radius=(0.34, 0.46),
    reach=(0.52, 0.68),
    min_angle_deg=70.0,
    plane_deg=12.0,
)
STYLES = {"irregular": IRREGULAR, "lobate": LOBATE}
HOSTS = {"irregular": "sedimentary", "lobate": "volcanic"}


def _directions(rng: np.random.Generator, n: int, style: LumpStyle) -> FloatArray:
    """n unit directions at least min_angle_deg apart (the last draw kept if none is)."""
    if style.plane_deg is not None:  # lobes round one plane, evenly spread with jitter
        step = 2 * math.pi / n
        az = rng.uniform(0, 2 * math.pi) + step * (np.arange(n) + rng.uniform(-0.2, 0.2, n))
        el = np.radians(rng.uniform(-style.plane_deg, style.plane_deg, n))
        return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
    least = math.cos(math.radians(style.min_angle_deg))
    out: list[FloatArray] = []
    for _ in range(n):
        d = np.zeros(3)
        for _ in range(DIRECTION_TRIES):
            d = rng.normal(size=3)
            d /= float(np.linalg.norm(d))
            if all(float(d @ e) < least for e in out):
                break
        out.append(d)
    return np.array(out)


def draw_balls(rng: np.random.Generator, style: LumpStyle) -> tuple[FloatArray, FloatArray]:
    """Centres (k, 3) and radii (k,) of one stone's balls, the core first."""
    n = int(rng.choice(style.counts, p=np.asarray(style.weights) / sum(style.weights)))
    dirs = _directions(rng, n, style)
    reach = rng.uniform(*style.reach, n)
    radii = rng.uniform(*style.radius, n)
    core = rng.uniform(*style.core)
    centres = np.vstack([np.zeros(3), dirs * reach[:, None]])
    return centres, np.r_[core, radii]


def ball_level(centres: FloatArray, radii: FloatArray, grid: int = GRID) -> NDArray[np.float32]:
    """The union's level on the drawing grid (shape units, positive inside): the
    largest over balls of radius less distance."""
    c = (np.arange(grid) + 0.5) * (2 * EXTENT / grid) - EXTENT
    x, y, z = c[:, None, None], c[None, :, None], c[None, None, :]
    level = np.full((grid, grid, grid), -np.inf, dtype=np.float32)
    for (cx, cy, cz), r in zip(centres, radii, strict=True):
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
        np.maximum(level, (r - d).astype(np.float32), out=level)
    return level


def stored_cells(level: NDArray[np.float32], grid: int = GRID) -> FloatArray:
    """The level agatesim stores: in cells, clamped, on the grid cropped to the
    union's box plus PAD nodes (`crop_start` gives where); nodes that the 8-bit storage
    would read as a second piece are turned outside (as `thunderegg3d.union_cells`)."""
    cell = 2 * EXTENT / grid
    inside = largest_component_3d(level > 0)
    lo = crop_start(inside)
    hi = np.minimum(np.argwhere(inside).max(axis=0) + PAD + 1, grid)
    sub = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
    cells = np.clip(level[sub].astype(np.float64) / cell, -CLAMP_CELLS, CLAMP_CELLS)
    cells = np.where((cells > 0) & ~inside[sub], -cells, cells)
    stored = np.rint(cells / CLAMP_CELLS * QUANT) > 0
    stray = stored & ~largest_component_3d(stored)
    return np.where(stray, -np.abs(cells), cells)


def crop_start(inside: NDArray[np.bool_]) -> NDArray[np.int64]:
    """The drawing-grid index of the stored lattice's node 0 (`stored_cells`' crop)."""
    return np.maximum(np.argwhere(inside).min(axis=0) - PAD, 0)


def sampled_key(solid: Sampled) -> tuple[object, ...]:
    """A drawn cavity's identity: its lattice, scale and stored level (a 128-bit hash)."""
    digest = hashlib.blake2b(solid.raw, digest_size=16).digest()
    return (solid.shape, solid.cell_mm, digest)


# Each lumpy solid's balls in its own frame (mm), by `sampled_key`: for the cutter's rule
# (`draw.cut_obliqueness`), which agatesim's drawn cavity cannot answer.
_BALLS: dict[tuple[object, ...], tuple[FloatArray, FloatArray]] = {}


def balls_of(solid: Solid) -> tuple[FloatArray, FloatArray] | None:
    """A lumpy solid's balls, centres (k, 3) and radii (k,) in mm in its own frame, as
    drawn in this process; None for any other solid."""
    if not isinstance(solid, Sampled):
        return None
    return _BALLS.get(sampled_key(solid))


def lumps_solid(rng: np.random.Generator, kind: str, size_mm: float) -> SolidConfig:
    """A stone's own lumpy solid (`kind` "irregular" or "lobate"), longest side
    `size_mm`, drawn from a stream spawned off `rng` (so `rng`'s own draws stay as
    they were) and remembered by that stream's key."""
    parent = rng.bit_generator.seed_seq
    if not isinstance(parent, np.random.SeedSequence):
        raise TypeError("lumps are drawn from a stream seeded by a SeedSequence")
    seq = parent.spawn(1)[0]
    ent = seq.entropy
    if ent is None:
        raise TypeError("lumps need a stream with known entropy")
    entropy = int(ent) if isinstance(ent, int) else tuple(int(e) for e in ent)
    return _remembered((entropy, tuple(seq.spawn_key), seq.pool_size), kind, size_mm)


@functools.lru_cache(maxsize=256)
def _remembered(
    key: tuple[int | tuple[int, ...], tuple[int, ...], int], kind: str, size_mm: float
) -> SolidConfig:
    entropy, spawn_key, pool = key
    rng = np.random.default_rng(
        np.random.SeedSequence(entropy, spawn_key=spawn_key, pool_size=pool)
    )
    style = STYLES[kind]
    for _ in range(TRIES):
        centres, radii = draw_balls(rng, style)
        level = ball_level(centres, radii)
        try:
            cfg = SolidConfig(
                sampled=SampledLevel.of(stored_cells(level)),
                host=HOSTS[kind],  # type: ignore[arg-type]
                size_mm=size_mm,
            )
        except ValueError:
            continue
        solid = cfg.build()
        assert isinstance(solid, Sampled)
        # shape units to the solid's mm: node i of the stored lattice is drawing-grid
        # node crop_start + i, at origin + cell_mm i
        c0 = CENTRES[crop_start(largest_component_3d(level > 0))]
        scale = solid.cell_mm / CELL
        _BALLS[sampled_key(solid)] = (solid.origin + (centres - c0) * scale, radii * scale)
        return cfg
    raise RuntimeError(f"no {kind} lumps accepted in {TRIES} draws")
