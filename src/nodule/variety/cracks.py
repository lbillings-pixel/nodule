"""Healed cracks of a variety stone: their constants and how they are drawn
(ADR 0018 s.10).

Appearance only: never evidence for a formation mechanism (ADR 0002).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from agatesim.grid import FloatArray
from agatesim.volume.solids import Solid

from nodule.variety.recipe import (
    Fill,
    FractureLayer,
    inradius,
    log_uniform,
    rim_point,
    share_weights,
    unit_vector,
)

# Cracks (the owner, 2026-11-08: ruler-straight black cracks look fake). Open
# cracks are no longer drawn: the sources describe cracks healed by chalcedony,
# quartz or iron oxides [@goetze2020, §5.1; @mrozik2023, pp. 14, 27]. Each crack
# is a shallow bowl of one gentle curvature: it rises by a sagitta at its edge, a
# share of its radius (`effective`, for the eye; no source gives a form). The owner
# judged noise bends too strong (2026-11-08): only slight curves, rarely if ever
# switching direction. A bowl's trace in any face is one arc, never an S.
SAG_SHARE = (0.03, 0.1)  # steepest slope 2 x share, at most 0.2


# The owner on crack form and mix (2026-11-08, ADR 0018 s.10; all `effective`, for the
# eye: no source gives crack counts, lengths or fills for agates). Healed cracks in a
# stone usually share one shade; some are straight; most are short or partial, and
# long, central or crossing cracks are rare (cracking that deep usually splits the
# stone). Shares below are among cracked stones.
CRACK_FILLS: dict[str, float] = {"clear": 0.6, "iron": 0.4}  # red stones: RED_CRACK_FILLS


RED_CRACK_FILLS: dict[str, float] = {"clear": 0.9, "iron": 0.1}  # iron hardly shows on red


MIXED_FILL_CHANCE = 1 / 12  # one crack takes the other fill (a later crack)


STAIN_JITTER = 0.1  # each crack's iron within this of the stone's


STRAIGHT_SHARE = 1 / 3  # cracks with no bend


PARTIAL_CRACKS = (3, 8)  # per cracked stone: inward from the rim


PARTIAL_REACH = (0.2, 0.5)  # how far in, as a share of the inradius


LONG_CRACKS: dict[int, float] = {0: 24, 1: 11, 2: 1}  # per 36 cracked stones


# The owner on the strip of cracked stones (2026-11-08): two long cracks crossing
# through the centre very rare (2 long was 1 in 12 cracked stones); iron stain too
# intense (was 0.4-1.0, the iron fill's own trace 0.9); no long crack through a
# cavity; and the cutter avoids cutting along a crack (a slab would split there),
# since a crack near the face's plane shows as a wide smear. All `effective`.
STAIN = (0.25, 0.6)


IRON_TRACE = 0.2  # the iron fill's trace: impurity = stain + this, at most 0.8


CRACK_CUT_ANGLE = 30.0  # degrees: least angle between the cut and a crack it meets


CAVITY_MARGIN_MM = 0.5  # cracks keep this far from the open cavity (estimate)


CRACK_TRIES = 20  # redraws of a crack that would enter the cavity; then it is left out


LONG_RADIUS = (0.6, 1.2)  # share of the inradius


LONG_OFFSET = (0.2, 0.6)  # centre's distance from the stone's centre, share of inradius


def enters_cavity(solid: Solid, f: FractureLayer, cavity_mm: float) -> bool:
    """Whether crack f comes within CAVITY_MARGIN_MM (plus its largest bend) of the
    open cavity: the points more than `cavity_mm` from the nearest wall."""
    n = np.asarray(f.normal) / float(np.linalg.norm(f.normal))
    e1 = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= float(np.linalg.norm(e1))
    e2 = np.cross(n, e1)
    rho = f.radius_mm * np.sqrt(np.linspace(0.0, 1.0, 16))[:, None]
    phi = np.linspace(0.0, 2 * math.pi, 48, endpoint=False)[None, :]
    a = (rho * np.cos(phi)).reshape(-1, 1)
    b = (rho * np.sin(phi)).reshape(-1, 1)
    pts: FloatArray = solid.centre() + np.asarray(f.centre_mm) + a * e1 + b * e2
    margin = CAVITY_MARGIN_MM + SAG_SHARE[1] * f.radius_mm
    return bool(np.any(solid.wall_distance(pts) > cavity_mm - margin))


def draw_fractures(
    rng: np.random.Generator,
    solid: Solid,
    red: bool,
    cavity_mm: float | None = None,
    scale_mm: float | None = None,
) -> tuple[FractureLayer, ...]:
    """A cracked stone's cracks (ADR 0018 s.7, s.10): 3-8 partial cracks running in
    from the rim, 0-3 nicks 1-4 mm long at the rim and, on 1 cracked stone in 3,
    one or (very rarely) two long cracks off the centre. All share one fill and about
    one iron strength, except that 1 cracked stone in 12 has one crack of the other
    fill. None is open; 1 in 3 is straight and the rest bend gently. The bends are
    drawn last, so the cracks lie where they were first drawn. With an open cavity
    (`cavity_mm` in from the wall), a crack that would reach it is redrawn, and
    left out after CRACK_TRIES. Crack sizes are shares of `scale_mm` (default the
    inradius; see `crack_scale`)."""
    r = inradius(solid) if scale_mm is None else scale_mm
    shares = RED_CRACK_FILLS if red else CRACK_FILLS
    fill: Fill = str(rng.choice(list(shares), p=share_weights(shares)))  # type: ignore[assignment]
    stain = float(rng.uniform(*STAIN))

    def shade() -> float:
        return float(np.clip(stain + rng.uniform(-STAIN_JITTER, STAIN_JITTER), 0.0, 1.0))

    def from_rim(radius_mm: float, aperture: tuple[float, float]) -> FractureLayer:
        radial = unit_vector(rng)
        normal = np.cross(radial, unit_vector(rng))  # the crack runs inward from the rim
        return FractureLayer(
            centre_mm=tuple(rim_point(solid, radial).tolist()),
            normal=tuple((normal / float(np.linalg.norm(normal))).tolist()),
            radius_mm=radius_mm,
            aperture_mm=log_uniform(rng, *aperture),
            fill=fill,
            stain=shade(),
            halo_mm=float(rng.uniform(0.0, 0.4)),
            pool_mm=float(rng.uniform(0.5, 2.0)),
        )

    def clear_of_cavity(draw: Callable[[], FractureLayer]) -> FractureLayer | None:
        for _ in range(CRACK_TRIES if cavity_mm is not None else 1):
            f = draw()
            if cavity_mm is None or not enters_cavity(solid, f, cavity_mm):
                return f
        return None

    def long_crack() -> FractureLayer:
        offset = float(rng.uniform(*LONG_OFFSET)) * r
        return FractureLayer(
            centre_mm=tuple((offset * unit_vector(rng)).tolist()),
            normal=tuple(unit_vector(rng).tolist()),
            radius_mm=float(rng.uniform(*LONG_RADIUS) * r),
            aperture_mm=log_uniform(rng, 0.05, 0.4),
            fill=fill,
            stain=shade(),
            halo_mm=float(rng.uniform(0.0, 0.5)),
            pool_mm=float(rng.uniform(1.0, 3.0)),  # half the 2-6 mm pools
        )

    drawn: list[FractureLayer | None] = []
    n_long = int(
        rng.choice(
            list(LONG_CRACKS), p=[v / sum(LONG_CRACKS.values()) for v in LONG_CRACKS.values()]
        )
    )
    for _ in range(n_long):
        drawn.append(clear_of_cavity(long_crack))
    for _ in range(int(rng.integers(PARTIAL_CRACKS[0], PARTIAL_CRACKS[1] + 1))):
        reach = float(rng.uniform(*PARTIAL_REACH)) * r
        drawn.append(clear_of_cavity(lambda reach=reach: from_rim(reach, (0.05, 0.3))))
    main = sum(f is not None for f in drawn)  # long and partial; the nicks follow
    for _ in range(int(rng.integers(0, 4))):
        size = float(rng.uniform(1.0, 4.0))
        drawn.append(clear_of_cavity(lambda size=size: from_rim(size, (0.05, 0.2))))
    out = [f for f in drawn if f is not None]
    if not out:
        return ()
    if rng.uniform() < MIXED_FILL_CHANCE:
        k = int(rng.integers(max(main, 1)))  # a long or partial crack, so it shows
        other: Fill = "iron" if fill == "clear" else "clear"
        out[k] = out[k].model_copy(update={"fill": other})
    return tuple(
        f.model_copy(
            update={
                "sagitta_mm": 0.0
                if rng.uniform() < STRAIGHT_SHARE
                else float(rng.uniform(*SAG_SHARE)) * f.radius_mm
            }
        )
        for f in out
    )
