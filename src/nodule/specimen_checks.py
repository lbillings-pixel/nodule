"""The N2 slice's checks (ADR 0005 section 4), run on a whole specimen.

Each check has a negative control in tests/test_specimen.py: a specimen
lacking the property, which the check must reject.

| Check | Negative control |
|---|---|
| bands do not fold (time rises away from wall and tubes) | an uncapped wobble, slope > 1 |
| each fracture's offset is zero at its tip line | a flat slip profile ending in the stone |
| each fracture's backward map is one-to-one | one backward step with tau·|grad v| > 1 |
| a preview agrees with a full cut where pixels coincide | noise drawn per pixel |

The fifth check of ADR 0005 (no band-thickness steps at grid spacing) applies
only once a coarse grid caches a coordinate; the slice has none yet.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from agatesim.folds import FoldCheck, fold_check_points
from agatesim.grid import FloatArray
from agatesim.volume.fractures import overlaps, tip_offset_mm
from agatesim.volume.growth import segment_distance

from nodule.specimen import Specimen, SpecimenCut

TIP_TOLERANCE_MM = 1e-9


def growth_surface_distance(specimen: Specimen) -> Callable[[FloatArray], FloatArray]:
    """Distance (mm) to the nearest surface that bands grow from: the wall or a
    filament (tube). This is the lining coordinate of agatesim's growth model
    (`GrowthSolid.distance` with no bed or aggregates, which the slice lacks)."""
    solid = specimen.volume.solid
    tubes = specimen.volume.setting.tubes

    def distance(points: FloatArray) -> FloatArray:
        d = solid.wall_distance(points)
        for tube in tubes:
            d = np.minimum(d, segment_distance(points, tube[:3], tube[3:6]) - tube[6])
        return d

    return distance


def fold_check(specimen: Specimen, count: int = 4000, seed: int = 0) -> FoldCheck:
    """Growth time must not fall going away from the nearest growth surface, at
    random points in the unfractured nodule (bands are laid down before any
    fracture). Measured from the wall alone, a tube's own rings would count as
    folds: going inward toward a tube, growth time falls."""
    solid = specimen.volume.solid
    lo, hi = solid.bounds()
    points = np.random.default_rng(seed).uniform(lo, hi, (count, 3))
    return fold_check_points(
        specimen.volume.evaluate, growth_surface_distance(specimen), solid.contains, points
    )


def tip_offsets_mm(specimen: Specimen) -> list[float]:
    """Largest offset on each fracture's tip line (zero passes)."""
    return [tip_offset_mm(f) for f in specimen.fractures]


def overlap_counts(specimen: Specimen) -> list[int]:
    """Places where each fracture moved rock onto rock (zero passes)."""
    return [overlaps(f) for f in specimen.fractures]


@dataclass(frozen=True)
class Agreement:
    compared: int  # coincident pixels
    differing: int  # of those, where inside, phase, band or growth time differ


def coincident_agreement(fine: SpecimenCut, coarse: SpecimenCut) -> Agreement:
    """Compare two cuts of one plane where their pixel centres coincide."""
    if not np.allclose(fine.plane.origin, coarse.plane.origin):
        raise ValueError("the cuts lie on different planes")
    i = _match(fine.s, fine.spacing_mm, coarse.s)
    j = _match(fine.t, fine.spacing_mm, coarse.t)
    jj, ii = np.meshgrid(j, i, indexing="ij")  # shaped like the coarse cut
    keep = (jj >= 0) & (ii >= 0)
    cj, ci = np.nonzero(keep)
    fj, fi = jj[keep], ii[keep]
    a, b = fine.fields, coarse.fields
    differ = (fine.inside[fj, fi] != coarse.inside[cj, ci]) | (a.phase[fj, fi] != b.phase[cj, ci])
    differ |= a.band[fj, fi] != b.band[cj, ci]
    ta, tb = a.t_dep[fj, fi], b.t_dep[cj, ci]
    differ |= ~((ta == tb) | (np.isnan(ta) & np.isnan(tb)))
    return Agreement(compared=int(keep.sum()), differing=int(differ.sum()))


def _match(fine_axis: np.ndarray, spacing: float, coarse_axis: np.ndarray) -> np.ndarray:
    """Index into fine_axis of each coarse pixel centre, or -1 where none lies there."""
    k_fine = np.rint(fine_axis / spacing).astype(np.int64)
    k = coarse_axis / spacing
    k_int = np.rint(k).astype(np.int64)
    exact = np.abs(k - k_int) < 1e-6
    idx = k_int - k_fine[0]
    ok = exact & (idx >= 0) & (idx < len(fine_axis))
    return np.where(ok, idx, -1)
