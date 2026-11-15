"""Checks on a variety cut, each with a negative control in the tests
(ADR 0018 s.7).

Appearance only: never evidence for a formation mechanism (ADR 0002).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from agatesim.fields import Phase
from agatesim.grid import FloatArray
from numpy.typing import NDArray

from nodule.specimen import Specimen, SpecimenCut
from nodule.variety.cracks import LONG_RADIUS
from nodule.variety.layers import BandRoughness, VarietyLayers
from nodule.variety.recipe import FractureLayer


def has_cavity(cut: SpecimenCut, least: float = 0.002) -> bool:
    """Open space inside the stone, beyond a share `least` of the face."""
    inside = cut.inside
    return bool(inside.any() and (cut.fields.phase[inside] == Phase.VOID).mean() > least)


def _perimeter(mask: NDArray[np.bool_]) -> int:
    return int((mask[1:] != mask[:-1]).sum() + (mask[:, 1:] != mask[:, :-1]).sum())


def _box_blur(a: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    for axis in (0, 1):
        c = np.cumsum(
            np.pad(a, [(k + 1, k) if i == axis else (0, 0) for i in (0, 1)], mode="edge"), axis=axis
        )
        size = int(c.shape[axis])
        hi = np.take(c, np.arange(2 * k + 1, size, dtype=np.intp), axis=axis)
        lo = np.take(c, np.arange(0, size - 2 * k - 1, dtype=np.intp), axis=axis)
        a = (hi - lo) / (2 * k + 1)
    return a


def lining_roughness(cut: SpecimenCut, smooth_mm: float = 1.0) -> float:
    """Faceted-lining check: the cavity's boundary length over that of the same
    cavity smoothed over `smooth_mm`. A saw-tooth quartz lining scores well above 1;
    a smooth chalcedony wall near 1. NaN without a cavity."""
    void = cut.inside & (cut.fields.phase == Phase.VOID)
    if void.sum() < 50:
        return math.nan
    k = max(1, round(smooth_mm / cut.spacing_mm))
    smooth = _box_blur(_box_blur(void.astype(np.float64), k), k) > 0.5
    return _perimeter(void) / max(_perimeter(smooth), 1)


def flat_bands(cut: SpecimenCut, least_span: float = 0.4, most_spread_mm: float = 0.25) -> int:
    """Waterline check: bands whose top edge stays level (spread under
    `most_spread_mm`) across at least `least_span` of the stone's width. Height is
    measured along the plane's projection of world up."""
    up = cut.plane.in_plane(np.array([0.0, 0.0, 1.0]))
    if float(np.linalg.norm(up)) < 0.5:
        return 0
    up = up / float(np.linalg.norm(up))
    ss, tt = np.meshgrid(cut.s, cut.t)
    height = ss * up[0] + tt * up[1]
    across = ss * up[1] - tt * up[0]
    col = np.floor(across / cut.spacing_mm).astype(np.int64)
    inside = cut.inside & (cut.fields.phase == Phase.CHALCEDONY)
    if not inside.any():
        return 0
    width_cols = col[inside].max() - col[inside].min() + 1
    band = cut.fields.band
    count = 0
    for b in np.unique(band[inside]):
        m = inside & (band == b)
        c, h = col[m], height[m]
        order = np.lexsort((h, c))
        c, h = c[order], h[order]
        last = np.r_[c[1:] != c[:-1], True]
        tops, cols = h[last], c[last]
        if len(cols) < least_span * width_cols:
            continue
        if float(np.std(tops)) < most_spread_mm:
            count += 1
    return count


def floor_bands(cut: SpecimenCut, layers: VarietyLayers) -> tuple[float, float]:
    """Floor check: the floor's height and its mean band width (mm), from the
    chalcedony laid down before the walls start. Height is measured along the
    plane's projection of world up; (0, 0) without a floor."""
    up = cut.plane.in_plane(np.array([0.0, 0.0, 1.0]))
    t = cut.fields.t_dep
    floor = cut.inside & (cut.fields.phase == Phase.CHALCEDONY) & (t < layers.start)
    if not floor.any() or float(np.linalg.norm(up)) < 0.5:
        return 0.0, 0.0
    up = up / float(np.linalg.norm(up))
    ss, tt = np.meshgrid(cut.s, cut.t)
    height = (ss * up[0] + tt * up[1])[floor]
    span = float(height.max() - height.min())
    return span, span / len(np.unique(cut.fields.band[floor]))


def fracture_stain(cut: SpecimenCut, layers: VarietyLayers, near_mm: float = 0.5) -> float:
    """Stained-fracture check: mean impurity of chalcedony within `near_mm` of a
    crack, less the median impurity beyond 3 mm. NaN without a crack in the face."""
    ss, tt = np.meshgrid(cut.s, cut.t)
    d = layers.fracture_distance(cut.plane.points(ss, tt)).reshape(ss.shape)
    chal = cut.inside & (cut.fields.phase == Phase.CHALCEDONY)
    imp = np.nan_to_num(cut.fields.impurity, nan=0.0)
    near, far = chal & (d < near_mm), chal & (d > 3.0)
    if near.sum() < 20 or far.sum() < 20:
        return math.nan
    return float(imp[near].mean() - np.median(imp[far]))


def _trace_centre(
    cut: SpecimenCut, layers: VarietyLayers, i: int, bin_mm: float
) -> tuple[FloatArray, FloatArray] | None:
    """Crack i's trace in the face as a centre line: the mean offset from the
    trace's main axis in bins of `bin_mm` along it. None if under 3 bins."""
    f = layers.recipe.fractures[i]
    ss, tt = np.meshgrid(cut.s, cut.t)
    d = layers.crack_distance(i, cut.plane.points(ss, tt)).reshape(ss.shape)
    trace = cut.inside & (d < max(f.aperture_mm / 2, cut.spacing_mm))
    xy = np.stack([ss[trace], tt[trace]], axis=1)
    if len(xy) < 10:
        return None
    xy = xy - xy.mean(axis=0)
    axes = np.linalg.svd(xy, full_matrices=False)[2]
    along, off = xy @ axes[0], xy @ axes[1]
    k = np.floor((along - along.min()) / bin_mm).astype(np.int64)
    counts = np.bincount(k)
    keep = counts >= 3
    if keep.sum() < 3:
        return None
    sums = np.zeros(len(counts))
    np.add.at(sums, k, off)
    x: FloatArray = (np.arange(len(counts))[keep] + 0.5) * bin_mm
    return x, sums[keep] / counts[keep]


def _fit_rms(x: FloatArray, y: FloatArray, degree: int) -> float:
    resid: FloatArray = y - np.polyval(np.polyfit(x, y, degree), x)
    return float(np.sqrt(np.mean(resid**2)))


def crack_bend(cut: SpecimenCut, layers: VarietyLayers, i: int, bin_mm: float = 1.0) -> float:
    """Bent-crack check: the RMS offset (mm) of crack i's trace in the face from
    its best straight line, so a straight trace of any width gives about 0. NaN
    if the trace spans less than 3 bins."""
    line = _trace_centre(cut, layers, i, bin_mm)
    return math.nan if line is None else _fit_rms(*line, 1)


def graded_share(layers: VarietyLayers, points: FloatArray) -> float:
    """Soft-edge check (ADR 0018 s.10): among points inside mottle patches, the share
    whose stain lies between 0.05 and 0.55 x strength, which a hard patch (0, then a
    rim of at least 0.76 x, then 0.6 x) never shows."""
    m = layers.recipe.mottle
    if m is None:
        return math.nan
    a = layers.mottle_amount(points) / m.strength
    patch = a > 0
    return float(np.mean((a[patch] > 0.05) & (a[patch] < 0.55))) if patch.any() else math.nan


def crack_traces(cut: SpecimenCut, layers: VarietyLayers, least_mm: float = 1.0) -> list[float]:
    """Crack-length check (ADR 0018 s.10): each crack's trace in the face, its
    length along its main axis as a share of the face's widest extent. Traces
    shorter than `least_mm` (or missing) are left out."""
    ss, tt = np.meshgrid(cut.s, cut.t)
    pts = cut.plane.points(ss, tt)
    inside = cut.inside
    s_in, t_in = ss[inside], tt[inside]
    width = max(float(s_in.max() - s_in.min()), float(t_in.max() - t_in.min()))
    out: list[float] = []
    for i, f in enumerate(layers.recipe.fractures):
        d = layers.crack_distance(i, pts).reshape(ss.shape)
        trace = inside & (d < max(f.aperture_mm / 2, cut.spacing_mm))
        if trace.sum() < 4:
            continue
        xy = np.stack([ss[trace], tt[trace]], axis=1)
        xy = xy - xy.mean(axis=0)
        along: FloatArray = xy @ np.linalg.svd(xy, full_matrices=False)[2][0]
        length = float(along.max() - along.min())
        if length >= least_mm:
            out.append(length / width)
    return out


class CrackMix(NamedTuple):
    mixed: float  # share of cracked stones whose cracks take more than one fill
    stain_spread: float  # largest spread of iron strength within a one-fill stone
    straight: float  # share of cracks with no bend
    long: float  # share of cracked stones with a long crack
    two_long: float  # share with two or more


def crack_mix(stones: list[tuple[tuple[FractureLayer, ...], float]]) -> CrackMix:
    """Crack-mix check (ADR 0018 s.10) over cracked stones, each given as its
    cracks and its inradius (mm). A long crack reaches LONG_RADIUS[0] x inradius."""
    cracks = [f for fs, _ in stones for f in fs]
    one_fill = [fs for fs, _ in stones if len({f.fill for f in fs}) == 1]
    longs = [sum(f.radius_mm >= LONG_RADIUS[0] * r for f in fs) for fs, r in stones]
    n = len(stones)
    return CrackMix(
        mixed=1 - len(one_fill) / n,
        stain_spread=max(
            (max(f.stain for f in fs) - min(f.stain for f in fs) for fs in one_fill), default=0.0
        ),
        straight=sum(f.sagitta_mm == 0 for f in cracks) / len(cracks),
        long=sum(k >= 1 for k in longs) / n,
        two_long=sum(k >= 2 for k in longs) / n,
    )


def crack_wiggle(cut: SpecimenCut, layers: VarietyLayers, i: int, bin_mm: float = 1.0) -> float:
    """No-S-turn check: the RMS offset (mm) of crack i's trace from its best
    parabola, the part of the bend that one curve cannot follow. An arc gives
    about 0; a trace whose curvature switches sign (an S-turn, a wiggle) does not.
    NaN if the trace spans less than 4 bins."""
    line = _trace_centre(cut, layers, i, bin_mm)
    return math.nan if line is None or len(line[0]) < 4 else _fit_rms(*line, 2)


def band_swelling(
    cut: SpecimenCut, layers: VarietyLayers, growth_mm: float, plain: SpecimenCut
) -> float:
    """Wobble swelling check (ADR 0018 s.10): the 99th percentile of band width over
    the same stone cut without wobble (`plain`, same plane and spacing), pixel by
    pixel, in chalcedony from 1 mm below the wall to 1 mm short of its end. Band
    width is 1 / |grad depth|, so the stone's outline and the cut's obliqueness
    cancel. Not for floor stones: a floor's flat bands give near-zero gradients."""
    widths: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    for c in (cut, plain):
        f = c.fields
        t = np.asarray(f.t_smooth if f.t_smooth is not None else f.t_dep, dtype=np.float64)
        ok = c.inside & np.isfinite(t) & (t < layers.chalcedony_end)
        depth = np.where(ok, t * growth_mm, np.nan)
        gy, gx = np.gradient(depth, c.spacing_mm)
        widths.append((depth, np.hypot(gx, gy)))
    (_, g), (d0, g0) = widths
    end = np.nanmax(d0)
    m = np.isfinite(d0) & np.isfinite(g) & (d0 > 1.0) & (d0 < end - 1.0) & (g > 0) & (g0 > 0)
    return float(np.percentile(g0[m] / g[m], 99)) if m.sum() > 100 else math.nan


def shift_by_depth(
    cut: SpecimenCut, plain: SpecimenCut, growth_mm: float, depths_mm: tuple[float, float, float]
) -> tuple[float, float]:
    """Roughness-grows-inward check: the RMS shift of the bands (mm), from the same
    stone cut without roughness (`plain`, same plane and spacing), in chalcedony at
    depth `depths_mm[0]`-`[1]` from the wall and at `[1]`-`[2]`. Depth is read from
    the plain cut's growth time."""
    f, g = cut.fields, plain.fields
    tf = f.t_smooth if f.t_smooth is not None else f.t_dep
    tg = g.t_smooth if g.t_smooth is not None else g.t_dep
    both = cut.inside & (f.phase == Phase.CHALCEDONY) & (g.phase == Phase.CHALCEDONY)
    both &= np.isfinite(tf) & np.isfinite(tg)
    depth = np.asarray(tg, dtype=np.float64) * growth_mm
    shift = (np.asarray(tf, dtype=np.float64) - np.asarray(tg, dtype=np.float64)) * growth_mm
    out: list[float] = []
    for lo, hi in ((depths_mm[0], depths_mm[1]), (depths_mm[1], depths_mm[2])):
        m = both & (depth >= lo) & (depth < hi)
        out.append(float(np.sqrt(np.mean(shift[m] ** 2))) if m.sum() > 20 else math.nan)
    return out[0], out[1]


def fold_share(
    specimen: Specimen, samples: int = 4000, step_mm: float = 0.05, seed: int = 0
) -> float:
    """No-fold check: at random points in the stone, step `step_mm` inward along
    the unwarped band normal and read growth time through the warp at both ends.
    Returns the share of steps (among those where the unwarped bands rise inward)
    where the warped bands fall instead: bands met in reverse, a fold. Zero for an
    unwarped stone by construction."""
    warp = specimen.warp
    if not isinstance(warp, BandRoughness):
        raise ValueError("fold_share reads a stone warped by BandRoughness")
    solid = specimen.volume.solid
    rng = np.random.default_rng(seed)
    lo, hi = (np.asarray(b, dtype=np.float64) for b in solid.bounds())
    pts = lo + rng.uniform(size=(samples, 3)) * (hi - lo)
    pts = pts[solid.contains(pts)]
    gt = specimen.growth_time
    g, _, _ = warp.frame(pts, gt)
    ahead = pts + step_mm * g
    rising = gt(ahead) - gt(pts) > 0
    if not rising.any():
        return 0.0
    w0, w1 = gt(warp(pts, gt)), gt(warp(ahead, gt))
    folded = rising & np.isfinite(w0) & np.isfinite(w1) & (w1 < w0)
    return float(folded.sum() / rising.sum())


def stray_quartz(cut: SpecimenCut) -> float:
    """Sharp-contact check: the share of quartz pixels with fewer than two quartz
    neighbours (of four), i.e. quartz scattered into the chalcedony."""
    q = cut.inside & (cut.fields.phase == Phase.QUARTZ)
    if q.sum() < 50:
        return math.nan
    p = np.pad(q, 1)
    neighbours = p[:-2, 1:-1].astype(int) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    return float((q & (neighbours < 2)).sum() / q.sum())
