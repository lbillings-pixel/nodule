"""Set-level compare of one stone's cuts with the library photos (ADR 0008,
sections 3 and 4; ADR 0006 section 10).

The question is whether one stone's cuts lie within the library's spread,
not whether they match its distribution (ADR 0008 section 1):

- **Scaling.** Each compared descriptor is scaled by the photos' median and
  interquartile range (robust z). A descriptor with fewer than two photo
  values, or an interquartile range of zero, is left out of the score
  (assumption: it cannot be scaled).
- **Block scores.** Values are missing in blocks, as gates fail (`BLOCKS`,
  one per measuring module). Within a block the distance between two faces
  is the mean squared scaled difference over the face's values, and a photo
  is a neighbour only if it has every value of the block the face has, so a
  face cannot hide an odd value beside photos that lack it (ADR 0008
  amendment 2). A face's block score is its mean distance to its `K`
  nearest such photos; photos are scored leave-one-out.
- **Joint score.** Each block score becomes a tail share *p*: how rarely a
  photo's block score would be as high, from a normal with the robust
  centre and spread of the cube roots of the photos' block scores
  (`effective`; Wilson-Hilferty). A rank
  would stop at the most extreme photo, so no face could ever be less
  typical than all of them; *p* has no such floor. The photos take part in
  their own block's fit; with robust estimates that leak is small
  (reasoning). The face's *m* scored blocks combine as 1 - (1 - min *p*) **
  *m* (Sidak's correction, so a face with more blocks is not less typical by
  chance; it assumes the blocks independent and is conservative where they
  are correlated). The joint score is -log10 of that: higher is less
  typical.
- **Verdict.** The stone lies within the spread if its median cut's joint
  score is at or below the `LEVEL` percentile of the photos' joint scores.
- **Block presence.** The share of faces with each block, cuts beside
  photos, is reported too: how often the gates pass is itself a finding.
- **Rejection rate.** A fake (F1-F3) must fail the verdict in at least
  `REJECT_SHARE` of `RESAMPLES` bootstrap resamples, resampling both its cuts
  and the photos (ADR 0008 section 4). The rate is reported, not just yes or
  no.
- **Null control N1.** Each photo held out in turn, scored against the rest,
  fails at about 1 - `LEVEL` by construction; this checks the code.
- **Where the line falls.** The verdict is a line, not a test whose power
  grows with the photos: more photos make the rates settle toward 0 or 1,
  and few photos add spurious fails (ADR 0008 amendment 1). So the compare
  also reports, per descriptor and for all together, the smallest shift of
  the cuts (in the photos' interquartile ranges) that crosses the line.

The method is reasoned, not taken from a source (*speculative*, ADR 0008
section 3). One stone is one sample: a pass reads only "not refuted by these
descriptors". Appearance only: never evidence for a formation mechanism
(ADR 0002). Every number here is `effective`.

C6 is recorded but not compared, as it describes the photo at its *L*
(ADR 0006 section 10). `C16_corr_censored` is a flag, not a value; C16's
correlation is compared as measured, alike for photos and cuts.
"""

from __future__ import annotations

import argparse
import time
import tomllib
import tracemalloc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from nodule._scipy import normal_cdf
from nodule.tags import (
    FROM_CONTINUITY,
    FROM_CURVE,
    FROM_DEFECTS,
    FROM_ORDER,
    FROM_RINGS,
    FROM_SPECTRUM,
    IMPORTED,
    TAGS_TOML,
)

# The 25 compared values, C7-C16 and C26's summaries (ADR 0006 section 10),
# in blocks by measuring module (ADR 0008 amendment 2). C14-C16 share the
# face image and fail together in the library, so they are one block.
BLOCKS: dict[str, tuple[str, ...]] = {
    "shape": tuple(n for n in IMPORTED if n != "C17_core"),  # agatesim.specimen.shape
    "persistence": ("C10_persistence",),  # nodule.bands
    "continuity": tuple(FROM_CONTINUITY),  # nodule.bands, C12
    "rings": tuple(FROM_RINGS),  # nodule.concentric, C13
    "image": (
        *(n for n in FROM_ORDER if n != "C16_corr_censored"),
        *FROM_CURVE,
        *FROM_DEFECTS,
    ),  # nodule.orientation, .curvature, .defects: C16, C15, C14
    "spectrum": tuple(FROM_SPECTRUM),  # nodule.spectrum, C26
}
COMPARED: tuple[str, ...] = tuple(n for block in BLOCKS.values() for n in block)
# The photo strata (ADR 0008 section 1): view kind and lighting in the tags.
STRATA = {
    "reflected": ("cut face", "reflected"),
    "transmitted": ("slab in transmitted light", "transmitted"),
}
K = 5  # nearest photos in a face's score
LEVEL = 95.0  # percentile of the photos' scores a stone's median cut must not exceed
REJECT_SHARE = 0.80  # share of resamples a must-reject fake must fail
RESAMPLES = 200  # bootstrap resamples for a rejection rate
SIZES = (15, 30, 60)  # photo subsets for verdict stability (ADR 0008 s. 4, amendment 1)
CUT_SIZES = (10, 20)  # cut subsets for the fakes' pilot (ADR 0008 amendment 6)
MOST_SHIFT = 20.0  # largest shift searched, in interquartile ranges (`effective`)
MIN_REF = 4  # fewest photo block scores to fit a block's tail (assumption)
IQR_PER_SD = 1.349  # interquartile range of a unit normal
FLOOR = 1e-300  # smallest joint tail share kept
MIN_CUTS = 5  # fewest cuts with a block for its own verdict (ADR 0008 amendment 7; assumption)
STEPS = 20  # bisection steps for a shift (resolution MOST_SHIFT / 2**STEPS)

Floats = NDArray[np.float64]
EXCLUDED_TOML = TAGS_TOML.parent / "compare-excluded.toml"  # ADR 0008 amendment 3


@dataclass(frozen=True)
class Table:
    """Faces by compared descriptor: `values[i, j]` is face i's `names[j]`,
    NaN where it is missing (a gate failed or it was never measured)."""

    ids: tuple[str, ...]
    names: tuple[str, ...]
    values: Floats
    rows: int  # rows given, including those with no compared value

    def __len__(self) -> int:
        return len(self.ids)


def _number(value: object) -> float:
    if isinstance(value, bool | int | float):
        return float(value)
    return float("nan")  # None, "unknown" and anything else


def table(
    rows: Sequence[Mapping[str, object]],
    ids: Sequence[str] | None = None,
    names: Sequence[str] = COMPARED,
) -> Table:
    """A table of `rows` (dicts by descriptor name, as `nodule.descriptors`
    writes them or the tags hold them). Rows with no compared value are left
    out, as they take no part in any score."""
    keys = list(ids) if ids is not None else [str(k) for k in range(len(rows))]
    values = np.array([[_number(r.get(n)) for n in names] for r in rows], dtype=np.float64)
    values = values.reshape(len(rows), len(names))
    keep = np.isfinite(values).any(axis=1)
    kept = tuple(k for k, y in zip(keys, keep, strict=True) if y)
    return Table(kept, tuple(names), values[keep], len(rows))


def excluded(path: Path = EXCLUDED_TOML) -> frozenset[str]:
    """The library ids the owner left out of the compare (ADR 0008
    amendment 3)."""
    photos: dict[str, object] = tomllib.loads(path.read_text("utf-8"))["photos"]
    return frozenset(photos)


def library_table(
    stratum: str = "reflected",
    finishes: Sequence[str] | None = None,
    path: Path = TAGS_TOML,
    leave_out: frozenset[str] | None = None,
) -> Table:
    """The library photos of one stratum (`STRATA`), optionally of some
    finishes only (ADR 0008 section 7c), without those in `leave_out`
    (default: `excluded()`; pass an empty set to keep every photo)."""
    view, light = STRATA[stratum]
    images: dict[str, dict[str, object]] = tomllib.loads(path.read_text("utf-8"))["images"]
    out = excluded() if leave_out is None else leave_out
    chosen = {
        k: e
        for k, e in images.items()
        if e.get("view_kind") == view
        and e.get("lighting") == light
        and (finishes is None or e.get("finish") in finishes)
        and k not in out
    }
    return table(list(chosen.values()), list(chosen))


def robust_scale(photos: Floats) -> tuple[Floats, Floats]:
    """Median and interquartile range per descriptor; the range is NaN where
    the descriptor cannot be scaled (module docstring)."""
    centre = np.full(photos.shape[1], np.nan)
    width = np.full(photos.shape[1], np.nan)
    for j in range(photos.shape[1]):
        col = photos[:, j][np.isfinite(photos[:, j])]
        if col.size < 2:
            continue
        q1, q2, q3 = np.percentile(col, [25, 50, 75])
        centre[j] = q2
        width[j] = q3 - q1 if q3 > q1 else np.nan
    return centre, width


def distances(a: Floats, b: Floats) -> Floats:
    """Mean squared difference between each row of `a` and each of `b`
    (already scaled), over the columns both have; NaN where none."""
    fa, fb = np.isfinite(a).astype(np.float64), np.isfinite(b).astype(np.float64)
    za, zb = np.where(fa > 0, a, 0.0), np.where(fb > 0, b, 0.0)
    shared = fa @ fb.T
    # sum over shared columns of (za - zb)^2, as three matrix products
    sq = (za**2) @ fb.T + fa @ (zb**2).T - 2.0 * za @ zb.T
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.maximum(sq, 0.0) / shared
    out[shared == 0] = np.nan
    return out


def knn_scores(d: Floats, k: int = K) -> Floats:
    """Each row's mean distance to its `k` nearest columns, ignoring NaN
    (excluded pairs); NaN where fewer than `k` are left."""
    filled = np.where(np.isfinite(d), d, np.inf)
    near = np.sort(filled, axis=1)[:, :k]
    out = near.mean(axis=1)
    out[~np.isfinite(out)] = np.nan
    return out


def _scaled(values: Floats, centre: Floats, width: Floats) -> Floats:
    return (values - centre) / width  # NaN columns drop out of every distance


def blocks(names: Sequence[str] = COMPARED) -> list[tuple[str, NDArray[np.intp]]]:
    """`BLOCKS` as column indices into `names`; a name in no block is a
    block of its own."""
    out: list[tuple[str, NDArray[np.intp]]] = []
    for block, cols in BLOCKS.items():
        idx = [names.index(n) for n in cols if n in names]
        if idx:
            out.append((block, np.array(idx, dtype=np.intp)))
    known = {n for cols in BLOCKS.values() for n in cols}
    out += [(n, np.array([j], dtype=np.intp)) for j, n in enumerate(names) if n not in known]
    return out


def block_scores(
    faces: Floats, photos: Floats, exclude: NDArray[np.bool_] | None = None, k: int = K
) -> Floats:
    """Each face's mean distance to its `k` nearest photos within one block
    (columns already chosen and scaled). A photo is a neighbour only if it
    has every value the face has; `exclude[i, p]` also rules a pair out.
    NaN where the face has none of the block or fewer than `k` neighbours."""
    d = distances(faces, photos)
    fa = np.isfinite(faces).astype(np.float64)
    lacks = fa @ (1.0 - np.isfinite(photos).astype(np.float64)).T > 0
    d[lacks] = np.nan
    if exclude is not None:
        d[exclude] = np.nan
    out = knn_scores(d, k)
    out[fa.sum(axis=1) == 0] = np.nan
    return out


def _tail(x: Floats, ref: Floats) -> Floats:
    """Each `x`'s upper tail share *p* among the photos' block scores `ref`:
    a normal with the robust centre and spread of the scores' cube roots
    (`effective`; Wilson and Hilferty's transform, which brings a mean of
    squares near normal and keeps the exact zeros of tied whole-number
    values finite, where a log does not). NaN where `x` is NaN, or fewer
    than `MIN_REF` photos are scored, or their scores have no spread."""
    r = ref[np.isfinite(ref)]
    out = np.full(len(x), np.nan)
    if r.size < MIN_REF:
        return out
    q1, q2, q3 = np.percentile(np.cbrt(r), [25, 50, 75])
    if q3 <= q1:
        return out
    ok = np.isfinite(x)
    z = (np.cbrt(x[ok]) - q2) / ((q3 - q1) / IQR_PER_SD)
    out[ok] = normal_cdf(-z)  # computed directly, so tiny shares keep their digits
    return out


def _joint(p: Floats) -> Floats:
    """-log10 of 1 - (1 - min p) ** m over each row's m finite block tail
    shares (Sidak); NaN where none."""
    finite = np.isfinite(p)
    m = finite.sum(axis=1)
    low = np.where(finite, p, 1.0).min(axis=1)
    out = np.full(len(p), np.nan)
    has = m > 0
    joint = -np.expm1(m[has] * np.log1p(-low[has]))
    out[has] = -np.log10(np.maximum(joint, FLOOR))
    return out


@dataclass(frozen=True)
class Spread:
    """One spread test (ADR 0008 section 3, amendment 2)."""

    cut_scores: Floats  # joint scores; NaN where a cut has no scored block
    photo_scores: Floats  # joint scores, leave-one-out
    threshold: float  # the photos' LEVEL percentile
    median: float  # the stone's median cut score
    within: bool
    outside_share: float  # share of scored cuts above the threshold
    block_names: tuple[str, ...]
    cut_blocks: Floats  # cuts by block: tail share p among the photos
    photo_blocks: Floats  # photos by block, leave-one-out
    scaled_names: tuple[str, ...]  # descriptors that could be scaled
    joint_within: bool  # the joint verdict alone (amendment 2)
    block_medians: Floats  # by block: the cuts' median -log10 p; NaN if untested
    block_lines: Floats  # by block: the photos' LEVEL percentile of -log10 p
    block_outside: NDArray[np.bool_]  # by block: median above line (amendment 7)


def spread_test(
    cuts: Floats,
    photos: Floats,
    names: Sequence[str] = COMPARED,
    origin: NDArray[np.intp] | None = None,
    k: int = K,
) -> Spread:
    """The spread test of `cuts` against `photos` (rows by `names`).
    `origin` gives each photo row's source photo, so that a resampled
    duplicate is left out of its own leave-one-out scores."""
    centre, width = robust_scale(photos)
    zp = _scaled(photos, centre, width)
    zc = _scaled(cuts, centre, width)
    src = np.arange(len(photos)) if origin is None else origin
    same = src[:, None] == src[None, :]
    parts = blocks(names)
    ub_c = np.full((len(cuts), len(parts)), np.nan)
    ub_p = np.full((len(photos), len(parts)), np.nan)
    for b, (_, cols) in enumerate(parts):
        sp = block_scores(zp[:, cols], zp[:, cols], same, k)
        sc = block_scores(zc[:, cols], zp[:, cols], None, k)
        ub_p[:, b] = _tail(sp, sp)
        ub_c[:, b] = _tail(sc, sp)
    photo_scores, cut_scores = _joint(ub_p), _joint(ub_c)
    ref = photo_scores[np.isfinite(photo_scores)]
    got = cut_scores[np.isfinite(cut_scores)]
    threshold = float(np.percentile(ref, LEVEL)) if ref.size else float("nan")
    median = float(np.median(got)) if got.size else float("nan")
    joint_within = bool(median <= threshold)  # False if either is NaN
    medians, lines = _block_lines(ub_c, ub_p)
    block_outside = medians > lines  # False where untested (NaN)
    return Spread(
        cut_scores=cut_scores,
        photo_scores=photo_scores,
        threshold=threshold,
        median=median,
        within=joint_within and not block_outside.any(),
        outside_share=float(np.mean(got > threshold)) if got.size else float("nan"),
        block_names=tuple(name for name, _ in parts),
        cut_blocks=ub_c,
        photo_blocks=ub_p,
        scaled_names=tuple(n for n, w in zip(names, width, strict=True) if np.isfinite(w)),
        joint_within=joint_within,
        block_medians=medians,
        block_lines=lines,
        block_outside=block_outside,
    )


def _block_lines(cut_p: Floats, photo_p: Floats) -> tuple[Floats, Floats]:
    """By block, over the faces that have it: the cuts' median -log10 tail
    share and the photos' `LEVEL` percentile of theirs (ADR 0008 amendment 7).
    NaN where fewer than `MIN_CUTS` cuts or `MIN_REF` photos have the block."""
    n = cut_p.shape[1]
    medians, lines = np.full(n, np.nan), np.full(n, np.nan)
    for b in range(n):
        c, ph = cut_p[:, b], photo_p[:, b]
        c, ph = c[np.isfinite(c)], ph[np.isfinite(ph)]
        if c.size >= MIN_CUTS and ph.size >= MIN_REF:
            medians[b] = np.median(-np.log10(np.maximum(c, FLOOR)))
            lines[b] = np.percentile(-np.log10(np.maximum(ph, FLOOR)), LEVEL)
    return medians, lines


def block_presence(t: Table) -> dict[str, float]:
    """Share of `t`'s rows (all given, gate failures included) with any
    value of each block."""
    f = np.isfinite(t.values)
    parts = blocks(t.names)
    return {name: float(f[:, cols].any(axis=1).sum() / t.rows) for name, cols in parts}


def rejection_rate(
    cuts: Floats,
    photos: Floats,
    rng: np.random.Generator,
    resamples: int = RESAMPLES,
    photo_count: int | None = None,
    k: int = K,
    cut_count: int | None = None,
) -> float:
    """Share of bootstrap resamples (cuts and photos both resampled) whose
    verdict is "outside". `photo_count` and `cut_count` draw that many
    photos or cuts per resample, for the power curves; default all."""
    m = len(photos) if photo_count is None else photo_count
    n = len(cuts) if cut_count is None else cut_count
    fails = 0
    for r in rng.spawn(resamples):
        ci = r.integers(len(cuts), size=n)
        pi = r.integers(len(photos), size=m)
        fails += not spread_test(cuts[ci], photos[pi], origin=pi, k=k).within
    return fails / resamples


def power_by_size(
    cuts: Floats,
    photos: Floats,
    rng: np.random.Generator,
    sizes: Sequence[int] = SIZES,
    resamples: int = RESAMPLES,
) -> dict[int, float]:
    """Rejection rate with `sizes` photos drawn per resample, and with the
    full stratum (ADR 0008 section 4, power by size). Sizes above the
    stratum's are skipped."""
    wanted = [s for s in sizes if s < len(photos)] + [len(photos)]
    streams = rng.spawn(len(wanted))
    return {
        s: rejection_rate(cuts, photos, g, resamples, photo_count=s)
        for s, g in zip(wanted, streams, strict=True)
    }


def power_by_cuts(
    cuts: Floats,
    photos: Floats,
    rng: np.random.Generator,
    sizes: Sequence[int] = CUT_SIZES,
    resamples: int = RESAMPLES,
) -> dict[int, float]:
    """Rejection rate with `sizes` cuts drawn per resample, and with all the
    cuts: how many cuts a fake needs before its verdict settles (ADR 0008
    amendment 6). Sizes at or above the cuts' count are skipped."""
    wanted = [s for s in sizes if s < len(cuts)] + [len(cuts)]
    streams = rng.spawn(len(wanted))
    return {
        s: rejection_rate(cuts, photos, g, resamples, cut_count=s)
        for s, g in zip(wanted, streams, strict=True)
    }


def crossing_shift(
    cuts: Floats,
    photos: Floats,
    column: int | None = None,
    sign: float = 1.0,
    k: int = K,
) -> float:
    """The smallest shift, in the photos' interquartile ranges, that moves
    the cuts outside the spread: added to one descriptor (`column`) or to all
    (None), upward (`sign` 1) or downward (-1). 0 if they are outside
    already; inf if no shift up to `MOST_SHIFT` does it (the compare is
    blind there, e.g. where the cuts lack the descriptor). Found by
    bisection, assuming the verdict flips once (assumption)."""
    _, width = robust_scale(photos)
    step = np.where(np.isfinite(width), width, 0.0) * sign
    if column is not None:
        step = np.where(np.arange(len(step)) == column, step, 0.0)

    def outside(shift: float) -> bool:
        return not spread_test(cuts + shift * step, photos, k=k).within

    if outside(0.0):
        return 0.0
    if not np.any(step) or not outside(MOST_SHIFT):
        return float("inf")
    lo, hi = 0.0, MOST_SHIFT
    for _ in range(STEPS):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if outside(mid) else (mid, hi)
    return hi


def crossing_shifts(
    cuts: Floats, photos: Floats, names: Sequence[str] = COMPARED, k: int = K
) -> dict[str, tuple[float, float]]:
    """`crossing_shift` down and up, for each descriptor and for "all"."""
    out: dict[str, tuple[float, float]] = {}
    for j, name in [*enumerate(names), (None, "all")]:
        down = crossing_shift(cuts, photos, j, -1.0, k)
        up = crossing_shift(cuts, photos, j, 1.0, k)
        out[name] = (down, up)
    return out


def held_out_rate(photos: Floats, k: int = K) -> float:
    """N1: each photo held out, scaled and scored against the rest; the share
    above the rest's LEVEL percentile. About 1 - LEVEL/100 by construction."""
    fails, judged = 0, 0
    for i in range(len(photos)):
        rest = np.delete(photos, i, axis=0)
        s = spread_test(photos[i : i + 1], rest, k=k)
        if np.isfinite(s.median) and np.isfinite(s.threshold):
            judged += 1
            fails += not s.within
    return fails / judged if judged else float("nan")


@dataclass(frozen=True)
class Single:
    """One descriptor, descriptive only (ADR 0008 section 3)."""

    name: str
    cuts: int
    photos: int
    percentile: float  # of the cuts' median within the photos' values
    outside_share: float  # cuts outside the photos' central LEVEL percent


def held_out_set_rate(
    photos: Floats, size: int, rng: np.random.Generator, draws: int = 200
) -> float:
    """N3 (ADR 0008 amendment 7): `size` photos drawn without replacement are
    tested as if they were cuts against the rest; the share of draws whose
    verdict is "outside". A null control of the set verdict, block lines
    included; about 0 to 5 % expected."""
    fails = 0
    for r in rng.spawn(draws):
        pick = r.permutation(len(photos))
        fails += not spread_test(photos[pick[:size]], photos[pick[size:]]).within
    return fails / draws


def per_descriptor(cuts: Floats, photos: Floats, names: Sequence[str] = COMPARED) -> list[Single]:
    """Each descriptor on its own; no pass or fail."""
    tail = (100.0 - LEVEL) / 2
    out: list[Single] = []
    for j, name in enumerate(names):
        c = cuts[:, j][np.isfinite(cuts[:, j])]
        p = photos[:, j][np.isfinite(photos[:, j])]
        if c.size == 0 or p.size < 2:
            out.append(Single(name, int(c.size), int(p.size), float("nan"), float("nan")))
            continue
        med = float(np.median(c))
        below, tied = np.count_nonzero(p < med), np.count_nonzero(p == med)
        pct = 100.0 * (below + 0.5 * tied) / p.size
        lo, hi = (float(v) for v in np.percentile(p, [tail, 100.0 - tail]))
        outside = float(np.mean((c < lo) | (c > hi)))
        out.append(Single(name, int(c.size), int(p.size), pct, outside))
    return out


def main(argv: list[str] | None = None) -> None:
    """N1 on the library's strata; with `--cuts`, the spread test of a
    `nodule.descriptors` run against them. Reports time and peak traced
    memory."""
    import json

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--cuts", type=Path, help="a nodule.descriptors JSON")
    parser.add_argument("--stratum", default="reflected", choices=sorted(STRATA))
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cut-sizes", type=int, nargs="*", help="also report rejection rates at these cut counts"
    )
    args = parser.parse_args(argv)

    tracemalloc.start()
    t0 = time.perf_counter()
    photos = library_table(args.stratum)
    print(f"{args.stratum}: {len(photos)} photos with a compared value")
    print(f"N1 held-out failure rate: {held_out_rate(photos.values):.3f} (expect ~0.05)")
    for size in (10, 20):
        n3 = held_out_set_rate(
            photos.values, size, np.random.default_rng(args.seed), args.resamples
        )
        print(f"N3 held-out sets of {size} photos: {n3:.3f} outside (expect <= ~0.05)")
    ref = spread_test(photos.values[:0], photos.values)
    print(f"photos above the bar ({ref.threshold:.1f}), with their least typical block:")
    for i in np.argsort(-np.nan_to_num(ref.photo_scores, nan=-np.inf)):
        if not ref.photo_scores[i] > ref.threshold:
            break
        worst = ref.block_names[int(np.nanargmin(ref.photo_blocks[i]))]
        print(f"  {ref.photo_scores[i]:6.1f} {worst:12s} {photos.ids[i]}")
    if args.cuts:
        rows = json.loads(args.cuts.read_text("utf-8"))
        cuts = table(rows)
        s = spread_test(cuts.values, photos.values)
        rng = np.random.default_rng(args.seed)
        rate = rejection_rate(cuts.values, photos.values, rng, args.resamples)
        print(
            f"{len(cuts)} cuts: median score {s.median:.3f} vs threshold {s.threshold:.3f}: "
            f"{'within' if s.within else 'outside'}; {s.outside_share:.0%} of cuts outside; "
            f"fails in {rate:.0%} of {args.resamples} resamples"
        )
        print(
            f"joint verdict: {'within' if s.joint_within else 'outside'}; by block (median / line):"
        )
        for name, med, line, out in zip(
            s.block_names, s.block_medians, s.block_lines, s.block_outside, strict=True
        ):
            verdict = "untested" if np.isnan(med) else ("OUTSIDE" if out else "within")
            print(f"  {name:12s} {med:7.2f} / {line:6.2f} {verdict}")
        if args.cut_sizes:
            by = power_by_cuts(cuts.values, photos.values, rng, args.cut_sizes, args.resamples)
            print(
                "rejection rate by cuts drawn: " + ", ".join(f"{n} {p:.0%}" for n, p in by.items())
            )
        shifts = crossing_shifts(cuts.values, photos.values)
        print("descriptor: cuts, photos, percentile, outside; shift to cross down / up (IQR)")
        for d in per_descriptor(cuts.values, photos.values):
            down, up = shifts[d.name]
            print(
                f"  {d.name:18s} {d.cuts:3d} {d.photos:3d} "
                f"{d.percentile:5.1f} {d.outside_share:4.0%}; {down:.2f} / {up:.2f}"
            )
        print(f"  all together: {shifts['all'][0]:.2f} / {shifts['all'][1]:.2f}")
        pc, pp = block_presence(cuts), block_presence(photos)
        shares = ", ".join(f"{b} {pc[b]:.0%} / {pp[b]:.0%}" for b in pc)
        print(f"block presence, cuts / photos: {shares}")
    elapsed = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    print(f"{elapsed:.1f} s; peak traced memory {peak / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
