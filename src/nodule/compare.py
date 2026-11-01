"""Set-level compare of one stone's cuts with the library photos (ADR 0008,
sections 3 and 4; ADR 0006 section 10).

The question is whether one stone's cuts lie within the library's spread,
not whether they match its distribution (ADR 0008 section 1):

- **Scaling.** Each compared descriptor is scaled by the photos' median and
  interquartile range (robust z). A descriptor with fewer than two photo
  values, or an interquartile range of zero, is left out of the score
  (assumption: it cannot be scaled).
- **Joint score.** The distance between two faces is the mean squared scaled
  difference over the descriptors both faces have. A face's score is its
  mean distance to its `K` nearest photos. Photos are scored leave-one-out,
  which gives the reference distribution.
- **Verdict.** The stone lies within the spread if its median cut's score
  is at or below the `LEVEL` percentile of the photos' leave-one-out scores.
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

# The 25 compared values: C7-C16 and C26's summaries (ADR 0006 section 10).
COMPARED: tuple[str, ...] = (
    *(n for n in IMPORTED if n != "C17_core"),
    "C10_persistence",
    *FROM_CONTINUITY,
    *FROM_RINGS,
    *(n for n in FROM_ORDER if n != "C16_corr_censored"),
    *FROM_CURVE,
    *FROM_DEFECTS,
    *FROM_SPECTRUM,
)
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
MOST_SHIFT = 20.0  # largest shift searched, in interquartile ranges (`effective`)
STEPS = 20  # bisection steps for a shift (resolution MOST_SHIFT / 2**STEPS)

Floats = NDArray[np.float64]


@dataclass(frozen=True)
class Table:
    """Faces by compared descriptor: `values[i, j]` is face i's `names[j]`,
    NaN where it is missing (a gate failed or it was never measured)."""

    ids: tuple[str, ...]
    names: tuple[str, ...]
    values: Floats

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
    return Table(tuple(k for k, y in zip(keys, keep, strict=True) if y), tuple(names), values[keep])


def library_table(
    stratum: str = "reflected", finishes: Sequence[str] | None = None, path: Path = TAGS_TOML
) -> Table:
    """The library photos of one stratum (`STRATA`), optionally of some
    finishes only (ADR 0008 section 7c)."""
    view, light = STRATA[stratum]
    images: dict[str, dict[str, object]] = tomllib.loads(path.read_text("utf-8"))["images"]
    chosen = {
        k: e
        for k, e in images.items()
        if e.get("view_kind") == view
        and e.get("lighting") == light
        and (finishes is None or e.get("finish") in finishes)
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


@dataclass(frozen=True)
class Spread:
    """One spread test (ADR 0008 section 3)."""

    cut_scores: Floats  # NaN where a cut shares too little with the photos
    photo_scores: Floats  # leave-one-out
    threshold: float  # the photos' LEVEL percentile
    median: float  # the stone's median cut score
    within: bool
    outside_share: float  # share of scored cuts above the threshold
    scaled_names: tuple[str, ...]  # descriptors that could be scaled


def spread_test(
    cuts: Floats,
    photos: Floats,
    names: Sequence[str] = COMPARED,
    origin: NDArray[np.intp] | None = None,
    k: int = K,
) -> Spread:
    """The spread test of `cuts` against `photos` (rows by `names`).
    `origin` gives each photo row's source photo, so that a resampled
    duplicate is left out of its own leave-one-out score."""
    centre, width = robust_scale(photos)
    zp = _scaled(photos, centre, width)
    zc = _scaled(cuts, centre, width)
    dp = distances(zp, zp)
    src = np.arange(len(photos)) if origin is None else origin
    dp[src[:, None] == src[None, :]] = np.nan
    photo_scores = knn_scores(dp, k)
    cut_scores = knn_scores(distances(zc, zp), k)
    ref = photo_scores[np.isfinite(photo_scores)]
    got = cut_scores[np.isfinite(cut_scores)]
    threshold = float(np.percentile(ref, LEVEL)) if ref.size else float("nan")
    median = float(np.median(got)) if got.size else float("nan")
    return Spread(
        cut_scores=cut_scores,
        photo_scores=photo_scores,
        threshold=threshold,
        median=median,
        within=bool(median <= threshold),  # False if either is NaN
        outside_share=float(np.mean(got > threshold)) if got.size else float("nan"),
        scaled_names=tuple(n for n, w in zip(names, width, strict=True) if np.isfinite(w)),
    )


def rejection_rate(
    cuts: Floats,
    photos: Floats,
    rng: np.random.Generator,
    resamples: int = RESAMPLES,
    photo_count: int | None = None,
    k: int = K,
) -> float:
    """Share of bootstrap resamples (cuts and photos both resampled) whose
    verdict is "outside". `photo_count` draws that many photos per resample,
    for the power curves; default all."""
    m = len(photos) if photo_count is None else photo_count
    fails = 0
    for r in rng.spawn(resamples):
        ci = r.integers(len(cuts), size=len(cuts))
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
    args = parser.parse_args(argv)

    tracemalloc.start()
    t0 = time.perf_counter()
    photos = library_table(args.stratum)
    print(f"{args.stratum}: {len(photos)} photos with a compared value")
    print(f"N1 held-out failure rate: {held_out_rate(photos.values):.3f} (expect ~0.05)")
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
        shifts = crossing_shifts(cuts.values, photos.values)
        print("descriptor: cuts, photos, percentile, outside; shift to cross down / up (IQR)")
        for d in per_descriptor(cuts.values, photos.values):
            down, up = shifts[d.name]
            print(
                f"  {d.name:18s} {d.cuts:3d} {d.photos:3d} "
                f"{d.percentile:5.1f} {d.outside_share:4.0%}; {down:.2f} / {up:.2f}"
            )
        print(f"  all together: {shifts['all'][0]:.2f} / {shifts['all'][1]:.2f}")
    elapsed = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    print(f"{elapsed:.1f} s; peak traced memory {peak / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
