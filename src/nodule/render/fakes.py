"""Obvious fakes for the set-level compare (ADR 0008 s.4 and amendment 4).

Each fake changes only the band record (`BandRecord`: what fills each growth
step) and goes through the same renderer and descriptors as the stone. A fake is
built once from the whole stone's record (`stone_record`), never per cut: a
cut sees only the steps it crosses, so a fake built per cut would draw a
different stone in every cut.

The record is read as runs: stretches of steps with the same deposit (one run =
one band). A fake moves only the chalcedony runs before the last one. The last
run is open (it runs to the step where the cavity closed, so its width is set
by the geometry), and other phases (quartz) are not bands under test; both stay
on the steps where they were. Runs between two fixed runs form a segment, and a
fake keeps each segment's first and last steps.

- F1 `smoothed`: no structure finer than the bands. Impurity and translucency
  are smoothed by a Gaussian whose frequency width is the segment's typical
  band pair, P = twice the median run width (sigma = P / 2 pi steps): such a
  pair keeps e^-0.5 of its amplitude and its third harmonic about 1 %, while
  a one-step line inside a band keeps about 1 / (sigma sqrt 2 pi) of its
  contrast (a quarter at n2-slice's 10-step pair). The median, not the
  shortest pair, so that sub-bands (short runs themselves) cannot set the
  scale that is to remove them. Phase and pigment are categories and stay.
- F2 `remapped`: wrong spacing ratios. Within a segment each band takes the
  width of the band at the opposite rank (the thickest becomes the thinnest),
  so the deposits keep their order while thick and thin trade places.
- F3 `regular`: every band of a segment equally wide, to the step (widths
  differ by at most one step, the record's resolution).
- F4 `phase_randomised` (diagnostic only): impurity and translucency of a
  segment replaced by an iterated amplitude-adjusted Fourier (IAAFT) surrogate
  [@schreiber2000, s. 4.3], multivariate, with the same random phase shift in
  both fields so that their cross-spectrum is kept [@schreiber2000, s. 4.6,
  eqs. 18-20]. The result of the rank-ordering step is kept, so each field
  keeps exactly its own values (ranges hold) and approximately its spectrum
  [@schreiber2000, s. 4.3]; runs, edges and the order of bands are lost.
  The method treats a segment as periodic; the jump from its last to its
  first value leaks power to high frequencies [@schreiber2000, s. 4.5], which
  `end_jump` measures. Phase and pigment stay.

Choices of sigma, rank reversal and the fixed runs are `effective`, made here
(ADR 0008 amendment 4), not taken from a source. A fake is procedural and is
never evidence about agate; it tests whether the compare can see an axis.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from agatesim.fields import Phase
from numpy.typing import NDArray

from nodule.render.record import FIELDS, STEPS, BandRecord
from nodule.specimen import Specimen

STONE_POINTS = 1_000_000  # points drawn through the stone's bounds (~1 s, ~80 MB)
CHUNK_POINTS = 250_000
CONTINUOUS = ("impurity", "translucency")

IntArray = NDArray[np.intp]


def stone_record(
    specimen: Specimen, rng: np.random.Generator, points: int = STONE_POINTS
) -> BandRecord:
    """The whole stone's record, from points drawn uniformly through its bounds.
    Steps too thin to catch a point stay unseen and take their nearest neighbour,
    as in a cut."""
    lo, hi = specimen.volume.solid.bounds()
    out = BandRecord.empty()
    for a in range(0, points, CHUNK_POINTS):
        p = rng.uniform(lo, hi, (min(CHUNK_POINTS, points - a), 3))
        f, inside = specimen.step_fields(p)  # one material per step (ADR 0019)
        out = out.merge(BandRecord.of(f, inside))
    return out


@dataclass(frozen=True)
class Runs:
    """The record as runs of equal deposit over all STEPS (unseen steps filled
    from their nearest seen step)."""

    start: IntArray  # first step of each run
    width: IntArray  # steps in each run
    movable: NDArray[np.bool_]  # chalcedony and not the last run

    def segments(self) -> list[IntArray]:
        """Indices of the runs in each stretch of movable runs."""
        idx = np.flatnonzero(self.movable)
        if not len(idx):
            return []
        return [s for s in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(s)]


def dense(record: BandRecord) -> dict[str, NDArray[np.generic]]:
    """Every field at every step, unseen steps from their nearest seen one."""
    src = record.nearest_seen()
    return {name: getattr(record, name)[src] for name in FIELDS}


def runs(record: BandRecord) -> Runs:
    d = dense(record)
    keys = np.stack([d[name].astype(np.float64) for name in FIELDS], 1)
    start = np.flatnonzero(np.r_[True, (np.diff(keys, axis=0) != 0).any(1)])
    width = np.diff(np.r_[start, STEPS])
    movable = d["phase"][start] == Phase.CHALCEDONY
    movable[-1] = False
    return Runs(start=start, width=width, movable=movable)


def _from_dense(d: dict[str, NDArray[np.generic]]) -> BandRecord:
    return BandRecord(
        seen=np.ones(STEPS, bool),
        phase=d["phase"].astype(np.uint8),
        pigment=d["pigment"].astype(np.uint8),
        impurity=d["impurity"].astype(np.float32),
        translucency=d["translucency"].astype(np.float32),
        shifts=d["shifts"].astype(bool),
    )


def _rewidth(record: BandRecord, widths: Callable[[IntArray], IntArray]) -> BandRecord:
    """Each segment's runs, in their order, at new widths with the same sum."""
    r, d = runs(record), dense(record)
    out = {name: col.copy() for name, col in d.items()}
    for seg in r.segments():
        new = widths(r.width[seg])
        if new.sum() != r.width[seg].sum() or (new < 1).any():
            raise ValueError("new widths must be positive and fill the segment")
        k = r.start[seg[0]]
        for run, w in zip(seg, new, strict=True):
            for name in FIELDS:
                out[name][k : k + w] = d[name][r.start[run]]
            k += w
    return _from_dense(out)


def reversed_ranks(width: IntArray) -> IntArray:
    """Each band takes the width of the band at the opposite rank (ties by order)."""
    order = np.argsort(width, kind="stable")
    out = np.empty_like(width)
    out[order] = width[order[::-1]]
    return out


def equal_widths(width: IntArray) -> IntArray:
    """The same total in equal widths, to the step (most regular split)."""
    edges = np.round(np.linspace(0, width.sum(), len(width) + 1)).astype(np.intp)
    return np.diff(edges)


def gaussian(x: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    """x smoothed by a Gaussian of `sigma` samples, mirrored at both ends."""
    half = max(1, int(np.ceil(4 * sigma)))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    padded = np.pad(x, half, mode="symmetric")
    return np.convolve(padded, k / k.sum(), mode="valid").astype(np.float64)


def band_pair(width: IntArray) -> float:
    """A segment's typical band pair in steps: twice its median run width."""
    return 2.0 * float(np.median(width))


def smoothed(record: BandRecord) -> BandRecord:
    """F1: impurity and translucency smoothed below the band scale (module docstring)."""
    r, d = runs(record), dense(record)
    out = {name: col.copy() for name, col in d.items()}
    for seg in r.segments():
        if len(seg) < 2:
            continue
        a, b = r.start[seg[0]], r.start[seg[-1]] + r.width[seg[-1]]
        sigma = band_pair(r.width[seg]) / (2 * np.pi)
        for name in CONTINUOUS:
            out[name][a:b] = gaussian(d[name][a:b].astype(np.float64), sigma)
    return _from_dense(out)


def _rank_ordered(s: NDArray[np.float64], sorted_values: NDArray[np.float64]):
    """Each row of s replaced by its own row of sorted values, in s's rank order."""
    out = np.empty_like(s)
    np.put_along_axis(out, np.argsort(s, axis=1, kind="stable"), sorted_values, axis=1)
    return out


def iaaft(
    x: NDArray[np.float64], rng: np.random.Generator, most: int = 1000
) -> tuple[NDArray[np.float64], int]:
    """Multivariate IAAFT surrogate of rows x (channels, n) [@schreiber2000, ss. 4.3,
    4.6]: each row keeps its values exactly and its amplitude spectrum
    approximately; one phase shift per frequency is shared by all rows. Returns
    the surrogate and the iterations taken (`most` if no fixed point was reached)."""
    n = x.shape[1]
    spec = np.fft.rfft(x, axis=1)
    amp, rho = np.abs(spec), np.angle(spec)
    sorted_values = np.sort(x, axis=1)

    def with_shift(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
        alpha[0] = 0.0  # the mean stays real
        if n % 2 == 0:
            alpha[-1] = 0.0  # so does the Nyquist term
        return np.fft.irfft(amp * np.exp(1j * (rho + alpha)), n, axis=1)

    r = _rank_ordered(with_shift(rng.uniform(0.0, 2 * np.pi, amp.shape[1])), sorted_values)
    for i in range(most):
        d = np.angle(np.fft.rfft(r, axis=1)) - rho
        alpha = np.arctan2(np.sin(d).sum(0), np.cos(d).sum(0))  # eqs. 18-20
        new = _rank_ordered(with_shift(alpha), sorted_values)
        if np.array_equal(new, r):
            return r, i
        r = new
    return r, most


def end_jump(x: NDArray[np.float64]) -> float:
    """(first - last)^2 over the sum of squared deviations [@schreiber2000, eq. 14]:
    how far a series is from periodic."""
    var = float(((x - x.mean()) ** 2).sum())
    return float((x[0] - x[-1]) ** 2 / var) if var > 0 else 0.0


def phase_randomised(record: BandRecord, rng: np.random.Generator) -> BandRecord:
    """F4: impurity and translucency of each segment as one multivariate IAAFT
    surrogate (module docstring). Fields constant over a segment stay."""
    r, d = runs(record), dense(record)
    out = {name: col.copy() for name, col in d.items()}
    for seg in r.segments():
        a, b = r.start[seg[0]], r.start[seg[-1]] + r.width[seg[-1]]
        names = [n for n in CONTINUOUS if np.ptp(d[n][a:b].astype(np.float64)) > 0]
        if len(seg) < 2 or not names:
            continue
        x = np.stack([d[n][a:b].astype(np.float64) for n in names])
        sur, _ = iaaft(x, rng)
        for name, row in zip(names, sur, strict=True):
            out[name][a:b] = row
    return _from_dense(out)


def remapped(record: BandRecord) -> BandRecord:
    """F2: wrong spacing ratios (module docstring)."""
    return _rewidth(record, reversed_ranks)


def regular(record: BandRecord) -> BandRecord:
    """F3: perfectly regular spacing (module docstring)."""
    return _rewidth(record, equal_widths)
