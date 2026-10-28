"""Band count and persistence along rim-to-centre profiles (ADR 0004, C6 and C10).

agatesim's profiles (`agatesim.specimen.shape`) are short straight lines that
reach a few band spacings each way; they serve band shape but cannot count the
bands of a face. Here each profile starts on the face's rim and follows the
band normal inward (a streamline of the structure tensor's dominant direction,
as the reading pass's descriptor D1 asks), so it crosses each band once and
bends round with them. Inside a wide, even band there is no direction to
follow, so the path coasts straight on. It stops where it leaves the face,
where the bands lose their direction for more than `LOST_SPACINGS` (a quartz
core, a cavity), or once it has gone past the deepest point it reached (it
crossed the centre); a coasted stretch at the end is dropped. A path that
coasts into bands at a slant can turn and run along a band edge, crossing few
bands: that loses counts rather than adding them. Stopping at sharp turns was
tried and cut too many good paths (the direction flickers pixel to pixel). Like
agatesim's profiles it starts `MARGIN_SPACINGS` inside the outline, so
segmentation edges are not counted.

- **C6, band count** [@weber2010, §3.2]: the bright bands crossed, by two of
  the PEAK tool's rules: (1) maxima of a Gaussian-smoothed profile, one per
  light-dark couplet; (2) upward zero crossings of the smoothed profile about a
  wide moving average, also one per couplet. Each rule needs a minimum swing
  (agatesim's turn threshold: so many noise sigmas, or a share of the profile's
  spread, whichever is larger). Weber et al. advise reporting the count's range
  over a parameter sweep and the agreement between rules as its uncertainty, so
  the sweep covers the smoothing width (`FWHM_SPACINGS`), the minimum swing's
  share of the spread (`AMPLITUDES`) and, for rule 2, the moving average's
  width (`WINDOWS`). Reported per face: the median over profiles of the central
  rule-1 count, of the lowest and highest count over the sweep and both rules,
  and of the gap between the rules at central parameters (|n1 − n2| over the
  larger). The count runs from the margin to where the profile stops, so on an
  outline that cuts inside the rim (ADR 0004 section 8) it is a lower bound.
  On library photos (2026-10-28) the central count missed fine bands that the
  eye sees, and the range over the sweep was wide (often 3-16 on one profile):
  a photo's count is uncertain by about that much.
- **C10, persistence** [@baker2008]: lag-1 autocorrelation of the thickness
  series (distance between successive rule-1 maxima at central parameters).
  Each series is taken as log thickness with its own straight-line trend
  against band number removed, so the rim-to-centre trend (C8) is not read as
  persistence, as Baker et al. ask that detrending be stated. Series are
  pooled (the sum of lag-1 products over the sum of squares). Short series
  bias a lag-1 estimate below 0, so the mean of the same estimate on shuffled
  copies of each series is subtracted: a shuffled series gives about 0.
  Error in placing each maximum enters two neighbouring widths with opposite
  signs, which pulls lag-1 towards −0.5: evenly drawn bands with noise give
  about −0.25 here. A negative value on a face with low thickness spread (C7)
  may be that error, not alternation.

Each value is None where too few profiles or bands were found to judge it.
Every parameter here is `effective`: a choice of measurement, not a property of
the stone.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from agatesim import _scipy as agatesim_scipy  # pyright: ignore[reportPrivateUsage]
from agatesim.specimen.quality import MIN_COHERENCE, TURN_NOISE, turns, vertex
from agatesim.specimen.shape import MARGIN_SPACINGS
from numpy.typing import NDArray

STARTS = 24  # effective: profiles per face, started evenly round the rim
FIELD_SPACING_PX = 4.0  # effective: the direction field is read at a scale where a band
# spacing is at least this many pixels, so large photos stay quick
STEP_PX = 1.0  # effective: streamline step, field pixels
LOST_SPACINGS = 4.0  # effective: stop after this long without a clear band direction
# (wide even bands have none inside them, so the path coasts straight on through them)
PAST_SPACINGS = 1.0  # effective: stop once this far shallower than the deepest point
MIN_PROFILE_SPACINGS = 4.0  # effective: a profile shorter than this is not used
MIN_SWING = 1.0  # effective: grey levels; a swing smaller than this is never a band
FWHM_SPACINGS = (0.25, 0.5, 1.0)  # effective: smoothing; weber2010 ¶22 starts at about
# half a bright band's width, and a band spacing is one band
AMPLITUDES = (0.25, 0.5, 1.0)  # effective: minimum swing as a share of the spread
WINDOWS = (4.0, 8.0, 16.0)  # effective: rule 2's moving average, in band spacings
CENTRAL = (0.5, 0.5, 8.0)  # effective: the middle of each sweep
MIN_PROFILES = 8  # effective: profiles needed for C6
MIN_SERIES = 5  # effective: widths a series needs to enter C10
MIN_PERSISTENCE_WIDTHS = 40  # effective: widths needed for C10
SHUFFLES = 50  # effective: shuffled copies for C10's baseline


@dataclass(frozen=True)
class Bands:
    """Band count and persistence of one face."""

    count: float | None  # C6: median rule-1 count at central parameters
    count_low: float | None  # C6: median of the lowest count over the sweep and rules
    count_high: float | None  # C6: median of the highest
    rule_gap: float | None  # C6: median |n1 − n2| / max(n1, n2) at central parameters
    persistence: float | None  # C10: pooled lag-1, less its shuffled baseline
    profiles: int
    widths: int

    def to_json(self) -> dict[str, float | int | None]:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(self).items()}


@dataclass(frozen=True)
class Count:
    """One profile's counts."""

    rule1: int  # central parameters
    rule2: int  # central parameters
    low: int  # over the sweep and both rules
    high: int
    widths: NDArray[np.float64]  # rule 1's thickness series, central parameters, pixels


def moving_average(profile: NDArray[np.float64], window: float) -> NDArray[np.float64]:
    return agatesim_scipy.uniform(profile, max(1, round(window)))


def threshold(smooth: NDArray[np.float64], noise: float, amplitude: float, spacing: float):
    """Smallest swing that makes a band: so many noise sigmas, or `amplitude`
    of the smoothed profile's spread about the central moving average,
    whichever is larger."""
    rest = smooth - moving_average(smooth, CENTRAL[2] * spacing)
    q1, q3 = np.quantile(rest, [0.25, 0.75])
    return max(TURN_NOISE * noise, amplitude * float(q3 - q1), MIN_SWING)


def maxima(smooth: NDArray[np.float64], swing: float) -> list[float]:
    """Rule 1: positions of the maxima among the alternating turns."""
    at = turns(smooth, swing)
    if not at:
        return []
    if len(at) > 1:
        first_is_max = smooth[at[0]] > smooth[at[1]]
    else:
        first_is_max = smooth[at[0]] > float(np.median(smooth))
    start = 0 if first_is_max else 1
    return [vertex(smooth, i) for i in at[start::2]]


def crossings(smooth: NDArray[np.float64], base: NDArray[np.float64], swing: float) -> int:
    """Rule 2: upward crossings of the moving average, with hysteresis of half
    the minimum swing either side, so noise about the line is not counted."""
    rest = smooth - base
    state = 0  # +1 above, -1 below, 0 not yet known
    ups = 0
    for v in rest:
        if v >= swing / 2 and state <= 0:
            ups += state < 0
            state = 1
        elif v <= -swing / 2 and state >= 0:
            state = -1
    return ups


def count(profile: NDArray[np.float64], noise: float, spacing: float) -> Count:
    """Both rules over the parameter sweep on one profile (`spacing` in its pixels)."""
    rule1: dict[tuple[float, float], list[float]] = {}
    rule2: dict[tuple[float, float, float], int] = {}
    bases = {w: moving_average(profile, w * spacing) for w in WINDOWS}
    for fwhm in FWHM_SPACINGS:
        smooth = agatesim_scipy.gaussian(profile, fwhm * spacing / 2.3548)
        for amp in AMPLITUDES:
            swing = threshold(smooth, noise, amp, spacing)
            rule1[(fwhm, amp)] = maxima(smooth, swing)
            for w in WINDOWS:
                rule2[(fwhm, amp, w)] = crossings(smooth, bases[w], swing)
    counts = [len(v) for v in rule1.values()] + list(rule2.values())
    central = rule1[CENTRAL[:2]]
    return Count(
        rule1=len(central),
        rule2=rule2[CENTRAL],
        low=min(counts),
        high=max(counts),
        widths=np.diff(np.asarray(central, dtype=np.float64)),
    )


def _lag1_terms(series: list[NDArray[np.float64]]) -> tuple[float, float]:
    num = den = 0.0
    for w in series:
        x = np.log(w)
        k = np.arange(x.size, dtype=np.float64)
        x = x - np.polyval(np.polyfit(k, x, 1), k)
        num += float(np.dot(x[:-1], x[1:]))
        den += float(np.dot(x, x))
    return num, den


def persistence(series: list[NDArray[np.float64]], rng: np.random.Generator) -> float | None:
    """C10: pooled lag-1 autocorrelation of detrended log thickness, less the
    mean of the same on shuffled copies."""
    series = [w for w in series if w.size >= MIN_SERIES and bool(np.all(w > 0))]
    if sum(w.size for w in series) < MIN_PERSISTENCE_WIDTHS:
        return None
    num, den = _lag1_terms(series)
    if den <= 0:
        return None
    baseline = 0.0
    for _ in range(SHUFFLES):
        n, d = _lag1_terms([rng.permutation(w) for w in series])
        baseline += n / d if d > 0 else 0.0
    return num / den - baseline / SHUFFLES


def direction_field(
    lum: NDArray[np.float64], spacing: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Unit band normal (row, column parts) and coherence at each pixel, from
    the structure tensor integrated over one band spacing."""
    smooth = agatesim_scipy.gaussian(lum, max(1.0, 0.25 * spacing))
    gy = np.asarray(np.gradient(smooth, axis=0), dtype=np.float64)
    gx = np.asarray(np.gradient(smooth, axis=1), dtype=np.float64)
    jxx, jyy, jxy = (agatesim_scipy.gaussian(g, spacing) for g in (gx * gx, gy * gy, gx * gy))
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    coherence = np.hypot(jxx - jyy, 2 * jxy) / np.maximum(jxx + jyy, 1e-12)
    return np.sin(theta), np.cos(theta), coherence


def streamline(
    field: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    depth: NDArray[np.float64],
    inside: NDArray[np.bool_],
    start: tuple[int, int],
    spacing: float,
) -> NDArray[np.float64]:
    """Path (n, 2) of (row, column) from `start` inward along the band normal."""
    nr, nc, coherence = field
    h, w = depth.shape
    gr = float(np.asarray(np.gradient(depth, axis=0), dtype=np.float64)[start[0], start[1]])
    gc = float(np.asarray(np.gradient(depth, axis=1), dtype=np.float64)[start[0], start[1]])
    r, c = float(start[0]), float(start[1])
    prev: tuple[float, float] | None = None
    path = [(r, c)]
    best, best_at, lost, last_clear = float(depth[start]), 0, 0.0, 0
    for _ in range(int(3 * depth.max() / STEP_PX) + 1):
        i, j = round(r), round(c)
        if coherence[i, j] >= MIN_COHERENCE:
            dr, dc = float(nr[i, j]), float(nc[i, j])
            ref = prev if prev is not None else (gr, gc)
            if dr * ref[0] + dc * ref[1] < 0:
                dr, dc = -dr, -dc
            lost, last_clear = 0.0, len(path) - 1
        elif prev is not None:
            dr, dc = prev
            lost += STEP_PX
            if lost > LOST_SPACINGS * spacing:
                break
        else:
            break
        r, c = r + STEP_PX * dr, c + STEP_PX * dc
        i, j = round(r), round(c)
        if not (0 <= i < h and 0 <= j < w and inside[i, j]):
            break
        path.append((r, c))
        prev = (dr, dc)
        if depth[i, j] > best:
            best, best_at = float(depth[i, j]), len(path) - 1
        elif best - depth[i, j] > PAST_SPACINGS * spacing:
            break
    return np.asarray(path[: min(best_at, last_clear) + 1], dtype=np.float64)


def rim_starts(
    inner: NDArray[np.bool_], depth: NDArray[np.float64], n: int
) -> list[tuple[int, int]]:
    """`n` points on the rim of `inner`, evenly spaced in angle about its deepest point."""
    edge = inner & ~agatesim_scipy.erode(inner, 1)
    rows, cols = np.nonzero(edge)
    if rows.size == 0:
        return []
    cr, cc = np.unravel_index(int(np.argmax(np.where(inner, depth, -1.0))), depth.shape)
    order = np.argsort(np.arctan2(rows - cr, cols - cc), kind="stable")
    picks = order[np.linspace(0, rows.size, n, endpoint=False).astype(np.int64)]
    return [(int(rows[k]), int(cols[k])) for k in picks]


def measure(
    lum: NDArray[np.float32],
    mask: NDArray[np.bool_],
    scale: float,
    noise: float,
    spacing_px: float,
    *,
    rng: np.random.Generator,
    starts: int = STARTS,
) -> Bands:
    """C6 and C10 of a face. Arguments as `agatesim.specimen.shape.measure`:
    `lum` is brightness at full resolution, `mask` the face at the working size
    (`scale` working px per full px), `noise` and `spacing_px` the quality
    gate's noise (grey levels) and band spacing (full px)."""
    k = max(1, int(spacing_px // FIELD_SPACING_PX))
    h, w = lum.shape[0] // k, lum.shape[1] // k
    coarse = lum[: h * k, : w * k].astype(np.float64).reshape(h, k, w, k).mean(axis=(1, 3))
    # The working mask, read at each coarse pixel's centre.
    centres_r: NDArray[np.float64] = (np.arange(h, dtype=np.float64) + 0.5) * (k * scale)
    centres_c: NDArray[np.float64] = (np.arange(w, dtype=np.float64) + 0.5) * (k * scale)
    rows = np.clip(centres_r.astype(np.int64), 0, mask.shape[0] - 1)
    cols = np.clip(centres_c.astype(np.int64), 0, mask.shape[1] - 1)
    face = mask[np.ix_(rows, cols)]
    spacing = spacing_px / k
    inner = agatesim_scipy.erode(face, max(1, math.ceil(MARGIN_SPACINGS * spacing)))
    depth = agatesim_scipy.distance_to_false(inner, 1.0)
    field = direction_field(coarse, spacing)
    counts: list[Count] = []
    for start in rim_starts(inner, depth, starts):
        path = streamline(field, depth, inner, start, spacing)
        if path.shape[0] < 2:
            continue
        # Resample at one full pixel along the path, and read full-resolution brightness.
        full = (path + 0.5) * k - 0.5
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(full, axis=0).T))])
        if arc[-1] < MIN_PROFILE_SPACINGS * spacing_px:
            continue
        s = np.arange(0.0, arc[-1], 1.0)
        fr, fc = np.interp(s, arc, full[:, 0]), np.interp(s, arc, full[:, 1])
        profile = agatesim_scipy.bilinear(lum, fr, fc).astype(np.float64)
        counts.append(count(profile, noise, spacing_px))
    if len(counts) < MIN_PROFILES:
        return Bands(None, None, None, None, None, len(counts), 0)
    series = [c.widths for c in counts]

    def median(values: list[float]) -> float:
        return float(np.median(values))

    return Bands(
        count=median([c.rule1 for c in counts]),
        count_low=median([c.low for c in counts]),
        count_high=median([c.high for c in counts]),
        rule_gap=median([abs(c.rule1 - c.rule2) / max(c.rule1, c.rule2, 1) for c in counts]),
        persistence=persistence(series, rng),
        profiles=len(counts),
        widths=sum(int(w.size) for w in series),
    )
