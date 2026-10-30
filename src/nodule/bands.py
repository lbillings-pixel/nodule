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

- **Resolving limit *L*** (ADR 0006 section 1, choice 7a): the finest band
  period the photo shows, from the power spectrum of the same profiles,
  averaged over them in quarter-octave period bins. The noise floor is white
  noise at the quality gate's sigma, read along the same paths with the same
  bilinear sampling (and, on the ladder, the same blur), so the floor carries
  what the reading does to noise. *L* is the shortest period whose bin and the
  next coarser bin both hold `FLOOR_RATIO` times the floor's power. It is a
  couplet period (light plus dark band), C10's unit, in full pixels and as a
  share of the face's inner radius, stored with the number of profiles
  averaged (choice 7h). It reads content: a face without fine bands reports its
  finest bands, which is still a valid limit for "at least n coarser than L".
  JPEG blocking or noise correlated between pixels can put power above a white
  floor and make *L* optimistic (risk, unmeasured).
- **`C6_blank`** (choice 7g): the share of each rim-to-centre line where the
  band signal (the profile smoothed to *L*, less rule 2's central moving
  average) has local power under `FLOOR_RATIO` times the same reading of the
  noise profile; the median over the face. Where a path stopped because the
  bands lost their direction, the straight distance from its end to the face's
  deepest point counts as blank too, since the path stops before a band-free
  zone wider than `LOST_SPACINGS`; so bands beyond such a zone also count as
  unresolved. Nothing is imputed into the blank stretches. Sharp band edges
  put harmonic power at finer periods, so *L* can come out finer than the
  finest couplet (seen on drawn bands with a sharp onset).
- **N(L)** (section 2, choice 7c): the face recounted on a ladder of coarser
  limits. Rung r blurs the photo by `RUNG_BLUR` f (f = sqrt(2)^r), re-reads
  the same paths every f pixels, and adds fresh white noise of the native
  profiles' noise spread, as a coarser photo of the same stone would hold its
  own pixel noise (assumption; without it the blurred floor falls towards zero
  and window leakage reads as resolved power). *L* and the noise are found
  again at each rung (the gate's noise times the change in the noise
  profiles' spread), and the
  count runs at a band spacing of max(gate spacing, L/2). Rung 0 is C6 as
  above. The ladder stops once *L* reaches a quarter of the face's diameter;
  a rung with no bin above the floor is stored as 0 bands at that rung's own
  limit (blur and sampling) and ends the ladder. Below the native *L* nothing
  is extrapolated. On two drawn band scales of equal swing, the finer drops out
  of the count once blur takes its swing below the central share of the spread
  (`AMPLITUDES`), rungs before *L* passes its period: a step marks where a
  scale fades under the counting rule, not exactly its period.

Each value is None where too few profiles or bands were found to judge it.
Every parameter here is `effective`: a choice of measurement, not a property of
the stone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
MIN_PERIOD_PX = 2.0  # sampling: the shortest period a profile can hold (Nyquist)
BIN_STEP = 2.0**0.25  # effective: quarter-octave period bins
MIN_CYCLES = 2.0  # effective: a period is read only where a profile holds this many
FLOOR_RATIO = 2.0  # effective: power over the noise floor that counts as resolved
HALF_POWER = 5.34  # a Gaussian of sigma s halves the power of period ~5.34 s
RUNG_STEP = math.sqrt(2.0)  # effective: ladder step (ADR 0006 section 2)
RUNG_BLUR = 1.0  # effective: blur sigma per unit of a rung's sampling step; at 0.5,
# bands just under the rung's Nyquist period folded into false coarse bands (tried)
STOP_SHARE = 0.25  # effective: the ladder stops once L reaches this share of the diameter
MAX_RUNGS = 16  # effective: a guard; the stop above ends the ladder first
SHRINK_SIGMAS = 2.0  # effective: a rung's blur runs on the photo block-averaged by
# sigma / this, so wide blurs stay quick


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
    limit_px: float | None = None  # L: finest resolved couplet period, full px
    limit_share: float | None = None  # L over the face's inner radius
    blank: float | None = None  # C6_blank: median share of a profile with no resolved band
    ladder: tuple[Rung, ...] = ()  # N(L), rung 0 first


@dataclass(frozen=True)
class Rung:
    """The face counted at one limit of the ladder."""

    limit_px: float  # L at this rung (or the rung's own limit where nothing is resolved)
    count: float  # median rule-1 count at central parameters
    low: float
    high: float
    rule_gap: float


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


def spectrum(
    profiles: list[NDArray[np.float64]], step: float
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Mean power per period bin over `profiles` (sampled every `step` full px):
    each profile less its straight-line trend, Hann-windowed, scaled so white
    noise of variance s^2 gives s^2 in every bin. Bin b holds periods from
    MIN_PERIOD_PX * BIN_STEP^b; also the number of frequencies in each bin."""
    bins = int(math.log(1e6) / math.log(BIN_STEP))
    total = np.zeros(bins)
    n_in = np.zeros(bins, dtype=np.int64)
    for x in profiles:
        n = x.size
        if n < 8:
            continue
        k = np.arange(n, dtype=np.float64)
        x = x - np.polyval(np.polyfit(k, x, 1), k)
        w = np.hanning(n)
        power = np.abs(np.fft.rfft(x * w))[1:] ** 2 / float(np.sum(w * w))
        period = n / np.arange(1, power.size + 1) * step
        keep = (period >= MIN_PERIOD_PX * step) & (period <= n * step / MIN_CYCLES)
        b = np.floor(np.log(period[keep] / MIN_PERIOD_PX) / math.log(BIN_STEP)).astype(np.int64)
        b = np.clip(b, 0, bins - 1)
        np.add.at(total, b, power[keep])
        np.add.at(n_in, b, 1)
    return total / np.maximum(n_in, 1), n_in


def resolving_limit(
    signal: list[NDArray[np.float64]], noise: list[NDArray[np.float64]], step: float
) -> float | None:
    """L: the centre period (full px) of the shortest bin that holds
    `FLOOR_RATIO` times the noise profiles' power. (Asking the next coarser bin
    to pass too was tried: it missed a narrow band scale whose power sits in
    one bin.)"""
    power, n_in = spectrum(signal, step)
    floor, _ = spectrum(noise, step)
    above = np.nonzero((n_in > 0) & (power >= FLOOR_RATIO * np.maximum(floor, 1e-12)))[0]
    if above.size == 0:
        return None
    return MIN_PERIOD_PX * BIN_STEP ** (float(above[0]) + 0.5)


def blank_share(
    profile: NDArray[np.float64],
    noise: NDArray[np.float64],
    limit: float,
    spacing: float,
    unread: float = 0.0,
) -> float:
    """Share of a rim-to-centre line with no band above the floor. Along
    `profile` (one sample per full px), band signal is the profile smoothed to
    `limit` less the central moving average; its local power over one couplet
    is compared with `FLOOR_RATIO` times the mean of the same on the noise
    profile. `unread` px past the profile's end, towards the centre, count as
    blank."""

    def local(x: NDArray[np.float64]) -> NDArray[np.float64]:
        smooth = agatesim_scipy.gaussian(x, limit / HALF_POWER)
        rest = smooth - moving_average(x, CENTRAL[2] * spacing)
        return agatesim_scipy.gaussian(rest * rest, 2 * spacing)

    floor = float(np.mean(local(noise)))
    blank = float(np.sum(local(profile) < FLOOR_RATIO * floor))
    return (blank + unread) / (profile.size + unread)


def blurred(image: NDArray[np.float64], sigma: float) -> tuple[NDArray[np.float64], int]:
    """`image` blurred by `sigma` full px, at 1/m of its size: block means of m
    by m pixels (their variance (m^2 - 1)/12 counted in the blur), then a
    Gaussian for the rest. Returns the image and m."""
    m = max(1, int(sigma // SHRINK_SIGMAS))
    if m > 1:
        h, w = image.shape[0] // m, image.shape[1] // m
        image = image[: h * m, : w * m].reshape(h, m, w, m).mean(axis=(1, 3))
    rest = math.sqrt(max(sigma**2 - (m**2 - 1) / 12, 0.0)) / m
    return agatesim_scipy.gaussian(image, rest), m


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
) -> tuple[NDArray[np.float64], bool]:
    """Path (n, 2) of (row, column) from `start` inward along the band normal,
    and whether it stopped because the bands lost their direction."""
    nr, nc, coherence = field
    h, w = depth.shape
    gr = float(np.asarray(np.gradient(depth, axis=0), dtype=np.float64)[start[0], start[1]])
    gc = float(np.asarray(np.gradient(depth, axis=1), dtype=np.float64)[start[0], start[1]])
    r, c = float(start[0]), float(start[1])
    prev: tuple[float, float] | None = None
    path = [(r, c)]
    best, best_at, lost, last_clear = float(depth[start]), 0, 0.0, 0
    strayed = False
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
                strayed = True
                break
        else:
            strayed = True
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
    return np.asarray(path[: min(best_at, last_clear) + 1], dtype=np.float64), strayed


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
    ladder: bool = True,
) -> Bands:
    """C6 and C10 of a face, with L, C6_blank and (if `ladder`) N(L).
    Arguments as `agatesim.specimen.shape.measure`: `lum` is brightness at full
    resolution, `mask` the face at the working size (`scale` working px per
    full px), `noise` and `spacing_px` the quality gate's noise (grey levels)
    and band spacing (full px)."""
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
    paths: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    unread: list[float] = []  # px from where a path lost the bands to the centre
    centre = np.unravel_index(int(np.argmax(np.where(inner, depth, -1.0))), depth.shape)
    for start in rim_starts(inner, depth, starts):
        path, strayed = streamline(field, depth, inner, start, spacing)
        if path.shape[0] < 2:
            continue
        full = (path + 0.5) * k - 0.5
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(full, axis=0).T))])
        if arc[-1] >= MIN_PROFILE_SPACINGS * spacing_px:
            paths.append((full, arc))
            gap = float(np.hypot(path[-1, 0] - centre[0], path[-1, 1] - centre[1])) * k
            unread.append(gap if strayed else 0.0)
    if len(paths) < MIN_PROFILES:
        return Bands(None, None, None, None, None, len(paths), 0)

    def read(
        image: NDArray[np.float32] | NDArray[np.float64], step: float, m: int = 1
    ) -> list[NDArray[np.float64]]:
        """Brightness along every path, one sample per `step` full px, from an
        image at 1/m of full size."""
        out: list[NDArray[np.float64]] = []
        for full, arc in paths:
            s = np.arange(0.0, arc[-1], step)
            fr, fc = np.interp(s, arc, full[:, 0]), np.interp(s, arc, full[:, 1])
            if m > 1:
                fr, fc = (fr + 0.5) / m - 0.5, (fc + 0.5) / m - 0.5
            out.append(agatesim_scipy.bilinear(image, fr, fc).astype(np.float64))
        return out

    def median(values: list[float]) -> float:
        return float(np.median(values))

    def rung(limit: float, cs: list[Count]) -> Rung:
        return Rung(
            limit_px=limit,
            count=median([c.rule1 for c in cs]),
            low=median([c.low for c in cs]),
            high=median([c.high for c in cs]),
            rule_gap=median([abs(c.rule1 - c.rule2) / max(c.rule1, c.rule2, 1) for c in cs]),
        )

    image = lum.astype(np.float64)
    # A stream of its own, so C10's shuffles below draw what they drew before.
    stream = rng.spawn(1)[0]
    white = stream.normal(0.0, max(noise, 1e-6), lum.shape)

    def fresh(profiles: list[NDArray[np.float64]], sigma: float) -> list[NDArray[np.float64]]:
        return [p + stream.normal(0.0, sigma, p.size) for p in profiles]

    profiles, noise_profiles = read(lum, 1.0), read(white, 1.0)
    counts = [count(p, noise, spacing_px) for p in profiles]
    series = [c.widths for c in counts]
    radius = float(agatesim_scipy.distance_to_false(face, 1.0).max()) * k
    limit = resolving_limit(profiles, noise_profiles, 1.0)
    blank = None
    rungs: list[Rung] = []
    if limit is not None:
        at = max(spacing_px, limit / 2)
        lines = zip(profiles, noise_profiles, unread, strict=True)
        blank = median([blank_share(p, q, limit, at, u) for p, q, u in lines])
        rungs.append(rung(limit, counts))
        spread = float(np.std(np.concatenate(noise_profiles)))
        last = limit
        for r in range(1, MAX_RUNGS if ladder else 1):
            if last >= STOP_SHARE * 2 * radius:
                break
            f = RUNG_STEP**r
            sigma = RUNG_BLUR * f
            (small, m), (small_white, _) = blurred(image, sigma), blurred(white, sigma)
            signal = fresh(read(small, f, m), spread)
            floor = fresh(read(small_white, f, m), spread)
            found = resolving_limit(signal, floor, f)
            if found is None:
                rungs.append(Rung(max(MIN_PERIOD_PX * f, HALF_POWER * sigma), 0.0, 0.0, 0.0, 0.0))
                break
            rung_noise = noise * float(np.std(np.concatenate(floor))) / max(spread, 1e-12)
            at = max(spacing_px, found / 2) / f
            rungs.append(rung(found, [count(p, rung_noise, at) for p in signal]))
            last = found

    return Bands(
        count=median([c.rule1 for c in counts]),
        count_low=median([c.low for c in counts]),
        count_high=median([c.high for c in counts]),
        rule_gap=median([abs(c.rule1 - c.rule2) / max(c.rule1, c.rule2, 1) for c in counts]),
        persistence=persistence(series, rng),
        profiles=len(counts),
        widths=sum(int(w.size) for w in series),
        limit_px=limit,
        limit_share=limit / radius if limit is not None and radius > 0 else None,
        blank=blank,
        ladder=tuple(rungs),
    )
