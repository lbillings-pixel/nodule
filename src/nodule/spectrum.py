"""C26, the profile spectrum of a face (ADR 0004 footnote 3; N2 step 3).

ADR 0006 section 10 compares synthetic stones with the library by image
statistics at set level, "C7-C16 and the profile spectrum". C26 is that
spectrum: how a face's brightness varies with period along the rim-to-centre
profiles that C6 reads (`nodule.bands`). It describes the photo at its
resolving limit, not the stone's growth: profiles are records in depth, not
time [@holten1997], and depth spectra are distorted by a varying growth rate
[@lantink2022]. Appearance only: never evidence for a formation mechanism
(ADR 0002).

- **Estimate** (owner, 2026-11-01): the one L uses. Each profile loses its
  straight-line trend and is Hann-windowed; its periodogram is scaled so white
  noise of variance s^2 gives s^2 at every frequency (`periodogram`). Power is
  averaged over the profiles in quarter-octave bins. Multitaper and wavelet
  estimates [@lantink2022; @torrence1998] stay later candidates.
- **Curve**: each bin's share of the profiles' total power (their variance
  after detrending), on a fixed grid of period over the face's inner radius
  (owner: photos have no scale). Only bins a profile can hold, at least
  `MIN_CYCLES` periods long, are kept. A share, not the power per frequency,
  since a band's power per frequency grows with the samples a profile holds:
  the same face at half the resolution gave 2-5 times less (measured
  2026-11-01), while its shares agree.
- **Tilt** (ADR 0017, the owner's choice; it replaces a fine-side slope that
  ended at the peak's bin): the least-squares slope of log share against log
  period over a fixed band of scales, the resolved bins (at or coarser than L,
  so above the noise floor) up to `TILT_TOP_SHARE` of the inner radius, at
  least `MIN_TILT_BINS` of them. It is not a fine-side slope: on most stone
  cuts this band runs past the strongest peak, so it reads the spectrum's
  overall tilt. Which way a fake moves it depends on the spectrum: smoothing a
  stone's bands raised it on 12 of 12 planes (ADR 0016 s.16), while removing
  sub-bands from one narrow line lowers it, as the band then runs past the line
  into a steep fall. So tilts are read as a distance from the photos, never as
  a direction. White noise gives about -1, as a quarter-octave holds more
  frequencies the finer it is. The band starts at L, so tilts compare only at a
  matched limit, as ADR 0006 section 10 renders them.
- **Peak**: the largest ratio of a bin's power to a red-noise background, and
  that bin's period over the radius. The background has the AR(1) form of
  [@torrence1998, eq. 16], fitted to the face's own resolved bins by least
  absolute log residuals, so one strong peak barely moves it. The fit to the
  curve, not a lag-1 coefficient read between pixels, is this project's own
  choice: at one sample per pixel the lag-1 measures blur. [@lantink2022]
  likewise test against a robust background fitted to the spectrum. No
  significance level is claimed: neighbouring profiles of one face are far
  from independent, so the degrees of freedom are unknown; the peak is
  compared between sets as a number.

Every parameter here is `effective`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

GRID_STEP = 2.0**0.25  # effective: quarter-octave bins, as L's
FINEST_SHARE = 2.0**-12  # effective: finest period on the grid, over the inner radius
BINS = 48  # effective: the grid runs from FINEST_SHARE to 1 radius
MIN_PERIOD_PX = 2.0  # sampling: the shortest period a profile can hold (Nyquist)
MIN_CYCLES = 2.0  # effective: a period is read only where a profile holds this many
MIN_BINS = 4  # effective: resolved bins needed for the peak
TILT_TOP_SHARE = 0.25  # effective: the tilt's band ends here, over the inner radius (ADR 0017)
MIN_TILT_BINS = 3  # effective: bins needed for the tilt
ALPHAS = 1.0 - np.geomspace(1.0, 1e-4, 200)  # effective: AR(1) coefficients tried, 0 to 0.9999


@dataclass(frozen=True)
class Spectrum:
    """C26 of one face; summaries None where too few bins are resolved."""

    period_share: tuple[float, ...]  # bin centres, period over the inner radius
    power: tuple[float, ...]  # share of the profiles' total power in each bin
    tilt: float | None  # d log share / d log period, L up to TILT_TOP_SHARE
    peak: float | None  # largest power over the red-noise background
    peak_share: float | None  # that bin's period over the inner radius


def periodogram(
    x: NDArray[np.float64], step: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Periods (in the units of `step`, the spacing of the samples) and power of
    one profile, less its straight-line trend and Hann-windowed, scaled so white
    noise of variance s^2 gives s^2 at every frequency. Periods run from the
    profile's length down to two samples."""
    n = x.size
    k = np.arange(n, dtype=np.float64)
    x = x - np.polyval(np.polyfit(k, x, 1), k)
    w = np.hanning(n)
    power = np.abs(np.fft.rfft(x * w))[1:] ** 2 / float(np.sum(w * w))
    period = n / np.arange(1, power.size + 1) * step
    return period, power


def curve(
    profiles: list[NDArray[np.float64]], step: float, radius: float
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Each grid bin's share of the total power of the detrended `profiles`
    (sampled every `step` full px, on a face of inner radius `radius` full
    px), pooled over them; also the number of frequencies in each bin."""
    total = np.zeros(BINS)
    n_in = np.zeros(BINS, dtype=np.int64)
    whole = 0.0
    for x in profiles:
        n = x.size
        if n < 8:
            continue
        period, power = periodogram(x, step)
        whole += float(power.sum())
        keep = (period >= MIN_PERIOD_PX * step) & (period <= n * step / MIN_CYCLES)
        b = np.floor(np.log(period[keep] / radius / FINEST_SHARE) / math.log(GRID_STEP))
        inside = (b >= 0) & (b < BINS)
        np.add.at(total, b[inside].astype(np.int64), power[keep][inside])
        np.add.at(n_in, b[inside].astype(np.int64), 1)
    return (total / whole if whole > 0 else total), n_in


def centres() -> NDArray[np.float64]:
    """Bin centres of the grid, period over the inner radius."""
    return FINEST_SHARE * GRID_STEP ** (np.arange(BINS) + 0.5)


def red_noise(
    share: NDArray[np.float64],
    power: NDArray[np.float64],
    n_in: NDArray[np.int64],
    step: float,
    radius: float,
) -> NDArray[np.float64]:
    """The AR(1) background [@torrence1998, eq. 16], (1 - a^2) / (1 + a^2 -
    2a cos(2 pi f)) per frequency, f in cycles per sample, times the `n_in`
    frequencies of each bin, scaled and fitted to the bins' shares `power` at
    periods `share` (over the radius) by least absolute log residuals."""
    f = step / (share * radius)
    a = ALPHAS[:, None]
    per = (1 - a**2) / (1 + a**2 - 2 * a * np.cos(2 * math.pi * f[None, :]))
    shape = np.log(per * n_in[None, :])
    resid = np.log(power)[None, :] - shape
    scale = np.median(resid, axis=1, keepdims=True)
    best = int(np.argmin(np.abs(resid - scale).sum(axis=1)))
    return np.exp(shape[best] + scale[best, 0])


def measure(
    profiles: list[NDArray[np.float64]], step: float, radius: float, limit_px: float | None
) -> Spectrum | None:
    """C26 of the `profiles` (sampled every `step` full px) of a face of inner
    radius `radius` full px, whose resolving limit is `limit_px` (None: nothing
    resolved, so no summaries). None where no bin holds a period."""
    if radius <= 0:
        return None
    power, n_in = curve(profiles, step, radius)
    held = n_in > 0
    if not held.any():
        return None
    share = centres()
    out = Spectrum(
        tuple(round(float(s), 6) for s in share[held]),
        tuple(round(float(p), 5) for p in power[held]),
        None,
        None,
        None,
    )
    if limit_px is None:
        return out
    resolved = held & (power > 0) & (share * radius * GRID_STEP**0.5 >= limit_px)
    if resolved.sum() < MIN_BINS:
        return out
    s, p = share[resolved], power[resolved]
    ratio = p / red_noise(s, p, n_in[resolved], step, radius)
    top = int(np.argmax(ratio))
    band = s <= TILT_TOP_SHARE
    tilt = None
    if band.sum() >= MIN_TILT_BINS:
        tilt = float(np.polyfit(np.log10(s[band]), np.log10(p[band]), 1)[0])
    return Spectrum(out.period_share, out.power, tilt, float(ratio[top]), float(s[top]))
