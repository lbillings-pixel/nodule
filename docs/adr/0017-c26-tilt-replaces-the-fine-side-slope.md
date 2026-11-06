# 17. C26: an overall tilt replaces the fine-side slope

- Status: accepted (owner, 2026-11-06). The slope question is closed.
- Date: 2026-11-06
- Builds on: [ADR 0006](0006-band-counts-as-censored-records.md) section 10
  (set-level comparison), [ADR 0016](0016-renderer-materials-that-vary-with-place.md)
  sections 13, 14 and 16 (the slope's instability, the fit ranges, the F1 control).

## Context

C26's slope ran from L to the spectrum's strongest peak. Section 13 of ADR 0016
found it jumps as the peak changes bin. Two tests then judged it against other
fit ranges on the same planes:

| fit range | steadiness: median change, set 4 vs plain (25 cuts) | F1 change over ordinary change (12 cuts) | F1 upward | F1 rule |
|---|---|---|---|---|
| L to the peak (old) | 0.28 | 2.16 | 7/12 | fail |
| L to 0.1 | 0.19 | 7.54 | 12/12 | pass |
| **L to 0.25** | **0.13** | 4.27 | 12/12 | pass |

The old range failed both. L to 0.25 was the only one whose gain over it cleared
the steadiness rule, which was fixed in advance; L to 0.1 did better on the F1
test, but choosing it would have meant changing the rule after the results.

## Decision

1. **`C26_tilt` replaces `C26_slope`** among the 25 compared values: the
   least-squares slope of log share against log period over the resolved bins, from
   L to `TILT_TOP_SHARE` = 0.25 of the inner radius, at least `MIN_TILT_BINS` = 3
   (both `effective`). The curve, peak and peak's period are unchanged.
2. **It has a new name because it measures something else.** The band runs past the
   strongest peak on most stone cuts, so this is an overall tilt across a fixed band
   of scales, not a fine-side slope.
3. **It is read as a distance from the photos, never as a direction** (below).
4. **The 25 fresh cuts offered in ADR 0016 section 16 are not run** (the owner's
   choice). They could re-check the choice of range only; they cannot test the slope
   gap, as half B is spent and no photos are held out.

## What was found while adopting it

- **Direction is not a rule** (*verified*). Smoothing raised the tilt on 12 of 12
  stone planes, but on the synthetic rings of `tests/test_spectrum.py` removing
  sub-bands lowers it: −6.6 without, −1.6 with. There is one narrow line there, so
  the band runs past it into a steep fall (*plausible* as the reason). The old slope
  gave 13.2 and 2.5 on the same rings.
- **The fakes lie outside the photos** (half A, 21 photos with a tilt, range 0.15 to
  1.66, median 0.88; already seen, so *speculative*). F1's median is 1.92, above all
  of them, range over its 12 cuts 1.11 to 3.16. Both synthetic rings lie below all of
  them. Set 4's median is 0.98 on its first 12 and 25 cuts (57th and 62nd percentile),
  plain's 0.85 on 12 (43rd).
- **Photo values** (*verified*). A full re-tag of all 261 photos changes `C26_slope`
  to `C26_tilt` and nothing else. 72 photos have a tilt, the same number that had a
  slope. Rebuilt from the saved, rounded curves, 70 lie within 0.02; the other two
  (gaps 0.16, 0.09) have a finest bin within L's rounding of the limit, and match to
  0.0004 without it. 1,687 s, peak traced 443 MiB, against the 730 s noted earlier.

## Limits

- **This does not show the slope gap was an artefact, and it does not show the
  stone's fine bands are right.** The gap shrinks because the measure changed. The
  owner's eye and the other statistics stand as they were.
- *Plausible* as a choice of measure: one stone, one rendered fake, 12 planes for the
  control and 25 for steadiness, with the range chosen on those same cuts and no
  fresh repeat. F2 to F4 have not been judged by it. Its lead over L to 0.1 is not
  settled (ADR 0016 section 14), and a blunter measure can ignore something real.
- A tilt compares only at a matched limit, as the slope did. Appearance only: never
  evidence for a formation mechanism (ADR 0002).

## Consequences

- Reports from before this ADR quote the old slope and stand as written. Their rows
  hold no tilt. Only 62 stored cuts hold curves that could be rescored: set 4 and
  plain, 25 each, and F1, 12.
- `compare.table` reads a row without a compared value as missing, so old rows would
  not raise an error. The run script's checkpoint key now names the measure, so an
  old checkpoint is refused rather than mixed. A 100-cut set 4 run under the tilt
  starts afresh: about 5 h at today's speed, not needed now.
- 467 tests pass. The control is `test_fine_structure_moves_the_tilt`; a new test
  covers too few bins, with its control.
