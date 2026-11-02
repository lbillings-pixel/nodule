# 11. Band regularity for the stone: a tuning step on the record

- Status: proposed
- Date: 2026-11-02
- Builds on: [ADR 0008](0008-set-level-render-and-compare.md) choice 7b
  (tuning limited to optics and exposure, "not band geometry"),
  [ADR 0009](0009-when-photos-run-out-sub-resolution-structure.md) point 4
  (band regularity is resolved structure, so photos can constrain it),
  [ADR 0010](0010-optics-tuning-step-1-pores.md) (optics step 1, run at
  0.7 / 0.7) and [ADR 0002](0002-founding-nodule.md) (graded claims).

## Context

- **The stone failed on its record, not its render.** The `n2-slice`
  preset keeps agatesim's defaults: `period_variability` 0.35 (the
  log-normal sigma of band durations) and `alternation` 0.7 (how strictly
  milky and clear bands alternate). Its cuts were outside the photos on the
  spectrum block alone. The spectrum's peak was 9.70 in renders, against a
  photo 95th percentile of 4.78–5.54. Blur and sizing were ruled out as the
  main cause; the peak was already there with no blur or noise (6.88)
  (*verified*; runs/stone-2026-11-02/diagnosis.md, not committed).
- **A 1D toy pointed at the record.** Profiles straight from agatesim's
  `make_bands`, 30 faces each, gave a median peak of 18.8 at 0.35 / 0.7 and
  4.7 at 0.7 / 0.7 (*plausible*: a relative proxy with no wobble, 2D or
  render).
- **Four variants on the CPU stand-in** (30 cuts each, seed 0; looks from
  half A of the 97 reflected photos; judged on A and held-out B; 291–357 s
  and 841–855 MB each; *verified*):

| pv / alt | spectrum vs A (score / line) | vs B | slope pct A / B | peak pct A / B |
|---|---|---|---|---|
| 0.35 / 0.7 (default) | 3.0 / 1.4 **outside** | 4.6 / 2.0 **outside** (continuity also outside) | 86 / 90 | 95 / 95 |
| **0.7 / 0.7** | 0.33 within | 0.20 within | 76 / 71 | 62 / 48 |
| 1.0 / 0.7 | 0.31 within | 0.30 within | 10 / 14 | 52 / 29 |
| 0.7 / 0.3 | 0.50 within | 0.49 within | 24 / 29 | 76 / 67 |

- **Rendered, 0.7 / 0.7 is within both halves** after optics step 1, on
  all 20 cuts and the noise-matched ones (ADR 0010, *verified*).
- **The photos give a range, not a value.** Any of the three changed
  settings is within. Doubling the spread (0.7) or tripling it (1.0) both
  pass at alternation 0.7; loosening alternation instead (0.3) passes too.
  At 1.0 the slope sits low in the photos (10th–14th percentile), but still
  within.
- **What the sources say.** No source we hold measures how much agate band
  durations or widths vary.
  - Holten, Jamtveit and Meakin found agate greyscale profiles self-affine
    over about two decades, with Hurst exponents 0.36 ± 0.05 and
    0.49 ± 0.05 [@holten1998]. Read as abstract only: the paper is closed
    and has not been obtained, so it is not known whether the profiles ran
    across the bands, and the exponents cannot be turned into a sigma.
  - In zoned crystals (not agate), the same group found no distinct
    periodic components; a periodic pattern would give a width-method slope
    near 1 below its period and near 0 above it [@holten1997]. That fits
    irregular bands in kind; it is an analogue, not a measurement of agate.
  - Band width in agate slices "is not constant" and changes fastest near
    the rim, a systematic trend from rim to centre [@goldbaum2019]. Some of
    the photos' irregularity may therefore be a trend, which agatesim's
    record does not have, rather than random spread.
- **Order of events.** As in ADR 0010, the variant runs were diagnostic,
  made before this ADR. All four were reported against half B as well as A.

## Decision

### 1. This is tuning of band geometry

ADR 0008 7b allowed optics and exposure only. Changing the record's
regularity because the stone failed against the photos is band-geometry
tuning, so it is recorded as its own step, R1, and it changes what the
verdict can claim:

- The verdict states it comes after tuning of optics (ADR 0010) **and**
  band regularity (this ADR).
- The spectrum block, and the continuity block, which the default failed on
  half B, no longer count as passed tests of the stone. They were used to
  choose the value. The other blocks still count.
- The default's failure stays on the record: agatesim's 0.35 / 0.7 is too
  regular for these photos at their resolution (*verified* for the stand-in
  and the render on this stone). That is a finding in its own right, and it
  shows the spectrum block can reject an over-regular record (a negative
  control in effect, though it was not planned as one).

### 2. The value (choice 11a)

- *(Recommended)* **A. `period_variability` 0.7, `alternation` 0.7.**
  - The rule, stated now but after the runs: the smallest change from
    agatesim's defaults, among the settings tried, that puts every block
    within on half A. One parameter changes, from 0.35 to 0.7; 1.0 changes
    it more, and 0.7 / 0.3 changes two.
  - The rule uses half A only. Half B has been seen and can no longer be
    called untouched for these two parameters.
  - It is also the setting the rendered 20-cut check and the optics step
    already ran, so the stone run needs no new check first.
- B. **Keep the defaults and report the stone outside on the spectrum.** It
  is the most honest verdict, but the N2 slice then ends with a stone known
  to fail on its record, and the blind test (ADR 0008 section 5) would be
  run on a stone that fails by machine.
- C. **1.0 / 0.7.** It has the lowest joint score on A (0.5 against 0.9),
  but the slope sits in the photos' lowest tenth to seventh, and it is a
  larger step from the default with no source behind it.

### 3. Where the value lives

In nodule's `n2-slice` preset (`[volume.model]`), written out explicitly
with a comment pointing here. agatesim's defaults stay as they are: they
serve agatesim's own model, and nothing in this project's evidence is
about agatesim's stones.

### 4. Labels

- `period_variability` 0.7: `effective`. It is imposed in agatesim, a
  stand-in for whatever makes agate bands irregular, which may include a
  rim-to-centre trend (assumption, stated because no source gives a
  figure; see section 5).
- `alternation` 0.7: `effective`, unchanged.

### 5. Assumption stated (sourcing stalled)

Agate band durations vary about as much as a log-normal spread of sigma
0.7 gives, at alternation 0.7. No source gives this; Holten et al. 1998
could not be obtained. The value is the photos' choice at their resolution,
within a range that runs at least from 0.7 to 1.0.

### 6. Grades

- The run results: *verified* (measurement; one seed, 30 stand-in cuts per
  variant and 20 rendered cuts at 0.7 / 0.7).
- That the photos reject the default and accept 0.7–1.0: *plausible* (one
  seed, stand-in for three of the four settings).
- That real agate band durations spread with sigma near 0.7: *speculative*.
- That the irregularity is random rather than partly a trend, or forced
  from outside rather than internally driven: not addressed. A
  procedural choice of `period_variability` is never evidence about
  either (ADR 0002). The owner's question of whether irregular bands carry
  an environmental signal stays open.

## Alternatives considered

- **Add a rim-to-centre band-width trend to agatesim** [@goldbaum2019]
  and see how much spread is left. Closer to the sources, but it is a new
  agatesim feature with its own ADR and tests there; it can follow
  without reopening this choice.
- **Fit sigma to Holten's Hurst exponents.** Needs the full paper to know
  how the profiles were drawn; the abstract alone cannot carry a number.
- **Hold half B out and choose again on A with more seeds.** Better
  practice, but B has already been seen, so no rerun can make it untouched.
  More seeds would narrow the range, not the principle. Not taken.

## Risks

- **Fitted, not tested.** The spectrum block now measures the fit, not the
  stone. A claim that the stone's bands look right on that block would be
  circular.
- **One seed.** The ranking between 0.7 and 1.0 could change with another
  seed; both are within.
- **The range may hide a trend.** If agate's irregularity is mostly a
  trend, a random spread of 0.7 gets the statistics right but the
  arrangement wrong. The current blocks may not tell the two apart.

## Consequences

- On acceptance: `n2-slice.toml` gains `period_variability = 0.7` and
  `alternation = 0.7` with a pointer here, a test pins them, and ADR 0008
  7b gains a note that band regularity was tuned once, by this ADR.
- Then the stone run: 100 cuts at 2 mm depth, about 10 hours of GPU, which
  is announced before it starts and reports progress.
- Then optics tuning step 2, the translucency-to-optics mapping (ADR 0007
  7a C).
- References added: none (holten1997, holten1998 and goldbaum2019 are
  catalogued; holten1998 stays at abstract).
