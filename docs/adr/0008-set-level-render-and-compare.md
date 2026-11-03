# 8. Set-level render-and-compare, and the owner's blind test

- Status: accepted (owner, 2026-11-01). Section 1 records the owner's earlier
  choices; in section 7 the owner took the recommendations for 7a (staged),
  7b (tune on all photos, parameters listed first) and 7c (cut faces of every
  finish). Amendment ¹ (owner) rereads the power-by-size runs as verdict
  stability and adds the crossing shift. Amendment ² (owner) scores the
  descriptors in blocks, so no value can hide. Amendment ³ (owner) leaves
  six extreme photos out. Amendment ⁴ (owner) makes the fakes
  concrete. Amendment ⁵ (owner) replaces the noise rule of section 2,
  step 5. Amendment ⁶ (owner) pilots the fakes at 20 cuts.
  Amendment ⁷ (owner) adds a verdict per block.
- Date: 2026-11-01
- Builds on: [ADR 0006](0006-band-counts-as-censored-records.md) section 10
  (the fallback: band realism is tested by rendering synthetic stones at each
  photo's resolution and lighting, comparing image statistics at set level,
  and the owner's blind real-or-synthetic judgement, with obvious fakes as
  the negative control) and [ADR 0007](0007-renderer-cut-face-monte-carlo.md)
  (the renderer and its three lights). This is the last part of N2
  (docs/roadmap.md).

## Context

- **What exists.** `nodule.descriptors.survey` already cuts the n2-slice
  stone at random, makes each cut look like a library photo drawn at random
  (size so its bands show as the photo's do, then the photo's blur and
  noise), and measures C6–C17 with the photos' own code. It draws a flat
  stand-in image, not a rendering. `nodule.render` now renders a cut under
  L1 (dome), L2 (disc lamp) or L3 (back light through a slab).
- **The photos available.** The library holds 261 photos, but only reflected
  cut faces are compared here. Each of the 25 compared values stands on 24–103 of them,
  depending on the descriptor (counted 2026-11-01 from
  references/library-tags.toml: C13's drift and completeness on 24, C11 on 91). The 30
  transmitted-light slabs give 2–15 per value.
  Exteriors and broken surfaces are left out because the synthetic image is
  a sawn and polished face.
- **One stone against many.** The synthetic set is many cuts of one stone,
  while the library is mostly one photo per stone, of many stones. A test of
  "same distribution" would compare one stone's spread from cut to cut with
  the spread between stones. It would reject even if the simulator were
  right, because one stone is not a population (reasoning). In predictive
  checking the replication must be chosen to match the question; "a new
  stone" is a prior-predictive replication, which needs many stones drawn
  from a parameter prior [@gelman1996, p. 755] (*plausible*).
- **Reading.** Done and catalogued: two-sample tests [@gretton2012;
  @lopezpaz2017], predictive checks [@gelman1996], simulated images degraded
  to survey conditions and measured by the survey's code
  [@rodriguezgomez2019], texture synthesis and psychophysics [@portilla2000;
  @wallis2017], human real-or-fake protocols [@zhou2019], and single-observer
  designs [@smith2018]. Notes are in notes/n2-reading/compare-methods.md
  (helper output, unreviewed; claims enter docs only through this ADR).
  Still unread, for the owner to fetch if wanted: Székely & Rizzo on energy
  statistics, BDA3 chapter 6, Meyer et al. 1986, and Balas 2006 (listed in
  the notes, section 5).
- **Evidence is asymmetric.** A synthesis that visibly differs is strong
  evidence that the statistics are insufficient. Looking alike is weaker
  evidence [@portilla2000, p. 52]. Simulated galaxies passed on single
  statistics but failed on joint relations [@rodriguezgomez2019, abstract].
  So here a failure counts as a finding, and a pass reads only "not refuted
  by these descriptors and this judge" (*plausible*).

## Decision

### 1. Choices already made (owner, 2026-11-01)

- **A spread test, not a distribution match.** The question asked is whether
  the one stone's cuts lie within the library's spread. Matching the full
  distribution waits until the generator can draw a family of stones; then
  the two-sample tests of section 6 apply, permuting stones rather than cuts.
- **Lights.** Reflected photos are rendered under L1, the dome. L2 costs
  about 15 minutes per cut at full size, against about a minute for L1, and
  the photos do not say which light was used. Transmitted slabs are rendered
  under L3 through a 3 mm slab (`effective`, ADR 0007 amendment 5). L2 stays
  built and is not used here.
- **The blind-test protocol is Claude's recommendation** (the owner left it
  to Claude); it is section 5.

### 2. The image pipeline

Each synthetic cut goes through the steps `survey` uses now. The only change
is that the stand-in image is replaced by a rendering:

1. A random plane through the stone, its normal uniform over the sphere
   (as now).
2. A photo drawn at random from the stratum, either reflected cut faces or
   transmitted slabs. The photo's `Look` (band scale, blur, noise) is matched
   as now.
3. The pixel size comes from the stand-in probe, since band spacing is
   geometry, which the stand-in already gets right. Only the final image is
   rendered. At most `MOST_PX` (2048) pixels across.
4. Render: the growth grid is filled for the plane (the slab depth for L3),
   rendered under the stratum's light, and converted to sRGB at one fixed
   exposure per run (`effective`; 2.2 for reflected faces,
   [ADR 0013](0013-optics-tuning-step-3-exposure.md)).
5. Samples per pixel⁵: the fewest that keep the render's own Monte Carlo
   noise at or below a third of the matched photo's noise level. That adds
   at most about 5 % to the noise in quadrature (assumption, measured on the
   first cuts).
6. Blur and noise as now (`measure_cut`, `photo_like`). The renderer's
   subsurface spread already softens edges. Where a render is softer than
   its photo it cannot be sharpened; the cut is kept, flagged `matched =
   False`, and the share of such cuts is reported. A high share is itself a
   finding: the bands look softer than real ones.
7. Measure with the photos' code, as now. Compared: C7–C16 and the profile
   spectrum C26, as ADR 0006 section 10 names them. C6 is recorded but not
   compared, since it describes the photo at its *L*.

**Colour.** All the compared descriptors read luminance. Colour realism is
judged only in the blind test (section 5).

### 3. The spread test (one stone)

- **Scaling.** Each descriptor is scaled by the photos' median and
  interquartile range (robust z).³
- **Joint score.** For any face, the mean squared scaled difference to each
  photo, taken over the descriptors both faces have (descriptors are
  missing where a gate fails, as for photos). The score is the mean distance
  to its k = 5 nearest photos. Photos are scored leave-one-out, giving the
  reference distribution.²
- **Verdict.** The stone lies within the spread if its median cut's joint
  score is at or below the 95th percentile of the photos' leave-one-out
  scores. A random real photo fails that 5 % of the time by construction.
  One stone is one sample, so this is a typicality check, not a test with
  power against small differences (*speculative*: the method is reasoned
  from [@gelman1996] and [@rodriguezgomez2019]'s descriptive comparison, not
  taken from a source).
- **Per descriptor (descriptive, no verdict).** For each descriptor:
  - the percentile of the stone's median within the photos' values;
  - the share of cuts outside the photos' central 95 %.

  These show which descriptor is off. They are not combined into a pass or
  fail: with 25 values, a typical real stone would land outside
  on at least one of them by chance more often than not (reasoning).
- **Reported beside it:** the share of cuts outside the spread, the
  `matched = False` share, and the same scores for the stand-in images, as
  a point of reference.
- **Transmitted slabs.** At 2–15 photos the slab compare is a screen for
  gross failure only (a toy power run in the notes, section 3, finds almost
  nothing short of a large difference at n = 30; *speculative*).

### 4. Negative and null controls

Every fake changes only the band record (the per-growth-step material fed
to the renderer, `BandRecord`) and runs through the same pipeline:⁴

- **Must be rejected:**
  - F1: bands without fine structure (the record smoothed below the band
    scale);
  - F2: wrong spacing ratios (growth time remapped, so thick and thin
    bands trade places in sequence);
  - F3: perfectly regular spacing.
- **Diagnostic only:**
  - F4: phase-randomised record (same spectrum, other structure lost);
  - F5: the image left undegraded (no blur or noise).

  These show what the descriptors can see; they need not be rejected.
- **Criterion.** A must-reject fake fails the verdict of section 3 in at
  least 80 % of 200 bootstrap resamples, resampling both its cuts and the
  photos. Each fake's rate is reported, not just yes or no. If a fake
  passes, the compare is blind on that axis, and a pass of the real stone
  says nothing there [@portilla2000, p. 52]. That gets written down; a
  descriptor is added, or the limit is recorded.
- **Null controls:**
  - N1: photos held out one at a time fail at about 5 %. This checks the
    code, since the rate is 5 % by construction.
  - N2: a second set of the stone's cuts on fresh RNG streams reaches the
    same verdict, with scores within the resampling spread.
- **Power by size.** The F1–F3 rejection rates are recomputed on photo
  subsets of 15, 30 and 60 photos and on the full stratum. This shows what
  the slab stratum can catch and how many more photos each axis would need
  (*plausible*, after the notes' section 4.4).¹

### 5. The owner's blind test

Run only after the machine compare has been run and any tuning (section 7,
7b) is frozen.

- **Task.** Two images side by side; the owner picks the synthetic one
  (two-alternative forced choice: chance 50 %, free of response bias). Each
  synthetic image is paired with the photo whose `Look` it was rendered to
  match, so resolution and noise cannot give it away. Unlimited viewing
  time; inspection exposes syntheses that brief viewing hides
  [@wallis2017].
- **Trials.**
  - 200 main trials and 20 catch trials (F1–F3), shuffled together.
  - Four sessions of about 55 trials, roughly 10–15 minutes each (estimate).
  - Synthetic images come from the compare's cuts, two different crops
    each where needed.
- **Against recognition.** The owner has seen every library photo, so both
  images get the same random square crop size inside the face, a random
  flip and a quarter-turn rotation. The aim is that a photo is not known by
  memory (*speculative*).
- **Against learning.** No feedback per trial, and no score until the end
  [@wallis2017]. The first session is reported separately, along with a
  trend in accuracy over trial number. The synthetic set is one stone, so
  its look can be learnt.
- **Catch trials.** At least 18 of 20 correct. Otherwise the session shows a
  problem with the task or the display, not with the stone (after the
  qualification step of [@zhou2019]; *plausible*).
- **Decision rule, fixed now.** Exact one-sided binomial at α = 0.05: 113
  or more correct out of 200 means the owner can tell synthetic from real.
  This has power 0.86 if the true accuracy is 60 % (computed 2026-11-01;
  *verified* as arithmetic). Fewer correct is not equivalence: the 95 %
  interval is reported, and at 100 of 200 its upper end is about 57 %. One
  judge supports a claim about that judge's eye under these conditions
  only [@smith2018].
- **Privacy.** A local page in the style of `nodule.zonecheck`. Answers are
  saved locally as counts and image ids. Photos and crops never leave the
  PC or enter the repository; derived files carry no metadata.

### 6. Later: the distribution match (with a stone family)

Once stones can be drawn from a parameter prior, the spread test is joined
by:

- per-descriptor KS or Anderson–Darling with Holm correction;
- one multivariate test (energy, or MMD with the median-distance kernel
  [@gretton2012, p. 748]) on scaled descriptors, with 999 or more
  permutations of stone labels;
- a classifier test whose held-out accuracy is the effect size
  [@lopezpaz2017], in the same units as the owner's blind score.

Recorded here so the spread test's code fits this later step; not decided
now (*plausible*).

### 7. Choices for the owner

**7a. Run budget.**
- Cost per cut (estimate from ADR 0007's measured speeds): a grid fill of
  3–9 s plus about 8 s under L1 at 2048 px and 32 spp, so roughly 15–20 s.
- Planned cuts: the stone 100, F1–F3 60 each, F4–F5 40 each, about 360 cuts
  in all. That comes to roughly 1.5–2 hours of GPU.
- *(Recommended)* **Staged.** Time three cuts first and report the measured
  figure, then the stone (about 30 minutes), then the fakes, each
  announced before it runs, with progress through `agatesim.progress`.
- Alternatives: everything in one overnight run; or half the counts
  (verdicts noisier, power curves coarser).

**7b. Tuning the look.** Milky bands are not white yet (ADR 0007), and the
optics are `effective`, so some tuning will be needed. Tuning against the
same photos that test the result can be fooled (ADR 0007, risks).
- *(Recommended)* **Tune on all photos, with limits.** The parameters that
  may be tuned (optical coefficients and exposure, not band geometry) are
  listed before tuning starts. Each tuning step is recorded. The verdict
  states that it comes after tuning. The band descriptors are mostly
  geometric, so tuning optics touches them weakly (reasoning).
  (Note, 2026-11-02: band regularity was tuned once, as a stated
  exception, by [ADR 0011](0011-band-regularity-for-the-stone.md); the
  spectrum and continuity blocks then no longer count as passed tests.)
- Alternatives: a seeded half for tuning and the other half for the verdict
  (honest, but 12–50 photos per value leaves little power); or no
  tuning in this ADR.

**7c. Which photos.**
- *(Recommended)* Reflected cut faces of every finish (polished, sawn,
  unknown), with polished faces also reported on their own.
- Alternative: polished only, which is closer to the render but leaves
  about two thirds of the photos.

### 8. Grades

- Pipeline (section 2): *plausible* (after [@rodriguezgomez2019]).
- Spread test and its verdict (section 3): *speculative*.
- Controls (section 4): *plausible*.
- Blind-test arithmetic: *verified*; its protocol: *plausible*.
- Any outcome is a procedural result about appearance: never evidence for a
  formation mechanism (ADR 0002).

## Alternatives considered

- **A two-sample test on one stone's cuts now.** Rejected by the owner: it
  answers "is this stone the population", which no single stone is.
- **L2 for reflected photos, or both lights.** Rejected for cost (about 15
  times L1); the photos do not record the light. It may return if the
  verdict depends on the light.
- **Single-image real-or-fake (HYPE∞ [@zhou2019]) as the main blind test.**
  Kept as a possible second block. It is closer to "would this pass as a
  photo", but it mixes in response bias and needs more trials.
- **The three-patch oddity task [@wallis2017].** Defeats "pick the natural
  one" strategies, but chance is 1/3. Crops from one source must be
  different regions, and agate faces are small.

## Risks

- **Few photos.** At 24–103 photos per value, only large departures are
  caught. The power curves (section 4) say how large.
- **The joint score is new.** It is reasoned, not taken from a source.
  The null controls check its calibration, and the fakes its power.
- **One stone.** A pass is about the n2-slice preset only. A fail may come
  from the stone's parameters rather than the renderer; the per-descriptor
  report and the fakes help tell which.
- **Renders softer than photos.** If many cuts cannot be matched in blur,
  the comparison is uneven. The `matched = False` share is reported.
- **Lighting assumption.** Reflected photos under unknown lights are
  compared with dome renders. A failure could be the light.

## Consequences

- `nodule.descriptors` gains a rendered image path beside the stand-in, and
  a `nodule.compare` module for the scores, controls and power curves, with
  tests (CPU tests on synthetic descriptor tables; GPU tests skip in CI).
- The fakes F1–F4 are `BandRecord` transforms in `nodule.render.fakes`,
  each with a test; F5 is a switch in the rendered descriptor path.⁴
- A `nodule.blindtest` local page follows the compare.
- References added: [@schreiber2000] (amendment ⁴), [@gretton2012], [@lopezpaz2017], [@gelman1996],
  [@rodriguezgomez2019], [@portilla2000], [@wallis2017], [@zhou2019],
  [@smith2018]. All are at "sections" level, read by a helper agent.

---

¹ Amended 2026-11-01 (owner chose option A). Building `nodule.compare`
showed that the verdict of section 3 is a line, not a test whose power grows
with the photos: a fake is rejected in the limit of many photos only if its
median cut is less typical than the photos' 95th percentile. On synthetic
tables (25 unit-normal descriptors, 40 % missing, 120 photos, 80 cuts,
100 resamples; the property is kept as a test in tests/test_compare.py), a
fake 1.2 standard deviations off was rejected in 93-98 % of resamples at
15, 30, 60 and 120 photos. One 0.9 off fell from 63 % at 15 photos to 20 %
at 120. One 0.6 off (inside the line) fell from 20 % to 0 %. At few photos
the noisy threshold adds spurious fails. So:

- The power-by-size runs of section 4 are kept, but they show how settled
  the verdict is at a library size. They do not show how many more photos
  an axis would need.
- The compare also reports where the line falls: the smallest shift of the
  stone's cuts, in the photos' interquartile ranges, that takes the verdict
  outside. It is given per descriptor (down and up) and for all descriptors
  together (`compare.crossing_shift`, found by bisection to 20 interquartile
  ranges; `effective`). "Infinite" marks an axis the compare cannot see.
  The negative controls are cuts already outside (shift 0) and a descriptor
  the cuts lack (infinite).
- Grade: *speculative*, as for section 3 (reasoning checked on synthetic
  tables, not taken from a source).

² Amended 2026-11-01 (owner chose block scores). The crossing shift of
amendment ¹ found that the joint score of section 3 could not see C7, C9,
C10, C11, C12 or C13 alone: shifted by 20 interquartile ranges on one of
them, three stand-in cuts stayed within. Values go missing in blocks, as
gates fail (only 26 of the 103 reflected photos have all; 35 have the shape
and image values but none from the band profiles). A cut found its 5
nearest photos among those lacking the shifted value, so that value was
never compared. Scored against only the photos that have it, each crossed
at a finite shift (C7 5.7, C11 4.8, C12 3.2, C13 depth 1.2 interquartile
ranges, upward). Section 3's joint score is replaced by:

- **Blocks** by measuring module: shape (C7, C8, C9, C11), persistence
  (C10), continuity (C12), rings (C13), image (C14-C16, which fail
  together) and spectrum (C26).
- **Block score.** Within a block, a photo is a neighbour only if it has
  every value of the block the face has. The score is the mean squared
  scaled difference to the `K` = 5 nearest such photos, photos
  leave-one-out.
- **Tail share.** Each block score becomes *p*, its upper tail share under
  a normal fitted (median, interquartile range) to the cube roots of the
  photos' block scores (`effective`). A rank was tried first. It stops at
  the most extreme photo, so with six blocks the photos' 95th percentile
  sat at the ceiling and a fake 2 standard deviations off passed. A
  log-normal was tried next. Ties among whole-number values (C13) give exact
  zeros, whose logs blew up the fitted spread, and rings could again never
  cross. Cube roots (Wilson-Hilferty) keep zeros finite.
- **Joint score:** -log10 of 1 - (1 - min *p*) ** *m* over the face's *m*
  scored blocks (Sidak), so that more blocks are not less typical by chance.
  It assumes the blocks independent and is conservative where they are
  correlated. The verdict of section 3 is unchanged: the median cut against
  the photos' 95th percentile.
- **Block presence.** The share of faces with each block, cuts beside all
  photos of the stratum (gate failures included), is reported, since how
  often the gates pass is itself a finding.

Checks (tests/test_compare.py; synthetic tables with whole blocks missing):
a cut set 5 interquartile ranges off on rings alone, beside photos 70 % of
which lack rings, fails. Under the old score it passed (the negative
control). Cuts with every block, against photos missing half, stay within.
N1: 0.05 synthetic, 0.068 on the 103 reflected photos, 0.067 on the 15
slabs. The amendment ¹ figures were for the old score. Under this one, a
fake shifted on all values is rejected in about 93 % of resamples at 2
standard deviations, 63 % at 1.5 and 25 % at 1.2 (120 photos). It
rejects less than the old score on such broad shifts, because the worst
block counts, not the sum over all 25. On the real library with three
stand-in cuts, every value now crosses at a finite shift: 1.1-1.8
interquartile ranges for C10, 3.6-8.9 for C12, 5.5-8.9 for shape, 12-15
for C13, and 1.1-1.8 for all values together.

The bar is wide because about five library photos are themselves extreme in
one block, most in the image block. Their 95th-percentile joint score is
15. They are listed by `python -m nodule.compare` runs (ids only).
Whether they are measurement artefacts, such as detail crops, is an open
check for the owner. Grade: *speculative*, as for section 3.

³ Amended 2026-11-01 (owner). The six reflected cut faces above the bar in
amendment ² are left out of the compare. They are listed, each with its
least typical block, in references/compare-excluded.toml, which
`compare.library_table` reads; the tags are unchanged. Basis: suspected
measurement artefacts (several are detail crops). They were not checked by
eye. Effect, on 97 photos: N1 0.072; the bar falls from 15.3 to 10.0; the
three stand-in cuts cross at 0.9-1.0 interquartile ranges on all values
together (from 1.1-1.8).

- **No further rounds.** The bar is the photos' 95th percentile, so about
  5 % of photos always lie above it (five do now). Cutting them each time
  would shrink the library's spread toward its middle, and real stones
  would start to fail. A further exclusion needs a reason found in the
  photo itself, not its score.
- The excluded photos stay in the library for every other use. A run can
  restore them by passing an empty `leave_out` (`library_table`).

⁴ Amended 2026-11-01 (owner accepted as written). Building the fakes
(`nodule.render.fakes`) needed choices section 4 leaves open. All are
`effective`, made here, and cheap to change.

- **One fake stone, not one per cut.** A cut's record holds only the steps
  that cut crosses (a centre cut of n2-slice reaches step 151; the stone
  reaches 161). A fake built per cut would draw a different stone in each
  cut, so each fake is built once from the whole stone's record, read from
  1,000,000 points drawn through its bounds (~1.5 s, ~80 MB, one spawned
  RNG stream).
- **What moves.** The record is read as runs of equal deposit (one run =
  one band; n2-slice has 32). Only chalcedony runs before the last move.
  The last run is open, ending where the cavity closed, so the geometry
  sets its width; quartz is not a band under test. Each stretch of movable
  runs keeps its first and last step.
- **F1** smooths impurity and translucency with a Gaussian of
  sigma = P / 2π steps, P = the typical band pair (twice the median run
  width; 10 steps for n2-slice). A typical pair keeps e^-0.5 of its
  amplitude, its third harmonic about 1 %, and a one-step line inside a
  band about a quarter. The median, not the shortest pair, so that sub-bands
  cannot set the scale meant to remove them. Phase and pigment are
  categories and stay.
- **F2** gives each band the width of the band at the opposite rank (the
  thickest becomes the thinnest), keeping the order of deposits.
- **F3** makes the bands of a stretch equally wide to the step. Widths still
  differ by one step (5 or 6 for n2-slice), the record's resolution
  (400 steps over growth time, about 0.075 mm a step in n2-slice).
- **F4** is a multivariate iterated amplitude-adjusted Fourier (IAAFT)
  surrogate of impurity and translucency, sharing one phase shift per
  frequency [@schreiber2000, ss. 4.3, 4.6]. It keeps each field's values
  exactly (so ranges hold) and its spectrum approximately: within 9–19 %
  (relative RMS amplitude) on n2-slice. Tied values from the runs limit the
  match; the paper says nothing about them. The end jump that the method's
  periodicity would leak [@schreiber2000, s. 4.5] is small there (0.05).
- **F5 is not a record change.** It skips the photo-like blur and noise, so
  it is a switch in the rendered descriptor path (next step), not a
  `BandRecord` transform.
- **A limit to state.** agatesim's record has no structure finer than its
  bands, since each band is uniform. So F1 differs from n2-slice only in
  soft edges and faded thin bands. If F1 passes the compare, that may mean
  the descriptors are blind to fine structure, or that the stone has none
  to lose. The compare of the stone itself against the photos (C26's
  fine-side slope) is the place to see which.

⁵ Amended 2026-11-01 (owner accepted as written; they chose option A after option B was
measured). Timing the first rendered cuts showed that step 5's rule cannot
be met. All figures are from the n2-slice under L1 on the RTX 4070
(*verified* as measurements; one stone, three cuts).
- **The render's noise.** At 128 samples per pixel, 6.6 grey levels of
  luminance per pixel. It is white: a 4 × 4 box lowers it fourfold, and its
  median absolute deviation matches its standard deviation, so there are no
  bright outliers. The photos have 1.0–4.6 (10th–90th percentile, median
  1.5). About one path in five leaves through the face; the rest end in
  the stone. That all-or-nothing outcome explains the noise (each path's
  value varies about twice its mean). A third of a photo's noise would
  take about 90,000 samples per pixel, roughly an hour a cut.
- **Option B, a less noisy renderer, measured first.** The kernel's own
  light sampling (the L2 disc's, as a 90° disc that equals the dome, mean
  within 0.08 %) is about 9 times *less* efficient than plain path
  tracing here (noise² × time). Each aim needs a shadow ray through the
  bands, and the grains' narrow forward lobes seldom point at the face.
  Absorption is already weighted, not played out (`spectral_step`). Even a
  10-fold gain would leave about 12 minutes a cut, so B was dropped.
- **What the noise does to the descriptors.** Three cuts were rendered at
  128 samples (two seeds) and at 1,024 and measured with the same draws.
  Shifts, in the photos' interquartile ranges: most values move less than
  0.5 between 128 and 1,024 samples. The defect values C14 (up to 2.1)
  and the spectrum C26 (slope 2.2 and 0.8, peak 2.5–4.3) move most.
  C26's peak also moves 1.4–2.0 between two seeds at 128, so it is
  unstable in itself. Noise reads as fine structure, as expected.
- **Slab depth under L1.** 20 mm against 10 mm changed one cut by 0.02
  grey levels on average (0.02 standard deviation), against 9.5 between
  two seeds: almost no light returns from below 10 mm. 10 mm stays
  (`effective`). Later 2 mm ([ADR 0010](0010-optics-tuning-step-1-pores.md)
  10b).

**The new rule (replaces step 5).** A rendered cut's own noise counts
toward the photo's.
1. Samples are added in batches to two halves on independent seeds. The
   image is their mean; their difference gives its noise exactly.
2. 256 samples first (`effective`). The photo's blur is then found as
   before (`measure_cut`). The render's noise after that blur is its noise
   times the blur's factor for white noise (`blur_noise_factor`, checked
   against blurred noise within 3 %).
3. If that exceeds 0.9 of the photo's noise (`effective`, room for the
   photo's own noise), samples are added once, as noise falls with the
   square root of the samples, up to 1,024 (`effective`, cost). The blur is
   found again.
4. Step 6 adds only the noise the cut lacks: the square root of the
   photo's noise squared minus the render's blurred noise squared. A cut
   still noisier than its photo gets none and is flagged
   (`noise_matched = False`). Its share is reported beside the
   `matched = False` share of step 6.
5. F5 renders by the same rule and is measured without blur or noise, so
   it keeps the render's own noise.

Measured with the rule (three cuts, 280 s, 438 MB traced CPU memory, 65 MB
GPU pool). Two cuts drawn against photos at noise 1.0 needed about 1,000
samples (fill 10–12 s, render 81–120 s); their noise after the blur was
0.82–0.87. One drawn against a photo at noise 6.3 stayed at 256 samples
(10 s); its face fails the gate at any sample count, so it is not a noise
effect. All three were matched.

**Cost.** Over the 97 reflected photos the rule asks for about 580 samples
on average; 35 % stay at 256 and none should reach the cap (estimate, for
a typical blur). That is about 70 s a cut, so the stone's 100 cuts take
about 2 hours and all 360 about 7 hours. This is more than section 7a's
1.5–2 hours. The staged plan stands: each stage is announced first.

**Limits.** The render's noise is white per pixel, but a photo's is shaped
by its camera and compression. The rule matches the amount of noise, not
its texture. That was already true of the noise added to the stand-in.
After the blur both are blurred alike. Grades: the measurements are
*verified*; that matching the total noise is enough for the descriptors is
*plausible*. The 128 against 1,024 comparison shows which values are
sensitive. At the full run, the stone's values against `noise_matched` will
show whether the matched cuts still differ.

⁶ Amended 2026-11-01 (owner chose the pilot and accepted as
written). The stone's run measured 104 s a cut, not 70 s, so the planned
fake counts of section 7a (F1–F3 60 cuts, F4–F5 40) would take about 7.5
hours. `power_by_size` cannot set those counts: it varies the photos, not
the cuts, and it needs fake cuts to exist. Running it on the stone's cuts
would score the stone before the fakes, which the owner ruled out.

1. **Pilot.** Each fake is first rendered at 20 cuts on the reflected
   stratum. `compare.power_by_cuts` (CLI `--cut-sizes 10`) gives its
   rejection rate with 10 and 20 cuts drawn per resample (200 resamples).
2. **Stopping rule.** A fake that fails in at least 95 % of resamples at 20
   cuts is rejected and stops there (95 %: `effective`, chosen to match the
   compare's level). Otherwise it is rerun at its planned count on the same
   seed, and that run gives the verdict.
3. **Seeds.** Fake *i* uses `--seed i --record-seed i` (F1 seed 1 to F5
   seed 5), independent of the stone's seed 0.
4. **Stone scoring (owner's choice, 2026-11-01).** 24 of the stone's 100
   cuts are not `noise_matched`; 22 of these are at the 1,024-sample cap.
   The stone is scored twice: all cuts, and the noise-matched cuts only.
   Both verdicts are reported. It is scored only after the fakes are
   rejected.

Cost: about 35 minutes a fake at 20 cuts, so about 3 hours if all five
stop at the pilot. The worst case is about 10.5 hours. Negative control:
in tests/test_compare.py a fake inside the line is not rejected once all
its cuts are drawn, while one beyond it is rejected at 10. Grade:
*speculative*. A fake rejected at 20 cuts says the compare catches it. It
does not say by how much.

⁷ Amended 2026-11-02 (the owner chose option b after the fakes' runs;
accepted with the two notes on resampling and the test added). Of the five fakes, only F3 was rejected
(runs/fakes-2026-11-02/). The joint verdict of amendment ² takes one median
over all cuts, but cuts carry different blocks: in F5's 40 cuts all had
shape values and only 13 had any band block. The cuts with shape alone
score near typical and set the median, so a band block can be atypical in
nearly every cut that has it and still not move the verdict. This is the
dilution amendment ² fixed for neighbours (a value compared only against
photos that have it), left open for the median.

This change was proposed after seeing the fakes' results, so it is fixed
here before any fake is rescored. It is guarded by null controls, and its
cutoffs are the ones already used.

1. **Block verdicts.** For each block, take the cuts that have it and the
   photos that have it. The block's score for a face is -log10 *p*, its
   tail share (amendment ²). The block is outside if the cuts' median
   score exceeds the photos' `LEVEL` (95th) percentile of theirs,
   leave-one-out. This is the line of section 3, applied block by block.
   A block is judged only if at least `MIN_CUTS` = 5 cuts have it
   (assumption). Otherwise it is reported as untested, a limit.
2. **Verdict.** Outside if the joint verdict is outside, or any block's
   verdict is. The joint verdict stays because it sees a cut moderately
   atypical in several blocks at once.
3. **Multiplicity.** Up to six block lines add chances of a false
   rejection. Under the null a median of *n* cuts rarely passes a photo's
   95th percentile (for 10 cuts with uniform tail shares, at most about 6 in
   100,000). Null controls measure this rather than assume it.
4. **Null controls.**
   - Synthetic cuts from the photos' population, with blocks missing as
     in the photos, stay within.
   - N3: a real reflected photo subset (10 and 20 photos) held out and
     tested against the rest is rejected in at most about 5 % of draws.
5. **Negative control.** In a synthetic table, the cuts are shifted on
   rings alone, and only 30 % of cuts and photos have rings. The joint
   verdict passes this; the block verdict must reject it.
6. **Rescore.** F1–F5 are rescored on CPU from their saved runs (no
   rendering), with power by cuts. Each fake's verdict and its failing
   blocks are reported.

Grade: *speculative*. The line is reasoned, as in amendment ¹, not taken
from a source. The null and negative controls are its only checks.

**Results (2026-11-02; runs/fakes-2026-11-02/rescore-am7/, 15–17 s a
fake on CPU).** Null controls: synthetic sets stay within. N3 on the 97
reflected photos is 0.005 for sets of 10 and 0.015 for sets of 20. The
rings-only negative control fails under block verdicts and passes under
the joint one. Every fake is now outside at the point verdict. The fail
rates over 200 resamples are F1 59 %, F2 72 %, F3 100 %, F4 51 % and
F5 77 %, so only F3 is settled. Each fake is outside on spectrum (C26).
The block's median against its line of 2.39 is F1 2.49, F2 8.14, F3 144,
F4 3.40 and F5 7.56. F5 is also just outside on rings (0.72 against a line
of 0.64). Small resampled sets reject more often by chance. With 40 % of
blocks missing (about 6 cuts a block), synthetic sets with no shift failed
in 8 % of resamples at 10 cuts and 1 % at 40, against 0 % under the joint
verdict. So fail rates at 10 cuts are somewhat inflated. The fakes had 13
or more banded cuts. The power-by-cuts test's near fake moved from 0.6 to
0.3 standard deviations. At 0.6 it crossed no block line on the full data
(0 of 10 seeds), but resampled it failed in 11–20 % of resamples, so it
was no longer clearly inside the line. Two limits, both *plausible*:

- F5 keeps the stone's record unchanged and still fails on spectrum, so
  the stone may itself be outside there. If so, the fakes' spectrum
  failures are inherited, not caught. The stone's own verdict decides
  this.
- The rings line is low (0.64; ties among C13's whole-number values
  compress its tail shares), and every fake's median lies at 0.45–0.72.
  A rings verdict is fragile.

**The stone's verdict (2026-11-02; runs/stone-2026-11-02/, 22 s a score
on CPU).** Scored twice as amendment ⁶ point 4 asks, with power at 10, 20,
40 and 60 cuts. All cuts: 89 of 100 compared, median 14.2 against the bar
of 9.96, outside, failing in 93 % of 200 resamples (91–94 % at every cut
count). Noise-matched cuts only: 65 of 76 compared, median 13.1, outside,
failing in 92 % (94–96 %). Both are outside on spectrum alone (block
median 18.4 and 20.5 against its line of 2.39). The other five blocks are
within, rings included (0.50 and 0.48 against 0.64). Against the 42
reflected photos with a spectrum (5th / 50th / 95th percentile), the
cuts' median C26_slope is 3.24 (0.25 / 1.37 / 2.93) and C26_peak is 9.70
(1.69 / 2.47 / 5.51). C26_peak_share, 0.05, is within (0.02 / 0.21 /
0.55). The measurements are *verified*. What follows from them:

- The stone's profiles have one strong spacing and less fine-scale power
  than the photos (*verified* for these descriptors; whether a viewer
  would see it is untested).
- Matching the noise does not change the verdict, so the failure is not
  the unmatched cuts' extra noise (*plausible*).
- The fakes' spectrum failures are inherited. F1, F2, F4 and F5 are all
  nearer the photos on spectrum than the stone is (block medians 2.5–8.1
  against 18.4). So F1, F2 and F4 differ from the stone on no block, and
  F5 only on rings, which is fragile. Only F3 is caught on its own trait
  (*verified* from the scores above). The negative control shows the
  compare catches equal widths; it does not yet show it catches the other
  four.
- F5 (the same record, no blur or noise) has C26_slope 2.96 and C26_peak
  6.88, so the record is already peaked before the photo-like blur, and
  the blur and noise add to both (*verified* for medians; the cause is
  *speculative*: the record itself, the blur estimate per photo, or both).
