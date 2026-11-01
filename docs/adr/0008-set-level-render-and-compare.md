# 8. Set-level render-and-compare, and the owner's blind test

- Status: accepted (owner, 2026-11-01). Section 1 records the owner's earlier
  choices; in section 7 the owner took the recommendations for 7a (staged),
  7b (tune on all photos, parameters listed first) and 7c (cut faces of every
  finish). Amendment ¹ (owner) rereads the power-by-size runs as verdict
  stability and adds the crossing shift.
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
   exposure per run (`effective`).
5. Samples per pixel: the fewest that keep the render's own Monte Carlo
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
  interquartile range (robust z).
- **Joint score.** For any face, the mean squared scaled difference to each
  photo, taken over the descriptors both faces have (descriptors are
  missing where a gate fails, as for photos). The score is the mean distance
  to its k = 5 nearest photos. Photos are scored leave-one-out, giving the
  reference distribution.
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
to the renderer, `BandRecord`) and runs through the same pipeline:

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
- The fakes F1–F5 are `BandRecord` transforms in `nodule.render`, each with
  a test.
- A `nodule.blindtest` local page follows the compare.
- References added: [@gretton2012], [@lopezpaz2017], [@gelman1996],
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
