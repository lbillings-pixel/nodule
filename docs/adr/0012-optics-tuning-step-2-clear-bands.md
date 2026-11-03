# 12. Optics tuning step 2: how fast pores fade toward clear bands

- Status: accepted (owner, 2026-11-02): 12a A (γ 4), as recommended.
- Date: 2026-11-02
- Builds on: [ADR 0007](0007-renderer-cut-face-monte-carlo.md) choice 7a C
  (grain part plus pore part, both coupled to agatesim's translucency, an
  assumption), [ADR 0010](0010-optics-tuning-step-1-pores.md) (tuning step
  1 and the list of tunable fields; contrast left low, step 2 named as the
  next knob), [ADR 0011](0011-band-regularity-for-the-stone.md) (record
  0.7 / 0.7 and the 100-cut results) and [ADR 0002](0002-founding-nodule.md)
  (graded claims).

## Context

- **Contrast is the open gap.** After step 1 the stone's renders sit near
  the photos' 5th to 20th percentile of lightness spread (L sd). In the
  photos' centre window (half A, 46 photos) the 5th / 50th / 95th
  percentiles are 4.4 / 9.7 / 27.8 (ADR 0010).
- **The stone is mostly milky.** In the `n2-slice` record (0.7 / 0.7,
  record seed 0, 400 steps, all chalcedony, IRON pigment) agatesim's
  translucency t has percentiles 0 / 5 / 25 / 50 / 75 / 95 / 100 of
  0.14 / 0.22 / 0.26 / 0.26 / 0.31 / 0.81 / 0.88. About a fifth of the
  steps are clear (t ≈ 0.8). The rest are milky (*verified*, CPU, 2 s).
- **Clear bands still scatter a lot.** The step-1 mapping lowers the pore
  fraction linearly in t, (1 − t) × 1.75%. At 550 nm the reduced
  scattering is 3.45 per mm at the median band and 0.99 per mm at t 0.81.
  That is a ratio of only 3.5 between the milky bands and the clear ones.
- **Grains barely matter in reflected light.** The grain part gives 1% of
  the reduced scattering at t 0.2 and 9% at t 0.8. The grains scatter
  almost straight ahead (g 0.93–0.996). Retuning grain radius therefore
  cannot raise contrast under L1. It matters for transmitted light (L3)
  (*verified* within the model).
- **What the sources say about pores in each band.** This draws on a
  helper's reading, with page checks, in notes/n1-reading/ (not committed).
  - *No source gives a pore fraction or a pore size for each band.* No
    source gives paired scattering coefficients for clear and milky bands
    of one specimen either.
  - *Graetsch and colleagues.* Their translucent bands hold more water
    than their milky bands (1.20–1.36 against 0.65–0.72 wt%). The milky
    bands have more BET surface (1.9 against 0.4–1.1 m²/g) and are 0.5–1%
    denser (p. 302). TEM found no 0.1 µm pores [@graetsch1985].
  - *Folk and Weaver.* They tie milkiness to water-filled bubbles, and
    the bubbly zones have the lower index (p. 505). Their material is chert
    and novaculite, not agate [@folk1952].
  - *Pelto.* Low-index bands that do not scatter "probably also contain
    pores, but of considerably smaller dimensions" (p. 40). This is an
    opinion without numbers [@pelto1956].
  - So the sources disagree about which way water goes. On the question
    asked here, the leaning is that clear bands hold fewer or smaller
    scatterers, not none [inference].
- **A probe on the look cuts.** The usual 4 look cuts (0, 5, 8, 18) were
  rendered with half-A looks, 2 mm depth, step-1 values otherwise
  (runs/stone-2026-11-02/variants/look_mapping.py; not committed). The
  pore fraction was taken as (1 − t)^γ, scaled so the median band
  (t 0.264) keeps its step-1 fraction. Milky bands are therefore unchanged
  and clear bands get clearer.

  | γ | μs′ 550 at t 0.81 | L sd, cuts 0/5/8/18 | mean | spacing, cuts 8/18 | time |
  |---|---|---|---|---|---|
  | 1 (step 1) | 0.99 per mm | 5.1 / 5.6 / 5.0 / 3.7 | 4.9 | 13.0 / 10.9 px | 212 s |
  | 2 | 0.33 | 6.5 / 7.0 / 5.5 / 4.4 | 5.9 | 9.6 / 9.4 px | 214 s |
  | 4 | 0.11 | 7.6 / 8.3 / 5.8 / 5.0 | 6.7 | 9.1 / 9.0 px | 240 s |

  - Stand-in spacing on cuts 8 / 18 is 6.4 / 6.8 px. Peak memory is
    549 MB in each run.
  - Contrast rises on every cut and the bands sharpen toward the stand-in
    (*verified* within the model: the cuts, looks and random streams are
    the same in each run).
  - At γ 4 the clear bands reach the floor that grains and matrix give
    (μs′ about 0.10 per mm). A larger γ would change little more.
  - Lightness median is 43–46 in all three runs, against the photos' 62.
    The renders are darker than the photos whatever γ is. That is a
    separate gap, not this step's.

## Decision

### 1. The change

The pore part's fraction becomes `pore_milky × (1 − t)^pore_gamma`, plus
WHITE pigment's `pore_white × impurity` as before. The new field is
`OpticsTable.pore_gamma` (`effective`). `pore_milky` is rescaled so that
the record's median band (t 0.264) keeps its step-1 fraction (1.29%):
`pore_milky = 0.0175 / 0.736^(γ − 1)`. The grain mapping is unchanged.
This is tuning step 2 under ADR 0010's list, which named "the mapping from
agatesim's translucency to grain radius and pore fraction". It touches
optics only, not the record.

### 2. Choice for the owner

**12a. How steeply pores fade toward clear bands.**
- *(Recommended)* **A. γ 4: clear bands at the grain floor.** This gives
  the most contrast of the tested settings (L sd mean 6.7 against 4.9).
  It is the end of this lever, since clear bands then scatter about as
  little as the grains and matrix allow. Read physically, it means
  clear bands carry few or very small pores. Small pores at the same
  fraction scatter far less, which fits Pelto's "considerably smaller"
  pores [inference]. Costs:
  - `pore_milky` at t 0 becomes 4.4%. The record's milkiest band
    (t 0.14) gets 2.4%, a little above the about 2% that Folk and
    Weaver's figures give [inference, ADR 0010].
  - The look cuts take 13% longer (240 against 212 s).
- B. γ 2. About half the contrast gain (mean 5.9) and nearly the same
  sharpening (9.6 / 9.4 px). Clear bands keep 0.09% pores, a "fewer, not
  none" reading taken literally. `pore_milky` at t 0 is 2.4%. Same time
  as now.
- C. Keep step 1 (γ 1). Contrast stays near the photos' 5th to 20th
  percentile, and step 3 has to find all of the gain elsewhere.

Neither A nor B closes the gap to the photos' median (9.7). The next
candidates are the milky side (whiter milky bands) and the overall
darkness (L median 44 against 62). Those would be a step 3, after the
20-cut check.

### 3. Labels and stated assumption

- `pore_gamma` and the rescaled `pore_milky` are `effective` and
  *speculative* as optics.
- The coupling between translucency and pore fraction stays an
  **assumption** (ADR 0007 7a C). Its shape is now an assumption too. No
  source measures pores band by band, and the sources disagree on which
  way water goes. Sourcing stalled here: one helper read the relevant
  catalogued sources, with page checks. So the shape is set by the
  photos, as the owner's rule allows (state the assumption and proceed).

### 4. Grades

- The probe numbers are *verified* within the model (4 cuts, paired
  streams).
- That γ 4 brings the stone's set closer to the photos is *plausible*
  until the 20-cut check on half A and B confirms it.
- Nothing here is evidence about what makes agate bands milky (ADR 0002).
  A match supports only "this parameter set looks right".

## Alternatives considered

- **Retune grain radius.** It gives under 10% of the reduced scattering in
  reflected light. Rejected for L1; to be revisited with L3.
- **A two-state threshold mapping** (smooth step between milky and clear).
  It is not needed, since the record's t is already split into two groups
  (0.22–0.31 and about 0.8). The sources describe gradual change inside
  bands with sharp edges between them [@lu1994]. A power curve keeps that
  continuity with one parameter. Not tried.
- **Raise pore fraction everywhere.** Step 1 tried 1% air (about 6% water).
  It sharpened the bands but cost 2.4 times the render time and went beyond
  the one measurement (ADR 0010). Not repeated.

## Risks

- **Edges may get more orderly.** Sharper bands may push edge coherence C16
  (already at the photos' 98th percentile) and curvature C15 further out.
  The 20-cut check watches both.
- **Tuned on the look cuts, which are half-A looks.** Half B stays held
  out for the 20-cut check.
- **Darkness unexplained.** L median 44 against the photos' 62 is untouched
  here. It may be part of why contrast is low.
- **GPU-only.** The checks run locally; CI cannot repeat them.

## Consequences

- On acceptance:
  - `OpticsTable` gains `pore_gamma`, and `pore_milky` is rescaled, with a
    test pinning both values.
  - A test checks that the median band's fraction is unchanged.
  - A negative control checks that the step-1 value (γ 1) still
    reproduces the step-1 optics.
  - ADR 0007 section 3 gets a pointer to this ADR.
- Then the 20-cut check (0.7 / 0.7, half A and B), about 25 minutes, with
  C16, C15, C* and L sd reported. The 100-cut stone run follows only if the
  owner asks: about 2.2 hours, scaled from 7111 s by the look cuts' +13%
  [inference].
- References added: none (pelto1956, lu1994, folk1952, graetsch1985 are
  catalogued).

## Results: the 20-cut check (2026-11-03)

- **What was run.** γ 4 (`pore_milky` 0.043894), band regularity 0.7 / 0.7,
  2 mm depth, 20 rendered cuts, looks from half A. The cuts were judged
  against half A and against the held-out half B. Run time 1611 s, peak
  traced memory 1049 MB (*verified*; the report is in runs/, not
  committed).
- **What it is compared with.** The comparator is ADR 0010's 20-cut check
  (0.3% air pores at 10 mm depth, 1471 s). ADR 0010 judged that setting
  equivalent to step 1's water pores. It is not an exact rerun of step 1.
- **Verdict.**
  - All 20 cuts are within against A and against B. So are the 12
    noise-matched cuts (11 before). No block is outside.
  - The joint distance against A is 0.5 (was 1.1) and against B 3.4 (was
    2.8). The line for "within" is 15.9 against A and 7.9 against B.
- **Gains.**
  - The spectrum distance is lower against both halves: 0.43 (was 0.50)
    against A, 0.71 (was 0.91) against B.
  - Edge curvature C15 is 0.08, the photos' 74th percentile against A
    (was 0.09 at the 87th).
  - Noise-matched C15 is 0.10, the 91st percentile against A (was 0.11 at
    the 98th). That watch item eases.
- **Edge coherence worse, as the Risks expected.** C16 is still 0.91 (the
  photos' 98th percentile against both halves). On the noise-matched cuts
  it rose to 0.89 (was 0.85), from the 96th / 92nd to the 98th / 98th
  percentile. It stays within, but sharper bands make edges a little more
  orderly than in the photos.
- **Contrast.** This check does not record L sd. The evidence for the gain
  is the 4-cut probe (Context).
- **Grade:** *plausible*. Step 2 keeps the stone within the photos on both
  halves and improves the spectrum and curvature. Edge coherence remains
  the open watch item, with contrast and darkness (L median) still to do.
