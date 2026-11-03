# 10. Optics tuning, step 1: what may be tuned, and stronger pore scattering

- Status: accepted (owner, 2026-11-02): 10a A (water-filled pores at
  1.75 %) and 10b (2 mm slab depth), as recommended. Section 5 records the
  4-cut look check run on acceptance.
- Date: 2026-11-02
- Builds on: [ADR 0008](0008-set-level-render-and-compare.md) choice 7b
  (tune on all photos, with the tunable parameters listed and each step
  recorded), [ADR 0007](0007-renderer-cut-face-monte-carlo.md) section 3
  (the optics table, amendment ²) and [ADR 0002](0002-founding-nodule.md)
  (graded claims).

## Context

- **Rendering lost the bands.** The first GPU check of the stone's cuts
  (band regularity 0.7 / 0.7, 20 cuts, 1232 s) was outside half A of the
  reflected photos on the image block (14.3 against a line of 3.4; edge
  curvature C15 0.164 / 0.402). On four cuts the mean band spacing was
  17.6 / 8.6 / 25.0 / 28.6 px, against 4.6–6.8 px for the same cuts drawn
  by the CPU stand-in. Bands were present in the record but not visible in
  the render (*verified* for these cuts).
- **The cause: milky bands scatter too weakly.** With the ADR 0007 pores
  (0.1 % × (1 − translucency), air, radius 0.1 µm) the milkiest band's
  reduced scattering is 1.5 per mm at 550 nm. The chalcedony's own
  translucency in the record (0.21–0.82) varies it only about fourfold
  (3.6–0.9 per mm), so light spreads 0.3–1 mm sideways under the face and
  blurs neighbouring bands together. Raising the pore fraction brings the
  bands back: spacing 5.3 / 7.7 / 15.4 / 12.4 px at 0.3 %, 5.2 / 7.4 /
  10.6 / 9.2 px at 1 % (*verified* within the model).
- **Contrast is still low.** Across half A (46 photos, centre window) the
  lightness spread L sd has 5th / 50th / 95th percentiles 4.4 / 9.7 /
  27.8. Renders give 2.5–3.5 at 0.1 % pores, 3.3–4.7 at 0.3 % and 4.0–5.7
  at 1 %; adding three times the iron oxide at 0.3 % gives 3.7–5.5, mostly
  by darkening (median L 45, against 58 without; photos' median 62).
  Mixing light deeper (a 2 mm slab instead of 10 mm changes the image by
  0.14 grey levels) and matrix absorption (0.05 and 0.13 per mm) only dim
  the image (*verified*).
- **What the sources say about pores.** Folk and Weaver saw water-filled
  spherical cavities of "very nearly 0.1 micron" in chalcedonic quartz,
  with bubble zones "milky white" in reflected light [@folk1952]. Their
  densities (clear novaculite 2.6424, bubbly chalcedony 2.6103 g/cm³) give
  about 2 vol% pores if the pores hold water [inference: computed here].
  Graetsch and colleagues found no 0.1 µm pores by TEM in Brazilian agate
  and put the inner-surface water at twin boundaries instead
  [@graetsch1985]. The sources disagree.
- **The model's pores are air, the sources' are water.** Pores filled with
  water (n 1.333) scatter much less than air pores in chalcedony (n 1.54).
  By the renderer's Mie code, 0.3 % air pores have the same reduced
  scattering as 1.7–1.8 % water-filled pores (450 / 550 / 650 nm), close to
  Folk and Weaver's 2 %. Matching 1 % air would need about 6 % water-filled
  pores, three times their figure (computed, *verified* as arithmetic).
- **The 20-cut check with the change.** Pores at 0.3 % and iron oxide at
  three times the old loading (volume fraction 9 × 10⁻⁴), band regularity
  0.7 / 0.7: within half A and within the held-out half B, for all 20 cuts
  and for the 11 noise-matched ones. No block outside (the image block was
  outside before). C15 curvature 0.09 / 0.21, where it was 0.164 / 0.402.
  The spectrum was measured on 19 cuts, not 12. Time 1471 s (+19 %), peak
  memory 1049 MB (*verified*; report in runs/, not committed).
- **Two watch items in that check.** Edge coherence C16 is 0.90, the photos'
  98th percentile in both halves (0.85 noise-matched, 96th / 92nd). Noise-
  matched C15 curvature is 0.11, the 98th percentile against half A. Both
  blocks stay within (*verified*). Sharper renders may now be slightly too
  orderly at the edges.
- **Order of events.** ADR 0008 7b asks for the tunable parameters to be
  listed before tuning starts. These runs were diagnostic, made to find why
  the bands vanished, and were done before that list. They used half A
  only for calibration and kept half B out, which is stricter than 7b
  asked. This ADR writes the list now and records the steps taken as
  tuning step 1, so the stone's verdict is reported as after tuning.

## Decision

### 1. What may be tuned (ADR 0008 7b)

Optics and exposure only, never band geometry or the record. All
`effective` unless stated. Fields of `nodule.render.materials.OpticsTable`:

- Pores: `pore_milky`, `pore_white`, `pore_r_um`, `pore_n`.
- Grains: `grain_r_clear_um`, `grain_r_milky_um`, and the mapping from
  agatesim's translucency to grain radius and pore fraction (log-linear and
  linear now).
- Pigments: `phi_iron_oxide`, `mua550_shape`, `mus_blue550`; the hematite
  n, k source (Go by default, Bedidi as the option, ADR 0007 ¹).
- Host and quartz: `matrix_mua`, `quartz_mus`, `quartz_g`.
- Exposure, as the per-photo `Look` already matches it.

Render settings that change cost, not the look by intent (slab depth,
samples per pixel), are listed with each step, and a change to one counts
as a step.

### 2. Step 1 (recorded)

Taken on half A, judged on A and B. From the ADR 0007 values:
`pore_milky` 1 × 10⁻³ → 3 × 10⁻³ and `phi_iron_oxide` 3 × 10⁻⁴ →
9 × 10⁻⁴, at slab depth 10 mm, with band regularity 0.7 / 0.7 (its own
ADR comes next). Variants tried and not taken: pores at 1 %; iron at ten
times; matrix absorption 0.05 and 0.13 per mm; 2 mm depth (choice 10b).

### 3. Choices for the owner

**10a. How to state the pores.** The 20-cut check used 0.3 % air pores.
- *(Recommended)* **A. Water-filled pores at 1.75 %** (`pore_n` 1.333,
  `pore_milky` 0.0175). The same reduced scattering as the tested setting
  within 3.5 % per channel, but matching Folk and Weaver in kind and amount.
  `pore_white` scales by the same factor (0.2 % → 1.17 %) so white
  pigment keeps its strength. Its
  forward-scattering cosine is a little higher (0.62 against 0.56 in blue),
  so a 4-cut look check (about 4 minutes) compares spacing and L sd with the
  tested setting before the stone run. If they agree within run-to-run
  noise, the 20-cut check stands.
- B. Keep 0.3 % air, exactly as tested. Simpler, but the number has no
  source and the physical form is wrong.
- C. 1 % air. Bands crisper (spacing nearer the stand-in), but it needs
  about 6 % water-filled pores, beyond the one measurement, and costs
  2.4 times the old render time (stone run about 26 h).

Either A or B can later be varied together with the translucency mapping
(step 2) without reopening this choice.

**10b. Slab depth for reflected renders.**
- *(Recommended)* **2 mm**, checked in the same 4-cut look run. It saves
  about 22 % (206 against 265 s on the look cuts; stone run about 10 h
  instead of 13 h). The image differs by 0.14 grey levels, well under the
  photos' noise.
  *Correction (2026-11-02):* the run times here were mis-scaled from the 4
  look cuts (206 s for 4 cuts is about 52 s a cut, so 100 cuts take about
  1.4 hours, not 10; the 13 h and 26 h of option C are wrong in the same
  way, about 1.8 h and 4.4 h). The look run's log was not kept, so the slip
  cannot be traced exactly. The 22 % saving was measured directly and
  stands. Measured on the stone run itself: 7111 s for 100 cuts, about
  71 s a cut (2.0 hours).
- Keep 10 mm. The stone run is then exactly what the 20-cut check ran.

### 4. Grades

- Run results and the air/water equivalence: *verified* (measurement and
  arithmetic).
- That weak pore scattering is why the render lost the bands: *verified*
  within the model; nothing outside it.
- That agate's milky bands hold about 2 % water-filled 0.1 µm pores:
  *plausible* on one source [@folk1952], disputed by [@graetsch1985]. The
  alternative (light scattered at twin boundaries or grain edges, with no
  such pores) is kept open. Both give the same look here, so the compare
  cannot choose between them.
- All values stay `effective` and *speculative* as optics; a match to the
  photos supports only "this parameter set looks like these photos"
  (ADR 0007 section 5). Never evidence for a formation mechanism
  (ADR 0002).

### 5. The 4-cut look check (run on acceptance)

Water-filled pores at 1.75 % (`pore_white` 1.17 %), iron 9 × 10⁻⁴, at
2 mm depth, on the same four cuts, looks and render streams as the earlier
look runs (`look_water.py` in runs/, not committed; 212 s, peak 549 MB):

| cut | spacing, px: tested (air 0.3 %, 10 mm) | air 0.3 %, 2 mm | **water 1.75 %, 2 mm** | L sd: tested / **new** | C*: tested / **new** |
|---|---|---|---|---|---|
| 0 | 5.26 | 5.27 | **5.32** | 5.0 / **5.1** | 12.8 / **13.7** |
| 5 | 7.74 | 7.77 | **7.86** | 5.5 / **5.6** | 13.2 / **14.1** |
| 8 | 12.47 | 12.64 | **13.01** | 4.9 / **5.0** | 13.6 / **14.5** |
| 18 | 11.16 | 11.17 | **10.91** | 3.7 / **3.7** | 13.5 / **14.4** |

- Spacing changes by −2 % to +4 % with no common direction; L sd by at most
  0.1; median L stays 45–46 (*verified*). Run-to-run noise was not
  measured separately: the same streams were used, so these differences
  are the setting's own. Judged small enough that the 20-cut check stands
  for the stone run.
- Chroma C* rises by about 0.9 on every cut (*verified*), a small
  systematic shift (water pores scatter blue a little more forward, so a
  little less blue returns). The photos' C* runs 2.7 / 7.3 / 33.7 (5th /
  50th / 95th percentile), so renders sit a little further above the
  median. Added as a third watch item.

## Alternatives considered

- **Raise contrast by pigment alone.** Iron at ten times darkened the image
  (median L 34) without restoring the bands' spacing. Rejected.
- **Deeper mixing or matrix absorption.** Only dimmed the image. Rejected.
- **Retune grain radius instead of pores.** The grains scatter almost
  straight ahead (g ≈ 0.998), so they carry light sideways rather than
  back; raising them would not separate bands in reflected light. Not
  tried; reasoning only.
- **Wait for the translucency mapping (step 2) before any change.** The
  bands vanish without this step, so no other knob can be judged first.

## Risks

- **Tuned to half A.** Half B stays held out and within, but B has been
  seen once now. Later steps should report both halves.
- **Contrast still low.** Renders sit near the photos' 5th to 20th
  percentile of L sd. The next knob is the translucency to optics mapping
  (ADR 0007 7a C), as step 2.
- **Edges may be too orderly** (C16, the watch items above).
- **Chroma a little higher** with water pores (section 5).
- **GPU-only.** The checks ran locally; CI cannot repeat them.

## Consequences

- On acceptance: `OpticsTable` defaults change as chosen in 10a (and the
  reflected depth in 10b), with a test pinning the values, and ADR 0007
  section 3 gains a pointer to this ADR.
- The 4-cut look check runs first if A or the 2 mm depth is chosen (about
  4 minutes). Then the band-regularity ADR (pv 0.7), then the stone run.
- The stone's verdict is reported as after tuning step 1.
- References added: none (folk1952 and graetsch1985 are catalogued).
