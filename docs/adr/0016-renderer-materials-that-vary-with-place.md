# 16. Renderer: band materials that vary with place (lenses and drift)

- Status: accepted (owner, 2026-11-04): 16a A, 16b A, 16c B (section 2).
  All three as recommended except 16c, where A was recommended.
- Date: 2026-11-04
- Builds on: [ADR 0007](0007-renderer-cut-face-monte-carlo.md) section 3 (materials
  as a function of the growth step), [ADR 0014](0014-optics-tuning-step-4-colour.md)
  4m (current optical defaults, `iron_floor` 0.28),
  [ADR 0015](0015-band-order-diagnosis-and-options.md) sections 1, 6 and 7
  (option C; the owner's eye first; the dark-grey risk over the black bottom);
  agatesim ADR 0078 (the fields; its sections 2, 3 and 5).

## Context

agatesim ADR 0078 (pinned at 35bcd89) makes each band's pigment and
translucency vary along the band. There are two independent fields: pigment
lenses and a translucency drift. Both read a point's place and its growth
time alone. `FeatureLayer.material_shift(points, t)` returns a pigment
multiple (`gain`, 1 off a lens, up to 1 + `lens_strength`) and a
translucency change (`change`, up to ±`translucency_drift`). agatesim
applies them as `impurity × gain` and `translucency + change`, both clipped
to [0, 1], and only to banded chalcedony (phase CHALCEDONY, band ≥ 0). The
clear first layer is left as it is.

Nodule's renderer cannot show this today. `BandRecord` reads one set of
materials per growth step and refuses a step whose values differ between
points (ADR 0007 section 3). The GPU camera then looks up one material per
step. Three facts set the design:

- **Thin lenses.** The owner chose 0.1 mm lenses. The time grid has 0.2 mm
  spacing, so a lens can't be stored on it. Interpolated growth time is
  accurate far below the spacing (agatesim ADR 0078 section 2), so the
  field must be worked out at each sample from the sample's place and time.
- **Cost on the CPU** (measured, scratchpad `shift_cost.py`, n2-slice
  settings, 0.1 mm lenses with drift): agatesim's `material_shift` handles
  0.5 million points a second, with a peak of 144 MB per million points.
  A grid baked at 0.05 mm over a 50 × 50 × 2 mm slab is 4 × 10⁷ points:
  about 80 s per cut, and a 0.1 mm lens there is only two nodes thick.
- **The iron floor is a step** (ADR 0014 4m). Below impurity 0.28 iron
  oxides absorb nothing. At 0.28 the iron absorption jumps to
  (0.83, 0.29, 0.16) /mm, about half its value at 0.45, which is
  (1.61, 0.60, 0.29) /mm (measured with `OpticsTable.pigment_mua`).
  Band impurity runs 0.06 to 0.45, and 57% of bands lie between 0.20 and
  0.40. A lens that multiplies impurity would carry many faint bands
  across the step, giving a hard-edged core inside the lens.

### The owner's first look (2026-11-04)

A rough stand-in drawn from agatesim's `Fields`, not the renderer
(`runs/stone-2026-11-02/variants/lens-first-look/look.png`), showed four
panels. The owner: **"0.1 mm with drift" looks best, but the effect should
be more sporadic and sparse across the cut face.** The owner suggested
lowering lens cover to 20%, 10% and 5% (`lens_cover` 0.2, 0.1, 0.05).

Lower cover also makes each lens smaller. This was measured on the noise
(setting 4 mm, thickness 0.1 mm; `cover.py` beside the panel):

| cover | length p10 / p50 / p90 (mm) | thickness p50 (mm) | lenses per 100 mm of band |
|---|---|---|---|
| 0.20 | 1.7 / 4.0 / 7.1 | 0.100 | 4.5 |
| 0.10 | 1.4 / 3.3 / 5.6 | 0.080 | 2.8 |
| 0.05 | 1.1 / 2.8 / 4.8 | 0.072 | 1.7 |

agatesim ADR 0078 section 3 records cover 0.2 only. To keep the size fixed
while cover falls, the length setting must be raised by hand.

**Lens cover thins only the lenses.** The drift still runs across the
whole face and may account for much of the busyness the owner saw. This
was put to the owner (16c below).

### Sources

The sources are as in ADR 0015 section 7, narrowed. [@raman1953] is
context only, since the lighting behind its dark and bright sectors isn't
stated and transmission there depends on the ray's angle to the bands
(agatesim ADR 0078 section 6). That leaves [@goetze2020, §4.3]
("laminated and lenticular agglomerates" of hematite along the banding) and
[@howard2018, pp. 99, 104] (widths and inclusions that vary along a band).
Both are qualitative: they give no sizes and no amounts.

## Decision

### 1. What is recorded now

The owner's verdict and the cover plan above, and the narrowed source list.

### 2. The owner's choices (2026-11-04)

**16a. How the renderer gets the fields at each sample.**

- **A (chosen, recommended): a GPU copy, tested against agatesim.**
  Nodule's CUDA code works out agatesim's sheet noise at each sample. Fractures are already
  handled this way (`cuda/scene.cuh` mirrors agatesim's `Fracture.restore`,
  checked by `tests/test_render_gpu.py`). The test compares it with
  agatesim's own `material_shift` at random points, so a change in
  agatesim shows up there. It is exact at any lens thickness. Its cost per
  sample has to be measured.
- B: a fine grid baked per cut by agatesim's own code. No copy, but about
  80 s per cut at 0.05 mm, and 0.1 mm lenses blur.
- C: the GPU code lives in agatesim and Nodule includes it. No copy in
  Nodule, but every change needs work in both projects.

**16b. What a lens multiplies.**

- **A (chosen, recommended): the amount of the band's own pigment.**
  `mua = matrix_mua + gain × pigment_mua(pigment, impurity)`. The band's
  impurity, and with it the hematite share (its hue), stays at the
  band's value. A lens is more of the same pigment, so it has no step and
  no change of hue. Bands below the iron floor get no visible lenses,
  which also makes lenses sparser, as the owner asked. This **departs
  from agatesim's field values** (`impurity × gain`). Nodule's
  descriptors that read agatesim's own colour (`descriptors.py`) will see
  agatesim's version, not the rendered one. That is noted, not fixed.
- B: impurity × gain, keeping the step. Lenses also turn redder, and may
  show hard edges.
- C: impurity × gain, with the floor softened into a ramp. This changes
  the ADR 0014 calibration, so the colour statistics need re-checking.

**16c. The drift in the first real renders.**

- A (recommended): keep drift at 0.25 and judge again on real renders.
- **B (chosen): add one set with lower drift.**
- C: a sparse drift, which would be a new agatesim setting and come first.

### 3. Design

- **Reading the record.** When agatesim gives `band_impurity` and
  `band_translucency`, `BandRecord.of` reads those, which are the step's
  own values, and still refuses a step that varies in them. It also
  records which steps take the shift: banded chalcedony, as in agatesim.
- **Per sample (GPU).** For each sample in a band that takes the shift:
  growth time `t` as now (trilinear grid lookup at the restored point),
  then growth depth from `t` by agatesim's step-edge table
  (`MaterialVariation.depth_mm`), then each field's two bracketing
  sheets, then two noise lookups per field (the sheet blend in agatesim
  `SheetNoise.__call__`), then `gain` and `change` as in `shift`. The
  noise is one octave of agatesim `BandNoise`: a rotated, offset,
  periodic cubic B-spline on a 128³ tile. It needs 8 MB per field on the
  GPU, plus the sheet positions and offsets.
- **Optics per sample.** Absorption follows 16b. Scattering follows the
  shifted translucency, clipped to [0, 1], through a table in translucency
  for each band material (grain radius and pore fraction, ADR 0007
  section 3). The table is checked against `OpticsTable.band` to a stated
  tolerance. The majorant `mubar` covers the largest gain and the whole
  translucency range. First scope: iron-stained and unpigmented bands
  only. Any other pigment with the shift on is refused, not drawn wrong
  (n2-slice uses iron only).
- **With the shift off nothing changes.** The per-step table path stays
  as it is, and the 442 existing tests must pass unchanged.

### 4. Controls and measurements

- GPU vs agatesim CPU `material_shift` at about 10⁵ random points across
  the slab: they must match to float precision. This is the copy's guard.
- Shift off: renders identical to the current renderer, with the same
  seed and samples.
- Negative control for the per-sample path: a stone whose shift is a
  constant gain (all one lens) must render as the per-step path does with
  that pigment amount, and the drift must show no structure finer than a band.
- Measure render time and memory against the current 6-cut probe (about
  8 min, 1 GB). Flag a slowdown above 2×.

### 5. Look cuts (owner's eye first, ADR 0015 section 1)

Real renders at the first-look settings: lens_strength 1.5,
lens_length_mm 4, lens_thickness_mm 0.1, drift length 20 mm and thickness
0.5 mm, seed 3, wobble as n2-slice, one closing channel (`channel_count`
1, kept from ADR 0015 section 6).

1. cover 0.2, drift 0.25
2. cover 0.1, drift 0.25
3. cover 0.05, drift 0.25
4. cover 0.1, drift 0.1 (16c B; the middle cover, which the owner may change)

Each set is about 8 min before any slowdown. Offered afterwards and not
dropped: length 10 mm; subtle (strength 0.5, drift 0.1); thickness 0.5 mm.

### 6. Labels and grades

- Every lens and drift setting is `effective` (agatesim ADR 0078). The
  coupling of gain to pigment amount (16b) is an **assumption**, and so is
  the coupling of translucency to both scattering parts (ADR 0007 section 3).
- GPU-CPU agreement will be *verified* by its test. The look of lenses
  and drift is *speculative*: renders are procedural and are never
  evidence for what makes agate lenses.

### 7. Built (2026-11-04)

- **Code.** `render/shift.py` and `cuda/shift.cuh` hold the GPU copy. The
  stone's own fields come from agatesim's documented route,
  `wall_features(model, setting, default_rng(seed))`. The pack reads
  agatesim's `SheetNoise` and `BandNoise` internals, guarded by the test
  below. `BandRecord` reads the band values and marks the steps that take
  the shift. It marks none when agatesim varies nothing, so such records,
  tables and images are as before. `OpticsTable.materials(record, shift)`
  adds each row's translucency and pigment absorption. With drift it also
  adds a 1,025-row table in translucency, and a drifted sample takes the
  nearest row. `CutRenderer` turns the shift on by itself when the record
  has steps that take it. Fakes carry the flag with their moved runs.
- **Checks** (`tests/test_shift.py`, `tests/test_shift_gpu.py`; 13 tests,
  each with a control). *verified*:
  - The rebuilt fields are the stone's own: they reproduce agatesim's
    varied impurity to 3e-5 and its translucency to 1e-6, the residue of
    float32 growth times. Control: another seed's fields miss by > 0.1.
  - GPU vs agatesim `material_shift` at 1e5 points: gain within 3.1e-4
    (p99 2.3e-5), change within 2.1e-6. This is float32 with fast math,
    not bit-exact. Control: a transposed rotation fails by > 100× the
    bounds.
  - With no step taking the shift, the image is bit-identical to the plain
    renderer's at the same seed. All 442 earlier tests pass unchanged.
  - Constant gain 2.5 per sample vs the per-step table with 2.5× pigment:
    mean difference 1.9e-9 against 3.3e-2 without the lens (the control).
  - Drift vs lenses half a band (0.225 mm) deeper at the same place:
    correlation 0.76 for the drift and 0.01 for the 0.1 mm lenses (the
    control). The drift shows no structure finer than a band.
  - The drift table's nearest row is within 1% of the exact optics (mus,
    1 − g, lobe share). Control: a 33-row table misses by > 1%.
- **Cost** (one n2-slice cut, 268 × 244 px, 64 spp, dome, black bottom;
  best of 3): 0.43 s plain; lenses only 0.71 s (1.65×); drift only 1.45 s
  (3.4×); both 1.96 s (**4.6×, over the 2× flag**). GPU memory is
  unchanged at ~1.1 GB. The drift is read at every sample in every banded
  layer, and the lenses only in bands above the iron floor. Making the
  milkiest bands milkier also raises the majorant, from 28.7 to 52.9 /mm,
  which alone costs ~1.7×.
- **Measured option, not built.** The drift could be baked once per cut
  on the 0.2 mm growth-time grid and interpolated. Its error in
  translucency would be p50 0.005, p99 0.036, and at worst 0.10 where
  growth time jumps (drift range ±0.25; 16k points). Lenses would stay
  exact. This departs from 16a, so it is the owner's choice.

### 8. The owner's verdict and the preset (2026-11-04)

- **Look cuts.** The run was stopped after cuts 0 and 1 (the PC ran low on
  memory); the owner judged from those two. **Set 4 is best** (lens cover
  0.1, drift 0.1). The owner's words: "the variance effect is subtle, as it
  should be, since ... the original unmodified images are already pretty
  good". Drift 0.25 had turned a milky eye clear and then dark brown-grey
  over the black bottom (the first risk below, cut 1). The 0.1 mm lenses
  are barely visible at full-sheet size (the second risk).
- **Preset.** `n2-slice` now carries set 4: lens_strength 1.5,
  lens_length_mm 4, lens_thickness_mm 0.1, lens_cover 0.1,
  translucency_drift 0.1, translucency_drift_length_mm 20,
  translucency_drift_thickness_mm 0.5. All `effective`, judged by eye.
  The look is *speculative* and is no evidence for how lenses form.
- **Tests.** A new test checks that the preset gives a variation at set 4;
  the plain stone (lens_strength and drift 0) still gives none, as the
  control. Two tests of the renderer's machinery now build that plain
  stone explicitly (fixture `plain_config`): the per-step table check,
  which assumes values constant along a band, and one rendered cut
  measured end to end. That cut sits on the quality gate's edge: without
  lenses, one of three render seeds already fails the gate ("not fit"),
  and with set 4 the test's seed fails it too. The gate is unchanged.
  Every other test passes with set 4 as it stands. 456 tests.
- **Cost of set 4** (section 7's cut and pattern; 269 × 245 px, 64 spp,
  best of 3): plain 0.49 s; lenses only 0.91 s (1.9×); drift only 1.66 s
  (3.4×); both 2.22 s (**4.6×**), the same as drift 0.25. The majorant
  is 52.9 /mm for both drifts. It takes the densest row of the whole
  drift table, so it does not depend on the drift's size. On full survey
  cuts (2048 px, 256–1,024 spp) set 4 took 2.5–3.4× the plain stone.
  The bake option (section 7) stays open.
- **Measured option, not built: a tighter majorant.** A bound over the
  translucency each band can reach (its own ± drift size) gives 42.8 /mm
  for set 4 and 1.61 s against 1.97 s (−18%). Drift 0.25 gains nothing
  (52.9 either way). The mean image is unchanged: −1.7e-4 against a
  per-pixel spread of 3.4e-2, within noise. Most of the cost is the drift
  read at every sample, not the majorant.
- **Statistics probe** (`lens_stats.py`, scratch): ADR 0014's four look
  cuts (k 0, 5, 8, 18), plain and set 4 on the same planes, looks and
  render streams, analysed by ADR 0014's `depth_compare.py` with the
  exposure fitted to median L\* 62.1. 1,435 s; peak traced Python memory
  671 MB (GPU not traced).

  | setting | Y p90/p10 | L\* sd | C\* median | corr(L\*, C\*) | band ΔL\* / Δa\*b\* | shuffled |
  |---|---|---|---|---|---|---|
  | plain | 2.53 | 8.75 | 9.01 | +0.10 | 12.2 / 7.1 | 1.5 / 1.2 |
  | set 4 | 2.67 | 9.35 | 9.12 | +0.05 | 10.6 / 7.6 | 1.5 / 1.3 |
  | photos (ADR 0014) | 2.49 | 9.7 | 7.3 | −0.24 | 14.4 / 6.3 | 2.2 / 0.7 |

  *Verified* as a model property (one stone, one seed, four cuts). The
  plain row repeats ADR 0014's defaults (the wall channel left them
  unmoved). Set 4 closes most of the lightness-spread gap (8.75 → 9.35
  against 9.7). The L\*–C\* correlation moves a little toward the photos'
  sign (+0.10 → +0.05) but stays positive. Band lightness contrast moves
  away from the photos (12.2 → 10.6 against 14.4); differences under
  ~0.5 are noise here (ADR 0014). Contrast and band colour move slightly
  away. Render-to-photo only: no evidence for how lenses form.

### 9. The drift baked per cut (the owner's choice, 2026-11-04)

The owner chose the bake measured in section 7. This departs from 16a for
the drift only. The lenses are still worked out at every sample, since
0.1 mm lenses are thinner than the grid.

- **What it does.** Once per cut, `bake_drift` evaluates the drift's
  translucency change at every node of the cut's 0.2 mm growth-time grid
  (restored coordinates). It uses the node's filled growth time and the
  tested GPU copy (`shift_points`). The camera reads the change by the
  same trilinear lookup as the growth time (`grid_at`, scene.cuh). The
  per-sample drift stays as the reference (`Rendering.drift = "exact"`).
  `"baked"` is the default. `render` refuses a baked grid made on another
  grid. The majorant is unchanged and still bounds every sample: the
  interpolated change stays within ± the drift size.
- **Accuracy** (*verified*; one cut, 2e5 points in banded chalcedony
  within the slab, against the per-sample drift at the grid's growth
  time). Error in translucency for drift 0.25: p50 0.0047, p99 0.034, max
  0.11, matching section 7's estimate. For set 4 (0.1): p50 0.0019, p99
  0.014, max 0.043. The largest errors sit where growth time jumps.
  Control: a wrong copy baked the same way (each sheet given another's
  pattern offset) misses by p50 0.10.
- **Image** (*verified*; drift alone, same seed and majorant, 32 spp,
  2 mm blocks of a 20 × 20 mm view). Baked and per-sample images differ by
  3.6e-3 in luminance, against 3.7e-2 between the per-sample drift and
  no drift (the control). The whole-image bias is −1.5e-3.
- **Cost** (section 7's cut; best of 3): set 4 renders in 1.57 s baked
  against 2.19 s per sample (−28%), still ~3.2× the plain stone. The
  lenses and the majorant, which still spans the whole drift table, are
  the rest. The bake takes 0.25 s and 8 MB per cut. The tighter majorant
  (section 8) is not built.
- **Tests.** Three GPU tests (bake accuracy, image, refusal of another
  grid's bake) and one CPU test of the setting. 460 tests.

### 10. Twenty-cut survey with set 4 (2026-11-04)

The plain stone's 20-cut run (ADR 0015 section 6, `gpu_check_defaults.py`)
repeated with set 4 and the baked drift: the same seed, cuts, looks and
render streams. Looks come from half A; half B is held out. **5,874 s**
against 1,670 s plain (3.5×); peak traced Python memory 1,171 MB; GPU
pool 197 MB at the end. Nine of 20 cuts were noise-matched in both runs.

| all 20 cuts | plain | set 4 | photos p50 / p95 (A; B) |
|---|---|---|---|
| joint distance vs A (line 15.9) | 1.1 | 0.4 | |
| joint distance vs B (line 7.9) | 3.2 | 2.8 | |
| spectrum block vs B (line 1.97) | 0.36 | 1.07 | |
| C26 slope (pct A / B) | 2.29 (86 / 90) | 2.66 (86 / 95) | 1.41 / 2.94; 1.23 / 2.55 |
| C26 peak (pct A / B) | 3.05 (67 / 67) | 2.68 (62 / 48) | 2.07 / 4.78; 2.83 / 5.54 |
| C15 curvature (pct A / B) | 0.08 (78 / 83) | 0.09 (87 / 87) | 0.06 / 0.10; 0.06 / 0.11 |
| C15 high (pct A / B) | 0.19 (52 / 53) | 0.21 (70 / 62) | 0.18 / 0.25; 0.19 / 0.30 |
| C16 coherence (pct A / B) | 0.91 (98 / 98) | 0.90 (98 / 98) | 0.67 / 0.84; 0.60 / 0.87 |

- Read with section 12: at 20 paired cuts none of the changes in this table
  exceeds cut-to-cut noise, and the reading below is withdrawn there.
- *Verified* as a model property (one stone, one seed, 20 paired cuts;
  no test of whether each change exceeds cut-to-cut noise). Both runs
  are within the joint lines, and set 4 is a little closer on both
  halves. No block falls outside.
- Moved away from the photos: the C26 spectrum slope, back to the 95th
  percentile of B (the open gap of ADR 0014 4o), the spectrum block
  against B, and C15 curvature. C16 coherence is unchanged and still
  above the photos' 95th percentile. The C26 peak moves toward B's
  median.
- *Speculative* reading: the drift adds slow change along the bands,
  which would put power at coarse scales and steepen the slope. Not
  tested. Render-to-photo only; never evidence for how lenses form.
- Cost for longer runs: the 100-cut plain run took 8,466 s, so set 4
  would take about 8 h at 3.5×.

### 11. A tighter majorant (the owner's choice, 2026-11-04)

The bound measured in section 8 is now built. It is exact: the renderer
draws the same image, only with less wasted work.

- **What it does.** `majorant` no longer pairs every band that takes the
  shift with the densest row of the whole drift table. Each band is paired
  with the densest row it can reach, its own translucency ± the drift
  size, clipped to [0, 1], widened by one row on each side for float
  rounding in the kernel's nearest-row pick. `DeviceShift` carries the
  drift size; its default of 1 gives the old whole-table bound. The baked
  drift stays within ± the drift size (section 9), so the bound covers it.
- **Bound** (*verified*; `tests/test_shift.py`). It is at least the
  largest mu_t the kernel's own float32 pick reaches over 20,001 changes
  in [−d, d], band by band. It is below the whole-table bound and equals
  it at drift size 1. Control: a bound that ignores the drift (size 0)
  misses samples the drift reaches.
- **Image** (*verified*; set 4's drift 0.1 on the look stone, 32 spp,
  2 mm blocks of a 20 × 20 mm view). Tighter and whole-table bounds differ
  by 5.4e-3 in block luminance, against 5.6e-3 between two seeds of the
  same bound; whole-image bias −1.7e-5. The two bounds draw different
  random paths, so render seeds are not comparable across this change.
- **Cost** (section 7's cut, 268 × 244 px, 64 spp, dome, black bottom,
  baked drift; best of 3; `majorant_cost.py`, scratch, 20 s). The
  majorant falls from 52.9 to 42.8 /mm. Set 4 renders in 1.11 s against
  1.34 s (**−17%**), 2.6× the plain stone (0.43 s); collision events per
  path 295 against 364. GPU memory is unchanged. Drift 0.25 still gains
  little (section 8). 461 tests.

### 12. The 100-cut run with set 4, and a corrected reading of section 10 (2026-11-06)

`stone_100_defaults.py` with the current preset: set 4, baked drift, tighter
majorant, one stone, seed 0, looks from half A, half B held out. It ran
without a crash or resume. The script's report still says "ADR 0014 4m
defaults"; that label is out of date, and the preset is what counts.

- **Cost** (*verified*). **25,494 s** (7.1 h); peak traced Python memory
  1,246 MB; GPU pool 327 MB at the end. Rendering alone took 23,059 s
  against 6,388 s for the plain run, **3.6×**. The 68 cuts at 1,024 spp in
  both runs also give 3.6×, and the median per-sample ratio is 3.56. The
  whole run is 3.0× only because about 1,600 s of other work is the same in
  both. Section 11's 2.6× came from a 268 × 244 px cut at 64 spp and does not
  carry over to full-size cuts; why is untested. Only two cuts share samples
  and pixels with section 10's run, so the majorant's gain at full size has
  thin support: 1,486 s to 1,174 s (−21%).
- **The baseline is not plain set 4's.** The plain 100-cut run (ADR 0014 4o)
  predates the closing channel (ADR 0015 section 6). This run differs from it
  by the channel, the lenses and the drift together. No 100-cut run exists
  with the channel alone, so the table below cannot separate them. The
  channel adds no measurable cost: 1,225 s against 1,240 s over 20 cuts.

| 100 cuts, same planes | no channel, no set 4 | channel + set 4 | photos p50 / p95 (A; B) |
|---|---|---|---|
| joint distance vs A (line 15.9) | 1.1 | 0.5 | |
| joint distance vs B (line 7.9) | 4.0 | 2.7 | |
| spectrum block vs A (line 1.40) | 0.46 | 0.49 | |
| spectrum block vs B (line 1.97) | 0.85 | 0.64 | |
| C26 slope (pct A / B) | 2.72 (86 / 95) | 2.66 (86 / 95) | 1.41 / 2.94; 1.23 / 2.55 |
| C26 peak (pct A / B) | 2.99 (67 / 62) | 2.73 (62 / 48) | 2.07 / 4.78; 2.83 / 5.54 |
| C15 curvature (pct A / B) | 0.09 (87 / 87) | 0.09 (87 / 87) | 0.06 / 0.10; 0.06 / 0.11 |
| C15 high (pct A / B) | 0.20 (65 / 55) | 0.21 (74 / 64) | 0.18 / 0.25; 0.19 / 0.30 |
| C16 coherence (pct A / B) | 0.90 (98 / 98) | 0.89 (98 / 98) | 0.67 / 0.84; 0.60 / 0.87 |

- *Verified* as a model property (one stone, one seed; the bootstrap below
  resamples cuts of this stone and says nothing about other stones). Both
  runs are within the joint lines, with no block outside. The joint distance
  falls on both halves beyond cut-to-cut noise (95% bootstrap over paired
  cuts: A −1.19 to −0.07; B −2.37 to −0.60). The spectrum block's change is
  within noise on both halves. C26 peak falls and C16 coherence falls very
  slightly, both beyond noise; C16 stays above the photos' 95th percentile on
  all cuts.
- **Noise-matched cuts** (32 here, 31 plain, 29 in both) do not confirm the
  joint gain. On the 29 cuts matched in both, the joint distance goes from
  3.0 to 2.6 vs A and from 3.5 to 3.4 vs B, well within noise (A −5.40 to
  +2.60; B −2.59 to +2.13). The other 68 or so cuts are capped at 1,024 spp
  and stay noisier than the photos, so the all-cuts gain may depend on how
  noise enters; untested.
- *Plausible*: set 4, not the channel, gives the joint gain. Over the 20-cut
  planes the channel left the joint distance almost unchanged (A 1.13 to
  1.13; B 3.02 to 3.18) and set 4 then lowered it (A 0.43; B 2.85), but those
  20-cut steps are within noise themselves. Render-to-photo only; never
  evidence for how lenses form.
- **Section 10's reading is withdrawn.** It said set 4 moved the C26 slope and
  the spectrum block away from the photos. Bootstrapped over its 20 paired
  cuts, every change it listed includes zero: slope 2.29 to 2.66 (−0.54 to
  +1.38), spectrum vs B 0.36 to 1.07 (−0.15 to +1.05), joint vs A (−1.78 to
  +0.26), joint vs B (−3.21 to +0.80). The speculative reading there, that
  drift steepens the slope, now has no result to explain. The 100-cut planes
  differ from the 20-cut planes, so the two runs cannot be paired.
- **The open gap (ADR 0014 4o) is the shape of the C26 slope spread, and it
  did not move** (*verified* for this stone). The renders have too many steep
  cuts, in both runs. Only 21 photos have a slope in each half.

| C26 slope | below 1.5 | 1.5 to 2.5 | above 2.5 | median (95%) |
|---|---|---|---|---|
| no channel, no set 4 (98 cuts) | 29% | 15% | 56% | 2.72 |
| channel + set 4 (100 cuts) | 26% | 16% | 58% | 2.66 (2.47 to 2.82) |
| photos, half A (21) | 57% | 29% | 14% | 1.41 |
| photos, half B (21) | 62% | 29% | 10% | 1.23 |

- **The slope is unstable on a single plane** (*verified*). Between the two
  100-cut runs it changes by more than 1 on 27% of the 98 paired cuts; their
  correlation is 0.39. On one plane it went from 5.74 to 0.95. That pair also
  differs by the channel, but section 10's pair shares the channel and changes
  by more than 1 on 30% of its 20 cuts. A median of 20 cuts can therefore swing
  on a few cuts. *Speculative*: the fit may flip between two regimes,
  which would be a property of the measure and not of the stone. Section 13
  tests this.
- **Lens variants: keep set 4** (the owner's verdict, 2026-11-04). They looked
  at 4L (length 10 mm), 4T (thickness 0.5 mm) and 4S (strength 0.5), cuts 0
  and 1 (`lens_variants.py`, 757 s). All look nearly the same at cover 0.1; 4T
  is a little patchier.
- Scratch analysis, gitignored: `compare100_*.py` beside the run scripts.

### 13. Why the C26 slope jumps (the owner's choice, 2026-11-06)

Section 12 found the slope changing by more than 1 on about a quarter of
paired cuts. The cause is how the measure is built. The slope is fitted from
the resolving limit L up to the bin with the strongest peak, so it depends on
which peak wins, and several peaks often come close.

- **Slope jumps follow peak jumps** (*verified*, 98 paired cuts of section 12).
  Their sizes correlate at 0.92. Where the peak kept its bin (65 cuts) no slope
  changed by more than 1; where it moved over 1.5 octaves (24 cuts) every slope
  did.
- **The tests reproduce the runs** (*verified*). The set 4 run's own random
  streams gave its stored slope on 12 of 12 cuts. Today's code with the old
  plain preset gave the plain run's stored slope on 8 of 8. Slopes rebuilt from
  saved curves match on all 168 render measurements and all 42 photos. The
  cuts are the cheapest ones that flipped or stayed put, not a random sample;
  the 12 cuts' stored median, 2.67, happens to sit close to the run's 2.66.
- **Noise alone tips few set 4 cuts** (*verified* for these cuts). A new render
  seed plus three noise draws moved one of 12 cuts by more than 1 (cut 1, 1.58);
  the median range of the others is 0.24. Noise draws alone tipped more cuts in
  the materials without set 4, as the second table shows, on small selected
  sets.
- **Many things tip it** (*verified* for these cuts). One factor at a time on
  the same planes; each entry is the strongest peak's period over the inner
  radius. "Size" is the image's pixels across, which the stand-in probe sets
  and which differs between the runs by up to 14%.

| cut | plain | plain, set 4's size | + channel | + lenses and drift | set 4, plain's size | what tipped it |
|---|---|---|---|---|---|---|
| 3 | 0.034 | 0.034 | 0.545 | 0.545 | 0.545 | channel |
| 20 | 0.545 | 0.545 | 0.545 | 0.068 | 0.068 | lenses and drift |
| 24 | 0.081 | 0.081 | 0.081 | 0.648 | 0.081 | lenses and drift, then size |
| 26 | 0.057 | 0.229 | 0.229 | 0.229 | 0.193 | size, near-tie under noise |
| 66 | 0.648 | 0.648 | 0.648 | 0.193 | 0.193 | lenses and drift |
| 71 | 0.162 | 0.162 | 0.029 | 0.029 | 0.029 | channel, near-tie under noise |
| 92 | 0.162 | 0.386 | 0.386 | 0.162 | 0.162 | size, then lenses and drift |
| 98 | 0.229 | 0.229 | 0.229 | 0.229 | 0.229 | nothing |

| noise draws alone change the slope by more than 1 | cuts |
|---|---|
| plain | 3 of 8 |
| plain, set 4's size | 2 of 8 |
| channel, set 4's size | 1 of 8 |
| set 4, plain's size | 0 of 8 |
| set 4 (12 cuts, 2 renders each) | 1 of 24 |

- **The margins are thin** (*verified* for these cuts). The winner often beats
  its best rival an octave or more away by only 5–50%. Cut 3's slope fell from
  5.40 to 0.75 when the channel moved its peak. Size alone changed cut 92 by
  2.4 for a 1.5% resize, on a cut already tied under noise.
- **Section 12's speculation is answered in part.** The slope does flip between
  fits, but noise alone is rarely the trigger in set 4. Any modest change to the
  image can do it. The change is real; the size of the jump comes from the
  measure.
- **The mix of peaks** (*verified* for this stone). With the peak finer than 0.1
  of the radius the slope is steep; with it coarser, renders and photos agree.

| | cuts with a fine peak | slope, median | cuts with a coarse peak | slope, median |
|---|---|---|---|---|
| set 4 | 63 of 100 | 3.02 | 37 | 0.93 |
| no channel, no set 4 | 64 of 98 | 2.89 | 34 | 0.82 |
| photos | 14 of 42 | 1.98 | 28 | 0.84 |

- **Which photos have a slope** (*verified*). A photo has one exactly when its
  rim outline was found and accepted: 42 of 97, half A and half B 21 each. Of
  the other 55, 42 have no usable rim and 13 have a rejected outline. All 100
  set 4 cuts have a known outline. So the slope compares every cut with the
  photos that show a whole rim; whether that subset differs is not tested.
- *Speculative*: the pile of fine peaks may be this stone's own band spacing.
  One stone has one spacing over its radius, and the photos are many stones.
  Not tested; it needs more than one simulated stone.
- **A fixed fit range is steadier** (*plausible*: 8 and 12 selected cuts). Range
  of the slope within a cut over every seed, draw, size and material above:

| fit range | median | largest | set 4 seeds only, median / largest |
|---|---|---|---|
| L to the peak (today's) | 2.75 | 4.99 | 0.26 / 1.58 |
| L to 0.1 of the radius | 1.08 | 2.57 | 0.34 / 2.12 |
| L to 0.25 of the radius | 0.61 | 1.51 | 0.20 / 1.48 |
| L plus 8 bins (2 octaves) | 1.10 | 1.92 | 0.39 / 1.32 |
| L plus 12 bins (3 octaves) | 0.88 | 1.33 | 0.22 / 0.89 |

  Part of what remains is real image change, so these ranges would not fall to
  zero for a good measure. On set 4's seeds alone, today's range is already
  about as steady as the others; the gain shows when the image changes.
- **The gap's size depends on the fit range** (*speculative*). The 12 cuts'
  median, against the photos:

| fit range | cuts' median | photos' medians (A; B) | percentile (A / B) |
|---|---|---|---|
| L to the peak | 2.67 | 1.41; 1.23 | 86 / 95 |
| L to 0.1 | 2.80 | 1.18; 1.17 | 100 / 100 |
| L to 0.25 | 1.66 | 0.88; 0.96 | 100 / 95 |
| L plus 8 bins | 2.42 | 2.13; 2.34 | 62 / 62 |
| L plus 12 bins | 1.57 | 1.49; 1.69 | 52 / 43 |

  Ranges counted from L close the gap and ranges ending at a fixed share of
  the radius keep or widen it. This does **not** show the gap is an artefact:
  12 selected cuts, five ranges tried, half B looked at. A range picked because
  it closes the gap would prove nothing. Any replacement must be chosen on
  stability and half A, needs a negative control (a result without the form
  that it must reject), and then needs fresh cuts, since half B has now been
  seen for this question.
- **Effect on what has been reported.** The numbers stand; their reading
  narrows. The C26 slope's median is steady across 100 cuts, but a single cut's
  slope, or a median of 20, is not a reliable way to judge a change. Section 12
  stays as the record of the slope gap *as measured*; whether the stone really
  lacks what the photos have is open. The slope is one value in the joint
  comparison and the spectrum block, which was fitted in ADR 0011, so changing
  it could move those results too. Nothing has been changed in the code.
- **Cost.** Seed test 1,140 s; factor test 1,232 s against an estimate of about
  720 s, cause not examined. Memory was not measured in these two tests; the
  same render path peaked at 1,246 MB in section 12. Renders only; no image
  saved, and no photo left the PC. Scratch: `slope_*.py` beside the run scripts,
  outputs in `slope-seeds/`.

### 14. Choosing a slope range on a fair sample (the owner's choice, 2026-11-06)

Section 13 could not judge the fit ranges: its cuts were selected and half B
had been seen. Here are the first 25 cuts of the set 4 run, rendered again with
curves saved, and the same 25 planes under the plain preset, as a second state.
Both reproduce their stored runs: all 25 slopes are identical, and so is every
other stored value but the timings. Four of the 25 (1, 3, 20, 24) were selected
in section 13; the rule's results are repeated without them.

**The rule, fixed before the plain pass finished.** Eligible: defined on at
least 90% of paired cuts and of half A's photos with a slope. It must beat
shuffled pairs of different cuts (the negative control: median change below the
5th percentile of 2,000 pairings). Rank by median change. Today's range is
replaced only if the best one's gain excludes zero (95% bootstrap over cuts).
The correlation between states is reported so a range steady only because it is
flat is seen. The gap to the photos has no part; half B was not read. A dry run
of set 4 against itself had printed half A's medians before the plain pass
finished, so they had been seen, though the rule does not use them.

| fit range | median change | over 1 | largest | r | Spearman (95%) | between-cut IQR (set 4 / plain) | beats shuffle |
|---|---|---|---|---|---|---|---|
| L to the peak (today's) | 0.28 | 32% | 4.96 | −0.01 | 0.27 (−0.22 to 0.70) | 2.01 / 1.49 | yes |
| L to 0.1 | 0.19 | 4% | 1.24 | 0.91 | 0.83 (0.59 to 0.94) | 1.26 / 1.02 | yes |
| L to 0.25 | 0.13 | 0% | 0.69 | 0.86 | 0.79 (0.54 to 0.91) | 0.74 / 0.55 | yes |
| L plus 8 bins | 0.28 | 16% | 2.15 | 0.47 | 0.52 (0.10 to 0.82) | 1.12 / 0.95 | yes |
| L plus 12 bins | 0.34 | 4% | 1.07 | 0.47 | 0.41 (−0.01 to 0.74) | 0.54 / 0.51 | **no** |

- **The rule's outcome** (*verified* for this stone, these planes and this one
  pair of states). L to 0.25 ranks first; its gain over today's is +0.14 (95%
  +0.05 to +1.28), and +0.13 (+0.02 to +0.64) without the selected cuts. Its
  change is smaller on 21 of 25 cuts. L to 0.1's gain includes zero (−0.06 to
  +1.24), as does L plus 8 bins' (−0.35 to +0.93). Allowing for the three
  comparisons, the gain still excludes zero, narrowly (+0.01 to +1.62).
- **Today's range hardly follows its own cut between the states** (*verified*,
  same limits). Spearman 0.27, r −0.01. Nine of the 25 peaks moved bin. This
  confirms section 13 on a less selected sample.
- **L plus 12 bins fails its control** (*verified*). Its change is no smaller
  than the 5th percentile of shuffled pairs. Section 13 listed it as the
  steadiest on selected cuts, and as closing the gap; that reading is withdrawn.
- **L to 0.25's lead over L to 0.1 is not settled** (*plausible*). L to 0.25 also
  has less spread between cuts, so its smaller change is partly a smaller scale;
  L to 0.1 has the higher correlation, and the two were not compared directly.
  The rule compares each with today's.
- **It measures something else** (*verified* as arithmetic). L to 0.25 runs past
  the strongest peak on 15 of 25 set 4 cuts and 18 of 25 plain cuts, so it is no
  longer a fine-side slope. It is the tilt of the spectrum across a fixed band of
  scales. Today's slope is 3.04 on set 4's fine-peak cuts and 0.95 on the coarse;
  L to 0.25 gives 0.83 and 1.26. Being steadier is not enough: a blunt measure is
  steady too, and part of what it ignores may be real. **No control yet shows L
  to 0.25 can reject a result without the form.** The shuffled pairs test only
  that it knows its cut, so it cannot be adopted on this.
- **The gap, for the record only** (half A, *speculative*). Cuts' median and its
  percentile among half A's 21 photos: L to the peak 2.46 at 86; L to 0.1 1.59 at
  81; L to 0.25 0.98 at 62; L plus 8 bins 2.86 at 62; L plus 12 bins 1.79 at 62.
  Section 13's L to 0.25 at the 100th percentile came from 12 selected cuts and
  does not hold on this sample. This does **not** show the slope gap is an
  artefact: one stone, 25 cuts, half A, the photos already seen, and photos with a
  slope are only the 42 with a whole rim. Fresh cuts renew only the cuts; half B
  is spent for this question, and no photos are held out.
- **Section 12's 3.6× mixes code and materials** (*verified*). With the same
  samples and pixels, today's code renders the plain preset 1.80× slower than the
  stored plain run (median; 10th to 90th percentile 1.76 to 1.81), while set 4 run
  again takes 1.00×, so the machine is not the cause. On today's code set 4 costs
  1.95× plain over these 25 cuts (2.06 per sample per pixel, median), against 3.50×
  from the stored runs. *Plausible*: this is also section 13's unexplained
  factor-test overrun. The cause is not examined; the materials that vary
  (section 7) and the majorant (section 11) came between the two plain runs.
- **Cost** (*verified*). Set 4 pass 5,307 s, peak traced 1,177 MB, GPU pool 240 MB
  at the end. Plain pass 2,922 s, 1,045 MB, 149 MB; the estimate of about 1,870 s
  used the old plain timings and missed the slowdown above. Judging takes seconds.
  No image saved, no photo left the PC. Rows now hold each cut's curve;
  `survey`'s `stop_after` keeps a 100-cut run's planes and can resume on to 100.
  Saved curves are rounded, so rebuilt slopes lie within 0.02 of stored (largest
  0.0127), after using the stored slope where L's rounding leaves the finest bin
  ambiguous. Scratch: `slope_range_judge.py`, `slope_range_followup.py`.
- **Nothing in the measure has changed.** Section 12's slope gap stands as
  measured. The owner decides whether to go on.

### 15. Why today's code is slower, and a register cap (the owner's choice, 2026-11-06)

Section 14 found the plain preset rendering 1.80× slower than its stored run, with
the same samples and pixels. Stored rows put it in the render stage (fill 1.05×).

- **Where** (*verified*; cut 1, 728 px, 128 spp, fixed seed, best of 3, kernel
  seconds, code of each commit). Plain takes 2.69 s before section 7 (a328f04) and
  4.85 s from section 7 on (1743581, 0b5a750, 4bd6159, cd3f798): 1.80×, the same
  as the 100-cut figure. Images are bit-identical and event counts equal at all five.
- **Why** (*verified* for the render kernel on this RTX 4070). Section 7 raised the
  kernel's peak registers from 113 to 211, read from the compiled binaries, though
  the lens code does not run for the plain stone. Removing the cold path restores
  the speed (113 registers, 2.66 s), and so does keeping the branch with a trivial
  body (113, 2.72 s). Moving it out of line does not (184, 5.05 s; set 4 +5%), nor
  does forcing `matid` inline (211, 5.08 s) or moving the spline lookup out of line
  (190, 4.83 s; set 4 +8%). Code size rose too, so the test that separates them is
  the cap below: registers fall with the code size unchanged, and the time falls.
- **The mechanism** (*plausible*). A kernel's peak register count would limit how
  many threads run at once. This is not measured directly. The one indirect sign is
  in the last-digit item below: the capped kernel's samples overlap at a wider
  spacing.
- **The cap** (`effective`; `--maxrregcount`, camera kernel only). Same cut, kernel
  seconds, plain / set 4; every image bit-identical to the uncapped one:

  | cap | none (211) | 160 | 128 | 112 | 96 | 80 | 64 |
  |---|---|---|---|---|---|---|---|
  | plain | 4.86 | 3.73 | 3.20 | 3.29 | 3.25 | 4.55 | 5.82 |
  | set 4 | 10.3 | 7.73 | 7.05 | 7.34 | 6.94 | 8.84 | 10.88 |

  **128 is chosen.** 96 to 128 is a plateau whose differences are small and point
  different ways by preset, and 96 sits next to the sharp slowdown at 80, which
  later kernel changes could reach. The best value may differ on another GPU.
- **On real cuts** (*verified*; the first 4 of each stored 25-cut pass, same seeds).
  Every stored value but the timings is identical on all 4 cuts of both presets,
  slopes included. Set 4 render time 716 s → 492 s (**0.688×**; per cut 0.684 to
  0.691); plain 380 s → 240 s (**0.631×**; 0.628 to 0.647). Peak traced 1,051 and
  931 MB, GPU pool at the end 129 and 63 MB. Plain's kernel stays about 1.2× the
  one before section 7; a second kernel without the shift would recover that but
  serves plain only, so it was not built. Section 14's set 4 / plain ratio moves to
  about 2.2 on the small cut. Stored timings from before this section do not
  compare with new ones.
- **The last digit, not new** (*verified*). The kernel adds samples into the image
  as float32 atomics, so thread order can move the last digit where a pixel's
  samples are few blocks apart (`npx` / 128). The cap exposed this:
  `test_shift_with_no_takers_renders_identically` failed on its 100 × 100 view
  (differences up to 2e-7 relative, event counts equal). The uncapped kernel does it
  too, on repeats at 100 px and at 128 spp, so that test's exact match was luck of
  timing. Over 4 repeats at 16 spp the differences end between 150 and 200 px
  capped and between 100 and 150 px uncapped; real cuts are about 3,700 blocks
  apart and matched exactly. The test keeps exact equality and moves to 400 × 400
  (1,250 blocks apart). New test: the cap leaves the lens and drift path's image
  unchanged; control, another seed differs. 466 tests pass; CI has no GPU, so it
  skips both.

## Alternatives considered

The B and C options of 16a to 16c. Also a cut-face-only shift held
constant through the slab's depth. It was rejected because bands cross the
cut at an angle, so a lens is not constant along the view ray.

## Risks

- **Clear stretches over the black bottom** may show the dark grey that
  failed for the clear layer (ADR 0015 section 6). *plausible*. The drift
  at +0.25 makes the clearest bands (about 0.875) fully clear.
- **0.1 mm lenses may blur away** under scattering in milky bands.
  *plausible*. Judge by eye.
- **The cost per sample**, from four cubic-spline lookups per sample, may
  slow renders noticeably. It is measured before any long run.
- **16b's departure from agatesim's numbers** could confuse later
  comparisons that read agatesim's `Fields`. It is stated here and in the
  code.

## Consequences

`BandRecord`, `OpticsTable`, the scene upload and the CUDA camera all
change. A new GPU test guards the copy. The look cuts in section 5 follow
the build, and statistics come only after the owner has judged the look.
