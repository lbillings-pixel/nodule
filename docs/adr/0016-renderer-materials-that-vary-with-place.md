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
