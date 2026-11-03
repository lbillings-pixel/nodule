# 13. Optics tuning step 3: one exposure for reflected cut faces

- Status: accepted (owner, 2026-11-03): 13a A (one fixed exposure), as
  recommended.
- Date: 2026-11-03
- Builds on: [ADR 0007](0007-renderer-cut-face-monte-carlo.md) section 2
  (linear reflectance, sRGB at a stated exposure, no tone curve),
  [ADR 0008](0008-set-level-render-and-compare.md) (one exposure per run,
  `effective`; 7b lists exposure as tunable), [ADR 0012](0012-optics-tuning-step-2-clear-bands.md)
  (step 2; darkness left open) and [ADR 0002](0002-founding-nodule.md)
  (graded claims).

## Context

After step 2 the rendered reflected cut faces were too dark: median
lightness L\* 44 against 62 for the photos (centre windows, half A). Their
contrast was also low: L\* standard deviation 6.7 against the photos'
median 9.7. Step 2 noted that the darkness might explain part of the
contrast gap. Two causes were possible:
- **exposure**, a property of the camera, which scales all bands alike; or
- **optics**: milky bands not white enough, which changes the contrast
  between bands.

**The probe** (`exposure_probe.py`, scratch, 47 s, 253 MB) took the four
saved γ 4 look renders and scaled them by a factor k in linear light. These
are raw renders, with no photo blur or noise added. It compared them with
the 46 half-A reflected photos in the same centre window. The test that
separates the two causes is Y p90/p10, the ratio of bright to dark
luminance in linear light. Exposure cannot change this ratio unless pixels
clip.

| | L\* median | L\* sd | L\* p10 / p90 | Y p90/p10 | C\* | clipped |
|---|---|---|---|---|---|---|
| photos, median (p5–p95) | 62.1 (14.7–79.8) | 9.7 (4.4–27.8) | 43.6 / 75.4 | 2.49 (1.49–565) | 7.3 (2.7–33.7) | 0% |
| renders, k 1.0 (now) | 44.1 | 6.7 | 34.1 / 50.8 | 2.46 | 13.2 | 0% |
| k 2.0 | 59.7 | 8.5 | 47.1 / 68.1 | 2.46 | 16.7 | 0% |
| **k 2.2** | **62.1** | **8.8** | 49.1 / 70.8 | 2.46 | 17.2 | 0% |
| k 2.5 | 65.5 | 9.2 | 52.0 / 74.6 | 2.46 | 17.9 | 0% |

The renders' contrast in linear light (2.46) already equals the photos'
median (2.49). So the darkness is exposure, not missing whiteness. Most of
the L\* sd gap is the same effect: L\* compresses dark values, so a dark
image shows less spread in L\* than a bright one with the same contrast.

## Decision

### 1. The change

Reflected cut faces (L1, the dome) are converted to sRGB at exposure
**2.2** (`effective`). This brings the median L\* to 62.1. It is one value
for every run, the stone and its fakes alike, so the fakes still differ
from the stone only in their records (ADR 0008). Transmitted slabs (L3)
keep 1.0 until they are calibrated against the slab photos. Exposure is
set per light because the two lights and photo sets differ. No optical
coefficient and no part of the record changes.

### 2. Choice for the owner

**13a. How exposure is set.**
- *(Recommended, chosen)* **A. One fixed exposure.** The factor is fitted
  to the half-A photos' median L\* and checked against half B by the
  20-cut check. ADR 0008 already allows this. Lightness per cut stays
  free, so it remains a test.
- B. Auto-exposure per cut: each cut is scaled to its paired photo's
  lightness, as a camera would do. This is closer to how the photos were
  taken, but lightness then matches by construction and stops being a
  test. It would also change ADR 0008.
- C. Neither yet; colour first. Brightening raises C\* (see Risks).

### 3. Labels and stated assumption

- Exposure is `effective`. It stands for the cameras' exposure and any
  tone curve the cameras applied, both unknown per photo. It is not a
  property of the stone.
- **Assumption:** the photos' sRGB values are close enough to linear
  reflectance after decoding that their Y p90/p10 can be compared with the
  renders'. Phone and camera JPEGs usually carry a tone curve
  [inference], which would raise the photos' ratio. If so, the renders'
  true contrast is somewhat below the photos', not equal.

### 4. Grades

- The probe numbers are *verified* within the model (4 cuts, 46 photos).
- That the darkness comes from exposure rather than optics is
  *plausible*. It rests on one ratio, the photos' spread in that ratio is
  very wide (1.49–565), and the tone-curve assumption above applies.
- That exposure 2.2 brings the stone's set closer to the photos is
  *plausible* until the 20-cut check on half A and B confirms it.
- Nothing here is evidence about what makes agate bands milky (ADR 0002).

## Alternatives considered

- **Whiter milky bands** (more pore scattering on the milky side). This
  would raise the linear contrast above the photos' median, and the probe
  found no gap there to close. Not tried.
- **A tone curve** (a camera-like S-curve). It would add an unknown
  per-camera shape as a second `effective` knob, and ADR 0007 chose no
  tone curve. Kept as a fallback if the L\* sd gap (8.8 against 9.7)
  matters after the check.
- B and C above.

## Risks

- **Colour gets stronger.** C\* rises from 13.2 to 17.2, against the
  photos' median 7.3 (p95 33.7). Colour becomes the largest gap in the
  look, and it is the candidate for step 4 (pigment and iron oxide).
- **Render time may change.** Noise is measured in grey levels, so
  brighter cuts may need more samples. The check reports the time.
- **Descriptors may shift.** Brightness-dependent steps (the quality
  gate, band finding) may keep different cuts. The check watches every
  block.
- **Tuned on half-A looks and photos.** Half B is held out for the check.
- **GPU-only.** CI cannot repeat the check.

## Consequences

- `cutface.REFLECTED_EXPOSURE = 2.2`. It is the default exposure of
  `Rendering.reflected`, and `Rendering.transmitted` keeps 1.0. The
  command line's `--exposure` falls back to the light's default when not
  given. A test pins both values and checks that an explicit exposure
  still overrides them (exposure 1.0 reproduces the step-2 renders).
- ADR 0007 section 2 and ADR 0008 point here.
- Then the 20-cut check (0.7 / 0.7, half A and B), about 27 minutes and
  1.05 GB, reporting every block, C16 and C15. Lightness and C\* come
  from the probe, since the survey keeps no images.
- References added: none.
