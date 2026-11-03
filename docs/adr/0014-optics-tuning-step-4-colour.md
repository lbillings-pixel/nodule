# 14. Optics tuning step 4: colour, and what darkens the stone

- Status: proposed (2026-11-03). The owner chose to record the measurements
  and read the sources before any optics change. Section 4 is filled in
  after the reading.
- Date: 2026-11-03
- Builds on: [ADR 0010](0010-optics-tuning-step-1-pores.md) (step 1: iron
  oxide tripled to 9 × 10⁻⁴; matrix absorption "only dims"),
  [ADR 0013](0013-optics-tuning-step-3-exposure.md) (step 3: exposure 2.2;
  C\* named as the next gap), [ADR 0007](0007-renderer-cut-face-monte-carlo.md)
  section 3 (the optics table) and [ADR 0002](0002-founding-nodule.md)
  (graded claims).

## Context

ADR 0013 left colour as the largest gap in the look: C\* 17.2 for the
renders against the photos' median 7.3 (p95 33.7). Step 4 measures that
gap before anything is changed. Two scratch probes ran on the four look
cuts (0, 5, 8, 18) of the n2-slice stone (record 0.7 / 0.7, γ 4).

**Probe 1** (`chroma_probe.py`, 43 s, 0.27 GB) compares the saved renders,
re-exposed by 2.2, with the 46 half-A reflected photos (centre window),
image by image.

| | photos p5 / p25 / p50 / p75 / p95 | renders (percentile among photos) |
|---|---|---|
| C\* median | 2.7 / 4.8 / 7.3 / 12.3 / 33.7 | 15.8–18.2 (87) |
| hue of median a\*, b\* (deg) | 26 / 80 / 200 / 271 / 316 | 84–87 (26) |
| corr(L\*, C\*) within the image | −0.73 / −0.58 / −0.24 / 0.18 / 0.70 | 0.19–0.56 (74–87) |
| share of pixels with C\* > 15 | 0.00 / 0.00 / 0.05 / 0.37 / 0.87 | 0.56–0.72 (87–89) |
| L\* median | 14.7 / 47.4 / 62.1 / 73.0 / 79.8 | 60.1–63.8 (41–54) |

Photos with a warm hue (20–90°, n 11) have C\* 8.5 / 12.1 / 27.3 (p25 /
p50 / p75). The renders' 16–18 lie inside that range.

**Probe 2** (`look_pigment.py` with `phi_iron_oxide` 0, then
`chroma_compare.py`; 295 s, 0.55 GB) renders the same cuts, looks and
streams with the iron oxide switched off. The stand-ins match the earlier
renders exactly. With iron off, 60–75 % of each face's pixels clip to white
at exposure 2.2, and the darkest tenth stays at 0.78–0.98 of white.

## Findings

1. **The colour is within the photos' range.** The comparison against the
   library median set one ochre stone against a library made mostly of grey
   and blue agates. Among photos of a similar hue, the stone's chroma lies
   inside the middle half. *Plausible*: only 4 cuts, and hue is a crude
   way to group photos.
2. **Iron oxide is the model's only darkening agent.** Without it the face
   is nearly white (*verified* within the model). The lightness and contrast
   matched in steps 1 and 3 therefore rest on the yellow pigment. In
   step 1 (ADR 0010) iron was tripled to raise contrast, and step 3
   brightened the image to match the photos' lightness. Grey matrix
   absorption and deeper slabs only dimmed the image in step 1. So
   lowering the iron would wash the stone out, unless something else
   darkens it.
3. **Colour sits in the bright parts.** Within a render, brighter pixels are
   more coloured (correlation +0.19 to +0.56). In most photos the darker
   pixels are (median −0.24), as pigmented bands would be. The renders
   still lie inside the photos' wide spread. *Speculative*: a lead that
   fits finding 2, not a failure.

## Decision (proposed)

1. **No optics change now.** The colour is graded *plausible*: within the
   photos' range, and carried by the same iron oxide that gives the stone its
   darkness. The 20-cut check's descriptors are measured on luminance
   (`descriptors.py`), so chroma does not enter them and the stone's
   verdict there is unchanged.
2. **Fitting one stone to the library median is rejected.** It would make a
   strongly stained stone into an average one.
3. **The open question goes to the sources:** what makes clear chalcedony
   bands look dark in reflected light? Candidates: light lost into the
   stone's depth, grey (broadband) absorbers, or iron itself. Section 4
   records the reading.

## 4. Reading and a renderer finding (2026-11-03)

Reading note: notes/n2-reading/step4-dark-bands.md (helper report, about
12 sources, unreviewed). Only catalogued sources are cited here. The
note's other leads (carbon, measured CIELAB of agate, iron by band colour)
enter only when catalogued at the decision.

**What the sources say.**
- Clear chalcedony is a weak scatterer that absorbs little. Blue and white
  bands of one agate hold about 20 ppm Fe, and a broad iron-centre band near
  570 nm takes about 0.02 in absorbance per ~1 mm [@lorenzi2022]. That
  gives a grey-ish μ_a of roughly 0.02–0.05 mm⁻¹ [inference, the note's],
  above our `matrix_mua` 0.01 and below Lorenzi's 0.13 mm⁻¹ bound.
- Carbon can blacken agate, but no source yet shows it greying ordinary
  bands. In Mali onyx the black bands hold 1.88 wt% disordered organic
  carbon in the pores of fibrous chalcedony, while the white bands hold
  0.22 wt% and stay white [@goetze2012] (Table 1, p. 120). Iron there is
  165 against ~40 ppm, as hematite, which the authors rule out as the
  darkener. The paper gives no visible-light optics. So a carbon-like grey
  absorber is a real mechanism for black and grey onyx. For ordinary
  agate it remains an assumption.
- Whether a band looks light or dark depends on what lies behind it.
  Jayaraman's colourless agate plates swap light and dark bands when
  viewed against a dark background under oblique light [@jayaraman1953],
  explained by diffusion.
- Diffusion theory [@jensen2001] with the model's own clear-band values
  (reduced scattering μ_s′ 0.09–0.19 mm⁻¹, computed from `OpticsTable`)
  gives a diffusion length of about 7–18 mm at μ_a 0.01–0.03 mm⁻¹. In a
  semi-infinite clear band, diffuse reflectance is then about 0.17–0.38, a
  grey, not a white [inference: computed here with the dipole's total
  diffuse reflectance, n 1.54; it agrees with the note's table].
  Milky bands (μ_s′ 3–13 mm⁻¹) return light within 1–3 mm and stay
  light (0.6–0.84 at the same μ_a).

**What the renderer does instead** (*verified*, `camera.cu`). The
reflected face is a slab 2 mm deep (ADR 0010 10b) with a smooth bottom
face over black air. Inside chalcedony (n ≈ 1.54), light that meets the
bottom at more than about 40° is totally reflected back up. So most
diffuse light that reaches 2 mm, about the depth where a clear band
should lose it, is sent back toward the face. A real cut nodule has more
agate below the face, with no such mirror: light that goes deep is lost to
the stone's interior and rind [inference]. The 2 mm depth was checked
against 10 mm (0.14 grey levels), but both depths had the same mirror, and
that check ran before γ 4 made the clear bands clear.

**So the likely reason the stone needs iron to be dark** is the trapped
light, not a missing absorber. *Plausible*: computed, not yet rendered.

### 4a. Choice for the owner: what to test first

Each test renders the four look cuts with iron off (about 5 minutes each,
under 1 GB), then measures L\*, contrast (Y p90/p10) and C\*. A test
changes nothing in the defaults.
- *(Recommended)* **A. A lossy bottom.** A renderer option (`effective`;
  default off, so every earlier render is unchanged) makes the bottom
  face index-matched and black, so light that goes down is lost as into
  the rest of the nodule. It is tried at depths of 2, 10 and 20 mm and
  `matrix_mua` 0.01 and 0.03. It needs a small kernel change, with tests,
  including the negative control below.
- B. Grey matrix absorption only (`matrix_mua` 0.03–0.05, bottom
  unchanged). No code change, but the mirror stays.
- C. Wait for more sources first (below).

**Negative control (for whichever is adopted).** A clear band with no
absorption over a lossless mirror bottom must render near-white. The
look comparison must then flag it as too light.

**Sources.** Götze et al. 2012 was supplied by the owner and read (above).
Rossman 1994 (*Rev. Mineral.* 29) could not be obtained. The
supplementary tables of Zhou et al. 2022 (doi 10.3390/min12060677) and
Zhang et al. 2025 (doi 10.3389/feart.2025.1548167) are optional.

**Owner's choice (2026-11-03): A.** The option is built: `Flag.BLACK_BOTTOM`
in the camera kernel, and `Rendering(bottom="black")` with "air" as the
default. Three new GPU tests are checked against exact values and the
forward slab:
- a pure absorber over a black bottom returns only the face's specular;
  its control, the air bottom, matches the exact round trip;
- a clear band matches the forward slab with a matched bottom, by
  reciprocity; its control, the air-bottom slab, is far off;
- the back light refuses a black bottom.

The forward slab gained an optional bottom index for this check.

## Alternatives considered

- **Swap part of the iron for grey matrix absorption** so that C\* falls
  while L\* holds. There is no source yet for grey absorption in chalcedony
  (`matrix_mua` is `effective`), and it would tune one stone to the
  library median. Not tried.
- **Lower the iron and raise exposure further.** Ruled out by probe 2: the
  face clips to white.
- **Group the photos by hue for the comparison.** A possible later
  criterion. It would need a negative control (for example a stone dyed a
  hue no photo has) before it could judge anything. Not built.

## Consequences

- Defaults are unchanged. The black bottom is an option (4a, above), so
  every earlier render and result stands. The probes are scratch
  (runs/stone-2026-11-02/variants/, gitignored).
- Open items carried forward: what darkens clear bands (section 4), edge
  coherence C16 (photos' 98th percentile), and L\* sd a little low (8.8
  against 9.7).
- References added: goetze2012, zhou2022, zhang2025 (owner PDFs, read in full; notes/n2-reading/goetze2012-onyx.md, step4-zhou-zhang.md).
  lorenzi2022, jayaraman1953 and jensen2001 were already catalogued, read
  in full. The step-4 note's other sources wait for the decision.

## Results: the black-bottom probe (2026-11-03)

`look_bottom.py` and `bottom_compare.py` (scratch): the four look cuts,
saved at exposure 1.0. Each setting is then given the one exposure k that
brings its median L\* to the photos' 62.1. In all, 4803 s (deep slabs cost
12–21 min a setting, against ~4 min at 2 mm); peak traced 0.70 GB. Medians
over the cuts:

| bottom, depth | `matrix_mua` | iron | k | Y p90/p10 | L\* sd | C\* |
|---|---|---|---|---|---|---|
| photos (median) | | | | 2.49 | 9.7 | 7.3 |
| air, 2 mm (control) | 0.01 | off | 0.57 | 1.72 | 5.5 | 6.3 |
| black, 2 mm | 0.01 | off | 0.71 | 2.26 | 7.5 | 6.6 |
| black, 2 mm | 0.03 | off | 0.77 | 2.32 | 7.6 | 6.4 |
| black, 10 mm | 0.01 | off | 0.49 | 1.36 | 3.4 | 5.7 |
| black, 10 mm | 0.03 | off | 0.62 | 1.53 | 4.4 | 5.8 |
| black, 20 mm | 0.01 | off | 0.49 | 1.36 | 3.3 | 5.6 |
| black, 20 mm | 0.03 | off | 0.62 | 1.53 | 4.4 | 5.8 |
| black, 10 mm | 0.01 | 9 × 10⁻⁴ | 2.20 | 2.50 | 8.9 | 17.1 |
| air, 2 mm (ADR 0013, re-exposed) | 0.01 | 9 × 10⁻⁴ | 2.2 | 2.46 | 8.8 | 17.2 |

All *verified* within the model (4 cuts):
1. **With iron, the bottom does not matter.** A 10 mm black bottom gives
   the same image as the 2 mm mirror (contrast 2.50 against 2.46, C\* 17.1
   against 17.2, same exposure). The iron absorbs light before it gets
   deep. The defaults' results stand.
2. **Without iron, contrast is set by what lies below the face.** The
   mirror at 2 mm gives 1.72, and a black plane at 2 mm gives 2.26. A stone
   that goes on below the face gives 1.36 (10 and 20 mm agree, so this is
   the deep limit). Light then spreads sideways under the face, from milky
   bands into clear ones. Tripling grey absorption to 0.03 mm⁻¹ raises this
   only to 1.53.
3. **So in a stone that continues below the face, photo-like contrast
   needs a strong absorber near the face.** In the model only the iron
   does this, and it is yellow. Grey absorption at the level the sources
   suggest so far (0.01–0.03 mm⁻¹) is several times too weak. A black plane
   at 2 mm imitates the effect, but a whole stone has no such plane. It is
   physical only for a thin slab on a dark backing.
4. **Without iron, C\* is about 6 and blue-ish** (median a\*, b\* about −1,
   −1.5; hue 224–241°). This comes from the scatterers, which return more
   blue, and from per-pixel noise in the chroma [inference].

Open for the owner: 4b in the next section.

### 4b. Choice for the owner: what darkens the stone

- *(Recommended)* **A. Measure grey absorption up to Lorenzi's bound
  first.** Lorenzi's total-transmittance loss, about 0.13 mm⁻¹, bounds the
  matrix absorption from above [@lorenzi2022]. One more probe (10 mm black
  bottom, iron off, `matrix_mua` 0.08 and 0.13, about 25 min) shows
  whether a grey absorber inside that bound can give the photos'
  contrast. If it can, iron can be lowered toward the stone's real stain,
  with grey absorption doing the darkening (a later ADR). If not, B.
- B. **Keep iron as the darkener** (the current defaults). Colour is graded
  *plausible* (within the photos' range), and the darkening mechanism is
  recorded as an open assumption. Move on to C16 and the 100-cut run.
- C. **Treat some photos as slabs on a dark backing**, where a black plane
  at the slab's thickness is physical. This needs the owner to say which
  reflected photos are thin slabs.

**Owner's choice (2026-11-03): A**, after reading Zhou et al. 2022 and
Zhang et al. 2025 in full (notes/n2-reading/step4-zhou-zhang.md).

### 4c. What Zhou and Zhang add, and the updated plan

- **A measured colour-against-iron line.** Zhou measured 24 Xunke agates
  as 8 × 8 × 5 mm blocks on a near-white Munsell N9.5 chip (integrating
  sphere, specular excluded). Lightness falls with iron:
  L\* = 76.04 − 487.4 w, where w is Fe₂O₃ in wt%, over 0.035–0.118
  (R² 0.83) [@zhou2022] (Fig. 3a; checked on the page image). The
  sample with the least iron, a "nearly colourless control", reads only
  about L\* 64. Extending the line to no iron gives L\* 76 [inference]. So
  even an iron-free 5 mm slab over white returns only about half the light.
  That is a direct measure of grey loss in agate, independent of the
  photos.
- **Real iron-darkened agates are far more chromatic than ours.** Yellow
  samples at L\* 60–64 have C\* 34–47 (read off Zhou's Fig. 10 swatches,
  approx.). The photos' median of 7.3 lies below even Zhou's colourless
  control (C\* ≈ 12). So most photographed stones are probably not
  darkened by iron-oxide pigment [inference].
- **Our iron is high.** φ 9 × 10⁻⁴ as hematite is about 0.18 wt% Fe₂O₃,
  about 1270 ppm Fe [inference: densities assumed]. That is above Zhou's
  darkest red (0.118 wt%). By laser-ablation spot analysis, white and
  translucent zones hold ~70–400 ppm Fe, and fibrous chalcedony ~130 ppm
  (medians read off figures, approx.) [@zhang2025].

**Updated plan for A:**
1. **A1, as chosen:** a deep stone (10 mm, black bottom), iron off,
   `matrix_mua` 0.08 and 0.13 (about 25 min). Does a grey absorber within
   Lorenzi's bound give the photos' contrast?
2. **A2 (recommended next): calibrate against Zhou's set-up.** A 5 mm slab
   on a white backing under the dome needs one more bottom option, "white":
   a diffuse reflector of the chip's reflectance (to be sourced). Without
   iron the slab should read about L\* 76, and at the control's iron
   (0.035 wt%, φ about 2 × 10⁻⁴ as goethite [inference]) about 64. Two
   measured numbers then fix the grey absorption without the photos.
   Negative control: the same slab with no absorption must read clearly
   above 76, and the check must reject it. *Speculative* until built: the
   Beihong blocks' scattering is not measured.
3. **If the grey absorber works,** iron is reset from measured ppm by band
   (Zhang, Möckel) through a stated conversion. That turns
   `phi_iron_oxide` from `effective` towards `physical`, in a later ADR.
4. **Supplements:** Zhou's Table S2 (Fe₂O₃ per sample) would give the
   exact iron-to-colour pairs. It is worth fetching. The others are mostly
   recovered from figures or are of low value here. (Fetched and read:
   see 4d.)

### 4d. Zhou's supplement (read 2026-11-03)

The owner fetched Zhou's supplement (Tables S1–S5, per-sample colour, XRF,
moganite, spectral troughs and heating) [@zhou2022]. Refitting S1 lightness
against S2 iron reproduces Fig. 3a exactly. Three findings change A2:

- **Same iron, same lightness, different colour.** The colourless control
  (BH-F-1: L\* 63.78, C\* 12.29) and a strong yellow (BH-F-4: L\* 64.17,
  C\* 46.63) both hold 0.035 wt% Fe₂O₃ (*verified*, S1–S2). Total iron
  sets neither chroma nor, at this level, lightness. So the control's
  L\* 64 is most likely grey loss from the slab itself, not iron
  [inference; *plausible*].
- **"L\* 76 with no iron" is weaker than it looked.** It extends the line
  0.035 wt% below the lowest measured sample. Five samples sit at
  0.035–0.041, perhaps the XRF floor (no detection limit is given). The
  two lightest samples lie about 5 L\* above the line. Treat 76 as a loose
  upper bound, not a target [*speculative*].
- **Heating lightens the control** (L\* 63.78 → 70.63 at 310 °C), while
  every iron-bearing sample darkens. The slab's own loss changes with
  heating, without pigment [inference; *speculative*].

**Effect on the plan.** A2 now has one measured target, not two: a 5 mm
slab with no iron pigment on the white chip should read about L\* 64,
C\* about 12. The negative control stands: a slab with no absorption must
read clearly lighter, and the check must reject it. A1 is unchanged.

### 4e. A1 result (2026-11-03)

Deep stone (10 mm, black bottom), iron off, four cuts per setting; exposure
fitted so the median L\* is the photos' 62.1. Time 1076 s for both
settings (59–344 s a cut); peak traced Python memory 549 MB (GPU memory
not traced). Earlier probe rows are repeated for comparison.

| `matrix_mua` (mm⁻¹) | Y p90/p10 | L\* sd | C\* | corr(L\*, C\*) |
|---|---|---|---|---|
| 0.01 | 1.36 | 3.35 | 5.66 | −0.06 |
| 0.03 | 1.53 | 4.37 | 5.84 | −0.03 |
| 0.08 | 1.87 | 5.98 | 5.58 | +0.04 |
| 0.13 (Lorenzi's bound) | 2.12 | 6.86 | 5.09 | +0.12 |
| 0.01, iron 9 × 10⁻⁴ | 2.50 | 8.85 | 17.06 | +0.43 |
| photos | 2.49 | 9.7 | 7.3 | −0.24 |

*Verified* (as a property of this model): contrast and lightness spread
rise steadily with grey absorption, and chroma stays near the photos'.
Even at Lorenzi's upper bound, grey absorption alone reaches 2.12 of the
photos' 2.49 contrast and 6.9 of their 9.7 L\* spread. The L\*–C\*
correlation moves the wrong way: it turns positive, while the photos'
is negative. So a grey absorber within the bound is part of the
answer at most, not all of it [inference; *plausible*]. These are model
results, not evidence about how real stones form (ADR 0002).

### 4f. Probe B: grey bound plus measured-level iron (2026-11-03)

Same set-up as 4e, `matrix_mua` 0.13 with `phi_iron_oxide` 2 and
4 × 10⁻⁴: about 280 and 560 ppm Fe as hematite [inference: densities
assumed, as in 4c]. That spans the upper end of clear-band Fe through the
coloured-layer medians [@zhang2025]. Time 723 s; peak traced Python
memory 549 MB.

| `phi_iron_oxide` | Y p90/p10 | L\* sd | C\* | corr(L\*, C\*) |
|---|---|---|---|---|
| 0 | 2.12 | 6.86 | 5.09 | +0.12 |
| 2 × 10⁻⁴ | 2.39 | 7.74 | 10.76 | +0.05 |
| 4 × 10⁻⁴ | 2.46 | 8.27 | 13.10 | +0.23 |
| photos | 2.49 | 9.7 | 7.3 | −0.24 |

*Verified* (model property): measured-level iron on top of the grey
bound brings contrast to within 4% of the photos'. Colour strength
overshoots, already 10.8 at the lower iron, and the lightness spread
stays short (8.3 against 9.7). The L\*–C\* correlation stays positive,
while the photos' is negative.

In the rendered cuts the band colours of the stand-in (brown, grey,
blue) mostly merge into one warm cream. The darkest features are thin
neutral lines that look like relief at band edges. So the model's
contrast comes mainly from light returning from different depths, not
from dark, coloured bands as in the photos [inference from viewing one
cut and the correlation; *plausible*].

No setting reaches all four photo statistics. Tuning these two
parameters further against the photos would fit, not explain, so it
is not proposed.

### 4g. Per-band check: how much band colour survives the render (2026-11-03)

The 4f merging was seen by eye on one cut. This check measures it on
every saved pair (12 settings × the same four cuts, stand-in and render
on the same pixels). Each stand-in's pixels are grouped into its band
colours (k-means in CIELAB, K = 4, groups under 3% dropped, interiors
only: masks eroded 3 px). Separation is the mean distance between group
means, split into lightness (ΔL\*) and colour (Δa\*b\*). Retention is
the render's separation on the same pixels divided by the stand-in's.
Exposure is fitted per setting as in 4e. The renders sit on the stand-in
grid: cross-correlation peaks at 0–1 px shift on three cuts and 4 px on
the fourth (r 0.37–0.61), with no flip. Time 178 s, almost all in an
unoptimised k-means; no rendering.

Controls (stand-in against an altered copy of itself): identity gives
1.00 / 1.00; colour removed gives 1.00 / 0.00; a Gaussian blur of
σ 30 px gives 0.11 / 0.27. The check responds as intended to each.

| setting | retained ΔL\* | retained Δa\*b\* | render ΔL\* / Δa\*b\* |
|---|---|---|---|
| stand-in (all settings) | 1 | 1 | 18.4 / 19.2 |
| 10 mm, `matrix_mua` 0.01, iron off | 0.16 | 0.02 | 2.9 / 0.5 |
| 20 mm, 0.01, iron off | 0.16 | 0.03 | 2.8 / 0.5 |
| 2 mm, 0.01, iron off | 0.40 | 0.03 | 7.2 / 0.7 |
| 10 mm, 0.13, iron off | 0.42 | 0.06 | 7.5 / 1.2 |
| 10 mm, 0.13, iron 2 × 10⁻⁴ | 0.47 | 0.15 | 8.4 / 2.8 |
| 10 mm, 0.13, iron 4 × 10⁻⁴ | 0.50 | 0.18 | 9.0 / 3.5 |
| 10 mm, 0.01, iron 9 × 10⁻⁴ | 0.54 | 0.21 | 9.6 / 4.1 |

One cut in detail (10 mm, 0.13, iron 2 × 10⁻⁴). The stand-in's bands
are a dark bluish grey (L\* 52, C\* 3), an ochre band (L\* 65, C\* 29,
hue 75°) and a pale band (L\* 78, C\* 5). In the render they read L\* 48,
67 and 69, with C\* 10, 11 and 7, all hues 95–114° (yellow).

*Verified* (as a property of this model and these cuts):
- The render keeps at most about half the bands' lightness differences
  and at most a fifth of their colour differences. Every setting merges
  colour more than the σ 30 px blur does.
- Without iron, almost no band colour survives (≤ 0.06). In the detailed cut,
  the bluish-grey band gets no hue of its own and comes out
  yellow-tinted.
- Lightness retention roughly triples (0.16 → 0.40–0.42) either with a
  thinner slab (2 mm) or with stronger grey absorption (0.13). Both
  shorten the light's path in the stone.

So the merging comes from light transport: light that returns to the
face has crossed several bands below it, and those bands tilt under the
cut [inference: thinner slabs and shorter paths both reduce it;
*plausible*]. The renderer gives each band its own pigment
(`OpticsTable.band`), so the bands do not share one material.
The ochre band's colour is diluted, and a yellow cast spreads into the
neutral bands. That fits 4f's positive L\*–C\* correlation
[inference; *plausible*].

Not yet known: how strongly the photos' own band colours separate. The
photos have no band labels, so the same grouping would have to be run on
the photos and the renders alike (each clustered on its own pixels).
These are model results, not evidence about how real stones form
(ADR 0002).

### 4h. Band-colour separation in the photos (2026-11-03)

The same measure on photos, stand-ins and renders, each grouped on its
own pixels: light Gaussian smoothing (σ 2 px) so that groups are
spatially coherent, then k-means in CIELAB (K = 4, groups under 3%
dropped). Separation is as in 4g. Photos: the half-A reflected set,
centre window as in 4e, raw sRGB; 46 photos have a usable window.
Renders: the four saved cuts per setting, exposure fitted as in 4e.
Control: each image pixel-shuffled before smoothing (same colours, no
bands). Time 132 s; peak traced Python memory 280 MB.

| image set | ΔL\* median [IQR] | Δa\*b\* median [IQR] | shuffled ΔL\* / Δa\*b\* |
|---|---|---|---|
| photos (46) | 14.4 [9.6–30.3] | 6.3 [3.3–12.2] | 2.2 / 0.7 |
| stand-in (4 cuts) | 12.4 [11.3–13.4] | 15.6 [15.1–16.0] | 2.1 / 2.2 |
| 10 mm, `matrix_mua` 0.01, iron off | 4.8 [3.5–7.7] | 0.6 [0.3–1.3] | 0.5 / 1.3 |
| 2 mm, 0.01, iron off | 12.2 [10.5–13.5] | 0.5 [0.5–0.7] | 1.3 / 1.1 |
| 10 mm, 0.13, iron off | 10.2 [9.0–11.4] | 1.3 [1.2–1.4] | 1.3 / 0.8 |
| 10 mm, 0.13, iron 2 × 10⁻⁴ | 11.1 [9.9–12.3] | 2.8 [2.1–3.5] | 1.6 / 0.6 |
| 10 mm, 0.13, iron 4 × 10⁻⁴ | 12.4 [11.1–13.4] | 3.8 [3.4–4.1] | 1.8 / 0.7 |
| 10 mm, 0.01, iron 9 × 10⁻⁴ (default) | 13.8 [12.8–14.6] | 4.8 [4.7–5.2] | 1.9 / 1.0 |

*Verified* (measurement; 46 photos, one specimen, one seed, four cuts):
- The control works: shuffled images score at most 2.2 / 2.2, well
  under the photos' and renders' band values. The deep iron-free render
  (4.8 / 0.6) falls below the photos' lower quartile on both, so the
  measure rejects it as an obvious fake.
- The stand-in separates band colours far more than most photos do
  (Δa\*b\* 15.6 against a photo median of 6.3 and upper quartile 12.2).
  4g's retention ratios therefore overstate the gap to the photos: they
  were measured against a stand-in that is more colourful than the photos.
- Against the photos, the renders with iron sit inside the photos'
  interquartile range for lightness (11.1–13.8 against 9.6–30.3) and
  near its lower edge for colour (2.8–4.8 against 3.3–12.2, median 6.3).

Read with 4f, the gap is more specific than "too little colour". The
renders carry more overall colour than the photos (median C\* 10.8–17
against 7.3) but smaller colour differences between bands. The photos
are less coloured overall, and their colour sits in some bands and not
others. In the renders a warm cast covers every band. That fits the
L\*–C\* correlation having the wrong sign
[inference; *plausible*]. A test that would separate the causes: if the
cast comes from light crossing iron bands below the face (4g), a thin
slab with iron should show more Δa\*b\* and less overall C\* than a deep
one at the same iron.

Caveats: pixel scale differs between the photos (no scale) and the
renders, so σ 2 px smooths over different physical widths. The photos
vary widely (ΔL\* IQR 9.6–30.3), and the renders are one specimen and
seed. Model results; not evidence about formation (ADR 0002).
