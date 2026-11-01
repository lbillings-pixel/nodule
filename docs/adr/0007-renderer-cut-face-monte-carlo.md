# 7. Renderer: a cut face in reflected light, by GPU Monte Carlo

- Status: accepted (owner, 2026-11-01). Section 1 records earlier choices
  from the same day. In section 6 the owner took the recommendations for
  7a (C), 7c (0.2 mm) and 7d (majorant grid first). For 7b they chose all
  three lights now, not L1 + L2 first. Amendment ³ defers the majorant
  grid (owner).
- Date: 2026-11-01
- Builds on: [ADR 0003](0003-scope-a-3d-specimen-simulator.md) section 3
  (the renderer as its own concern), [ADR 0005](0005-representation-implicit-hybrid.md)
  section 5 ("Renderer: its own ADR … its cost is measured before it is
  chosen") and [ADR 0006](0006-band-counts-as-censored-records.md) section 10
  (band realism is judged by rendering and comparing a set of images, which
  needs this renderer). This ADR is N2 step 4 (docs/roadmap.md).

## Context

- **Why the light has to be simulated inside the stone.** Agate bands are
  about as thin as the distance light travels before it scatters, and they
  cross the cut face at an angle. Light that enters through one band leaves
  through another, so a colour lookup at the face cannot give the look
  [@donner2005; @jensen2001] (*plausible*; ADR 0005 section 5).
- **Reading.** Done and catalogued (c90c497, 62a5ebf). Method: null-collision
  (delta) tracking with weighted and spectral variants, ratio tracking for
  shadow rays [@woodcock1965; @novak2018; @novak2014; @kutz2017; @pharr2023].
  Validation targets: MCML's tables [@wang1995a]. Agate optics:
  [@lorenzi2022; @apetz2003; @graetsch1985; @white1961; @bedidi1993;
  @scheinost1998; @go2022]. Working notes are in notes/n2-reading/
  (gitignored; claims enter docs only through this ADR).
- **Only one agate attenuation measurement exists that the reading found.**
  It is a mixed blue and white slab about 1 mm thick. Blue and white bands
  have the same diffuse spectra. White bands have grains of about 5 µm,
  blue ones about 300 nm, and the authors put the difference down to grain
  scattering [@lorenzi2022] (*verified* as reported; one sample).
- **Grain scattering cannot easily make an opaque white face (the central
  problem).** Grain boundaries hardly reflect light backwards. "Only the
  pores and not the grain boundaries are able to backscatter light"
  [@apetz2003] (*verified* for alumina; *plausible* for chalcedony).
  Apetz's model, applied to quartz, makes grains of 0.5–5 µm scatter almost
  straight ahead (g ≈ 0.96–0.997). Light then has to travel roughly 7–50 mm
  before it turns around, so in reflected light a band of coarse grains
  should look clear or faintly milky, not opaque white ([inference],
  *plausible*). Coarse grains make a slab look milky in *transmitted*
  light, which is what Lorenzi measured. Small pores, under about 100 nm,
  send about half their scattered light backwards, and 0.1% of 75 nm pores
  already looks "whitish" [@apetz2003] (*plausible* as a mechanism). Whether
  agate's white bands hold such pores is *speculative*. Milky wall bands
  have a larger internal surface (BET 1.9 against 0.4–1.1 m²/g), which fits
  dry pores, but electron microscopy found no 0.1 µm pores in them
  [@graetsch1985].
- **The GPU prototype (notes/n2-proto/RESULTS.md, gitignored).** It is a
  CuPy kernel with null-collision tracking, three colour channels sharing
  one path (spectral tracking), exact Fresnel at a smooth face, and
  counter-based random numbers, so results do not depend on batch size.
  - It passes the MCML slab and half-space tests [@wang1995a], an exact
    analytic half-space result (Chandrasekhar's H function), an
    energy-conservation (furnace) test, and invariance to the majorant.
    Each check has a negative control that fails as required (*verified*).
  - Finding: MCML's Table 2 value (Rd 0.2600) *includes* the 0.04 surface
    reflection. Without it the prototype gives 0.2198, and 0.2198 + 0.04 =
    0.2598 (z −0.76). The table's text says only "total diffuse
    reflectance" ([inference], *plausible*; strongly supported by the
    numbers). renderer-methods2.md had assumed the opposite.
  - The specimen runs on the GPU. The fracture backward map and the exact
    rim and tube edges are evaluated per point, and growth time t comes
    from a cached grid. Against `Specimen.evaluate` on 400 000 points: no
    material-class mismatches. Band-position error, in band periods: p50
    0.0016, p99 0.037, max 0.21. Controls fail: no backward map gives
    p99 1.67; grid-only edges give 0.29% class errors.
  - Camera and lights: an orthographic camera looking straight at the face;
    a dome (uniform sky); a distant directional light with exact shadow
    rays through the flat face, plus its mirror image in the back face.
    Three checks pass: reciprocity, single scattering against a formula,
    and directional light summed over the sky against the dome. Their
    controls fail.
  - Cost (RTX 4070): 0.1 mm pixels, 536 × 488, 64 samples per pixel. Dome
    3.2 s; directional about 30 s (estimated; shadow rays cost about 9× the
    path). The GPU uses about 20 MB besides the grid. A preview (4× larger
    pixels, same samples per area) matches the full render within noise.
  - Two bugs were caught by the checks: the sign of the phase function in
    the shadow rays, and direction vectors drifting in float32 after about
    300 scatterings.
  - **Look (stand-in coefficients, not a result):** every feature is
    present (lobes, bands, tube linings, the fracture offset). But the
    image is milky and soft, with a glow about 2 mm into the host rock.
    Real faces are sharp. Using white bands that scatter strongly but
    forward (μ_s 20 /mm, g 0.97) spreads light sideways. This is the
    problem the first bullet describes, now seen in an image.

## Decision

### 1. Choices already made (owner, 2026-11-01)

- **First target:** a polished cut face in reflected light. Ground
  finishes come later. A backlit slab joins the first build (choice 7b).
- **One engine, two speeds:** a physical GPU Monte Carlo volume renderer.
  The preview is the same code at larger pixels and fewer samples, so it
  has the same expected image (checked: rms 1.43× single noise, 1.41×
  expected; a cut shifted 1 mm gives 6.7×).
- **A few standard lighting setups** (which ones: choice 7b).
- **A cached growth-time grid:** smooth t in restored coordinates, under
  the face, filled once from agatesim (`Fields.t_smooth`, agatesim ADR 0076
  section 6). The GPU does the band lookup, the fracture backward map and
  the exact edges. Risk: band corners rounded at the grid spacing (choice
  7c).

### 2. Light transport

- Null-collision tracking against a majorant [@woodcock1965; @novak2018],
  weighted and spectral so that three channels share one path
  [@kutz2017]. Shadow rays use ratio tracking [@novak2014]. The surface is
  a smooth dielectric with exact Fresnel. n = 1.54, `physical` (bulk
  chalcedony 1.53–1.539 [@lorenzi2022; @white1961]).
- Channels: three, at 450, 550 and 650 nm. More wavelengths can come later
  without changing the path code.
- Random numbers are keyed by (seed, pixel, sample) with Philox, so a
  preview and a full render draw the same random streams (the
  resolution-independence rule).
- Output is linear reflectance relative to a white Lambertian standard
  under the same light, converted to sRGB with a stated exposure. There is
  no artistic tone curve, so real photos can be put through the same
  conversion.

### 3. Materials (coupling to agatesim, stated as assumptions)

- Band fields from agatesim `Fields` map to optical coefficients per band
  class: absorption μ_a per channel, scattering μ_s per channel, and phase
  function. The mapping is an **assumption**, `effective`, written in one
  table in code so it can be tested and changed.
- Pigments: hematite and goethite absorption from n, k and a volume
  fraction per band [@bedidi1993; @go2022; @scheinost1998], `physical`
  form with `effective` loading. Goethite's blue edge is `effective`
  (*speculative*), because the measured k does not rise in the blue.
- White-band scattering: choice 7a.²
- Hematite n, k: the five datasets Go et al. retained [@go2022] by
  default, with Bedidi and Cervelle's lower blue k [@bedidi1993] kept as an
  option.¹

¹ Amended 2026-11-01 (owner). The table is `nodule.render.materials`
(`OpticsTable`). It reads per-step band fields recorded by
`nodule.render.record`, and refuses open cavities, hollow crystals and
dendrites until there are materials for them. The two hematite sources
differ about threefold in k at 450 nm (Go: ochre-brown bands; Bedidi:
olive-ochre). The owner chose Go as the default. Scattering is a stand-in
until 7a. In the same change the sRGB output was found to write 450 nm as
red. It now maps 450/550/650 nm to B/G/R, each taken as the matching sRGB
primary (an approximation).

² Amended 2026-11-01 (7a C built). Each band has a grain part and a pore
part, and the renderer keeps them as **two Henyey-Greenstein lobes per
channel** instead of one lobe with their mean cosine. The choice text
said "mixed per point". A merged lobe keeps the transport coefficient,
so dome-lit faces barely change (mean 1% on the n2-slice). But it spreads
the grains' nearly straight-ahead scattering (g ≈ 0.998) over a g ≈ 0.9
lobe, and it puts about 3.5 times less light straight back in single
scattering (tests/test_render_gpu.py, *verified*). That is the difference
between reflected and transmitted light which 7a is about, and L3 will
show it. Cost: two phase evaluations per real collision; dome renders
take the same time. Three channels share one path, so the kernel samples
one channel's phase function and reweights each channel by p_c / p_mix
(an unbiased one-sample mixture). A check matches one-channel runs
(|z| < 3); its control, without the weight, fails (|z| 26–130).
- *Grain part:* Apetz's virtual composite (half the volume as spheres of
  radius r, Δn = (2/3) Δn_max = 0.0061 for quartz) [@apetz2003]. It is
  evaluated with the full Mie series [@bohren1983] rather than Apetz's
  Eq. 11. For Lorenzi's 0.3 µm grains, Eq. 11 lies outside the size range
  Pabst gives as safe [@pabst2013], and it comes out 20–40% too high (computed
  here). The Mie code reproduces Eq. 11 within 1.2% where Eq. 11
  holds, the Rayleigh limit, and a direct Bessel-function evaluation
  (tests/test_mie.py). Radius runs log-linearly with agatesim's
  translucency from 0.15 µm (clear) to 2.5 µm (milky), the radii of
  Lorenzi's blue-band and white-band grains [@lorenzi2022].
- *Pore part:* air-filled spheres of radius 0.1 µm, volume fraction
  0.1% × (1 − translucency), plus 0.2% × impurity for agatesim's WHITE
  pigment. BLUE stays Rayleigh-shaped in the same lobe.
- Every radius, index and fraction is `effective` and *speculative*, to be
  tuned at the set-level comparison. Coupling both parts to translucency
  is an assumption. Under the 45° point lamp the grain lobe gives
  coloured sparkle (renders, 64 samples); the disc lamp (L2) is the
  planned fix (7b).

### 4. Checks (each with its negative control)

Kept from the prototype as tests (sections above): MCML slab and
half-space, exact H, furnace, majorant invariance, band phantom; specimen
classification against `Specimen.evaluate`; reciprocity, single
scattering, directional-against-dome; preview against full render; two-lobe
single scattering and per-channel lobes (amendment ²). Their
controls, all of which fail as required: a non-bounding majorant in
unweighted tracking; no backward map; grid-only edges; no 1/n² at the
exit; no ratio tracking; no mirror term; a cut shifted 1 mm; no phase
weight. GPU tests
run locally (CI has no GPU) and skip in CI. CI runs the parts that do not
need a GPU.

### 5. Grades

Renderer checks against analytic and published values are *verified*.
Rendered images are procedural. They are never evidence for how agate
forms or for what its bands are made of, including the pores-or-grains
question in 7a. A match to photos supports only "this parameter set looks
right" (ADR 0002).

### 6. Choices for the owner

**7a. What makes white bands white.** Recommendation: **C**.
- *A. Pores only.* Small, near-isotropic scatterers set each band's
  whiteness. Likely result: sharp, opaque white bands. Against it: TEM
  found no 0.1 µm pores [@graetsch1985]. Pore presence is *speculative*.
- *B. Grain size only* (Lorenzi's account). μ_s and the phase function
  come from crystallite size by Apetz's formula [@apetz2003;
  @lorenzi2022]. Likely result: white bands look clear-to-milky in
  reflected light, not opaque ([inference]). This fits the one
  measurement, which was made in transmitted light.
- *C. Both, as two scattering parts per band* (recommended). A grain part
  (Apetz form, crystallite radius `effective`) and a pore part (pore
  fraction `effective`). Each band class gets its own mix, and set-level
  render-and-compare against the owner's photos tunes it. The ADR claims
  nothing about which mechanism is real. It costs little extra: both parts
  are just μ_s and g, mixed per point. It also leaves room for a future
  measurement (a backlit-against-reflected pair, or a thin section) to
  favour A or B.

**7b. Standard lights.** Recommendation was L1 + L2 now, L3 later.
**Owner's choice: all three now.** L3 therefore adds a slab geometry
(finite thickness, open back, a diffuse backlight) to the first build.
- L1 *dome:* uniform diffuse light, like overcast sky or a light tent. Built
  and fast (3 s).
- L2 *one lamp* at 45° elevation, as a small disc (angular radius 2.5°,
  `effective`) instead of a point. The disc removes the sparkle that
  forward scattering gives under a point light, at about the same cost.
- L3 *backlit slab* (transmitted light). This needs a thin-slab geometry
  and an open back. It is the light where Lorenzi's data apply and where
  the owner's notes report bands swapping light and dark (ADR 0006).⁴

⁴ Amended 2026-11-01 (L2 built; owner chose the truncation). **The disc
alone does not remove the sparkle.** L2 is `Light("disc")`. Each real
collision draws one direction in the cone (next-event estimation), and a
path leaving the face inside the cone also counts (phase sampling). The
balance heuristic blends the two. Its checks are in
tests/test_render_gpu.py, each with a control:
- the two strategies agree with each other and with the blend;
- a 90° disc at the zenith equals the dome;
- a 0.25° disc equals the directional light.

On the n2-slice (64 samples), 0.1% of pixels still had noise 49× the mean
(dome: 1×). The cause is that grain lobes up to g ≈ 0.998 are far narrower
than the disc seen from inside (about 1.3–1.6°). The fix the owner chose is
**truncation**. Scattering by less than a cut angle counts as no
scattering. μ_s keeps only the share beyond the cut, and the phase function
is exact outside it (`Materials.cut_deg`). This is the rule of Buras and
Mayer's DELTA comparator at 8.1°, which was biased +0.10% on average and
up to 1.4% per pixel in reflection views, but −42% looking toward the
light [@buras2011]. Radkevich reports errors within 4% at a 3° cut for a
variant that fills the cone [@radkevich2018].

Measured here (*verified* as numbers; the cut is `effective`):
- Bias: under the dome, ≤ 0.13% on n2-slice band materials for cuts up to
  4°, and pixel by pixel within noise. Larger cuts show it: −0.3% at 10°,
  −1.6% at 20° and −4% at 30° (the control). Under L2, cuts of 0.5–4°
  agree within about 0.3%.
- Noise at the 99.9th percentile under L2, in multiples of the mean: 49
  (no cut), 13 (1°), 4.1 (2°), 1.5 (4°).
- Speed: unchanged, 1.15 M paths/s (about 3.7 min for 1 Mpx at 256
  samples).

`L2_CUT_DEG` = 4° is used with L2 only. Dome and diffuse renders are clean
without a cut and keep the full phase function. L3 looks toward its light,
which is where truncation fails in Buras and Mayer. L3 therefore needs its
own check before any cut is used with it.

Read since [@wiscombe1977; @iwabuchi2009]. Our rule scales μ_s by 1 − f
as delta-M does, but delta-M was built for fluxes. It leaves radiance
alone and excludes the aureole, where the peak's shape matters
[@wiscombe1977, p. 1420]. Iwabuchi and Suzuki fill the cone with a
constant that keeps g, and test only one cut (about 8–9°). With one cut
for every order, reflection is biased by less than 0.7%, but transmission
looking toward the sun is off by −80 to −99% at optical depth 1–5. It is
fixed (all within 0.6%) by keeping the exact phase function while a path
is still in the forward peak [@iwabuchi2009, Tables 2 and 4, p. 1936].
- L2: their reflection bias matches ours. Our empty cone does not keep
  g. The grain lobe's μ_s(1 − g) falls 3.3% at 4°, but the pore lobe
  carries almost all of the material's μ_s(1 − g), so it falls only
  0.04–0.05% (milky band material; *verified* as a calculation). The fill
  is not needed.
- L3: both papers confirm that a fixed cut is unsafe there. L3 starts
  with cut 0. If it is too noisy, try the exact-peak scheme next, checked
  against an uncut render near the light and at band edges.

**7c. Growth-time grid spacing.** Recommendation: **0.2 mm now**, with
exact values stored near creases later if render-and-compare shows
rounded band corners.
- 0.2 mm: fill 9.4 s, 18 MB. Band corners are rounded within about one
  cell where fronts from different wall lobes meet. Errors: p99 0.037 of a
  band period, max 0.21.
- 0.1 mm: about 75 s fill (over the 30 s flag), about 145 MB. Halves the
  rounding.
- Exact crease cells: stores direct evaluations only in the 0.3% of
  cells with large error. Needs a crease detector, which is work.

**7d. Speed work, in order.** Recommendation: **(1) local majorant grid,
(2) disc lamp (with 7b), (3) wavefront restructuring only if still
needed.**
- About 45% of events are null collisions against one global majorant. A
  coarse majorant grid (cells about 10× the smallest detail
  [@novak2014]) should cut that several-fold, and shadow rays most of all
  ([inference]; to be measured).
- Target: 1 Mpx, 256 samples per pixel, under 3 minutes for L2
  (*speculative* until built).³

³ Amended 2026-11-01 (owner): **the majorant grid is deferred**, and the
disc lamp comes first. The several-fold estimate above was made with the
prototype's stand-in materials (milky 20 per mm, clear 2). The 7a table
spreads less: band materials 2.0–13.4 per mm, host rock 16.6. On the
n2-slice (0.2 mm grid, top 3 mm), 47% of tentative collisions are null
(measured, 8 samples per pixel). The bound per cell, from the t range of
its grid nodes, would cut tentative collisions only 1.46× (0.4 mm cells),
1.28× (1 mm) or 1.17× (2 mm) [inference: computed from the grid, before
the fractures' 1.5 mm slip widens each cell's bound and before the cost of
walking the cells]. The 45° directional light already runs at
1.29 M paths per second, about 3.3 minutes for the target image. The grid
is built only if L2, once measured, misses the target by more than the
grid could recover.

## Alternatives considered

- **Colour lookup at the face plus a blur** (diffusion profiles
  [@jensen2001; @donner2005]). Instant, but misses light carried under the
  face from one band to the next, which is the look we need. Kept only as
  a possible instant preview, labelled non-physical.
- **Free per-band scattering (μ_s, g) with no physical form.** Simpler than
  7a C, but it gives up the link to crystallite size and pores that a
  future measurement could test.
- **CPU path tracer.** About 40 h per image, estimated. Rejected.
- **Ray marching through bands.** Biased where features are thinner than
  the step [@novak2018]. Rejected.

## Risks

- **Coefficients are effective.** No agate phase function, no measured μ_s
  for white bands, and only one slab in the literature. The look can be
  tuned, but not predicted.
- **Render-and-compare can be fooled.** A parameter set can match photos
  for the wrong reason. Hence the obvious-fake negative control
  (ADR 0006 section 10) and the grades in section 5.
- **GPU-only tests.** CI cannot catch GPU regressions. Tests run locally
  before each push that touches the renderer.
- **Float32.** Direction drift has already caused one bug. Long paths in
  milky bands need renormalisation and roulette, both in place.

## Consequences

- The prototype moves from notes/n2-proto/ into a `nodule.render` package
  with the checks above as tests. CuPy comes through agatesim's `gpu`
  extra.
- References added: none beyond those catalogued in c90c497 and 62a5ebf.
- Next after acceptance: materials from agatesim fields (section 3), the
  7a scattering model, the three 7b lights (L3 with its slab geometry),
  the 7d majorant grid (deferred, amendment ³). Then the
  set-level comparison with the owner (ADR 0006 section 10).
