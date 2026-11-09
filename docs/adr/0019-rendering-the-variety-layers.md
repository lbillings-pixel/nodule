# 19. Rendering the variety layers

- Status: accepted (owner, 2026-11-09, after the decisions of section 6; built by
  Claude overnight with the owner's leave to choose the night's work). Option A2 of
  section 5 (air with refraction) stays open.
- Date: 2026-11-09
- Builds on: [ADR 0007](0007-renderer-cut-face-monte-carlo.md) (the GPU renderer and its
  growth-time grid), [ADR 0014](0014-optics-tuning-step-4-colour.md) (optics defaults, the iron
  floor), [ADR 0016](0016-renderer-materials-that-vary-with-place.md) (materials that
  vary along bands; the baked drift), [ADR 0018](0018-the-variety-gap-broad-mock-ups.md)
  (the variety layers).

## Context

ADR 0018 built its features as quick layers on agatesim's fields and judged them on
stand-in previews. Before any render (HANDOFF's warning), a check of the renderer
found it could show none of the variety stones:

1. **Outlines.** The GPU scene tested only sphere unions exactly; every variety stone
   is a rotated sphere union or a thunder-egg prism, so `pack` refused them all.
2. **Layers.** The grid held agatesim's growth time and the materials came from a
   per-step record, so mottle, rim and crack stains, crack fills, dust, quartz kept low
   in iron and white bands thinning inward never reached a render.
3. **Band roughness.** The grid read the growth model unwarped.
4. **Open cavities** (quartz-lined centres, a third of each sheet): refused, as the
   renderer has no void material.
5. **The stone record** was read through `Specimen.evaluate`, which now includes the
   layers; the record's own check refuses fields that vary within a growth step.

## Decision (built; proposed)

1. **Layers in two parts** (`nodule.specimen.Layers`): `per_step` changes what depends
   on the growth step alone (white fade by the step's own time `t_dep`, quartz iron);
   the call adds what varies from place to place. `Specimen.step_fields` /
   `restored_step_fields` give the first; `evaluate` gives both, as before.
2. **The record and the grid** read `step_fields` (through any warp), so each growth
   step keeps one material and band roughness reaches the render.
3. **Any outline:** for solids other than sphere unions the grid fill also bakes the
   solid's signed `level` at its nodes; the GPU reads it trilinearly for the inside
   test (`Scene.level`). Sphere unions keep the exact test (the old path, unchanged).
4. **Layers as absorption** (`nodule.render.layerbake`): the place-to-place part is
   baked, in the same pass as the grid fill, as the extra absorption (mm^-1, three
   channels) it gives at each node; the camera adds it to each sample's own and the
   majorant grows by its largest value (`Scene.absorb`). Absorption, not impurity, is
   averaged: nodes the layers touch are sampled at 2 x 2 x 2 sub-points with the
   layers at the sub-point spacing, because the optics' iron floor (ADR 0014 4l) would
   swallow a thin dust line whose impurity was averaged over a cell first (test).
5. **Close-ups:** the grid covers the view padded by the slab's depth (light wanders
   sideways about that far), so a 12 mm close-up at 0.03 mm fills a 16 mm window, not
   the whole stone.

**Tests** (tests/test_layers_gpu.py), each with a negative control: the level grid's
inside test against the solid (control: no level puts every point outside); the warped
growth time on the grid (control: the unwarped model differs); absorption averaged after
the floor (control: impurity averaged first loses the line); the bake acts only where
layers act (control: a stone without place-to-place layers bakes nothing); mottle
darkens a render where it lies and nowhere else (control: the same stone unlayered); the
fused bake equals the separate pass (control: no optics, no bake). All 562 earlier
tests still pass (sphere-union scenes and their images unchanged).

**Cost** (measured, stone 11 of sheet 1, 41 mm, band roughness): whole face at 0.1 mm
grid, 2 mm slab: fill + bake 48 s (88 s before the bake joined the fill), render
~2 s per 64 spp; 12 mm close-up at 0.03 mm: fill + bake 224 s (410 s before). Band
roughness costs about 7 x per node (its frame reads growth time 7 times).

## Limits

- **Scattering is not baked.** A clear-filled crack trace loses its iron but keeps its
  band's haze; translucency changes from place to place are not rendered (white fade,
  per step, is).
- **A pigment change** takes the new pigment's absorption at the point, at full weight.
- **The level grid** places the wall within a fraction of a grid cell (test: exact
  beyond 0.3 cells from the wall).

## 5. Options for the owner

**A. Open cavities** (quartz-lined stones cannot render).
A1: a dark absorbing stand-in material for the void, as the stand-in preview does
(cheap; the cavity reads as a black hole, which is how a lined geode's face often
photographs). A2: true air with refraction at the cavity's lining (right, but the
renderer's single-interface design needs a second interface: days of work).
*Recommended: A1 first, A2 only if the look asks for it.*

**B. Red stones render pale** (measured 2026-11-09, runs/variety-render-phi/). The
optics' hematite loading (`phi_iron_oxide` 4e-4, ADR 0014 4f) and iron floor (0.28, 4l)
were set on the n2-slice stone against its photos. Red variety stones render grey-brown:
mean face colour (sRGB, red minus blue) 15-19 against a deep red stand-in (stone 10,
carnelian: render 93/84/79, stand-in 135/58/45). The cause predates this ADR: stone 11
without layers or warp renders the same (122/113/102). **Tripling the loading darkens
without reddening** (red minus blue falls, 15 to 7 on stone 10), so more pigment is not
the fix. **Diagnosis** (same night, runs/variety-render-phi/): a homogeneous slab of the
band's own optics (hematite, impurity 0.4, 2 mm, black bottom) reflects red, R/B 1.7 to
3.7 at every translucency, so the material is red. Two things take the red away in the
render: (i) **the face's glare**: the dome adds its mirror reflection, F0 = 4.5 % white
(n 1.54), over body reflectance of only 1-3 % in translucent bands; removing it at the
same mean lightness raises red minus blue from 15 to 44 (stone 11) and 14 to 50 (stone
10). What is left reads dark brown-orange rather than red; (ii) **the bottom** matters
little: an air bottom instead of black, glare removed, raises red minus blue only from
28 to 34 (stone 11) and 34 to 37 (stone 10), so the rest of the gap to the stand-in's red
lies in the pigment's hue and darkness at this loading (untested). The glare comes from
the lighting chosen for the photo compare (ADR 0008 7c, ADR 0014), not from the pigment.

B1: keep (ADR 0014 is closed; the photo compare was matched under this lighting). B2
(dropped by the measurement): a stronger loading. B3: a **look preset** for the owner's
eye only, leaving the compare's settings alone: a light without the face's glare (as a
photographer's polariser or dark surround gives), shown beside the current render; then,
if red stones still read brown, a look at the pigment's hue (a separate question). B4: reopen ADR 0014's lighting against the
library's red-agate photos. *Recommended: B3*, cheap (a render setting, no new
physics), and it keeps the closed calibration intact.

## 6. The owner's decisions (2026-11-09)

The owner found every render washed out, not only the red stones: structure built in
the model no longer showed. A look sheet (runs/look-sheet/: four stones and a close-up;
as now, no face glare, no glare plus a display curve, a backlit 3 mm slice) was shown.

1. **B3 accepted: the owner's look.** Renders for the eye leave out the face's mirror
   reflection (`camera.render(glare=False)`, as a polariser or dark surround gives) and
   are exposed so the face's mean lightness equals the photo compare's
   (`Rendering.look`, `cutface.look_exposure`). The compare's renderings are unchanged.
2. **Backlit slices as a second view** (the owner's idea): a 3 mm slice (owner's
   choice of 3-4 mm) on a light box (L3, ADR 0007 7b, the existing back light), at one
   fixed exposure (3.0, `effective`) for every stone so slices differ in how much light
   they pass. Rendered on the look's grid fill (`CutRenderer.on`): no second fill.
3. **The display curve is dropped** (it warmed the host to orange more than it helped).
4. **Carnelian was too opaque** (owner's judgement; no photo test wanted). Two sweeps on
   stone 10 (runs/carnelian-sweep/): less pigment alone turns the slice tan and the look
   olive-grey, not lighter red; lowering hematite's red absorption as well gives
   carnelian red-orange (median red transmission of the slice 2.3 % -> 12 %). Owner's pick:
   **hematite pigment k(650 nm) = 0.05 (source 0.2) and carnelian loading x 0.5**
   (`varietyrender.optics_for`; `OpticsTable.hematite_red_k`), both `effective`,
   *speculative*. Reason (*plausible*, untested): one wavelength per channel puts the red
   channel beside hematite's 682 nm band (Scheinost Table 2, notes/n2-reading/
   agate-optics2.md); real red light passes mostly through the window between hematite's
   bands, so the source's k overstates the red absorption seen through thick stone. Only
   the HEMATITE pigment takes it (carnelian, one iron palette); the IRON pigment of the
   photo compare keeps the source's k, so ADR 0014's calibration stands. A later, fuller
   fix: several wavelengths within each channel, for every pigment, then recheck
   ADR 0014.
5. **A grey bottom for the look** (owner's pick after runs/grey-bottom/, 2026-11-09).
   Over the black bottom, clear quartz centres rendered black (seed 1, stones 1 and 6):
   light passing through them was lost. The look's slab (2 mm) now ends in an
   index-matched Lambertian reflector of albedo **0.15**, neutral (`Rendering.look`,
   `cutface.LOOK_BOTTOM_ALBEDO`; kernel: a nonzero `bottom_albedo` with
   `Flag.BLACK_BOTTOM`, dome only, as no next-event estimation reaches the bottom),
   `effective`: a stand-in for whatever lies below the face, not a measured rock.
   Trial: black, grey 0.15, grey 0.35 and a warm rock tint (0.33, 0.29, 0.24) on stones
   1, 5, 6 and 7. Any grey lit the quartz centres; at 0.35 the lightness matching washed
   out fine bands (stone 5) and clear zones (stone 1); 0.15 kept them. The warm tint
   looked like the grey; iron-stained stone 7 barely changed. Look renders take ~1.5x
   longer (paths return from the bottom). The photo compare keeps the black bottom
   (ADR 0014). Tests: the matte bottom matches a closed form for a pure absorber (with
   internal reflections at the face), per channel; the black bottom fails the same check
   (negative control); refusals without the black-bottom flag or the dome.
6. **A highlight roll-off for backlit slices** (owner's pick after runs/backlit-rolloff/,
   2026-11-09). At the fixed exposure 3.0, clear neutral slices blew out: 45 % of stone
   1's face and 15 % of stone 8's at full white. The backlit view now passes values
   through a shoulder (`camera.rolloff`, `varietyrender.BACKLIT_KNEE`): unchanged up to
   a **knee of 0.6**, then compressed so the bare light box (transmittance 1) is exactly
   white, slope 1 at the knee; `effective`, a display choice, not optics. The exposure
   stays one value for every stone. Carnelian's median colour is unchanged (stone 10),
   only its brightest glow softens; stone 1's near-white share (all channels >= 245)
   falls from 40 % to 6 %. Clear quartz itself stays pale and flat: it passes nearly
   all the light, so there is no hidden detail to recover. Bright bands get their
   shading back, though bands passing 40-90 % of the light span only grey levels
   239-254. Trial: knee 0.6 and 0.8 per channel, and 0.4 per channel or on the brightest
   channel (keeps hue); 0.4 greyed the whites. Only the variety backlit view uses it;
   the photo compare's L3 renders and the look are unchanged. Tests: below the knee
   unchanged, light box white, smooth and rising; bright bands stay distinct and not
   white, while plain clipping fails the same check (negative control); refused for
   the dome and for exposures that cannot exceed white.

7. **Open cavities, option A1** (owner's order of work, 2026-11-09; look sheet
   runs/void-look/, owner: "cavity looks right"). A void point (agatesim phase VOID, no growth time: an open cavity or a
   hollow crystal) fills no growth step: the record leaves it out
   (`record.void_points`) instead of refusing the stone; any other unfilled point is
   still refused. The grid fill marks void nodes (`GrowthGrid.void`) and gives each the
   growth time of its nearest filled node (`tgrid.fill_void`), the cavity's lining, so
   the camera reads the lining's material there; a stand-in absorption is added at the
   void nodes to the layers' absorption grid (`scene.with_void`), every channel alike,
   placed within a grid cell (trilinear). `Rendering.void_mua`, `effective`:
   **10 mm^-1 under the dome** (`cutface.VOID_MUA`; over the clear quartz lining, mus
   0.05, an albedo under 1 %: the cavity reads dark, as the stand-in preview and a
   lined geode's face often photograph), **0 under the back light** (the slice's hole
   passes the light box, through the lining's clear material). Not air: no refraction
   at the lining (A2). The void's absorption joins the majorant (bands' largest mu_t
   ~16-28 mm^-1), so stones with a cavity render somewhat slower. The look's lightness
   and the noise are measured over the face without the void, so a dark cavity does
   not brighten the rest. The backlit view on the look's fill swaps only the absorption
   grid (one more scene upload). `varietyrender` no longer skips quartz-lined stones.
   Finding: on sheet 1 only stone 2's cut shows a clear cavity; stones 3, 4 and 9 are
   quartz-lined by recipe but their cuts show a tiny void or none. The owner put why
   low on the wait list.
   Tests (tests/test_void.py), each with a negative control: the void fill takes the
   lining's t (the edge fill alone leaves a cavity's middle at the first band); the
   absorption lands on void nodes only (none without a void or a stand-in); sheet 1's
   stone 2 fills its void with quartz-lining steps (control: its cut holds a void); its
   look renders the cavity dark (luminance 0.00 against 0.40 around it) and its slice
   bright (0.80 against 0.50), while the look without the stand-in shows it lit (0.33,
   the control); the backlit view on the look's fill equals one on its own fill, sample
   for sample. tests/test_materials.py: a void is left out, a non-void hole refused.

Tests (tests/test_look.py), each with a negative control: glare off removes exactly
F0 Ld; the look's exposure matches the compare's lightness; one fill serves a backlit
slice sample for sample; the red setting touches only the HEMATITE pigment's red;
carnelian alone takes the lighter loading.

## Consequences

- Variety stones render with every ADR 0018 layer; open cavities with a stand-in (s.6.7).
- `Layers` implementations must split their work (`per_step` and the call).
- The renderer's record now holds white fade and quartz iron; earlier stones (no
  layers) are unchanged.
