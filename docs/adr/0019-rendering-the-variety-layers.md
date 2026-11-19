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

8. **Host rind and specks on the look, option B** (owner's pick, 2026-11-17; probe
   runs/rind-render-probe/, look sheet rind-render-sheet.png). The renderer has one
   host material (`OpticsTable.host`, a tan scatterer) for every stone, and draws
   neither rind nor specks (ADR 0018 s.16). Options probed on four existing renders
   (outlines matched the saved previews to the pixel, tubes stripped as the renders
   predate them):
   - A, the render's host kept, rind and specks at one scalar gain per stone (render
     host lightness / sheet host lightness): the gain was 3.4-7.2 on volcanic stones
     (render host sRGB ~140-200 against the sheet's basalt 68), so rinds glowed and
     basalt stayed tan. Rejected.
   - **B, the host redrawn** (`sheet.dressed_look`): host, specks and rind in the
     sheets' colours (`sheet.dress`, shared with `framed_rgb`), one gain for every
     stone (the sheet's values as displayed, `effective`), times the render's own
     grain there (pixel / the host's median); the rind's share of each edge pixel
     blended in linear light; agate clear of the rind kept as rendered. CPU, about
     2 s a stone. *Assumption*: no light passes between agate and rind (the render's
     own bleed into its host is kept only as grain).
   - Only on the look; never on the backlit slice (owner): its host is opaque black,
     so the slice is unchanged.
   - Room for the rind: `varietyrender` pads the cut by the rind's reach
     (`Specimen.cut(pad_mm=)`), so the frame no longer clips it (it did on seed 1
     stone 6's 3 mm shell). Sheets are unchanged (`framed_rgb` pads its own picture).
   Test (tests/test_rind.py): on a flat stand-in render, host pixels equal the sheet
   picture's to rounding, not the render's tan, and agate clear of the rind is kept.
9. **Tinted quartz centres and white needles** (work plan item 11, 2026-11-17; built,
   the owner's eye pending; look sheet from `runs/tint-probe/` and
   `runs/needle-grid-probe/`, local).
   - Cause of the grey amethyst: nothing in the renderer read the recipe's tint.
     `OpticsTable.quartz_tint_mua` now adds colour-centre absorption to quartz laid
     at or after `quartz_tint_from` (the first step after the early stage, so clear
     early crystals stay clear), its own material key so other quartz keeps its row.
     `varietyrender.QUARTZ_TINT_MUA` (`effective`, *plausible*): medium amethyst
     0.033, 0.079, 0.022 mm^-1 at 450, 550, 650 nm (a 6.2 mm Anahi slab, o-ray, from
     Rossman's Caltech spectra [@rossman-quartz2026], not peer-reviewed; a deep zone reads ~0.075, 0.165,
     0.07 off [@hassan1974, Fig. 3]); medium smoky 0.12, 0.093, 0.072 (a Saline Valley
     spectrum, same server, read off its plot; [@marshall1955] for a moderately
     coloured stone). Pale amethyst has no source. ADR 0014's grey absorber (0.13)
     stays on quartz; with it 1x reads mauve-grey and 2x purple-grey in the 3 mm look.
     Strength x TINT_STRENGTH: 2, the owner's pick (2026-11-17, after whole-face
     renders of sheet 1 stone 6 at 1x and 2x, `runs/item11-render/`); the owner kept
     the 0.05 mm needle grid too. Colour zoning (amethyst in r-sector growth zones) is
     not drawn.
     *Read later (2026-11-17, night; `notes/n1-reading/item11-quartz-colour-papers.md`,
     local).* [@nassau1977] (owner-supplied) puts smoky colour in the A3 band at 2.90 eV
     (428 nm, FWHM 1.5 eV; *verified*). In one natural crystal it grows from about 0.6
     to 2.6 cm^-1 (decadic, E parallel to c) between a low dose and saturation (*plausible*,
     read off Fig. 5). The smoky ratio of blue to red, 1.67, matches its saturated crystal
     (about 1.74, *plausible*), and 2x (0.24 mm^-1 at 450 nm) lies between its low-dose
     and saturated cases. So the smoky values stay. The paper has no amethyst data. The
     file supplied as Lehmann was [@lehmann-yellow1971] (yellow centres, no amethyst
     spectrum). Rossman 1994 could not be obtained.
     *Lehmann 1967 read (2026-11-18; `notes/n1-reading/item11-lehmann1967.md`, local).*
     [@lehmann1967] (owner-supplied, German) puts amethyst's colour in a band at 545 nm
     (*verified*). Its Fig. 3 amethyst reads about 0.85, 2.1, 0.66 cm^-1 at 450, 550,
     650 nm (*plausible*; decadic by convention, thickness and polarisation not given),
     so 450/550 0.40 and 650/550 0.31 against the tuple's 0.42 and 0.28: the shape holds.
     That stone is about 6x the tuple (0.49 mm^-1 Napierian at 550 nm, *speculative*), a
     single very dark stone, so it bounds the strength from above (TINT_STRENGTH up to
     about 6) and does not set a medium. Amethyst stays as above.
   - Needles: see ADR 0018 s.15 (correction): white remnants enter the record as milky
     chalcedony, and white fine needles render the whole face at NEEDLE_GRID_MM
     0.05 mm, coarsened to stay within NEEDLE_VOXELS (100 M voxels; ~11 GB RAM, ~5 GB
     GPU, an estimate scaled from the stone below, not measured). Measured on sheet 0 stone 2 (40 mm): fill 168 s against 24 s, peak RAM
     3.9 GB against 1.0 GB, GPU pool 1.8 GB against 0.24 GB.

Tests (tests/test_look.py), each with a negative control: glare off removes exactly
F0 Ld; the look's exposure matches the compare's lightness; one fill serves a backlit
slice sample for sample; the red setting touches only the HEMATITE pigment's red;
carnelian alone takes the lighter loading; tinted centres take their tint, untinted
stones none; white fine needles alone take the finer grid, within the budget
(test_materials.py: early-stage quartz stays untinted; tint off changes nothing).

10. **Depth through clear bands: probe** (work plan item 16, 2026-11-18; nothing decided;
   `runs/depth-probe/`, local). The look is a 2 mm slab over the grey matte bottom
   (s.6.5), so nothing deeper than 2 mm can show. Diffusion theory with the model's own
   clear-band values gives a diffusion length of about 7-18 mm (ADR 0014, from
   [@jensen2001]; *inference*), so a clear band should show stone well below 2 mm. Probe:
   one grid filled at the deepest slab, the look rendered at 2, 5, 10 and 15 mm from it,
   and at the deepest over a black bottom; same record, optics and seeds; compared at the
   2 mm look's exposure, luminance blurred 2 px, against a noise floor from the two
   halves of the 2 mm render. Whole faces on a 0.2 mm grid (coarser than the look's
   0.1 mm, so thin layers blur), 256 spp. Results (*verified* as measurements of the
   model, *speculative* as optics):
   - **Negative control** (sheet 2 stone 10 with every chalcedony step milky, translucency
     0.08): mean change 1.2 grey levels against a floor of 1.15, 0.1 % of pixels above 5.
     The same stone as drawn: 6.0 at 5 mm, 6.5 at 10 mm, 44 % of pixels above 5.
   - **Clear stones** (sheet 0 stones 1 and 3, sheet 2 stone 10): 6-13 grey levels, 43-60 %
     of the face. The change saturates by 10 mm (15 mm adds nothing measurable), and at
     10 mm or more the bottom no longer matters (black against grey: same to 0.05).
     5 mm gives most of it (6.0 of 6.5, 6.7 of 7.2, 8.9 of 13.2). By eye: clear zones
     darken and show the walls curving away below the face (soft gradients at lobe joins
     and round centres).
   - **Quartz centres go black** by 10 mm (sheet 0 stones 1 and 11; the amethyst tint of
     stone 11 vanishes; at 5 mm both are dark but keep some tone). The model's quartz is
     nearly clear (`quartz_mus` 0.05 mm^-1, g 0.9, `effective`), so light entering a
     centre runs down and is lost; at 2 mm the grey bottom returned it, the reason it was
     chosen (s.6.5). Whether real centres scatter more (grain boundaries, fluid
     inclusions, fractures) is not read: *speculative*. On stone 11 the look's matched
     exposure rises from 5.1 to 8.8, so the whole stone re-levels (8.7 grey levels
     between matched and fixed at 10 mm).
   - **Iron stone** (sheet 0 stone 8): 2.0 grey levels, 7 % of pixels: iron absorbs before
     depth tells. Carnelian not tried.
   - **Dendrites seen through** (sheet 80 stone 9, 12 mm close-up at the face's densest
     Mn, 0.08 mm grid, 2/5/10 mm): 3.2 grey levels, 21 % of pixels; the branches look the
     same, no new ones appear from below (they sit in milky bands that hide what is under
     them); only the clear bands between darken. So depth alone does not give the
     face-on fern view of ADR 0018 s.17.
   - **Cost** (sheet 2 stone 10, 64 x 51 mm face, the look's own 0.1 mm grid): 3 mm fill
     ~12 M voxels, 5 mm 23 M (measured: fill 22 s, peak RAM 2.1 GB, GPU pool 0.5 GB,
     look render 43 s against 22 s at 2 mm, 256 spp), 10 mm ~60 M (estimated, ~5 GB RAM).
     The window pads by the fill depth, so close-ups grow fastest. White-needle stones'
     voxel budget (NEEDLE_VOXELS) is padded by the 3 mm slice and would need revisiting.
   - **The owner's pick (2026-11-18):** 5 mm, full renders first; the default changes only
     on their verdict. *Full renders* (0.1 mm grid, 512 spp, matched lightness; sheet 2
     stone 10, sheet 0 stones 1, 3, 11; `out/depth-full-2v5.png`): change at 5 mm 5.9-8.7
     grey levels on the clear stones, 14.1 on the amethyst stone (its matched exposure 5.1
     to 6.7), against floors of 0.9-1.2. Fill at 5 mm 22-305 s (rough and eye stones the
     slowest), peak RAM 2.1-2.8 GB, GPU pool 0.5-0.7 GB; the 5 mm look renders in about
     twice the 2 mm look's time.
   - **Built (2026-11-18; the owner: "yes, lighter amethyst").** The look renders
     `cutface.LOOK_DEPTH_MM` 5 mm (`effective`) over the same grey bottom; the photo
     compare keeps 2 mm (ADR 0010 10b, 0014). `varietyrender` fills one grid FILL_MM
     (max of the look and the 3 mm slice) deep for both views, and the white-needle
     budget is padded by it (those faces coarsen a little more). Tinted centres: at 5 mm a
     centre is darker whatever its tint (less light returns); the tint sets its hue.
     Calibrated on sheet 0 stone 11 (0.2 mm grid, 128 spp): the 2 mm look at 2x gave a
     centre green/red of 0.79; at 5 mm, 2x gave 0.53, 1x 0.74, 0.6x 0.83, so
     `TINT_STRENGTH` 0.8 (*effective*): 2.0 x 2 mm and 0.8 x 5 mm give the same tint
     optical depth down to the bottom, which is also the basis for scaling smoky alike.
     Measured on the pipeline's own render (0.1 mm grid, 512 spp): centre green/red 0.78,
     RGB about (69, 54, 65) against (101, 79, 96) before. Tests: the look's
     depth against the compare's 2 mm (control); one fill serves both views, sample for
     sample (control: another seed), with and without a void.
   - **Side effects, the owner's calls (2026-11-18).** (i) The one tint strength also
     lightens a backlit slice's tinted centre (sheet 1 stone 6, `out/backlit-check.png`):
     kept, 0.8x for both views. (ii) Close-ups keep the 2 mm look
     (`varietyrender.CLOSE_LOOK_MM`) on the slice's 3 mm fill: at 5 mm their grid, padded
     by the fill, would grow from ~36 M to ~90 M points (est. 8-9 GB RAM, ~2.5x the fill
     time; not measured). Whole face and close-up then differ a little in depth (test:
     each call's look and fill depth; control: they differ). (iii) White-needle faces
     coarsen within NEEDLE_VOXELS: on sheets 0-11, 5 of 6 go from 0.05 to 0.054-0.063 mm
     (sheet 0 stone 2 stays 0.05, now 82 M points against 41 M): accepted, needles may be
     a little fainter. Cost: whole faces at 5 mm, sheet 0 stone 1 (rough, eyes) took
     550 s against about 5 min before.
11. **Renderer physics, order of work** (work plan item 19; the owner's choices,
   2026-11-19). Item 19 splits into three tracks: (A) the open hollow as air, so the
   crystal wall shows (option A2 above), then an optional small-lamp view for sparkle and
   six-faced crystal tips in agatesim; (B) scattering that varies by place (today only
   absorption is baked per grid node, s.6.7 and ADR 0018's plume bake; scattering comes
   from the band material's row); (C) the fuller red fix (several wavelengths per
   channel, then recheck ADR 0014). **The owner chose A first, and wants white (ghost)
   plumes on some stones**, so B stays in item 19 (the owner's 2026-11-09 condition for a
   plume material of its own, ADR 0018). Whether real quartz centres scatter more is
   still unread (s.6.10): B reads it before any change there.
   - **No cheap stand-in for A.** Dropping the void's stand-in absorption (`void_mua` 0)
     does not show the wall: void nodes carry the lining's growth time (`tgrid.fill_void`),
     so the hollow becomes clear lining quartz with no surface, as the backlit view's
     plain white hole shows (runs/void-look/seed1/stone-02-backlit.png). The wall shows in
     photographs through reflection and refraction at the air-to-crystal surface, which
     needs the second interface. Item 18's mock-up (runs/druse-centres/glint-probe.png;
     host-side shading, appearance only) stands for the look; the next step is the
     interface's design, then a measured build.
12. **The hollow as air: design** (track A; proposed 2026-11-19, nothing built).
   *Owner's verdict (2026-11-19 night): yes to all four points:* the agatesim change
   returning the CRYSTAL-stage level is approved; the void box reaches the hollow's bottom;
   the plan, order and ~8-session cost below are accepted; the design was pushed.
   - *Where the wall is.* Quartz-lined centres come from agatesim's CRYSTAL stage
     (`agatesim/volume/growth.py`, `evaluate`): a point stays open when its crystal
     arrival `metric.assign` (mm of full-rate growth) exceeds the stage's last reach.
     The difference, arrival minus last reach, is a signed level: positive in the hollow,
     negative in the crystals, zero on the wall made of the crystal tips. agatesim returns only the
     open/filled verdict, so nodule sees a bare mask; slopes from a mask are too coarse
     for glints (ADR 0018 s.24, the 3-bisection run). **Needs an agatesim change**: return this level with
     the fields, as agatesim ADR 0089 takes a level, not a mask (approved by the owner
     2026-11-19 night). Hollow crystals (stick
     moulds) have another level source and keep the stand-in (s.6.7).
   - *The hollow is deeper than the fill.* The wide hollow probed for s.24 has its far wall
     a median 9.2 mm below the face; the look fills 5 mm and close-ups 3 mm. Proposed:
     a **void box** of its own around each hollow, reaching its bottom, at a fixed world
     spacing (so face, close-up and needle grids see the same wall: the
     resolution-independence rule). It holds the level and, in a shell of lining around
     the hollow (thickness `effective`, 1-3 mm to be measured), the material; light
     leaving the shell below the slab is lost (black, as into the rest of a stone, ADR
     0014; *assumption*).
   - *Grid spacing, measured first.* Today's tips are round cones, curved about their axes,
     and the minimum over crystals has kinks, so trilinear normals are not exact. A probe
     (runs/, no agatesim edit: it reads the stage's metric from outside) compares
     normals from the level on 0.1, 0.05 and 0.025 mm grids with the bisection reference
     on the s.24 stone. It reports the share of wall within 1.5 deg, whether 30-74 glints
     a view come back, the bowl control, and any jumps in the level where agatesim's
     candidate crystals change between lattice cells. The spacing, memory and cost
     follow from it.
   - *The camera (`cuda/camera.cu`).* Each free flight in the stone that enters the void
     box is tested against the level (half-cell steps, then bisection); at a crossing the
     path reflects or refracts by Fresnel (n 1.54 to 1, normal from the level's
     gradient). In air it runs straight to the next wall, the face or the slab bottom, with
     no scattering. Radiance bookkeeping: x n^2 leaving stone for air, / n^2 back. A
     camera ray over an open pixel starts in air (no face Fresnel, no 1/n^2), and a path
     leaving through the open mouth adds the light's radiance without n^2; the host's
     face glare F0 Ld (photo compare) is masked there, since there is no polished face.
     The stand-in absorption is off on a hollow drawn as air (it bleeds a cell into the
     lining). Shadow rays cannot bend at the wall: a shadow leg that meets the wall (direct
     and both mirror legs, tested exactly) carries nothing, and light paths through the
     hollow are counted by phase sampling alone at weight 1, which stays unbiased. So glints
     need the DISC lamp (a small lamp, the sparkle view): a DIRECTIONAL light has no
     phase-sampled route and can never glint. The dome look needs no shadow rays.
     `slab.cu` has no void path and is unchanged; stones without a hollow run the old code.
   - *Order.* (1) the probe; (2) the agatesim level (with its own ADR and tests); (3)
     the void box and its fill; (4) the camera under the dome, shown to the owner;
     (5) the small-lamp view; (6) six-faced tips (agatesim), rechecking the probe's
     numbers.
   - *Tests planned, each with a negative control.* The wall test also works under flag 4
     (homogeneous material), so analytic cases exist: a flat air layer under the face
     against Fresnel's slab result (control: no level); a tilted flat facet glinting at
     the predicted lamp angle under the disc (control: an untilted facet stays dark); a
     white furnace (no absorption, white bottom, dome: every pixel Ld, hollow or not;
     control: the n^2 step left out breaks it); with no hollow the image equals today's,
     sample for sample.
   - *Cost, estimated.* (Claude's estimates have run about 2x low, so these are already
     doubled.) Work: probe 1 session; agatesim level 1; void box 2; camera and tests 3;
     renders for the owner 1: about 8 sessions. Run time per stone with a wide hollow: fill
     +1-3 min (narrow band at the fine spacing; agatesim's fill ran ~11 us a point in
     s.24), GPU memory +0.1-0.3 GB, render 1.3-2x the look's. Stones without a hollow:
     unchanged. All to be measured at each step.
   - *Step 1, the probe: results* (2026-11-19; runs/hollow-air/, gitignored: normals_probe,
     jumps_probe, edges_probe, look_and_cost; ~15 min of runs in all, peak 3.55 GB traced
     once, in a candidate check since split into chunks, otherwise <= 1.1 GB). One stone
     and one view (sheet 4, stone 5: 11 266 crystals, grain 0.5 mm, tip half-angle 38
     deg, elongation 4); *verified* for it, untested on others.
     - *Gate.* The level read from outside (arrival - last reach) has agatesim's VOID
       sign on 100 % of points 2 um either side of the s.24 wall; every wall found is the
       crystal level; |level| on it ~3e-5 mm. s.24's depths still hold after 740c2c4.
     - *Controls.* A sphere level passes on every grid and store (100 % within 1.5 deg,
       analytic glint share); its sign-only mask fails (0.5 %, median 18 deg).
     - *Spot counts do not choose a spacing.* Every grid returns 38-96 spots a view
       (exact normals 33-79; s.24's slopes 30-74, though they match the exact normals
       on only 38 % of the wall). The share within 1.5 deg does: 0.1 mm 12-18 %, 0.05 mm
       38-58 %, 0.025 mm 72-82 % (per-cell / node gradients). int16 storage equals
       float64; int8 harms the per-cell gradient. Rays with no crossing near the wall:
       1.4 % at 0.1 mm, 0.3 % at 0.05, 0.06 % at 0.025 (thin tips lost).
     - *Why the rest fail: edges between crystals.* At 0.025 mm, cells whose stencil holds
       one crystal: 96 % within 1.5 deg; within one cell of an owner change: 2 %; 1-2
       cells: 78 %; beyond: 96-98 %. A grid rounds each edge over about a cell (an
       `effective` rounding, shrinking with the spacing). The ~4 % still off far from
       edges are *plausibly* the tip-to-side kink inside a crystal (the owner test
       cannot see it). Six-faced tips would be exact inside faces but add edges.
     - *Candidate jumps in agatesim's `assign`.* Against a wider candidate set (16
       sites, 64 owner nodes) `assign` is late on 4.4 % of wall points, by up to 0.5 mm
       (0.6 mm at nodes). Not the normals' cause: the jump-free level scores the same
       (80 % at 0.025 mm). A jump-free wall would move > 0.1 mm on 3.1 % of rays (to at
       least the 1 mm search limit). Whether the steps show in an air hollow is
       untested.
     - *Cost of a level grid.* This stone's world-aligned void box is 26 x 25 x 27 mm
       (the cut is tilted). Dense: 0.1 mm 18 M nodes, 0.04 GB int16, fill ~1 min; 0.05
       mm 143 M, 0.29 GB, ~6 min; 0.025 mm 1 140 M, 2.3 GB, ~51 min (out of budget). A
       +-2-cell band at 0.025 mm is ~4 M nodes (from a lower-bound wall area), but needs
       sparse storage on the GPU.
     - *Alternative measured: an owner lattice.* Store each node's crystal (a dense box
       at 0.2 mm is ~2 M nodes, at 0.1 mm ~18 M) plus each crystal's few numbers (site,
       axis); at a point take the earliest of the 8 corners' crystals in closed form
       (`Crystals.pair_times`). Over all wall rays found, normals within 1.5 deg: 98.2 %
       (0.2 mm), 98.4 % (0.1 mm); where the wall did not move, exact (median 0.001 deg);
       edges stay sharp. Its wall is a third one, neither agatesim's nor the jump-free:
       it moves > 0.01 mm on ~2 % of rays, ~90 % of those making the hollow smaller (up
       to the 3 mm search limit); the shell's material comes from agatesim's fields, so
       those slivers need a rule (e.g. clear quartz). It has candidate jumps of its own:
       late on 3.9 % of its wall points, up to 0.7 mm, much as `assign`. About 1.2 % of
       rays found no wall within +-1 mm on every lattice and the jump-free level: cause
       unknown (likely the search window). Not measured: GPU time of up to 8 closed-form
       crystals a wall test against one trilinear fetch.
       *Copy rule:* the closed form must not be copied into nodule's CUDA. agatesim would
       ship it (a small CUDA function, checked against `pair_times` in its own tests)
       with its crystals, and six-faced tips would then change it there once. That is a
       larger agatesim change than the level approved: **the owner's choice.**
     - *Look* (runs/hollow-air/normals-look.png, mock-up shading, appearance only):
       its pixels are 0.06 mm, coarser than the 0.025-0.05 mm edge rounding, so the
       panels barely differ (0.1 mm scatters more glints). The look cannot decide this;
       cost and risk do.
     - *Owner's choice (2026-11-19 night): the owner lattice ("crystal list").* The
       larger agatesim change is approved: agatesim returns the crystal stage's crystals
       and ships their closed form for the GPU. Order step 2 is now this change; the
       slivers' rule and the GPU cost are measured as it is built.
   - *Step 2, the agatesim change: done* (2026-11-19; agatesim ADR 0090, 7b9fa04, pin
     moved). `VolumeSimulation.crystal_wall` gives the last crystal stage's crystals
     (`CrystalWall`: sites, axes, tip angle, side rate, last reach, grain offset;
     `arrival` = agatesim's `assign`, `level` = arrival − last reach, `packed()` rows), and `agatesim.volume.crystal_gpu.CRYSTAL_TIME_CUDA` ships the
     closed form as CUDA device functions (time and gradient: the exact normal),
     tested on the GPU against `pair_times` and finite differences, with controls.
     agatesim's fields are unchanged (hash before and after). The min over a point's
     corner crystals stays in nodule. *GPU cost, measured* (RTX 4070, 10 M random
     points, float32): the earliest of 8 crystals with the gradient 0.41 ns a point,
     one trilinear fetch from a 260^3 grid 0.88 ns; with the 8 owner ids to fetch, a wall
     test should cost about one fetch (*plausible*: synthetic points, one card; the id
     fetches themselves are untimed, measured in step 3). Smoke
     check on the probe stone (runs/hollow-air/wall_smoke.py, 33 s): the public wall is
     the stage the closure walk found. *The level's sign marks the hollow only where
     crystals could still grow* (quartz or hollow: there it matches VOID at every
     sampled point); it is also positive on all host and outside points and on 80 % of
     chalcedony points (behind the crystals' sites). So the void box must take the
     hollow's wall from the level only inside the crystal stage's region (the phase,
     or the stage's solid), not from its sign alone.
   - *Step 3, the void box: probe* (2026-11-19 late; runs/hollow-air/box_probe.py,
     box_fill.py, gpu_probe.py; nothing in src yet). The probe stone again (sheet 4, stone
     5); *verified* for it, untested on others (thin linings, hollows meeting chalcedony,
     tubes or needles).
     - *The hollow.* A coarse pass (0.4 mm, 3 s) finds it: one piece of 74 972 nodes plus
       three single nodes (not examined); it meets crystal quartz only. The box holds the
       whole hollow, both sides of any cut, + 3 mm, so it is one box per stone, shared by
       every cut and view: 36 x 37 x 33 mm. (Step 1's 26 x 25 x 27 mm was another
       measure: the part seen from one cut's face, + 1 mm.) Nodes are world-anchored
       (node i at i x 0.2 mm): 5.5 M.
     - *Crystal ids at every node* (int32, 22 MB): `wall.arrival` 14 s, 0.25 GB traced
       in the ids loop (the 0.27 GB of node coordinates, allocated before, not counted).
       0.1 mm would be 44 M nodes, ~106 s, for the same normals (step 1: 98.4 % against
       98.2 %), so 0.2 mm (`effective`). The wall's shape comes from the closed form, so
       the id spacing does not limit its sharpness in any view.
     - *Phase and growth time in a band* (0.8 mm into the hollow, out to the shell's
       edge), for the gate and as one material option: 1 mm 0.27 M nodes ~3 s; 2 mm
       0.56 M ~5 s; 3 mm 0.91 M ~7 s.
     - *The region gate holds at node resolution.* Among quartz and void nodes the
       level's sign equals VOID at every node; of 84 022 wall cells none has a corner
       outside quartz or void. The false hollow (level > 0 in chalcedony behind the sites)
       starts 2.8 mm from the real one (the stage's last reach). So a per-node flag
       (quartz or void; the hollow's inside) gates the wall.
     - *Slivers.* Of 300 000 points in wall cells, 0.22 % are VOID in agatesim but solid
       on the lattice (median 0.06 mm deep, at most 0.47 mm); 0.01 % the reverse (air
       where agatesim has quartz). *Assumption* (proposed rule): a sliver takes its
       lining's material, the nearest filled node's growth time, as `fill_void` gives
       open nodes today.
     - *Material below the slab: open.* Variety renders read material on the view's own
       grid: the whole face 0.1 mm (`WHOLE_GRID_MM`), needle stones 0.05, close-ups 0.03,
       down to 5 mm (close-ups 2 mm). A 0.2 mm material shell would be coarser than all of
       them. Measured on 0.2 mm in the quartz lining only (no chalcedony sampled): growth
       time read trilinearly is off by 2.6 growth steps typical, 19 at p99 (of 400; ~14
       steps a cell here); two samplings half a cell apart differ by 2.0 and 16 steps,
       the size of a seam between two grids (*plausible* as the slab grids' error too, at
       their spacing: not measured on them). Options: (a) one world 0.2 mm material shell
       for every view (~7 s, view-independent), coarser than every view, with a seam
       where each view's grid hands over, at a depth that differs by view (5 mm, close-ups
       2 mm); (b) each view's own grid continued down under the hollow (same nodes, so no
       seam, and the polished face unchanged); cost grows with the view (the whole face
       roughly a footprint of 26 x 25 mm to ~10 mm further down at 0.1 mm, ~6-8 M nodes;
       close-ups at 0.03 mm far more unless coarsened), measured when built. The choice
       is the owner's; it does not change the id box or the gate.
     - *GPU cost* (RTX 4070, 10 M points drawn from the real near-wall points; agatesim's
       string prepended, fast math as the camera): the wall test (8 ids, 8 closed forms
       with gradient, the earliest) 0.40 ns a point in random order; one trilinear read of
       a float32 grid of the same box 0.15 ns. (Sorted by cell, 0.06 and 0.03 ns, near
       the card's memory bandwidth for the harness's own reads and writes: those time the
       harness, not the test.) Its level matches the CPU's float64 to 3e-6 mm, sign on
       every point. Per point, not a risk; how many tests a flight needs is step 4's
       measure.
     - *Total per hollow stone:* ~25 s of fill (coarse 3, ids 14, band 7) and ~0.05 GB on
       the GPU for ids and gate, on top of the stone's build; material by the option
       taken. Not checked again: step 1's 98 % normals (`arrival` is agatesim's `assign`,
       at the same world nodes, and the GPU matches the CPU).
     - *Next stone for the gate:* the false hollow's 2.8 mm margin equals the stage's
       last reach, so it likely shrinks with a thinner lining; the build checks the stone
       with the smallest `last_reach` it can find.
     - *Owner's verdict (2026-11-19 late): all proposals accepted.* Material below the
       slab by option (b): each view's own grid continued down under the hollow (no
       seam, face unchanged; close-up cost measured when built). The 0.2 mm world id box
       per stone, its per-node region gate, a 3 mm shell (`effective`), and slivers taking
       the lining's material (*assumption*). agatesim's `CrystalWall` docs and ADR 0090
       now say where the level's sign marks the hollow (wording only, approved). Next:
       the build, unwired from `pack` and `cutface`, with tests and controls.
   - *Step 3, the build* (2026-11-19 night; `nodule.render.voidbox`, unwired;
     tests/test_voidbox.py, 9 s, each with its control; checks in runs/hollow-air/
     gate_check.py, gate_diag.py, thin_lining_map.py). The box holds ids and gate only
     (material by option (b) stays on each view's grid). The coarse hollow is open points
     with a positive crystal level (not every open point, so hollow crystals and tubes
     elsewhere do not widen the box). *Assumption:* a cell is air when all 8 corners are
     gated and the box's level is positive. The gate takes the stage's own quartz by grain
     (`grain_offset + j`): early wall crystals are quartz too, with a positive level, and
     passed the phase gate (977 of 400 000 points on sheet 0, stone 1).
     - *Measured* (6 stones of sheets 0-11; random points in the box; *verified* for
       them): gated air is never chalcedony or outside the stone, against 41 000-395 000
       chalcedony points a stone where the ungated level is positive. Box cost: probe
       stone 25 s, 0.35 GB traced (coordinates included); small stones 1-6 s, 0.21-0.25
       GB. A box with wider bounds agrees node for node.
     - *Where crystals cover the wall* (probe stone; sheet 0, stone 2): the hollow is air
       but for 0.02-0.06 % of open points (slivers).
     - *Where the hollow meets bare chalcedony it fails.* Thin crystal linings (sheet 3,
       stone 9, last reach 0.35 mm; sheet 8, stone 4, 0.25 mm) leave sparse crystals on
       a bare chalcedony wall; sheet 0, stone 1 (needles, a tube) has patches of it. Cells
       with a chalcedony corner are solid, so 38 %, 38 % and 14 % of open points are not
       air, within 0.16 mm (median) to 1.6 mm of air, and the wall there follows the
       0.2 mm cells in steps with the crystals' normals (runs/hollow-air/
       thin-lining-map.png). The bare wall is the chalcedony's last surface, which the
       box does not hold. How many stones have it: not counted (2 of the 6 chosen for
       thin linings). Options for the owner: (A) a second, smooth level for that surface
       (air beyond both), its source to be measured first (read from agatesim's fields
       from outside, as in step 1, or an agatesim change); (B) the gate on a finer grid,
       steps smaller but normals still wrong there; (C) such stones keep the stand-in.
     - *Owner's choice (2026-11-19 night): (A)*, starting with a probe of the surface's
       source; an agatesim change would be asked for separately.

## Consequences

- Variety stones render with every ADR 0018 layer; open cavities with a stand-in (s.6.7);
  the host rind and specks are drawn on the look after the render (s.6.8).
- `Layers` implementations must split their work (`per_step` and the call).
- The renderer's record now holds white fade and quartz iron; earlier stones (no
  layers) are unchanged.
