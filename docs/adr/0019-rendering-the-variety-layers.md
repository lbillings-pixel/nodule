# 19. Rendering the variety layers

- Status: **proposed** (built by Claude overnight, 2026-11-09, with the owner's leave to
  choose the night's work; awaiting the owner's acceptance). Options in section 5 are
  open for the owner.
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
10); (ii) **the black bottom** (ADR 0014): translucent red bands return little light, so
without glare carnelian reads dark brown, not red; red shows mostly where bands are
milky. Both come from the lighting chosen for the photo compare (ADR 0008 7c, ADR 0014),
not from the pigment.

B1: keep (ADR 0014 is closed; the photo compare was matched under this lighting). B2
(dropped by the measurement): a stronger loading. B3: a **look preset** for the owner's
eye only, leaving the compare's settings alone: a light without the face's glare (as a
photographer's polariser or dark surround gives) and a lighter bottom (a white backing
or air), each tried and shown side by side. B4: reopen ADR 0014's lighting against the
library's red-agate photos. *Recommended: B3*, cheap (a render setting, no new
physics), and it keeps the closed calibration intact.

## Consequences

- Variety stones without open cavities render, with every ADR 0018 layer.
- `Layers` implementations must split their work (`per_step` and the call).
- The renderer's record now holds white fade and quartz iron; earlier stones (no
  layers) are unchanged.
