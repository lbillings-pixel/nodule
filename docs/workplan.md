# Work plan (owner-approved 2026-11-15)

The order of work the owner approved on 2026-11-15: easiest and fastest first,
with time-saving code changes pulled forward. It merges HANDOFF.md, the
sessions of 2026-11-09/10 and the still-open ideas in `docs/ideas-2026-11.md`.
Consult it at pickup until every batch is done; tick items here as they close
(`[x]`, with the commit or ADR section). Times are rough estimates, not
measurements. Nothing here is a science claim; each new feature still starts
with reading (CLAUDE.md, "Literature first") and a look sheet for the owner
before any statistics.

## Batch 1: quick wins and time-savers

- [x] 1. Trim HANDOFF.md to its cap; delete the merged `render-layers` branch
  (owner agreed 2026-11-15).
- [x] 2. (owner picked B, 2026-11-15: `PLUME_MUA` 4, the 0.3 stop half hematite)
  Plume colour close-ups (ADR 0018 s.13 step 4, owner's option C): three
  variants of weaker plume absorption plus a warmer orange-gold, side by side
  on 12 mm close-ups of sheet 1 stone 3. The owner picks; record the pick in
  the ADR.
- [x] 3. DROPPED 2026-11-15 (halo-only nodes are 0.3 % of the grid; ADR 0018 s.13
  step 4a). Halo speed-up in the plume bake: plume stones take about 225 s of
  renderer setup against 10-70 s without plumes; halo-only nodes could skip
  sub-sampling. Measure before and after.
- [x] 4. (8.6 x per point; ADR 0018 s.13 step 4a) Measure the cost of band roughness (ideas doc: about 7 x per point,
  because its frame reads growth time seven times). Measurement only.
- [x] 5. (rendered 2026-11-15, 607 s for the three; local `runs/plume-render-b/`,
  the owner found green too faint, `GREEN_STRENGTH` 1.5, then "it's ok"; bd09109) Re-render the three plume stones with the owner's pick.

## Batch 2: short tasks (about one session each)

- [x] 6. (done 2026-11-15 in nodule, `variety.FrameLattice`: plume stone fill 218 s ->
  103 s; ADR 0018 s.13 step 4a) Roughness cache (a coarse cached grid of the growth-time gradient),
  only if item 4 shows it pays. If the code sits in agatesim, ask the owner
  first.
- [x] 7. (done 2026-11-15: package `nodule.variety` with `recipe`, `cracks`,
  `mottle`, `draw`, `layers`, `checks`; `window_cut` now in `nodule.specimen`;
  recipes of seeds 1-3 and one cut identical to the old module)
  Split `variety.py` (about 1,700 lines) into recipe, cracks, mottle and
  checks, and move `window_cut` from `sheet.py` beside `Specimen.cut`. No
  behaviour change.
- [x] 8. (2026-11-16; ADR 0018 s.14, agatesim ADR 0085-0086; moss now in the recipes,
  1 stone in 6; floor mats not drawn) Moss probe and look sheet: roof drapes growing down
  (`dendrite_tilt_deg` 180) and floor mats; mostly green, some yellow-brown,
  about 1 in 10 each red and black; same wrap draw as plumes. Negative
  control: a straight radial spray must be rejected.
- [x] 9 + 10 (merged 2026-11-16, the owner's pick; closed 2026-11-16 with 59ab2d3,
  combinations on 1 in ~150 crystal stones, ADR 0018 s.15). Early crystals: blades
  (calcite?) and fine needles (zeolite?) grown on the wall before the bands
  (agatesim's stick stage, wrapped at full rate), left white, turned to clear
  silica or dissolved to moulds. Reading done; probe and look sheet 1 done (all
  looks accepted). Recipe wiring (`nodule.crystals`, 1 stone in 6, an assumption),
  centre and floor compensation, wrap check with its negative control and ADR 0018
  s.15 done 2026-11-16; look sheet 2 judged "okay" 2026-11-16, share now 1 in 9.
  Owner's follow-ups (2026-11-16): thunder-egg length cap by arm width (knocked on:
  thunder eggs take needles only, done); agatesim floor fix (done, agatesim ADR 0087);
  a small share of crystals with plumes or moss (done; plume halos now keep faint
  bands, dda6964, s.13). Before a combination stone is rendered: plume roots take the
  crystals' material (`nearest_seen`, s.15); fixed 2026-11-17 (`BandRecord.plume`).
  Still open afterwards (for the owner, s.15): late white patches that cut across
  bands (a mottle-like layer), true radial needle sprays (would need agatesim).

## Batch 3: medium (one or two sessions each)

- [x] 11. (owner, 2026-11-17, on the moss renders: fix amethyst centres rendering grey; fix
  fine needles vanishing at whole face unless that proves resource-prohibitive, then skip it.
  2026-11-17: both fixed, the owner's picks taken (ADR 0018 s.15 correction; ADR 0019
  s.6.9): white remnants now milky in the render's record; white needles on a 0.05 mm
  grid within a voxel budget (kept); amethyst and smoky centres take colour-centre
  absorption at 2x medium amethyst (owner). Three papers for the owner to fetch.
  Recipes done with item 8; moss rendered 2026-11-16, sheet 0 stones 9 and 11,
  `runs/variety-render-moss/`, local: green tufts and red drapes show in their sheaths;
  waits for the owner's eye; amethyst centres render grey in the look, seen, not fixed;
  plume share 1 in 9 since 2026-11-16, the owner's pick) Moss in the recipes
  and the renderer; then drop plume frequency from
  1 in 4 to about 1 in 8-10 (owner, 2026-11-09). The owner's 3/4-up plume
  split depends on moss growing downward.
- [x] 12. (closed 2026-11-17 by judging on quick renders, below) Previews made from the renderer's own colour model, so the owner's
  eye on sheets carries over to renders. Measured 2026-11-16 (7 pixel-aligned
  preview/look pairs, leave one stone out; `runs/preview-vs-render/`, local): the
  preview sits a median dE 11-20 from the look; neither one colour map nor a lookup on
  the cut's own fields (phase, pigment, translucency, impurity, also blurred over 1 mm)
  closes it (dE 10-20 after). So no cheap fix: the gap likely comes from what the fields
  do not hold (the look's per-stone exposure, each pigment's optics, light from
  beneath; not yet separated).
  Options: model the renderer's materials in the preview (1-2 sessions), or judge looks
  on quick renders (whole face 1.5-5.5 min a stone on the GPU).
  2026-11-17: the owner left the pick to Claude ("whatever is faster and more
  efficient"): judge looks on quick renders. Nothing to build; previews stay a
  stand-in, and any look that matters goes to a quick render before the owner's verdict.
- [x] 13. (2026-11-17, ADR 0018 s.16 "Built"; `nodule.rind`; on the renders'
  look too, host redrawn in sheet colours, ADR 0019 s.6.8; no bleed, the owner's choice) Host rind, host speckle and pigment bleed
  into the host.
- [x] 14. (reading done 2026-11-17, ADR 0018 s.17: no agate study found; the owner's
  choices the same day: mostly from the rind inward and on cracks, thin-line view, at most
  5 % of stones; probe next) Black manganese dendrites in band partings.
  2026-11-18: BUILT (ADR 0018 s.17 "Built"; `nodule.dendrites`): 4 % of stones, half
  from the wall (largest eighth of the trees), half on cracks (a hairline added where
  needed); shows on about 1 cut in 100. Owner the same day: steer to the cut, darker
  crack dendrites, picks confirmed, moss + dendrites allowed. Done: every dendrite stone's
  cut shows them; Mn opaque in renders (30 mm^-1). CLOSED 2026-11-18: the owner kept
  30 for wall dendrites too (one strength for both carriers).
- [x] 15. (reading done 2026-11-17, ADR 0018 s.18; tubes BUILT 2026-11-17, 1 in 6
  non-pebble stones, `nodule.tubes`; eyes wait for the owner's choice of view, to be
  shown as a sketch) Eyes and escape tubes.
  2026-11-18: eyes probed and BUILT (ADR 0018 s.18 "Eye probe", "Built"; `nodule.eyes`):
  agatesim's spherulitic domes, owner "good, build them"; 1 in 10 volcanic stones, 1 in
  50 thunder eggs; size and spacing per stone; few-ring eyes rare; floor stones too.
  CLOSED 2026-11-18 by the owner.
- [ ] 16. Depth through clear bands (a render test).

## Batch 4: large (several sessions each)

- [ ] 17. True 3D thunder-egg cavity (a cut may gain or lose arms with depth;
  at Del Norte lens-shaped centres are reported more often than stars).
- [ ] 18. Druse crystal faces in open cavities.
- [ ] 19. Renderer physics: scattering in the camera code (ask the owner
  first), air with refraction in cavities (ADR 0019 A2), the fuller red fix
  (several wavelengths per channel; recheck ADR 0014), white ghost plumes.
- [ ] 20. Saw marks and polish level (last, by the owner's earlier order).

## Follow-on projects (when the feature work pauses)

In this order (ideas doc, section 4):

1. Bands as signals, tested on synthetic stones with a known forcing.
2. An interactive slicer web page.
3. A labelled dataset for band counts; photo to recipe; the cutter's choice
   as a tool; other banded stones.

## Needs the owner's word before it can be scheduled

- agatesim edits (each needs approval): gentler fade-in for lobed bands; plume
  directions between the six axes; different wrap rates per plume in one stone.
- The N2 blind test (ADR 0008 s.5), which the owner runs.

## Open small questions for the owner

- Do the reference plume agates usually have quartz or open centres?
- Is the sparse plume look on about 1 in 3 plume stones right (an assumption)?

## Low wait list (owner, 2026-11-09)

Why quartz-lined recipes rarely show a cavity on the cut; dust hard to see in
close-ups; stones with very thin drawn centres that rarely reach the cut.

## Standing limits

- Slope stays closed (ADR 0017); no work on blur of fine bands.
- No GPU render over about 30 s without flagging it first; the sheet 1
  re-render (about 80 min) is offered, not run unasked.
