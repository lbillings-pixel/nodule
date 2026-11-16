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
- [ ] 9 + 10 (merged 2026-11-16, the owner's pick). Early crystals: blades
  (calcite?) and fine needles (zeolite?) grown on the wall before the bands
  (agatesim's stick stage, wrapped at full rate), left white, turned to clear
  silica or dissolved to moulds. Reading done; probe and look sheet 1 done (all
  looks accepted). Next: recipe wiring, check with negative control, ADR section.
  Still open afterwards: late white patches that cut across bands (a
  mottle-like layer), true radial needle sprays (would need agatesim).

## Batch 3: medium (one or two sessions each)

- [ ] 11. (recipes done with item 8; renderer and plume share left) Moss in the recipes
  and the renderer; then drop plume frequency from
  1 in 4 to about 1 in 8-10 (owner, 2026-11-09). The owner's 3/4-up plume
  split depends on moss growing downward.
- [ ] 12. Previews made from the renderer's own colour model, so the owner's
  eye on sheets carries over to renders.
- [ ] 13. Host rind, host speckle and pigment bleed into the host.
- [ ] 14. Black manganese dendrites in band partings.
- [ ] 15. Eyes and escape tubes.
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
