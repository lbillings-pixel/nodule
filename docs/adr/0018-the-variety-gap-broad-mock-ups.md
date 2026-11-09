# 18. The variety gap: broad mock-ups of many features, varied per stone

- Status: accepted (owner, 2026-11-08; drafted from their choices of 2026-11-06).
  Fractures: option C (section 4). Section 11 (dust): accepted 2026-11-09, options
  A3, B1, C1.
- Date: 2026-11-08
- Builds on: [ADR 0008](0008-set-level-render-and-compare.md) section 5a (the
  trial and its notes), [ADR 0006](0006-band-counts-as-censored-records.md)
  section 10 (settled limits), [ADR 0017](0017-c26-tilt-replaces-the-fine-side-slope.md)
  (the slope is closed).

## Context

The small blind trial (ADR 0008 s.5a) ended 16 of 16: the fakes are spotted at a
glance. The fracture lines decided 3 pairs and are now off for look work. The other
notes name what the model lacks rather than what it gets wrong: "random-but-structured,
multidimensional" diversity, a bubbly layer, filigree and tubes, mottled warped bands,
stained fractures that cut features, a cavity, glints.

The owner's reading (2026-11-06) is that realism is the interplay of many features,
each varying, and that no single feature added first would close the gap. Adding one
feature at a time and rerunning the trial after each is therefore not the plan.

**The 5 pairs with no note** (owner, 2026-11-06): they were told by the same kinds of
thing as the noted pairs; there is no hidden tell behind them. *Verified* as the
owner's statement.

## Decision

### 1. Broad first, then tweak

1. Build a representative sample of features at "decent" fidelity, not one at a time
   to full fidelity.
2. Each stone gets its own recipe: a varied subset of features, with varied strength.
   **No stone carries every feature.**
3. Show the owner whole-stone sheets of about 12 stones as look
   cuts, before any statistics. Then tweak features one by one from the owner's notes.
4. No statistical run, critic score or trial rerun before the owner has seen a sheet.
   The photo-vs-render critic is not a realism judge (it preferred faded fakes) and is
   not used.

### 2. The feature list (owner-approved 2026-11-06, as drafted)

| # | Feature | Priority | Note |
|---|---|---|---|
| 1 | Centre: solid, quartz, open cavity | high (approved A) | The modelled stone has none (8 of 8 trial planes at least 99.9 % chalcedony) |
| 2 | Fractures: open, filled, stained | high (approved B) | Needs a real form (width, fill, staining) before it returns; see section 4 |
| 3 | Mottled, warped bands | high (approved B) | |
| 4 | Colour variation | high | Owner: near the top |
| 5 | Differences between stones: outline, band density, overall colour | high | The base of "varied per stone" |
| 6 | Inclusions: moss, filaments, odd ones | middle | |
| 7 | Filigree and tubes | middle | Escape tubes [@howard2018] were already queued as later work |
| 8 | Bubbly layer | middle | |
| 9 | Eyes | middle | |
| 10 | Flat waterline bands | middle | |
| 11 | Glints and finish | last | Owner: lowest |

Order of work follows priority, but the first sheet should already hold a rough form of
most of 1 to 10, as the point is their interplay.

### 3. How features are built

1. **Short reading per feature first** (literature first): enough to fix its form,
   size range and where it sits, read beyond the abstract. Helper agents do the bulk
   reading; claims enter docs only through ADR sections, within each source's
   `checked` level.
2. **Features read each other.** Where one feature depends on another (a cavity sets
   where quartz sits; a stained fracture tints nearby bands; tubes cross bands), the
   coupling is written down with its source or labelled an assumption.
3. **Parameters** are labelled `physical` or `effective`. Random choices use spawned
   streams and world coordinates, so a stand-in preview matches its render.
4. **Frequencies** come from tags, as a guide only. Library tags (261 photos, sampling
   unknown, not rechecked): quartz core 104, open cavity 75, fractures 85 of 199
   countable (offset 2 of 61), moss 55, eye 34, waterline 32, inclusions 23, tube 21,
   botryoidal 16, iridescence 7. The 2026-11-06 Commons harvest (559 distinct photos,
   254 machine-tagged as reflected cut faces, accuracy 0.81) can add counts, but its
   sampling is also unknown and some dyed stones, fordite and displays leak through.
   *Plausible* as rough shares; *not* population frequencies.
5. **Where they live.** Features may start as quick layers in Nodule. Those that earn
   their place move into agatesim (its own tests and ADRs), and Nodule imports them.
6. **Each new criterion** keeps a negative control (CLAUDE.md).

### 4. Fractures: sources disagree, so options

[@howard2018, p. 102] say "fractures through the bands are never seen". The owner's
note names, in a library photo, iron-stained fractures that cut through features. Both
can hold if the authors mean fractures during band growth and the photo shows later
breaks; that reading is *speculative*. Options, for the owner when fractures return:

- **A.** Later fractures only: they cross bands, are open, filled or stained, and leave
  bands unbroken where they formed.
- **B.** Follow the paper: no fracture crosses bands; breaks run only through the
  centre or rind.
- **C.** Both, by recipe: most stones without, a few with later stained breaks (tags
  give 85 of 199 countable with fractures, as a guide).

**Chosen (owner, 2026-11-08): C**, as it keeps variety and matches the photo the
owner noted. Fractures return only once they have a real form (section 2).

### 5. Stopping rule (owner, 2026-11-06)

Variety work stops when **a fresh small trial scores below 14 of 16** (the s.5a rule:
then the full test is worth its cost), **or when the owner calls it**. A fresh trial
uses new cuts and a new seed, and runs only after the owner has seen a sheet.

### 6. Costs (measured earlier; new work is measured as it goes)

`Specimen.build` about 4 s; stand-in cut about 6 s at 900 px (screening only); render
about 5 min a cut; a 12-stone sheet about 1 h on the GPU. Any run over 30 s or 2 GB is
flagged before it starts. `render_cuts` must take the preset for any rerun.

### 7. Reading for features 1 to 5 (2026-11-08)

Five helper agents read the catalogue's notes, the owner's PDFs and open web pages;
claims below stay within each source's `checked` level. No source counts how often
any of these features occurs, so every share in the recipe is `effective`, to be
judged on the owner's sheets. Figure readings are a helper's measurement against a
scale bar on one or two stones: *plausible* at best.

**Centre.**
- Many agates end in macro-quartz, often hollow: "most agates" [@howard2018, p. 99],
  "almost every agate nodule" [@pabian1994, p. 20], "very common" in large Brazilian
  agates [@moxon2014, p. 331], most Chihuahua specimens [@mrozik2023, p. 13]. All
  qualitative (*verified* as statements). The library tags (quartz core 104, open
  cavity 75 of 261) are the only counts.
- The chalcedony–quartz boundary is mostly sharp [@howard2018, p. 99; @moxon2020,
  p. 19]. Quartz is near-colourless and low in iron [@goetze2009; @mrozik2023,
  Table 5], at times amethyst or smoky [@goetze2020, p. 37] (*verified*).
- Grain width follows geometric selection: count falls about as t^-0.5, width grows
  as a power law and, in a closing cavity, levels off and then falls
  [@rodrigueznavarro2000, pp. 611, 613]. A model, not an agate measurement; quartz
  grains in one Chihuahua figure grow from about 0.25 mm to 1–2 mm over 3 mm
  (*plausible*).
- Centre width on 7 figured stones: 0 to about 0.6 of the nodule, most near 0.2
  (*plausible*, a hand-picked set).
- Flat (waterline) floors: in 9 Brazilian agates they fill 30–70 % of the volume, with
  quartz layers allowed in the stack [@moxon2020, pp. 15–16]; shares of banding types
  in 73 stones [@moxon2013] (*verified*). **Order is disputed** (options, not settled):
  floor first, wall lining above [@moxon2020, p. 16]; or concentric and flat packets
  alternating [@kigai2019, p. 872]. Kigai's genesis is a minority view; only his
  descriptions are used.

**Fractures** (option C, section 4).
- Rare, and put down to host-rock movement [@moxon2020, p. 15]; healed by later agate
  or quartz [@goetze2020, §5.1]; secondary cracks refilled with chalcedony or iron
  oxides [@mrozik2023, pp. 14, 27]; stain "discordant to the agate banding" along rims
  and cracks [@goetze2020, p. 37] (*verified*). One figured crack crosses every wall
  band with no visible offset [@moxon2014, Fig. 3 I] (*plausible*; a bought stone, a
  cutting crack not ruled out).
- From figures (*plausible*): traces 0.05–0.4 mm; little halo along the trace (0–0.5
  mm) but stain pools of 2–6 mm where a crack meets the cavity or a porous band; short
  rim-only cracks 1–4 mm. No offset is seen in any figure.
- Breccia: the sources read give data only for ore veins [@jebrak1997], and none for
  nodules [@pabian1994]. **Left out of the first sheet.**

**Mottled and warped bands.**
- Flow or plastic deformation is marked present at all 12 tabulated localities
  [@pabian1994, Table 1]; bands thin and converge at escape tubes [@pabian1994, p. 20;
  @howard2018, p. 99]; hematite lies in lenses along bands [@goetze2020, §4.3]; colour
  can change along a band [@mrozik2023, p. 11] (*verified*). Bands keep width and
  colour across sector boundaries [@howard2018, p. 104], which limits patchiness there.
- From figures (*plausible*): crenulated band edges, wavelength 1–5 mm, amplitude
  0.2–1 mm; kinks or lobes 0.3–1.5 cm; red mottle patches 5–30 mm with rims 0.6–1.2 mm.

**Colour.**
- Iron follows the band's texture: about 600 ppm cryptocrystalline down to about 70
  micro-granular [@zhang2025]; macro-quartz 78–141 ppm against 262–969 in bands
  [@mrozik2023, Table 5]; unbanded carnelian Fe2O3 about 1.1 wt% [@volkert2023, p. 12];
  black from carbon in the pores of fibrous chalcedony [@goetze2012] (*verified*).
- Pigment gathers near each band's growing front; red, yellow and brown bands are more
  even than white; white bands thin out from rim to core [@mrozik2023, pp. 13, 22, 23];
  Coyamito violet is a lattice defect, not iron [@mrozik2023, p. 29] (*verified*).
- Iron correlates only weakly with colour strength [@mrozik2023, p. 23], and some
  stones from iron-rich hosts are uncoloured [@moxon2020, p. 18]: palette is set per
  stone, not from host (*plausible* as a rule for the recipe).

**Differences between stones.**
- Outline follows the host: almond vesicles in basalt and andesite, star cavities in
  rhyolite thunder eggs [@pabian1994, pp. 11, 16]; amygdules flat below and convex
  above [@kigai2019, p. 870] (*verified*). Locality colour tendencies exist but almost
  all colours occur everywhere [@mrozik2023, p. 11].
- Visible band frequency 1 to about 5 per mm, narrowing inward [@goldbaum2019]
  (*verified* as reported; its fit is weak).

**First sheet recipe (all `effective` unless marked).** Twelve stones, each from a
spawned stream, each drawing a subset:
- *Centre:* solid, quartz-filled or quartz-lined cavity, about a third each; centre
  width 0–0.6, mass near 0.2; sharp boundary under 0.2 mm with a saw-tooth inner face;
  grain width w0 (1 + d/Δ)^0.5 from w0 about 0.25 mm, levelling off in a closing cavity
  (form from [@rodrigueznavarro2000]); amethyst or smoky tint in 10–20 % of quartz
  centres. Quartz low in iron (`physical`, sourced above).
- *Floor:* in 2–3 of 12 stones, 30–70 % of the volume (`physical`, small n), floor
  first by default, Kigai's alternation as the option.
- *Fractures:* 1–2 of 12 stones; 1–3 sheets each, high angle to bands; trace 0.05–0.4
  mm; fill clear or iron; stain pools at cavity and porous bands; no offset.
- *Warp and mottle:* on some stones, never all: crenulation 1–5 mm by 0.2–1 mm, 0–2
  kinks, mottle patches 5–30 mm with darker rims.
- *Colour:* per-stone palette, about 6 neutral, 3 iron, 1 onyx (carbon in fibrous
  bands), 1 stained, 1 carnelian-like (Fe2O3 near 1 wt%, `physical` anchor); iron
  weighted by band texture (`physical` ratios); pigment at band fronts; white bands
  fewer inward.
- *Outline:* about 7 basalt vesicles (1–2 lobed), 2 thunder eggs with star cavities,
  1 prolate, 1 irregular, 1 small unbanded pebble.
- **Couplings (assumptions unless cited):** centre sits inside the last band; floors
  perpendicular to a per-stone "down"; fracture stain pools at the cavity and porous
  bands (from figures); white-band share falls inward [@mrozik2023]; quartz takes
  little pigment [@mrozik2023; @goetze2009].
- **Negative controls:** a solid stone reports no cavity; a centre without facets fails
  a faceted-lining check; a curved floor fails a waterline check; an unstained crack
  fails a stained-fracture check.
- **Screening first:** stand-in cuts (about 6 s each) make a preview sheet for the
  owner before any render (about 1 h for 12).

### 8. First layers and preview sheets (2026-11-08)

`nodule.variety` draws a sheet's recipes from one seed (quotas exactly as section 7)
and turns each into an agatesim run plus quick layers in Nodule; `nodule.sheet` cuts
each stone and makes a stand-in preview sheet (`runs/variety-sheet`, gitignored).
Seeds 1 and 2 are drawn. Stand-in colours only: no render yet.

- **From agatesim, chosen per stone:** outline (its solid presets; a drawn star
  prism for thunder eggs, cut across), quartz centre (silica supply drops below
  agatesim's quartz threshold), quartz-lined open centre (the run ends as the lining
  reaches its thickness: agatesim allows no stage after quartz), floor (a settling
  pulse at the start), warp (its slope-capped wobble), palette and iron supply.
- **New layers in Nodule** (`VarietyLayers`, couplings labelled there): quartz iron
  at 0.2 of the bands' (`physical` ratio, [@mrozik2023, Table 5]); white bands fade
  inward; mottle patches with darker rims, milky bands taking more; a patchy rim
  stain; fracture traces (open, clear or iron fill, no offset) whose stain pools at
  milky bands and the cavity. Layers hook into `Specimen.evaluate`, so previews at
  any spacing agree (tested on shared pixels).
- **Checks and negative controls** (`tests/test_variety.py`), measured values:
  cavity (a solid stone reports none); faceted lining, boundary over smoothed
  boundary, 1.36 for a quartz lining against 1.04 for the same cavity lined by
  chalcedony (thresholds 1.15 and 1.08); waterline, level bands across 40 % of the
  width, 5 on a floor stone against 0 on a concentric one (threshold 3); stained
  fracture, impurity near the crack over the median beyond 3 mm, 0.41 against
  -0.01 unstained (thresholds 0.1 and 0.03).
- **Costs:** a 12-stone sheet takes 30 to 37 s at 600 px, peak traced memory under
  0.3 GB a stone. One early probe at 0.25 mm quartz grain spacing took 124 s for a
  45 mm stone (32 s at 0.5 mm), so grains start at 0.5 mm (`effective`).
- **Departures from section 7, for the owner:**
  - Crenulation is clamped to an amplitude of 0.08 × its scale, under agatesim's
    no-fold slope cap of 0.5 (ADR 0005 s.3); figures show steeper crenulation.
    Raising the cap (still below 1, so bands cannot fold) is the owner's call.
    Its scale starts at 2 mm, not 1 mm (cost of the noise lattice).
  - Kinks are not separate: "lobed" warp is a long wobble (8 to 15 mm).
  - Mottle patches are capped at 0.35 of the stone's size, so a face shows several.
  - On red stones most cracks are open or clear, as iron fill hardly shows.
  - Amethyst and smoky tints are in the stand-in colour only.
- **Slope-cap trial (owner's request, 2026-11-08).** The cap is now a per-run
  setting (`max_slope`, below 1; default 0.5) and the clamp scales with it
  (amplitude at most 0.16 × cap × scale). At 0.9 the crenulated stones of seeds 1
  and 2 take about twice the amplitude (0.16 to 0.29 mm, 0.26 to 0.46 mm), yet the
  stand-in pairs look nearly the same. agatesim's wobble moves all bands together
  as one smooth swell. The crenulation in the owner's reference photos is sharp
  chevrons and roughness that differs band by band, so the cap is not what limits
  it (*plausible*, from the pairs and the photos). **Owner: keep 0.5** (it looks
  better overall), and try a per-band roughness form as well.
- **Band roughness trial (2026-11-08).** `Roughness` / `BandRoughness`: the growth
  model is read at points shifted along the local band normal by world noise taken
  at the point's foot on the wall. The noise drifts with depth, so the pattern
  renews over a few mm, and its strength builds inward. Ridged (sharp creases) and
  smooth versions. The outline is not warped. The shift fades out near the centre
  line and over the last bands before quartz (an assumption, for the look; never a
  growth mechanism). Trial sizes (`effective`): 0.4 mm amplitude, 2 mm along the
  band, from one thunder egg's close-ups scaled by fingertips (field about 9-10 mm;
  chevron arms 1.5-3 mm; *plausible*, +-40 %). Checks and negative controls: band
  shift at 0.2-1.5 mm depth 0.035 mm RMS against 0.114 mm at 1.5-8 mm (full
  strength from the wall: 0.207 at both); no fold, 0.4-0.6 % of steps against 36 %
  for an extreme setting (threshold 2 %); quartz contact, 0.13 % stray quartz
  pixels against 3.25 % without the fade. Cost: about 2-4 times the plain cut (6
  more model reads a point for the normal); 18 stand-in cuts at 1200 px took 153 s.
  Comparison sheet: `nodule.sheet --roughness-trial 1 2`. The owner's verdict: s.9.
- **Owner's notes for planning (2026-11-08):** thunder-egg arms are much too
  regular and straight; real ones are sinuous and river-like, as of fluid
  infiltrating an irregular void. Higher fidelity may wait for later stages.
- **New reference photos (owner, 2026-11-08):** one thunder egg (face and three
  close-ups), one green moss agate with plumes (four views), one plume agate whose
  plumes vary in colour. Private: stored with the specimen photos in agatesim, never
  in this repository; notes in its `specimens/observations.md`. Features they show
  for the list: band-following particle dust and larger flakes seen at depth;
  chevron crenulation; healed cracks without stain; plumes growing from a base,
  with pale rims and colour that varies per plume (feature 6).
- **Before any render:** the GPU growth grid (`nodule.render.tgrid`) reads agatesim
  directly and must apply the same layers, and the quartz tint must reach the
  materials. No render or statistics before the owner has seen the sheets (s.1).

### 9. The owner's verdicts: rare, not removed (2026-11-08)

The owner judged the roughness trial and the plain stones: the plain cap-0.5 stones
look most real; smooth roughness beats ridged but was too strong at 0.4 mm. Two
plain stones of seed 2 looked less real: stone 1 (thick waterline bands, a small
nub on the outline) and stone 12 (bands swelling at the bulges). Each such feature
does occur in agates, but rarely, so each is kept and made rare (all `effective`):

- **Floor bands.** Bands are periods of growth time, and the floor rises faster
  than the walls grow, so at full supply floor bands came out about 4-5 times the
  wall bands. Silica supply now drops while the floor settles (agatesim gives
  settling precedence over quartz, so this cannot start quartz), and the run
  lengthens so the floor keeps its height and the walls their growth. Measured on a
  26 mm test stone with 0.5 mm bands: floor 10.9 mm at every supply, mean floor
  band 1.82 mm at supply 1, 0.91 at 0.35, 0.64 at 0.2, 0.34 at 0.12. Recipes draw
  supply 0.12-0.35, and 1 (the thick bands) for 1 floor in 8. Only the floor's
  phase stretches; the rest of the run is as at full supply.
- **Seen, not changed:** large floors (0.64-0.66 on seed 1's stones 8 and 10) rise
  above their target and can crowd out a quartz centre (stone 10 shows none). This
  was so before s.9 too; left for the owner's call. Fixed in s.10.
- **The cutter's choice.** Both stones are overlapping spheres (agatesim's lobed
  vesicle). Stone 1's cut grazed one sphere (offset 0.96 of its radius), leaving a
  nub; stone 12's passed 0.59 of a radius off one lobe's centre, where its shells
  widen toward the middle as in any off-centre cut. Taking the wobble out did not
  change it. Cuts are now redrawn (up to 64 tries) until every sphere they cross is
  cut within 0.3 of its radius from its centre, else the least oblique is kept;
  0-1 stones per 12 keep the first cut drawn. Over seeds 1-6: 28 stones of
  overlapping spheres, 2 above 0.8, both the kept odd cuts. Real slabs are cut for
  the best face, so this mimics the cutter, not growth.
- **Band roughness.** Smooth form only, now drawn into recipes, at most 0.25 mm (a
  limit on the setting): none on half the stones, faint (0.04-0.12 mm) on 35 %,
  marked (0.12-0.25 mm) on 15 %; not on cracked stones or the pebble. Over seeds
  1-6: 29 of 72 stones rough, 13 of them faint. Depth now counts from the walls'
  start, so floors stay flat.
- **Checks with controls:** `floor_bands` (thin at supply 0.2 against the thick
  control at 1, floor height within 25 %); `cut_obliqueness` (a grazing control
  above 0.9; the chosen cut within 0.3; 0 for a single-body solid); sheet shares for
  odd cuts, thick floors and roughness (a 0.4 mm control is refused).
- **Confirmed list:** the owner agreed the missing features to add next are
  band-following particles and dust, plumes and moss, and depth through clear bands.

### 10. Floors keep their centres; cracks heal and bend (2026-11-08)

The owner asked for lower floors (seed 1's 0.64-0.66 floors crowd out quartz) and
judged ruler-straight black cracks fake. All values `effective` unless cited.

- **Floors lost every centre.** Measured over sheets 1-6 before the change: none of
  the 11 floor stones with a quartz centre showed quartz or open space in its cut.
  The walls do not grow while the floor settles (agatesim fills the open space from
  its lowest point), but they were still sized to the whole cavity, so they always
  closed the space left above the floor. Now the walls are sized to that space: in
  a sphere of radius r a floor at a share f of the height leaves room for a sphere
  of radius r (1 - f) exactly (other outlines: an approximation), and the centre's
  width is a share of that. The floor runs first, then the walls' whole run.
- **Floor growth from volume.** The old scaling (6.6 / 16 r per unit share, from one
  probe) made low floors rise well above target once walls no longer shrank to make
  room (0.28 drawn, 0.43 reached). Settling conserves volume, so the floor's cap
  (pi h^2 (3r - h) / 3, h = 2 r f) now equals the shell of the growth it needs,
  which gives the old value at f = 0.5. Floors now reach their target within +0.06.
- **Lower draw:** f from 0.2-0.5 (was 0.3-0.7). Over sheets 1-6, the 15 floors reach
  0.28-0.55 of the cut's height; 10 of the 11 with a quartz centre show quartz or
  open space (0.1-6 % of the face; centres sit in the smaller space above the
  floor). The miss is an irregular outline with a narrow centre (seed 1 stone 9,
  width 0.11), where the sphere rule is only approximate.
- **No open cracks.** The sources describe cracks healed by later agate or quartz
  and refilled with chalcedony or iron oxides [@goetze2020, §5.1; @mrozik2023, pp.
  14, 27]; open cracks were drawn as voids, black in the stand-in. They are no longer
  drawn (the old open share goes to clear fill: red stones 0.9 clear, 0.1 iron;
  others 0.6 clear, 0.4 iron). Open fill stays available as a setting.
- **Bent cracks.** Each crack bends along the normal by smooth world noise (its own
  stream, so previews match full runs): long cracks at a 3-6 mm scale, rim cracks at
  2-4 mm, amplitude 0.35-0.6 of the scale (steepest slope about 0.75). No source
  gives a crack's form; this is for the eye. A first draw at 0.15-0.3 of a 3-8 mm
  scale bent 30 mm traces by only 0.2-0.3 mm RMS: within one face the noise spreads
  less than its full range. Now the median trace bends 0.73 mm RMS from its best
  line (0.11-2.2 over the 12 cracks seen in sheets 1-6), so a few stay nearly
  straight. Positions are drawn as before; bends are drawn after, so the cracks lie
  where they did. Branching is not modelled.
- **Gentler bends (the owner, 2026-11-08: they bend too much; only very slight
  curves, rarely if ever switching direction).** The noise bend is replaced by one
  gentle curvature per crack: a shallow bowl (paraboloid) that rises by a sagitta at
  the crack's edge, 0.03-0.1 of its radius (steepest slope 0.2), so a trace in any
  face is one arc and never an S. Over the 12 cracks seen in sheets 1-6 the median
  trace now bends 0.23 mm RMS from its best line (0.13-0.67; was 0.73), and none
  shows an S-turn: at most 0.014 mm departs from the best parabola. Sheets 1 and 2
  redrawn: runs/variety-sheet-s10b/.
- **Checks with controls:** `crack_bend` (RMS offset of the trace's centre line from
  its best line, in 1 mm bins: a bent test crack, sagitta 1.5 mm, above 0.08 mm, the
  flat control below 0.03); `crack_wiggle`, the no-S-turn check (RMS offset from the
  best parabola, what one curve cannot follow: the bowl crack below 0.03 mm, a control
  bent by sine waves of 16 mm period above 0.2); a floor keeps a quartz-filled or quartz-lined centre (over 20 % of its
  share without a floor), while the same floor on a solid centre shows none;
  sheets draw floors at most 0.5 and no open crack.
- **Half the wobble on mild warps (the owner, 2026-11-08: seed 1 stone 1's
  grey-blue band droops at lower right).** agatesim's wobble is independent noise at
  each depth, so a wide band's two edges drift apart. Measured over sheets 1-6 as
  the 99th percentile of band width over the same stone cut without wobble, pixel by
  pixel (so outline and cut obliqueness cancel; floor stones left out, their flat
  bands defeat the measure): the 28 mild stones swell by a median 1.10 (max 1.22),
  1.03 (max 1.12) at half wobble, all 28 less. Crenulated and lobed warps are left
  as drawn: they sit at the no-fold clamp, and there agatesim's fade-in from the
  wall takes up the slope the wobble leaves unused, so a weaker wobble fades in more
  steeply. Halved, crenulated stones barely change (median 1.09 to 1.10) and lobed
  ones swell more near the wall (1.11 to 1.17, though less beyond 6 mm). A gentler
  fade-in would be an agatesim change, not made. A first measure (band width P99
  over its median, per face) mixed the wobble with the cut's own widening and was
  dropped. Mild wobble is drawn as before (same stream) and then halved
  (`effective`); one mild stone in 12 keeps its full wobble ("full wobble" on the
  sheet), chosen last in the sheet's plan so that all other draws stand.
  Check: `band_swelling`, the same measure: the test stone at half wobble (0.1 mm,
  4 mm scale) below 1.08, the control at the drooping stone's full strength (0.2
  mm) above 1.12 (measured 1.03 and 1.20). Sheets 1 and 2 redrawn:
  runs/variety-sheet-s10c/ The owner (2026-11-08): the bands
  look good enough.
- **Crack form and mix (the owner, 2026-11-08, judging seed 2 stone 5: two long
  cracks of different shades crossing near the centre).** Such cracks occur but are
  rare, since cracking that reaches the centre usually splits the stone; healed
  cracks in one stone usually share a shade; some are straight; many are short. The
  drawing was 1-3 cracks of 0.6-1.5 x the inradius within 0.3 x of the centre, each
  with its own fill, all bent. Numbers proposed and approved (all `effective`, for
  the eye; no source gives crack counts, lengths or fills for agates; shares among
  cracked stones, which stay 1-2 in 12 by the owner's choice): one fill per stone at
  the old odds (clear 0.6 / iron 0.4; on red stones 0.9 / 0.1) and one iron strength
  within 0.1; 1 cracked stone in 12 has one crack of the other fill; 1 crack in 3 is
  straight; partial cracks run in from the rim 0.2-0.5 x the inradius, and the
  1-4 mm rim nicks stay; long cracks (0.6-1.2 x the inradius, centred 0.2-0.6 x off
  the centre) on 4 cracked stones in 12, two of them on 1 in 12. Partial cracks were
  proposed at 1-4 per stone; measured on the cut faces of the 91 cracked stones in
  sheets 1-60 (solid and plane only), 31 % of faces then showed no trace, so the
  count was raised to 3-8 (owner told): 9 % of faces show none, 88 % show 1-4; the
  median trace is 0.14 of the face's width (was 0.72), 79 % under a quarter (was
  8 %); 26 % of faces show a trace over half the width (was 91 %) and 1 % two (was
  55 %). Only the fractures' draw changed, so the cracked stones' other draws and
  every other stone stand (seed 1 stone 10, seed 2 stone 5 changed).
  Checks with controls: `crack_traces` (trace length over the face's widest extent:
  a crack running a third of the way in from the rim gives 0.1-0.3, the control
  through the centre, wider than the stone, over 0.7); `crack_mix` over 600 draws
  (mixed fills within 0.04 of 1/12, iron spread at most 0.2 in a one-fill stone,
  straight within 0.05 of 1/3, long on 0.25-0.42, two long on 0.03-0.15), its
  control the old drawing (mixed over 0.3, spread over 0.2, none straight, long
  over 0.9). Sheets 1 and 2 and a strip of 12 cracked stones from seeds 3-11
  redrawn: runs/variety-sheet-s10d/.
- **The owner on the strip of cracked stones (2026-11-08).** Four changes, all
  `effective`: (1) the cutter avoids cutting along a crack, since a slab would split
  there and a crack near the face's plane shows as a wide stain; a cut now also
  meets no crack at under 30 degrees, counting a crack within its stain's reach of
  the plane (halo + 2 x pool) and less its largest bend (`crack_cut_angle`), chosen
  among the same 64 tries; thunder eggs, cut across the star, move the cut along
  the axis only; cracks the chosen cut still runs along are left out (that stone
  would have split on the saw). (2) Iron stain lowered from 0.4-1.0 to 0.25-0.6,
  and the iron fill's own trace from 0.9 to the stain + 0.2 (at most 0.8). (3) No
  crack enters an open cavity: on quartz-lined stones a crack that comes within
  0.5 mm (plus its largest bend) of the points more than the lining's depth from
  the nearest wall is redrawn, up to 20 times, then left out (`enters_cavity`).
  (4) Two long cracks only on 1 cracked stone in 36 (was 1 in 12); long cracks stay
  on 1 in 3. Measured on the 91 cracked stones of sheets 1-60: traces meeting the
  face at under 30 degrees 0 (13 before at under 20); long cracks entering the
  cavity of the 33 open-cavity stones 0 (5); stones with two long cracks 1 (8);
  faces showing no crack 23 % (9 %), since more cracks are left out; median trace
  0.14 of the face's width, as before. Every uncracked stone is unchanged; cracked
  stones differ only in their cracks and cut. Checks with controls: a crack in the
  cut's plane gives under 1 degree, one across it over 89, and `clean_cut` moves off
  the first; a crack through the centre of a quartz-lined stone enters its cavity
  (the control), a rim nick does not, and no crack drawn for it does; drawn stain
  stays at or under 0.7, the old 0.9 above. Seen: the wide pink and red patches on
  strip stones 2, 7, 9, 11 and 12, first read as crack smears, are mottle patches
  (s.8), not cracks; they stand until the owner judges them. Sheets 1, 2 and the
  strip redrawn: runs/variety-sheet-s10e/.
- **The owner on the cracks after these changes (2026-11-08): "pretty good".**
- **Mottle: rarer, mostly small, rims varied (the owner, 2026-11-08).** Asked: a
  third to a half as often; most patches small, a large one (at least a quarter of
  the face) on 15-25 % of mottled stones; dark rims on about 1 patch in 4, slightly
  lighter on the rest. Built (all `effective`): mottle is kept on 1-2 stones in 12
  (was 4), the first of the first draw's stones, whose other draws stand; measured
  over sheets 1-60, 83 mottled stones (was 240, 0.35 x). Noise scale turned out
  not to be patch size: patches come out about 3-4 x wider than the scale, and at
  the old scales (5 mm to 0.35 x the stone's size, cover 0.15-0.35) the largest
  patch on a face reached a quarter of it in about 30-50 % of faces (solid and
  plane only, 22 stones, 3 noise draws each). Now 1 mottled stone in 5 is drawn
  large (scale 0.2-0.35 x the stone's size, cover 0.35-0.5), the rest small
  (0.04-0.06 x, cover 0.15-0.3); after the cut, the first of up to 16 noise draws
  whose face shows its class is kept (small: largest patch under a quarter of the
  face; large: a quarter to 0.6, beyond which a patch reads as a film over the
  face; one such film was seen). Measured: 14 % drawn large (12 of 83; 20 %
  expected), every small face under 0.22 and every large one 0.26-0.6. Rims: a
  second noise at twice the patch scale marks about a quarter of the stone dark
  (rim darkening as before); elsewhere the rim darkens 0.4 as much. Per stone the
  dark share runs 0.08-0.44 (12 stones). Checks with controls: `largest_patch` (a
  small pattern under 0.25, the control, a large one, above; sheets 1-6 draw 1-2
  mottled stones each, each showing its class, none above 0.6); `rim_darkness` (a
  test stone dark on 0.1-0.4 of points, the control, the first drawing, on all).
  Sheets 1, 2 and a strip of 9 small and 3 large mottled stones:
  runs/variety-sheet-s10f/.
- **Soft and hard mottle edges (the owner, 2026-11-08).** Not every patch should
  have a hard outline: about half of mottled stones soft, graded patches, the rest
  hard, and only about 5 % both. Built (all `effective`): a per-stone edge, hard
  0.475, soft 0.475, mixed 0.05, drawn last in the recipe's refine stream so that
  all else stands. A soft patch has no rim; its stain rises to 0.7 x strength over
  0.5-1.0 noise units centred on the patch's edge (about 0.4-0.8 x the noise scale
  in mm), so it keeps its size. A first ramp that started at the edge left soft
  patches smaller and fainter than hard ones and was dropped. On a mixed stone a
  noise at twice the patch scale makes about half its patches soft. Over 200
  sheets: 296 mottled stones, 150 hard, 124 soft, 22 mixed; nothing else changed.
  Check: `graded_share` (among patch points, the share whose stain lies between
  0.05 and 0.55 x strength): the control, a hard stone, 0; soft over 0.4; mixed in
  between, with 0.25-0.75 of points soft. Full renders read the same layer fields,
  once the renderer applies the layers (the open item on tgrid.py). Strip of 5
  hard, 5 soft and 2 mixed stones, and sheets 1, 2: runs/variety-sheet-s10g/.
- **Seen, open:** a clear-filled crack shows as a dark grey line in the stand-in
  (seed 2 stone 5); whether it reads darker than a real healed crack is for the
  render, not the stand-in.
- **Thunder-egg arms (2026-11-08, owner's note in s.9):** two arm styles, offered as
  options because the sources disagree. *Angular* (the first drawing): straight
  arms, as sections through Ross's rupture model of shear planes [@ross1941, p.
  729]. *Sinuous* (new): lobed, winding arms with rounded embayments, after the
  owner's private reference thunderegg-1 (one specimen; *plausible* as a look,
  never evidence for a mechanism). `nodule.thunderegg` draws a sinuous star as discs
  along winding arm midlines (heading off the radial by a sum of two sines, capped
  at 70 degrees so arms never turn back) around a lumpy body, with swelling widths
  and 0-2 side buds per arm; it is rasterised on a grid fixed in shape units,
  blurred to round the embayments, traced and checked by agatesim's
  `outline_problem`. Share at first 3 in 4 sinuous; the owner then chose sinuous
  for every stone (below). All sizes `effective`. Check:
  `arm_meander`, the upper quartile over arms of the largest sideways offset of an
  arm's midline from its best straight line, over the radial span followed; angular
  0.008-0.031 and sinuous 0.046-0.165 over 80 draws each, threshold 0.04; negative
  control: the angular star. Cost: 0.7 s an outline; stand-in cuts of the 8
  thunder eggs on sheets 1-4 took 1-5 s each, under 0.22 GB. Thunder-egg stones
  draw their other features afresh (the outline uses the stream differently).
  Cuts: runs/thunderegg-arms/ (cuts-1-2-3-4.png, outlines.png). Not yet: a cut
  still crosses an extruded prism, so every section shows the same outline
  ([@ross1941, p. 731]: the star depends on where the cut falls).
- **Owner's verdicts (2026-11-08):** winding arms "look very good"; every thunder
  egg winds, none straight (angular stays only as the check's negative control).
  Mottle (s10g) is "good enough" for now: it should improve with fuller renders
  in later stages, so it is closed for this round.
- **Spread of winding arms (owner, 2026-11-08: moderately more and less bending
  and thickness; a count from one fewer to two more, see the correction below):**
  each stone draws an `ArmShape`: bend x0.7-1.4 and width x0.75-1.3
  (log-uniform); first drawn as side lobes added to each arm's 0-2 by -1 to +2,
  now arms (below); all `effective`. Each arm and each
  lobe draws from its own spawned stream, so changing the lobe count leaves the
  other arms as they were (tested). Extremes stay sinuous: over 40 draws each,
  `arm_meander` 0.048 at least (least bend, thickest arms) against the 0.04
  threshold. Sheets: runs/thunderegg-arms/spread-one-at-a-time.png (one setting
  at a time on two stars), spread-random-12.png; cuts-1-2-3-4.png redrawn.
- **Correction, arms not lobes (owner, 2026-11-08):** the "one fewer to two more"
  was meant for the number of arms. Side lobes are back to 0-2 per arm; a sinuous
  star now has 5-8 arms plus an extra -1 to +2 drawn per stone (4-10, the ends
  rare; `ArmShape.extra_arms`).
- **Jagged mode (owner's choice 2026-11-08, after [@goetze2023]):** the paper's
  text gives no cavity geometry, but its Fig. 2 (p. 26; five Saxon lithophysae,
  checked by Claude) shows irregular, asymmetric cores with angular, stepped walls,
  3 to about 6 arms and bands that keep the wall's corners; no smooth winding arms.
  Sources disagree (the owner's specimen winds), so both are drawn: jagged on 1 in
  3 thunder eggs (`effective`; Claude's suggested share, accepted), sinuous on the
  rest. A jagged core is an irregular 5-8-cornered polygon (radius 0.32-0.52 of
  the reach) with 3-6 kinked wedges (kink up to 35 degrees, broad at the body),
  traced, cut into straight runs of 0.06-0.12, each corner pushed in or out by up
  to 0.02-0.04 (shape units; all `effective`), halving the push if the outline
  would cross itself. Check: `wall_corners`, the share of the wall's turning at
  corners sharper than 15 degrees (outline resampled at 0.005); over 40 draws
  each: jagged 0.70-0.87, sinuous 0.07-0.36, angular 0.40-0.62; threshold 0.66;
  negative controls: the sinuous and angular stars. Cost: 0.2 s an outline; cuts
  1-10 s to build, under 0.22 GB. Sheets: runs/thunderegg-arms/jagged-12.png,
  cuts-1-2-3-4-5-6.png (4 of 12 jagged). Later (owner agreed): the paper's rind,
  host speckle, pigment bleeding into the host, central druse, host crack nets.
- **Owner's verdict (2026-11-08):** the jagged cores "look good"; closed for now.
  For the open 3D cavity, the owner notes that a cavity varying with depth means a
  cut will sometimes gain or lose arms depending on where it falls (as Ross's
  cut-position remark implies [@ross1941, p. 731]); a 3D version must allow that.

### 11. Band-following dust: reading and plan (2026-11-09)

First item of the owner's next list (dust, then plumes and moss, then depth through
clear bands). Reading by helper agents, 2026-11-08/09 (notes/n1-reading/
dust-reading.md, wang2025.md, pabian1994-dust.md; gitignored); grades per ADR 0002.

- **What it is:** fine hematite (red) and goethite (yellow), particles under 10 µm
  in laminated or lenticular agglomerates [@goetze2020, p. 27]; in Nanhong agate,
  aggregates of nano-hematite in silica, mean 4.94 µm, 95 % within 1.0-10.2 µm,
  largest 43 µm [@wang2025, Fig. 7a]; Fe-Ti oxides 2-5 µm in fire agate
  [@natkaniec2020, p. 10] (*verified*).
- **Two populations:** fine dust along the crystallisation front; coarse aggregates
  up to several 100 µm at random in the inner chalcedony [@goetze2009, pp. 680-81;
  @goetze2020, p. 27]; orange-brown bubbles up to 100 µm at random within a band
  [@mrozik2023] (*verified*).
- **Where:** "for the most part, on the band boundaries" [@howard2018, Fig. 5];
  wall-parallel along crystallisation fronts, in only a small part of each band
  [@mrozik2023, p. 13, Fig. 14]; a brown belt 0.06 mm wide in a 0.3 mm layer
  [@wang1990, Fig. 2] (*verified*). White bands are mostly clean [@howard2018,
  Fig. 5; @mrozik2023, p. 23]; the quartz centre nearly so (Fe 7 against 483 ppm in
  chalcedony [@goetze2009]; 78-141 against 262-969 ppm [@mrozik2023, Table 5])
  (*verified*).
- **Along a band:** continuous lines and dotted strings both occur, and the amount
  varies "from band to band and along individual bands" [@howard2018, p. 104]
  (*verified*, qualitative). Red bands with fine, even dust look evenly coloured
  [@mrozik2023, p. 23; @wang2025, Fig. 6c].
- **Colour:** saturation rises with the pigment's projected area (S = 38.04 + 2.57 ×
  area %, R² 0.76, area 2-10 %), not with particle count or size [@wang2025,
  Figs 6-7] (*verified* for one locality; HSB from micrographs, not CIE Lab).
- **Floor:** earthy iron debris may settle on the vesicle floor before banding
  [@pabian1994, pp. 13-14, Fig. 26] (*plausible*; a collectors' book). No source
  describes dust lines in floor bands; Fe is below detection in Brazilian floor bands
  [@moxon2013], while "heavily pigmented" ones are possible [@heaney1993].
- **At render scale** (Claude's inference, *plausible*): a pixel covers about
  50-100 µm, so single particles and most lines (4-30 µm, measured on figures) are
  unresolved and show as a faint tinted line or a tinted band. Visible specks are
  about 25-70 µm on a polished face [@mrozik2023, Fig. 14b] and 0.07-0.4 mm under a
  hand lens [@howard2018, Fig. 5] (sizes measured on figures: *speculative*).
- **Mechanism** (context only, never tested by renders): iron rejected ahead of the
  front [@goetze2009, p. 686], or trapped when growth is fastest [@wang1990,
  p. 1635]; pigment supply varying independently of the micro-zoning
  [@frondel1985, p. 977].

**Gaps, so assumptions** (memory: state assumptions when sourcing stalls): no source
gives the share of bands carrying dust, lines per mm, number density in a section,
the length of dust patches along a band (one figure: at least 10 mm), or which side
of a layer the dust takes.

**Plan: four forms, varied per stone** (all `effective`; the numbers are starting
assumptions for look sheets, to be set by the owner's eye):

1. *Tinted band:* pigment even through a band. The band iron already does this; add
   nothing unless the look needs it.
2. *Dust line:* a thin strip at a band boundary, about 0.03-0.06 mm, drawn by area
   coverage so it can be thinner than a pixel; the stone's pigment, darker and more
   saturated; its strength varies along the band with noise in world coordinates
   (patches about 5-20 mm long).
3. *Speck string:* visible specks 0.03-0.3 mm, oval and drawn out along the band, in
   loose clustered rows on the same boundaries.
4. *Coarse blobs:* rounded aggregates 0.1-0.5 mm at random in the inner chalcedony,
   not on bands.

Rules: no dust on white bands, in the quartz centre or in floor bands; fewer dusty
bands toward the centre; dust colour from the stone's palette (red stones red,
others red, yellow or brown). Starting shares: dust lines on 1 stone in 2, on 20-40 %
of its coloured band boundaries; speck strings on 1 in 4; coarse blobs on 1 in 4.
No source gives these frequencies; coarse host debris is in over 90 % or under 5 %
of stones depending on locality [@pabian1994, Table 1], so a per-stone switch fits.
No stone gets every form (memory: realism is interplay).

**Options for the owner** (sources disagree or are silent):

- **A. Which side of a band boundary.** A1: the end of a layer, its inner, younger
  side (rejection [@goetze2009]). A2: the start of a fibrous layer [@pabian1994,
  Fig. 22] (trapping [@wang1990]). A3: either, drawn per stone. *Recommended: A3*;
  the sources split, and at render scale the side shows only as which shade the
  line borders.
- **B. Pore belts** (faint whitish haze lines in reflected light [@folk1952]). B1:
  leave out, as their nature is disputed (no 0.1 µm pores by TEM [@graetsch1985,
  p. 305]; no visible fluid inclusions in chalcedony [@goetze2020, p. 22]). B2: rare.
  *Recommended: B1.*
- **C. Settled iron debris on the cavity floor.** C1: later, with the floor work.
  C2: now, on 1 stone in 12. *Recommended: C1.*
- **Chosen (owner, 2026-11-09): A3, B1, C1**, as recommended.

**Checks:** `dust_on_bands`, the share of dust within a short distance of a band
boundary. Negative control 1: the same dust at random positions must fail. Negative
control 2: a stone with band contrast but no dust must not be flagged as dusty.
Specks are drawn per world-fixed cell from spawned streams, so previews match full
runs. **Cost:** measured as built; stand-in cuts about 6 s each. The owner sees
look sheets before any statistic or render (memory: owner's eye over statistics).

### 12. Dust built; look sheet to the owner (2026-11-09)

Built as planned in s.11 (`nodule.dust`; wired into `VarietyLayers`). How, where
s.11 left it open (all `effective` or assumptions):

- **Boundaries are exact.** agatesim's bands are periods of growth time; a table of
  each growth step's band, probed once per stone along 26 rays (0.2-0.7 s), gives
  every boundary as a step. A point's distance to it is its step coordinate less the
  boundary's, times the step's thickness (at the walls; wobble's slope cap keeps the
  error under about a third).
- **Area coverage.** Cuts pass their pixel size to the layers (`Specimen.cut`); a
  dust line covers the share of a pixel its strip covers, so a 0.04 mm line shows
  faintly in a 0.07 mm pixel. Dust per unit area agrees within 5 % from 0.02 to
  0.08 mm pixels (test); pixel values differ by design.
- **Specks and blobs** sit in world-fixed cells (0.15 and 0.25 mm), each cell's draws
  a counter-based hash of a spawned stream's seed and the cell: the same in any batch
  or resolution (test). A speck is a ball of its half-length cut to a strip of twice
  its half-thickness on the host side of the boundary: an oval drawn out along the
  band. Clusters thin the cells by smooth noise (1-3 mm).
- **White bands:** pigment white, or milky (translucency < 0.35) with iron < 0.15.
- **Pigment** from the stone's palette: hematite in it, red dust; otherwise hematite,
  goethite or iron at random.
- **A dusty stone shows dust:** with few bands the draw can pick no boundary; then
  the eligible boundary with the lowest draw is kept.
- **Sheet shares:** lines 6, specks 3, blobs 3 per 12 stones; never all three on a
  stone; no lines or specks on the pebble.

**Check** (`dust_on_bands`): dust read from the cut alone (impurity above its band's
median). On sheet 1's stones: share of dust within 0.1 mm of a boundary 0.97-1.0,
enrichment over area 2.4-3.9; negative control 1 (the same dust shuffled) 0.98-1.02,
not flagged; negative control 2 (bands without dust) not flagged.

**Cost** (measured): build +0.2-0.7 s, a 0.07 mm cut +0-0.4 s, peak under 0.4 GB.

**Look sheet** (runs/dust-sheet-s11/, `python -m nodule.sheet --seed 1 --dust`): each
dusty stone whole, then 12 mm without and with dust (0.03 mm pixels) and 4 mm with
dust (0.01 mm), windows where its dust is densest (so not typical views). Dust is
invisible at whole-stone scale, as expected from s.11. Hematite blobs on carnelian
barely show (red on red).

**Owner's verdict (2026-11-09):** "the dust looks good, for now". Kept as built;
closed for now, like mottle (s.10).

## Limits

- One judge and one stone stand behind the trial; the list is what that judge named,
  not a complete account of agate appearance (*plausible*).
- Settled limits stand (ADR 0006 s.10): photos give no counts, size distributions or
  connectivity, so features are judged by whether renders reproduce what photos show.
- Appearance only: none of this is evidence for a formation mechanism (ADR 0002).

## Alternatives considered

- **One feature at a time, trial after each.** Rejected by the owner: no single
  feature closes the gap, and each trial costs the owner's time.
- **Every feature on every stone.** Rejected: real stones vary; a full set on each
  would be its own tell.
- **Raise the render sample cap first.** Not now: no note names grain or noise
  (ADR 0008 s.5a).

## Consequences

- The roadmap's N3 generators (inclusions, defects, colour) start now, alongside N2's
  open blind test. The full test (ADR 0008 s.5) waits for the stopping rule.
- Next: reading for features 1 to 5, then the first rough sheet.
