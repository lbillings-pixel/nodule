# 18. The variety gap: broad mock-ups of many features, varied per stone

- Status: accepted (owner, 2026-11-08; drafted from their choices of 2026-11-06).
  Fractures: option C (section 4).
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
- **Before any render:** the GPU growth grid (`nodule.render.tgrid`) reads agatesim
  directly and must apply the same layers, and the quartz tint must reach the
  materials. No render or statistics before the owner has seen the sheets (s.1).

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
