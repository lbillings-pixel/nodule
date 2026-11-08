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
