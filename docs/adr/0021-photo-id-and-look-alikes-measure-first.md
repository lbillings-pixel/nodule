# 21. Photo ID and look-alike stones: measure today's tools first

- Status: proposed. The owner chose "measure first" (2026-11-21); the rules of part 1 are
  fixed here before anything runs.
- Date: 2026-11-21
- Builds on: [ADR 0003](0003-scope-a-3d-specimen-simulator.md) (goals),
  [ADR 0002](0002-founding-nodule.md) (graded claims),
  [ADR 0006](0006-band-counts-as-censored-records.md) section 10 (photo band counts are
  censored), [ideas doc](../ideas-2026-11.md) section 4 item 3 (photo to recipe),
  agatesim ADR 0067 (the form classifier), agatesim ADR 0073 section 4 (tier-2 photos are
  never tuned on).

## Context

After item 20 closed, the owner set a new direction (2026-11-21). Nodule should become a
stand-alone package that others can run (an executable on supported hardware and
systems), polished and presented, with features for its users. Above all, a user should be
able to upload a photo and have Nodule try to identify it and build a look-alike 3D stone.
Bands as signals, the fresh blind trial (ADR 0018 section 5) and the realism score (N3) are
deferred, not dropped. Platforms are open (owner: "not sure yet"); an options record
follows.

What exists (read in the code, 2026-11-21):

- agatesim's browser app: photo upload; a quality gate (agatesim ADR 0064); forms named by
  the archetype criteria (`specimen/identify.py`: a pass is reliable, a fail says little)
  and by a learned classifier (DINOv2-small through ONNX Runtime, one logistic regression
  per form; agatesim ADR 0067); a 2D replica whose band statistics are fitted to the
  photo's inside its outline (`specimen/fit.py`, agatesim ADR 0011).
- Nodule: outline shape (C1-C3) and band counts and spacing with their resolving limit
  (C6, C10) measured on the library photos (`nodule.tags`); a recipe of many varied
  features (ADR 0018) and a renderer (ADR 0019).
- Nothing sets Nodule's 3D recipe from a photo. Most of its features (cracks, rind,
  crystals, dendrites, plumes in Nodule's sense) have no detector.

Limits that hold whatever is built:

- One photo shows one face. A look-alike is a stone that could show such a face, not that
  stone, and never evidence of how it formed (ADR 0002).
- Photos censor fine bands (ADR 0006 section 10).
- The image model behind the classifier told renders from photos at AUC 0.999 (the critic
  test, 2026-11-06), so a test on renders alone can flatter it.

## Decision: measure first

Score today's tools against answer keys that already exist, before building anything. Part
1 runs first. Parts 2 and 3 are designed after it.

### Part 1. ID on real photos

The answer key is the owner's labels of the library photos in agatesim
(`references/library-labels.toml`, forms and features, labelled in agatesim's labelling
screen on 2026-10-24 and 2026-10-29). Only photos labelled as showing forms count.

- **1a. Forms, cross-validated.** agatesim's `cross_validate` with ADR 0067's settings
  unchanged: 5 folds by specimen group, seed 0, simulated slabs at weight 0.1. It runs on
  today's tuning labels, which leave out the never-tuned tier-2 photos as agatesim does,
  and without the owner's own photos. These are open-licence photos only, so this is the
  figure a shipped app could claim. ADR 0067 chose its settings on such folds, so the
  figure stays somewhat optimistic.
- **1b. Forms, held out.** The saved classifier (2026-10-24) on the friend's 31 photos of
  10 stones, against the owner's labels (agatesim 55b30c7). It was saved before those
  photos joined the library, and agatesim never tunes on them.
- **1c. Features, a new probe, reported apart.** The same embeddings and regression
  settings, one regression per library feature label, cross-validated as in 1a, with no
  simulated slabs. The labels are quartz core, open cavity, host rock, rough, inclusions,
  polished and iridescence. It asks whether this cheap route can name the centre type and
  host rock at all. It is not one of today's tools.

**Rules, fixed now:**

- **Scores.** For each form or feature: precision (right / reported) and recall (found /
  labelled), each set against its base rate (its share of the photos). The overall top
  answer is set against always naming the most common form, recomputed on the same photos.
  Over all 261 photos labelled with forms, fortification is on 220 (84 %).
- **Minimum.** A verdict needs at least 10 specimens (groups) that show the form or
  feature. Fewer reads "too few to judge".
- **Verdicts.** "Works now": held-out precision at least 0.8 and recall at least 0.5, both
  above the base rate. "Partly": precision at least 0.8 with lower recall. Otherwise "no".
  These thresholds are an assumption, and the owner may change them.
- **1b.** With 10 stones it gives counts, not verdicts.
- **Negative control.** 1a and 1c rerun with the label sets shuffled among specimen groups
  (seed 1). Precision must fall to about the base rate. If a shuffled run "works", the
  scoring is broken and no verdict stands.
- **Grade.** *Verified* as counts, for these photos under these conditions. Appearance
  only (ADR 0002).
- **No writes into agatesim.** The saved classifier and the embeddings cache are copied
  into `runs/photo-id-measure/` and used from there. No photo or crop is written.

### Part 2. Read-back on Nodule's own stones (next)

The answer key is the recipe. Part 2 uses stones whose recipe was stored with them, or
else fresh stand-in cuts from a new draw (time flagged first). Each is measured through
the photo-like path at a photo's resolution, and band counts are scored as counts coarser
than the limit (ADR 0006 section 10). It is designed after part 1, once agatesim's recorded
results (the ADR 0011 fit, the rim finder) are collected.

### Part 3. The owner's new photos (a fresh test)

The photos go in a subfolder of agatesim's `specimens/`, outside the app's training. They
are scored as they come and again after any build, and never tuned on. The owner chooses
whether Claude views them.

## Consequences

- A one-page summary goes to the owner: which parts of ID and look-alikes work now, which
  work partly, and which need building. The plan follows, with an options record on
  platforms, where the app lives and signing.
- Nodule's own features have no real-photo key, except its eye labels (C5 rind, C22
  fracture traces). Helpers proposed those labels, so a score against them is agreement,
  not accuracy.
