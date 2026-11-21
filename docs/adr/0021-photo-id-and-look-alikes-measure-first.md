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
- agatesim's roadmap marks its replication stage (M35: procedural look-alikes, with
  mechanistic fits as fidelity tests) "moved" to Nodule (agatesim ADR 0072). It was never
  started in agatesim, and Nodule has no module for it (its module list, 2026-11-21). A
  survey of agatesim's ADRs found no record of how well the 2D fit's replicas match.
  agatesim's rim finder drew outlines that followed the rim on 75 of 88 held-out photos
  (85 %, below its 90 % bar), so an outline is measured only after a helper's yes
  (agatesim ADR 0075).

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
- **Verdicts.** "Works now": held-out precision at least 0.8 and above the base rate, and
  recall at least 0.5. "Partly": the same precision with lower recall. Otherwise "no".
  Answering "yes" to every photo scores a precision equal to the base rate, so precision
  must beat it; recall needs no such baseline. These thresholds are an assumption, and the
  owner may change them. (This wording was corrected before any run: the first draft also
  asked recall to beat the base rate, which tests nothing.)
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

### Part 3. A private photo set (a fresh test, then training)

The owner supplies these photos privately (2026-11-21). They never enter a repository, a
commit or a distribution, and the records give only aggregate figures, never names or
descriptions. Claude may view them, and what they teach may be used. Nothing has seen
them yet, so they are scored first as a fresh test of today's tools. Then they are split
by stone: part becomes training material, and the rest stays a sealed test that nothing
is tuned on.

### Part 1 results (2026-11-21)

The run took 6.4 s on the processor, with a peak of 0.28 GB. agatesim was left untouched:
its status is clean, and both model files are unchanged since 2026-10-24 16:13. The
record is `runs/photo-id-measure/part1.json` (local).

**The negative control failed, so no verdict of part 1 stands.** With the label sets
shuffled, fortification still "worked": precision 188/218 (0.862) against a base rate of
197/230 (0.857). The verdict rule failed, not the scoring. Fortification is on more than 4
photos in 5, so the 0.8 floor is no bar, and "above the base rate" was passed by less than
one photo. The counts show the skill that the rule cannot see: the real run leaves
fortification off 30 of the 41 photos without it, the shuffled run off only 3 of 33. The
counts below stand, *verified* as counts for these photos.

Next comes a dated post-hoc amendment. It replaces only the base-rate clause, with a null
from many shuffles that move whole specimens, since photos of one stone move together. A
shuffle changes each form's photo count (eye went from 27 to 38 photos in the seed-1
run), so each form's margin over its own base rate is compared with the margins the
shuffles reach. The amendment also settles how forms marked unsure are scored (below).
The 0.8 and 0.5 floors and the 10-specimen minimum stay. The amendment is committed before
the rerun.

**1a, forms (230 photos, 178 specimens).** The top answer is among the labels for 0.909
of photos, against 0.822 for always naming fortification. This is not comparable with
ADR 0067's 87 % against 70 %: the labels have changed since (below), and the owner's own
photos are left out here.

| Form | Specimens | Right / reported | Found / labelled |
|---|---|---|---|
| fortification | 154 | 186 / 197 | 186 / 189 |
| waterline | 25 | 16 / 19 | 16 / 32 |
| fortification-waterline | 21 | 14 / 14 | 14 / 26 |
| botryoidal | 14 | 8 / 9 | 8 / 16 |
| moss | 38 | 31 / 40 | 31 / 53 |
| quartz-geode | 38 | 23 / 45 | 23 / 56 |
| eye | 23 | 9 / 18 | 9 / 27 |
| tube | 18 | 1 / 8 | 1 / 21 |
| iris, plume, shadow, stick, seam | 2 to 6 each | iris 3 / 4, stick 0 / 1, the rest never reported | too few to judge |

Waterline (16 of 32) and botryoidal (8 of 16) sit exactly on the recall line: one photo
would flip either.

**1b, held out (31 photos, 10 stones).** All 31 are labelled fortification, so the top
answer, always right, says nothing. The secondary forms:

- quartz-geode, on 14 photos of 4 stones: reported 7 times, right 3;
- eye, on 7 photos of 3 stones: reported 2 times, right 0;
- moss, on 2 photos of 1 stone: never reported.

**1c, the feature probe (230 photos).** Right of reported, then found of labelled:

| Feature | Right / reported | Found / labelled |
|---|---|---|
| rough | 17 / 21 | 17 / 25 |
| polished | 12 / 17 | 12 / 21 |
| quartz core | 49 / 85 | 49 / 85 |
| open cavity | 24 / 47 | 24 / 51 |
| host rock | 10 / 18 | 10 / 25 |
| inclusions | 4 / 12 | 4 / 21 |
| iridescence (6 specimens, too few) | 0 / 2 | 0 / 7 |

In the shuffled run, open cavity scored 17 of 38 against a base rate of 0.25. One seed
cannot show the null's spread; the amendment's many shuffles will.

**Forms marked unsure.** The scoring reads only each photo's forms and features, so one
the owner marked unsure counts as absent, as agatesim's classifier reads it (agatesim ADR
0071). Of the 230 photos, 11 carry unsure entries: tube 3, inclusions 3, plume 2, moss 2,
quartz core 2, host rock 2, open cavity 1. Up to 3 of the 7 wrong tube reports may
therefore fall on photos marked "unsure: tube", and the other counts may shift by up to
their unsure entries.

**The key's history, corrected.** The saved classifier's training counts match the
labels as of agatesim 67b0246 (2026-10-24: tube 26, moss 55, plume 9). Today's tuning
labels have tube 21, moss 53 and plume 5, because the labels were edited on 2026-10-25
(agatesim 443af3f, ADR 0071). The key is therefore the owner's labels of 10-24 and 10-29
as edited on 10-25, and the saved classifier saw the earlier version.

## Consequences

- A one-page summary goes to the owner: which parts of ID and look-alikes work now, which
  work partly, and which need building. The plan follows, with an options record on
  platforms, where the app lives and signing.
- Nodule's own features have no real-photo key, except its eye labels (C5 rind, C22
  fracture traces). Helpers proposed those labels, so a score against them is agreement,
  not accuracy.
