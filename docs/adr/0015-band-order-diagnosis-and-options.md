# 15. Band order (C16, C14, C7): what makes the stone too orderly, and what to do

- Status: accepted (owner, 2026-11-03): 15a B, then C, built modularly
  so that B carries what C needs (section 5). Recommended was A.
- Date: 2026-11-03
- Builds on: [ADR 0004](0004-characterization-schema-core-set.md) (C7, C14,
  C16 and their controls), [ADR 0006](0006-band-counts-as-censored-records.md)
  section 10 (photo statistics describe the photo; realism is tested by
  render-and-compare plus the owner's blind judgement),
  [ADR 0008](0008-set-level-render-and-compare.md) (set-level compare;
  section 5, the blind test), [ADR 0011](0011-band-regularity-for-the-stone.md)
  (band regularity 0.7 / 0.7; C16 named as an open gap),
  [ADR 0014](0014-optics-tuning-step-4-colour.md) section 4o (the 100-cut
  run with the defaults) and [ADR 0002](0002-founding-nodule.md) (graded
  claims).

## Context

- **The gap.** In the 100-cut stone run with the defaults (ADR 0014 4o),
  the mean structure-tensor coherence C16 is 0.90 on every cut. That is the
  photos' 98th percentile (half A p50 / p95 0.67 / 0.84, half B 0.60 /
  0.87). It stayed at 0.90–0.91 through the optics steps (ADRs 0010–0014).
  C16 measures how parallel the bands run at the scale of one couplet
  (`nodule.orientation`), not how sharp their edges are. The joint compare
  stays within both halves.
- **What tracks C16 in the photos** (Spearman, 97 reflected photos against
  the 100 cuts; *verified*, descriptive):
  - The banded share: r 0.95. The stone is 0.99 banded (93rd percentile).
  - Defect density C14, band ends and junctions per period²: r −0.70. The
    stone's is 0.235, at the 4th percentile; the photos' median is 0.51.
  - Among the 25 fully banded photos (banded share ≥ 0.95), C16 has a
    median of 0.81 and the stone is still at the 92nd percentile. There
    r(C16, C14) is −0.71.
  - Band-thickness spread C7 is also low, at the 4th percentile (0.53
    against 0.62). No ADR had recorded this gap until now.
- **Not the imaging.** The stone's C14 does not follow blur, noise,
  resolution or band spacing in pixels (r −0.04 to 0.11). The 31
  noise-matched cuts give 0.219 against 0.237 for the rest. The highest of
  the 100 cuts (0.51) only reaches the photos' median. The gap lies in what
  the stone shows at photo resolution. Per ADR 0006 section 10, it does not
  matter whether a photo's band ends are real terminations or fine bands
  fading below its resolution: the render must reproduce the image
  statistic. That question is not reopened here.
- **Why the model cannot do it** (*verified*, from the code).
  - agatesim's growth model gives each point its band from the distance to
    the wall plus one smooth wobble (agatesim `volume/growth.py`,
    `d + wobble × taper`, then one band table for the whole stone).
  - Bands are therefore contours of one smooth function. They cannot end,
    split or merge, and each keeps its thickness and its material along its
    whole length.
  - nodule's renderer also keys every material to the growth step alone
    (`render.materials.BandRecord`, ADR 0007 section 3). Even if agatesim
    varied a band along its length, the renderer would refuse the stone.
- **What the sources say.** Owner-fetched papers were read in full; the
  notes are in notes/n2-reading/step5-band-defects.md.
  - *Random breaks are absent.* "These defects are absent in agates"
    [@cartwright2026, §5.1.1, against Liesegang rings]. "Fractures through
    the bands are never seen" [@howard2018, p. 102]. Away from channels,
    layers grow "in relatively uniform thickness" [@walger2009, p. 128].
    Bands keep "the same width, consistency and colour" across sector
    boundaries [@howard2018, p. 104]. *verified* as the authors' statements.
  - *Bands do end by thinning at particular places.* At an infiltration
    channel the wall layers thin "from the normal thickness to zero" toward
    one focus [@walger2009, pp. 116, 136]. Such channels are absent below
    cherry size, "often one, rarely more" in a fist-sized amygdule, and
    "frequently several" above 1 dm³ (p. 128). They are often missed by a
    random section (p. 127). At escape tubes bands narrow "until … the
    banded structure is lost" [@howard2018, p. 99], and they are thinner at
    the apex of indentations (p. 98). *verified*, with no counts per area.
  - *A band's look varies along it.* Band widths vary within sectors, and
    inclusions vary "along individual bands" [@howard2018, pp. 99, 104].
    One iris strip has sectors that are "perfectly dark", "faintly banded"
    or "quite bright" [@raman1953]. Hematite gathers in "laminated and
    lenticular agglomerates" along the banding [@goetze2020, §4.3].
    *verified*, qualitative only.
  - *Unbanded areas:* a 1–2 mm clear outer layer [@howard2018, p. 99], and
    the quartz centre with band fragments in it (p. 100). No source gives
    an area share.
  - No source gives a number a generator could use directly: no lateral
    extent per band, no frequency per area, no unbanded share.
- **Cheap tests** (6 cuts each, half A looks, held-out B reported; about
  8 minutes and up to 1.0 GB each). These are exploratory, not tuning.
  - *Wavier bands, settings only:* wobble 0.36 mm at 4 mm, the most the
    no-folding cap of 0.5 allows (ADR 0005 section 3). The stone already
    uses about half of the allowed slope. C16 0.89, C14 0.20, C7 0.53:
    unchanged. Closed.
  - *Bands faded in patches, applied to the rendered image* (an image-level
    stand-in; `variants/fade_check.py`):
    - 90% fade in about 1 mm patches: C14 0.38 (15th–17th percentile), C7
      0.62 (40th–44th), C16 0.86 (92nd–96th), banded share 0.98.
    - Full fade in about 3 mm patches: C14 0.32, C7 0.57, C16 0.86, banded
      share 0.98. Smooth areas with no bands still read as coherent, since
      a gentle brightness slope has one direction.
  - **The owner judged every faded image obviously fake, and the unfaded
    renders much more like real agate, despite their statistics**
    (2026-11-03). The numbers moved toward the photos while the look moved
    away from them. Under ADR 0006 section 10 the owner's judgement is half
    of the realism test, so the stand-in is rejected as a direction.

## Decision

### 1. What is recorded now (whatever is chosen)

- C16 (98th percentile), C14 (4th) and C7 (4th) are recorded as the
  stone's open band-order gaps. The joint compare is within A and B.
- The cause in the model is recorded as above: bands are level sets of one
  smooth function, and the renderer's materials depend on growth step
  alone.
- **A rule from the owner's judgement.** No fix for these gaps is accepted
  on statistics alone. Each candidate is first shown to the owner as look
  cuts beside the current renders. Only a candidate that does not look
  more fake goes on to a statistical check.

### 2. Choice for the owner

**15a. What to do about band order now.**

- *(Recommended)* **A. Freeze and run the blind test.** Keep the stone as
  it is, with tuning frozen as ADR 0008 section 5 requires, and run the
  owner's blind real-or-synthetic test (200 trials plus 20 catch trials,
  four sessions of about 10–15 minutes). Its result decides whether band
  order matters to the eye.
  - If the owner tells synthetic from real above chance, the trials show
    whether band order is what gives it away. Then B or C follows, guided
    by that.
  - If not, the gaps stay recorded as statistical differences that the eye
    does not use.
  - Why: the owner's eye already prefers the current renders. The tested
    fix worsened the look. The sources give no numbers to build from, so
    any build would be tuned to these very statistics.
  - Cost: building the blind-test tool (crops, pairing, scoring, locally
    only; photos never leave the PC) and the owner's time.
- **B. Sourced local features in agatesim.** Build the features the
  sources describe:
  - fans of bands thinning to zero at a few channel foci [@walger2009];
  - bands thinning toward escape tubes [@howard2018]. These are channels
    through the wall, which [@goetze2020, §5.5] names together with
    infiltration channels. They are not the stone's filament tubes, which
    later growth lines;
  - a clear unbanded outer layer of 1–2 mm [@howard2018].
  Each needs agatesim code with its own tests and ADR, and a new pin here.
  Every amplitude and count is `effective`. Expected effect on C14 and C16:
  small, since the features are rare and local (*speculative*). Judged by
  the owner's eye first (section 1).
- **C. Band material that varies along each band.** Pigment lenses and
  translucency that change along a band, in 3D [@howard2018; @goetze2020;
  @raman1953]. This is the larger build: agatesim must vary band
  properties with position, and nodule's renderer (ADR 0007 section 3)
  must take materials that depend on position, not on growth step alone.
  It is the closest physical form of the rejected image-level fade, so it
  carries the highest risk of the same fake look (*plausible*). Judged by
  the owner's eye first.

B and C can be combined. Either can follow A.

### 3. Labels and stated assumptions

- No model parameter changes under A. Under B or C every new amplitude,
  count and length scale is `effective`. The couplings (pigment to
  position, thinning to distance from a channel or tube) are
  **assumptions** drawn from qualitative sources.
- The fade what-if's parameters (fade depth, patch scale, noise seed 777)
  were scratch only and are not adopted.

### 4. Grades

- The correlations, percentiles and test numbers are *verified* within the
  model: one stone, one seed, 6 cuts per test, unpaired with the 100-cut
  baseline.
- That the imaging does not cause the C14 gap is *verified* (C14 against
  blur, noise and resolution). That the cause is the level-set
  geometry and the per-step materials is *plausible*.
- The literature statements are *verified* as the authors' statements
  (full text or the sections read). Formation mechanisms are context only,
  and nothing rendered here is evidence for one (ADR 0002).

### 5. The owner's choice: B, then C, modular (2026-11-03)

The owner chose B followed by or alongside C: both at once if that is more
efficient, otherwise modular, with B built so that C follows readily.

**Modular, because B and C touch different layers.**
- B changes only *where* bands lie, through the growth-time field. Fans
  and the clear outer layer are still contours of one time field, and
  material still depends on growth step alone. So nodule's renderer takes
  B unchanged (`BandRecord`'s check holds), and B can be judged by eye on
  its own.
- C changes *what* a band is made of at each place. It needs agatesim's
  band properties to vary with position, and the renderer to take
  materials that depend on position (ADR 0007 section 3). That is the
  larger and riskier build.
- Built together, B would wait for the renderer change, and one look check
  would mix two risks. Built apart, each is judged by the owner's eye
  (section 1) before any statistics.

**How B prepares for C.** B's features are built in agatesim as one
world-fixed layer: seeded from spawned streams, in world coordinates,
independent of resolution. Each feature gives a per-point influence. B
reads that influence to thin bands. C will read the same influences
(Howard's spherical inclusions where bands are destroyed at tubes,
p. 104), plus new along-band fields, to vary materials. The interface C
needs is written into B's agatesim ADR but not built until C.

**Plan for B (owner-approved, 2026-11-03).**
1. agatesim (its own ADR and tests, in a session started there):
   - B1, a clear unbanded outer layer, off by default.
   - B2, wall channels: bands thin to zero toward each channel's focus;
     depth ≤ 2.5 cm [@walger2009]; lateral size 0.5–2 cm (rough figure
     measurements, *speculative*).
   - Both rest on a world-fixed feature layer that C will reuse.
   - Controls: no features gives the current stone exactly; the fold
     check holds; previews match full runs; a cut through a channel shows
     converging bands and a cut that misses it does not. Time and memory
     are measured.
2. nodule: move the pin and turn the features on in the `n2-slice`
   preset.
3. The owner judges about 4 look cuts beside the current renders before
   any statistics.
4. Only if the look passes: the 20-cut check of C16, C14, C7 and the joint
   compare.

**Starting values for this stone (owner):**
- Clear layer 1.5 mm: within Howard's 1–2 mm; Walger's pigment-free first
  layer is 2–2.5 mm.
- One channel: Walger's "often one, rarely more" for fist-sized
  amygdules.
Both are `effective` and judged by eye in step 3.

**Deferred: thinning at indentations** [@howard2018, p. 98]. It is needed
eventually, but it has no geometry to build from yet. It is to be added
once the project plans how to model and show nodule exteriors (README
goal 2), which will give the wall its indentations.

### 6. The look check: the channel passes, the clear layer does not (2026-11-04)

Step 3 of the plan. The owner compared look cuts of the current stone with
cuts through one channel, with and without the clear layer.

**Verdict (owner):**
- **The channel passes.** It is comparable in realism to the plain stone.
  The approved look is the channel-only render (run folder
  `variants/channel/look2-channel-only`, gitignored).
- **Channels close fully on the line.** `channel_opening_mm` = 0, the
  agatesim default since agatesim ADR 0077's update (7de45c6): every band
  in the stack thins to nothing on the channel's line.
- **The clear layer is off.** Its dark grey ring is an obvious fake tell.
  It returns only once that is solved (a cause and a fix are not yet
  known).

**Why channels close.** Walger's wall layers thin "from the normal thickness
to zero" toward the channel's focus (pp. 116, 136) [@walger2009]. That is
a closed fan, and the model now builds exactly that. *verified* (that the
source says so; qualitative).

**Open or fading band ends belong elsewhere.** The owner also sees band
ends that stop short or fade out. Howard's escape tubes fit them: bands
"narrowed and thinned and eventually peter out completely" (Fig. 6, p. 99)
[@howard2018]. In their model the tubes form after banding, as outflow
(stage 2), so they are a different feature from Walger's inflow channels.
The mechanism is context only; the geometry is what the stone uses.
- Escape tubes come **later, as a separate feature**, with their own
  agatesim ADR.
- Fading ends need band material that varies along a band, which is
  option C's renderer work. So the tubes follow C.
*plausible* (that tubes explain the open ends seen in photos; no counts).

**Now on in the preset.** `n2-slice` sets `channel_count = 1` (agatesim
pinned at 7de45c6). Lateral size, length and depth stay at agatesim's
defaults: 8 mm each side, 10 mm, 5 mm (`effective`). Step 4, the 20-cut
check of C16, C14 and C7 and the joint compare, runs on this preset.

**Step 4 result (2026-11-04).** The 20 cuts of ADR 0014 4n, rerun with the
channel: same planes, looks and seeds, so only the channel differs. 1670 s,
peak traced 1054 MB (4n: 1677 s, 1049 MB).

| Median of 20 cuts | Without channel (4n) | With channel | Photos |
|---|---|---|---|
| C16 coherence | 0.909 (pct 98) | 0.907 (pct 98) | 0.67 (A), 0.60 (B) |
| C14 defect density | 0.236 | 0.213 | 0.51 |
| C7 thickness spread | 0.530 | 0.517 | 0.62 |
| C26 slope, pct vs B | 2.68 (95) | 2.29 (90) | 1.23 |
| Joint, A / B (limit) | 1.1 / 3.0 (15.9 / 7.9) | 1.1 / 3.2 (15.9 / 7.9) | |

- 12 of the 20 cuts changed and 8 came out identical. That is more than
  expected, because the thinned zone is wide (about 16 by 10 by 5 mm in a
  50 mm stone). Within the changed cuts, C14 rose in some (cut 5: 0.29 to
  0.41) and fell in others (cut 11: 0.38 to 0.21).
- **The channel does not close the band-order gap.** C16 is unchanged.
  C14 and C7 moved slightly away from the photos, by less than the spread
  between cuts. *verified* (measured, descriptive; one stone, one seed).
- The joint compare stays within both halves. C26 slope moved toward the
  photos (90th percentile against B, down from 95th). *verified* that it
  moved; *speculative* that the channel is the cause and not chance, with
  only 12 changed cuts.
- This fits the sources: channels are local, and bands elsewhere grow "in
  relatively uniform thickness" (p. 128) [@walger2009]. B earns its place
  through the look, which passed, not through these statistics. The C16,
  C14 and C7 gaps are left to option C.

## Alternatives considered

- **Random pinch-outs and splits throughout the stone.** Against the
  sources ("absent in agates"; uniform thickness away from channels).
  Rejected.
- **Raise the no-folding cap toward 0.9.** Wavier bands at the current
  cap moved nothing. Not tried, and this would change ADR 0005's rule.
- **More image-level what-ifs** (for example textured patches to lower
  C16). Declined by the owner on 2026-11-03: the stand-ins look fake.
- **Change C16, for example by adding a texture control.** C16 is applied
  in the same way to photos and renders, so it is not biased by
  construction. Its negative control (isotropic noise) does not cover
  smooth blank areas, which it reads as coherent; that is noted under
  Risks rather than changed.

## Risks

- **C16's banded share partly measures texture, not bands.** A smooth
  area with no bands counts as banded. Photos' unbanded areas are probably
  textured (quartz grain, mottling; *plausible*), which a smooth render
  lacks.
- **Statistics can be gamed.** The fade test moved C14 and C7 toward the
  photos while the images looked worse. A pass on these measures alone is
  not evidence of realism.
- **The blind test may not single out band order.** If the owner detects
  the stone above chance for another reason, band order stays untested by
  the eye.
- **One stone, one seed.** The gaps may differ for other stones.

## Consequences

- With B chosen:
  - An agatesim ADR comes first, with its own tests, then the code, then a
    pin move here.
  - Look cuts go to the owner beside the current renders before any
    statistical run (section 1).
  - C follows with its own ADRs, one in agatesim and one here for the
    renderer.
  - The blind test (A) stays available. ADR 0008 section 5 runs it once
    tuning is frozen.
- References: howard2018 and walger2009 added to the catalogue (full
  text); goetze2020 §4.3 added to its sections read.
