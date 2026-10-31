# 6. Band counts as censored records, and what the eye adds

- Status: accepted (owner, 2026-10-29, with the recommendations in 7a–7e;
  the reading of 7e found no source that changes them). Amended
  2026-10-29 after three papers were read in full; no decision or grade
  changed (see Consequences). Amended again 2026-10-29 after ten more were
  read in full: the owner took choices 7f–7h (section 7); no grade changed.
  Amended 2026-10-30 with what was built for C23–C25 and the check
  (section 8), and the owner took choices 7j and 7k (section 7); no grade
  changed. Amended 2026-10-30 again with round 1's result and choices
  7l–7o for round 2 (section 9); no grade changed.
- Date: 2026-10-29
- Builds on: [ADR 0004](0004-characterization-schema-core-set.md) sections
  1, 4 (C6, C10, C13) and 8. Records the owner's choice of 2026-10-29 to
  "record differently", with a fallback. ADR 0005 stays reserved for the
  representation, as ADRs 0003 and 0004 name it.

## Context

- **The owner's check of 30 labels (2026-10-29, fd66ebf).** The owner gave a
  band count by eye on 11 faces and "unknown" on 19, and wrote notes on
  most. About 20 of the 30 notes report banding finer than the photo shows.
  Other notes report several band centres in one face, outer bands with a
  changed ("curdled") texture or colour, transmitted light inverting which
  bands look light, and shadow banding as a sign of tight fine bands
  (`references/library-view-labels.toml`, `checked.note`).
- **Fine bands are nearly universal.** Iris striations 0.1–5 µm apart are
  "present in virtually all agates", and defect content oscillates at three
  scales about two orders of magnitude apart: about 0.2 µm, about 700 µm and
  centimetres [@heaney1995]. Etched sections show ordered sequences of
  submicron to about 3 µm units, running to thousands [@frondel1985].
  Visible bands can hold up to 1000 micro-bands per mm [@goetze2020]. Zones
  of iris banding "exist in nearly every agate, although they are often
  concealed by pigmentation" [@cartwright2026].
- **A photo resolves far less.** Arithmetic, not a measurement: a 3000 px
  photo of a 60 mm face has 0.02 mm pixels, and a light–dark couplet needs at
  least 2 pixels, with blur and JPEG compression often 4 or more. A varve
  rule asks for at least three pixels per lamina, so 6 per couplet
  [@damci2016], which gives about 0.12 mm. A field review gives no image
  pixel rule at all; its one sampling rule, "at least ten measurements
  across each varve", is for describing layers by µ-XRF, not for seeing them
  [@zolitschka2015, p. 15]. To see a lamina's boundaries sharply, one CT
  study needed "a minimum of about 13 voxels per varve" [@jamba2025,
  pp. 94–95]: a measuring floor about four times Damci's detection floor.
  So the finest couplet a good
  library photo can show is about 0.04–0.12 mm, two orders of magnitude
  above iris spacing. Library photos have no scale, so even this limit is
  known only in pixels.
- **So a band count belongs to the photo, not the stone.** C6 (machine,
  `nodule.bands`) and `C6_eye` (owner) count the bands coarser than the
  photo's limit. The same face photographed sharper gives more. ADR 0004
  already reads C6 as a lower bound where the outline cuts inside the rim;
  the same holds at every face for bands below the limit. C6 can also
  overcount, from noise or texture, which its range over the sweep shows
  [@weber2010]. In thin chalcedony plates, an enlargement "exhibits
  numerous closely-spaced lamellae which are not visible" at the smaller
  scale, "a general feature" of both specimens [@jayaraman1953, p. 448]:
  one qualitative case of the same premise, without scales.
- **Layer counting elsewhere meets the same limit.** In ice cores, "limited
  measurement resolution" can "decrease the number of layers that can be
  recognised", and the GICC05 count enters each uncertain layer as ½ ± ½
  year, so the uncertainty is an interval, not a point [@winstrup2012].
  The rule comes from [@rasmussen2006], whose counters "agreed" before
  counting that uncertain marks "should be regarded and counted as 'half
  years'" (¶17, p. 5): a convention, not derived or tested. Its maximum
  counting error sums ½ year per uncertain mark, "assuming that the
  counting errors are correlated", and is called "highly conservative"
  (¶40, pp. 11–12). [@andersen2006] adds the definition that a layer is
  uncertain where its probability is judged to lie between 25 % and 75 %
  (p. 3249; the wording is not in [@rasmussen2006]). Units at the limit are
  a class of their own there: "shoulders, wide peaks, or double peaks that
  could represent two thin annual layers not fully resolved" count ½ ± ½,
  while layers "lost altogether due to insufficient resolution" lie outside
  the error, put at most at about 1 % (¶23, ¶38) [@rasmussen2006].
  The error of [@andersen2006] assumes "the data resolution
  is sufficient to resolve annual layers" (p. 3252), so layers below the
  limit lie outside it, as here. The same paper defines a record's
  effective resolution as "the shortest wavelength that can be identified
  from the data" (p. 3248), and finds single-record counts "highly
  dependent on the measurement resolution": coarse records miss thin
  layers, fine ones count events within a year (p. 3251) [@andersen2006].
  [@rasmussen2006] gives the same definition first (Table 1, p. 3), at 3–25
  times the sampling step, set "from inspection of the data"; its finest
  record is smoothed because it holds sub-annual peaks (¶6).
- **Counts err both ways, and repeat counts don't show it.** Varves below
  about 0.2 mm are missed or merged, while thick ones can hold "more than 10
  seasonal laminae" counted as years; one record had about 270 varves
  missing and 230 too many [@zolitschka2015, pp. 17, 23]. Varve errors are
  mostly two-sided percentages from repeat counts across cores, which
  measure precision, not accuracy, and miss varves absent from every core
  [@ojala2012, pp. 51, 53–54]. Distinctness matters more than thickness: a
  lake with 1.22 mm varves has ±8.4 % against ±1 % at 0.67 mm
  [@zolitschka2015, p. 18]. Where varves are indistinct, one study refuses
  to count them as zero years and fills them by resampling distinct
  sections, which assumes the unseen resemble the seen and needs a
  radiometric age to check against (p. 413) [@arcusa2022]. Observers there
  differed by 14.5–23.6 % on the same core (Table 2, p. 419).
- **No count-against-resolution test exists in what was read.** In varves,
  the three-pixel rule above is asserted without derivation, citation or
  test, ignores blur, and is applied to the average lamina, not the
  thinnest [@damci2016, p. 177, 180]. No source found (time-boxed search,
  2026-10-29) measures how a count changes over a ladder of resolutions, or
  explains agate's shadow effect. The nearest cases: on the same varves,
  1 µm SEM pixels and 45 µm CT voxels give counts within −1.4 to +5.6 %
  (Table 2, p. 88), finer voxels "did not provide additional information",
  and one lamina is shown at seven voxel sizes without counts (Fig. 19,
  pp. 94–96) [@jamba2025]; a tree-ring CT study scans at 60 to 4 µm, but
  each size on different cores, with no counts compared (Table 2)
  [@vandenbulcke2019]; a varve review finds counting error unrelated to
  mean varve thickness (r² = 0.0016), with methods chosen to suit the
  varves (pp. 51, 53) [@ojala2012]. [@ojala2002] compares two imaging
  methods on one shared count of 165 varves (p. 59, Table 1), not two
  counts.

The owner chose (2026-10-29) to try recording differently first, with a
fallback written in that loosens the requirement toward goals 1 and 2 (image
synthesis, realistic specimens) if this disappoints.

## Decision

### 1. Every band count carries its resolving limit

A count is stored as "at least *n* bands coarser than *L*", never as the
stone's band number:

- **C6** keeps its range over the sweep and rule agreement, and adds the
  **resolving limit** *L* of the face: the finest band spacing the photo
  could show, in pixels, as a share of the face's inner radius, and in mm
  where the view has a scale. *L* is the shortest spacing at which the
  face's rim-to-centre profiles keep power above their noise floor
  (choice 7a), so it shows blur and compression, not only pixel size; it is
  checked on synthetic faces of known blur. *L* is stored with the number
  of profiles averaged to find it, since averaging raises the signal over
  the noise; tree-ring CT filters along the rings for that reason
  [@vandenbulcke2019, ms p. 17] (choice 7h).
- **Bands finer than *L* are unknown, not zero.** No record may store or
  imply a count of them. The *n* above *L* still carries a two-sided error
  (C6's range), since counts err both ways (Context).
- **`C6_blank`** (choice 7g): the share of each rim-to-centre profile, and
  its median over the face, where no band is resolved above the noise
  floor. Two faces with the same "at least *n*" and very different blank
  stretches are then not read as alike. A varve study reports its
  indistinct share of the core (8.7–19.6 %) beside the count
  [@arcusa2022, p. 413]; nothing is imputed into the blank here (Alternatives).
  Grade: speculative (no source does this for agate).
- **`C6_eye`** is stored the same way: "at least *n*", with *L* of the
  photo it was given on. The eye's own limit isn't measured; it is taken to
  be no finer than the photo's (assumption). The owner recounts about 5
  faces a week or more later, unprompted, to estimate their own spread
  (choice 7h); until then `C6_eye` is one look.
- **Counts are compared only at a common limit and under one counting
  rule.** Two counts with different *L* are compared after the finer one is
  recounted at the coarser limit (section 2). A finer method can count
  fewer under a stricter rule: SEM counted laminae only where topped by a
  clay cap, so CT counted more [@jamba2025, p. 92]; reinterpreting the same
  ice-core data moved counts by up to 6.6 %, 4–5 times what the uncertain
  marks allowed (¶39) [@rasmussen2006]. C6 and `C6_eye` differ in rule as
  well as in *L*, so `C6_eye` below C6 need not mean a coarser eye.
  Counts are also compared only within one lighting kind,
  since transmitted light can invert which bands look light (owner's note on
  agate3; ADR 0004 section 1 already records the lighting). In thin
  chalcedony plates, bands dark against a window look bright under oblique
  light on a dark background, "and vice versa" [@jayaraman1953, p. 445,
  447]: a published case of the owner's note. Two iridescent plates show the
  same swap over their whole banded area [@raman1953, p. 200]. Thicker
  plates were less transparent [@jayaraman1953, p. 447; @raman1953,
  p. 200], so a slab's thickness also changes what shows; it isn't in the
  schema and is listed as a candidate. In varve image analysis, uneven
  illumination alone cut a per-varve grey-level correlation from 0.78 to
  0.10, and image acquisition is called "probably the most critical step"
  [@ojala2002, p. 63].

Negative controls: a synthetic face with fine bands below *L* drawn over
coarse ones must count only the coarse ones and report *L* above the fine
spacing; the same synthetic stone imaged at two resolutions, under one
counting rule, must give a higher-resolution count at least the
lower-resolution one, and equal counts once both are taken at the coarser
*L*; a synthetic face with a band-free zone must report that zone's share
in `C6_blank`, not only a lower *n*. Grade: plausible (the limit is
standard sampling, and ice-core counting defines a record's resolution the
same way, as its shortest identifiable wavelength [@andersen2006;
@rasmussen2006]; its use on agate photos is untested). A pixel floor after
[@damci2016] (*L* no finer than 6 px per couplet, flagged where the
spectrum says finer) is a candidate cross-check, not adopted: speculative,
since no source compares the two. If adopted, it must say whether it is a
detection floor (Damci's 3 px per lamina) or a measuring floor (about 13
voxels per lamina [@jamba2025, pp. 94–95]); spacing descriptors need the
stricter one.

### 2. Count against limit, N(L)

From one face, C6 is recounted at a ladder of coarser limits (the photo
blurred and downsampled in steps of √2 down to about 4 band spacings across
the face). The curve N(L) above the native limit is a measured descriptor;
below the limit it is not extrapolated. A scan of the same face at higher
resolution (section 5) tests whether the curve goes on rising.

[@heaney1995]'s three scales predict steps: a curve that stays flat over a
range of *L* and then rises would mark a gap between band scales. That is a
reading of one source, not a result. A step can also come from counting
rules or geometry, not band scales (reasoning), so the rule agreement is
kept beside N(L). On varves far coarser than both limits, counts at
1 µm and 45 µm agree closely [@jamba2025, p. 88]: a flat stretch, as N(L)
should show where *L* is well below the finest spacing (analogue,
plausible). Finer data can also add peaks that are not units: ice-core
counters smooth their finest record because it holds sub-annual peaks
(¶6) [@rasmussen2006], and varves hold "at least two or more seasonal
laminae" [@zolitschka2015, p. 6]. Those count as errors only because a year
is defined; agate bands have no such unit, so a rise of N(L) at fine *L* is
read as a finer band scale once rule agreement excludes noise (reasoning,
speculative).

Negative controls: rings of one spacing give a flat N(L) that drops to 0
once *L* passes the spacing; two drawn scales give two steps; red noise must
give a curve whose rule agreement fails (ADR 0004, C6's control). Grade:
speculative (no source applies count-against-resolution to laminae; choice
7c).

### 3. New descriptors from the owner's notes (photo rung)

| # | Domain | Descriptor | Source | Negative control | Grade |
|---|---|---|---|---|---|
| C23 | B, D | Centre count: separate band centres in the face, by eye, cross-checked against C13 where it is measured | owner's notes; [@marichal2025] (via C13) | One concentric set gives 1; two drawn separate sets give 2 | plausible |
| C24 | B, E | Fine-banding evidence, per zone (outer, middle, inner): each cue present, absent or unknown: (a) colour gradient within a band, (b) faint striations at the limit, (c) shadow-like bands in radial sectors, (d) iris colours | owner's notes; [@raman1953] for (c), as an analogue; [@heaney1995; @cartwright2026; @jones1952; @raman1953; @raman1955] for (d) | A synthetic face with no sub-limit bands must give no cue under the same rendering; a cue is never converted into a count | speculative |
| C25 | B, F | Zone sequence rim to centre: each zone's kind (even bands, curdled or mottled bands, plain: no visible bands, coarse quartz, cavity; 7k) and its share of the radius | owner's notes; ADR 0004 C17 for the core | A face of one even banded zone gives one zone; C17's core and C25's quartz zone agree | plausible |

C24's cues are signs, not measurements. Cue (c) is the owner's observation;
no peer-reviewed source found explains the shadow effect, so it stays
speculative. Neighbouring agate bands can differ in translucency through
crystallite size alone (about 5 µm against 300 nm, by scattering)
[@lorenzi2022], which a shadow effect would need, but that source says
nothing of viewing angle, sectors or band spacing. [@jayaraman1953], read
in full, doesn't describe it either: on thin colourless plates, light
scattered by the fibres spreads in a brush that "follows the course of
banding" (p. 447), and transmission falls quickly as the band planes are
tilted against the beam (p. 448). A shadow effect could arise from these
(reasoning, speculative). The nearest published picture is in its
companion paper: one iridescent plate whose band strip is "perfectly dark"
in one segment, "faintly banded" in the next and "quite bright" in the
third, the segments bounded by lines through the band corners; the
lighting isn't stated and no cause is given [@raman1953, p. 200, Fig. 1].
There, transmission changes strongly with the angle between the light and
the band planes, much less for tilt within them (p. 201), and a 1 mm plate
passes a sharp image only with a polaroid along the fibres
[@raman1955, p. 1, Plate I]. So sectors of differently oriented bands
could differ under one lamp (reasoning, speculative). In Brazilian geodes,
translucent and milky wall bands differ in water (about 1.3 against
0.7 wt %) and density, and heat turns translucent chalcedony milky as it
recrystallises [@graetsch1985, pp. 302–304]; neither source mentions
viewing angle. The owner's idea of reading fine-band spacing from shadow
banding is listed as a later candidate.

Cue (d) has two published causes, offered as options since the sources
disagree: a grating of thin lamellae of alternately higher and lower
refractive index [@jones1952], or a periodic change in the quartz c-axis
direction along each fibre, argued by ruling out other causes
[@raman1953, p. 203; @raman1955, p. 5], which rejects Jones's opal reading
(p. 6). The strata repeat about every 1 µm, as stated, not measured
[@raman1953, p. 199]. One iris strip holds "six different gratings" with
gaps, differing in colour (p. 202), and the diffraction is regular only in
"selected regions" [@raman1955, p. 4]; no source read turns a colour into a
spacing without measured angles and a thin plate.

### 4. Who labels

Helper agents propose C23–C25 for the banded faces from the outline-check
pictures and, on the 30 checked faces, from the owner's notes; the owner
checks those 30 on the check page and the check overrides the proposal
(as ADR 0004, 9b; choice 7b). The pre-filled proposals may anchor the
owner's answers, as the first check may have been (agreement 28–30 of 30);
the check page shows C24 blank, so that one is answered unprompted.

Blind first (choice 7f): the owner answers 10 of the 30, chosen at random
with a spawned RNG stream, with the proposals hidden, then the other 20
as above. Agreement is scored item by item (C23's count, each C24 cue per
zone, each C25 zone), not per face, and reported separately for the blind
10 and the shown 20; the gap between the two measures the anchoring. In
ice-core counting, counters work blind and then reconcile: their totals
agreed within 1.5 % while about 200 positions differed in 2,400 years
(¶18) [@rasmussen2006]. Grade: plausible (a standard check against
anchoring; one source's counters, not agate labels).

### 5. Cheap captures (data track, owner's stones)

These raise the limit or test a cue on the owner's own specimens. Their
images stay in agatesim's `specimens/` and never enter this repository.

- **Flatbed scan** (the owner has a scanner): a known scale and a finer
  limit (2400 dpi gives about 0.01 mm pixels), and, with a photo of the same
  face, a real N(L) test across two resolutions.
- **Turn test** under one lamp (owner's idea): sorts shadow sectors into
  those that move with the light and those fixed to the stone. A short
  video or a series of about 8 angles. It separates angle-dependent from
  angle-fixed appearance, not optics from structure: a translucency
  contrast between bands is structural, yet its shadow can move with the
  light (reasoning, speculative). Fixed sectors point to pigment or band
  boundaries; moving ones are an optical rendering of a structure not yet
  known. Record the lamp's direction against each sector's bands, since
  tilt across the band planes changes transmission far more than tilt
  within them [@raman1953, p. 201]. If a polarising filter is at hand,
  turning it in front of the camera tests for fibre-orientation contrast,
  which pigment wouldn't show (reasoning from [@raman1955, p. 1]).
- **Later, need a thin slab** (the owner has none): laser diffraction of
  iris banding, thin sections. CT may not separate bands that are all silica:
  an assumption. Bulk samples of translucent and milky wall bands differ in
  density by about 0.5–1 % (2.575–2.579 against 2.593–2.601 g/cm³, ±0.010)
  [@graetsch1985, p. 302, Table 1]; no source read gives a density profile
  across bands, or the contrast a CT boundary needs. CT is also costly:
  halving the voxel size gives eight times the data [@vandenbulcke2019,
  ms p. 12], and 45 µm voxels took 170 GB for a 1.5 m core [@jamba2025,
  p. 94], far above this project's 2 GB flag.

### 6. Fallback: measure the image, not the stone

Checked at the end of the re-tagging under this ADR. It is taken if any of:

- the machine count, at the owner's limit, disagrees with `C6_eye` on more
  than a third of the faces where both are given;
- the owner's check disagrees with the proposed C23–C25 on more than a third
  of the items over the 30, scored item by item, or on more than a third of
  the blind 10's items (section 4; choice 7f); each test is applied to C23,
  C24 and C25 alone as well as to all items pooled, and any one failing
  counts (choice 7j);
- N(L) doesn't separate the faces the owner notes as finely banded from the
  rest.

Then the band descriptors stop claiming anything about the stone. Instead,
synthetic stones are rendered at each photo's resolution and lighting, and
compared with the library by image statistics at set level (C7–C16 and the
profile spectrum), plus the owner's blind real-or-synthetic judgement.
Negative control: obvious fakes (bands drawn without fine structure, wrong
spacing ratios) must be caught. This serves goals 1 and 2 directly and
loosens goal 3. It needs N2's renderer, so until then counts stay ranges with
limits. The resolving limit of section 1 is needed either way, to render at
the photo's limit. Grade: plausible (forward comparison of image statistics
is what the core set already does for synthetic cuts).

### 7. Choices (owner took the recommendations, 2026-10-29)

- **7a. How *L* is found.**
  - *(Recommended)* **Per face, from the profile spectrum:** the shortest
    spacing at which the rim-to-centre profiles' power stays above the
    noise floor. Measured, so it shows blur and compression.
  - **2 pixels (Nyquist):** simple, but optimistic for blurred photos.
  - **From the sharpness of the rim edge:** measures blur only where an
    edge is visible.
- **7b. Who labels C23–C25.**
  - *(Recommended)* Helpers propose, the owner checks the same 30
    (section 4).
  - The owner labels all banded faces.
  - Only the owner's 30, nothing proposed.
- **7c. N(L).**
  - *(Recommended)* Adopt now as speculative, since it reuses C6's code and
    sets up the scan test.
  - List it as a later candidate.
- **7d. Fallback test.**
  - *(Recommended)* The three tests of section 6, at the end of
    re-tagging.
  - The owner's judgement alone at that point.
- **7e. Reading before acceptance.**
  - *(Recommended)* One time-boxed helper search for peer-reviewed work on
    the shadow effect and on counting laminae against resolution; papers
    that need downloading go on the owner's list. If none is found, the
    assumptions above stand as stated.
  - Accept as is.

Choices 7f–7h were put after ten more papers were read in full; the owner
took the recommendations (2026-10-29).

- **7f. Blind check.**
  - *(Recommended, taken)* The owner answers 10 of the 30 with proposals
    hidden, then the other 20; agreement is scored item by item (sections
    4 and 6).
  - All 30 blind.
  - Keep the check as it was.
- **7g. Blank share.**
  - *(Recommended, taken)* Adopt `C6_blank` now (section 1).
  - List it as a later candidate.
- **7h. Smaller additions.**
  - *Taken:* store with *L* the number of profiles averaged; the owner
    recounts about 5 faces after a delay (section 1).
  - *Listed as later candidates:* a shoulder or split peak at *L* counts as
    ½ ± ½ band, after [@rasmussen2006]'s marks for layers not fully
    resolved (needs a control that one wide band is never split); possible
    missed and possible extra bands reported separately per zone, after
    [@ojala2012, Fig. 4].

Choice 7i was put once *L* was built (`nodule.bands`, 12ead29) and checked
on 8 library photos with judged rim outlines (2026-10-29). Native *L* was
2.2–2.6 px; added blur raised it (σ 1, 2, 4 px gave 3.7–6.2, 6–10 and
10–17 px), and re-saving as JPEG at quality 90 to 30 never made it finer
(2.2–4.4 px). The spectrum can't tell bands from fine texture or processed
noise at those periods, and the quality gate's noise sits at its 1 grey-level
floor on 147 of 257 photos, so the floor there is a stand-in. The counting
rule's couplets (12–32 px on these photos) are far coarser than *L*, so the
rule, not *L*, censors their counts. Grade: plausible (one check, 8 photos).

- **7i. How *L* enters the record.**
  - *(Recommended, taken)* As measured, with the caveats above.
  - The candidate pixel floor of section 1 (no finer than 6 px per couplet,
    after [@damci2016]): more conservative, but nearly every photo would sit
    at it.

Choice 7j was put once C23–C25 were built (section 8): C24 gives 12 of
about 16 items per banded face, so it dominates a pooled test.

- **7j. How the fallback test weighs C23–C25.**
  - *(Recommended, taken 2026-10-30)* The one-third test on C23, C24 and C25
    each alone and on all items pooled; any one failing triggers the
    fallback.
  - As first written: all items pooled.

Choice 7k was put during the owner's check, before any answer was saved: the
proposals used the C25 kind "colour change" for a plain core, one broad
translucent band and broad coloured layers, so the kind meant three things.

- **7k. What the third C25 kind means.**
  - *(Recommended, taken 2026-10-30)* "plain: no visible bands": a stretch of
    fairly even colour with no banding seen at the photo's resolution (a
    lower bound, like the counts: not proof of no bands). A change of colour
    alone starts no new zone; bands that change colour stay "even bands".
    The ten proposals using the old kind were re-read from their notes
    alone: one became even bands (merged with its neighbour), six stay
    plain, and three stay plain marked unsure in their notes.
  - As first written: "colour change", undefined.

### 8. As built: C23–C25 and the check (2026-10-30)

`nodule.zones` holds the labels (`references/band-zone-labels.toml`, a
`proposed` and a `checked` table per photo, as ADR 0004's labels);
`nodule.zonecheck` is the check page. Choices made in building it, each an
assumption unless sourced:

- **`banded` first.** Each photo is first labelled banded yes or no: yes
  when bands can be judged on the face shown, whatever its view kind (two
  faces the owner labelled exterior and three labelled broken surface show
  bands, and the owner's notes count them). When no, C23–C25 are unknown.
- **C23** is a whole number from 0 to 50: 0 when bands show but no centre lies
  in the face. The cross-check against C13 waits, as C13 isn't measured yet.
- **C24's zones are thirds** of the rim-to-centre distance (outer, middle,
  inner), fixed whatever C25 finds, so every face has the same 12 items.
- **C25** lists zones from the rim inwards, each with its share of the
  radius; shares sum to 1 within 0.05; two neighbouring zones of one kind
  are one zone; an empty list means unknown. Where a face has several
  centres, the helpers read a typical path in the largest set.
- **Scoring, item by item:** `banded` is one item; when both sides say yes,
  C23 is one item, each C24 cue one (12), and each C25 zone position one
  (a zone one side lacks disagrees; two unknown sequences agree as one item).
  When either side says no, only `banded` is scored. Share differences of
  agreeing zones are reported apart (median), not scored.
- **Blind 10** are picked from the 30 by a stream spawned from seed 6,
  separate from the check sample's. They come first, and the page refuses the
  other 20 until they are saved. Proposals never show on them, even later.
  C24 starts blank on all 30. Proposal notes avoid C24 cues for that reason.
- **Delayed recount:** 5 of the faces with an eye count, picked by a second
  spawned stream, served by `--recount` from 2026-11-05 (a week after the
  first counts), without the earlier count.
- **Negative controls** (section 3): three synthetic faces drawn in world
  coordinates (`zones.CONTROLS`): one concentric set, two sets in one outline,
  and even bands around coarse quartz and a cavity (0.5, 0.3, 0.2 of the
  radius). Bands are at least 8 pixels per couplet, so no cue below the
  limit can show. One was put among each helper's ten pictures under a
  neutral name. Each must give its own C23 and C25 kinds (shares within
  0.1) and no C24 cue present.
- **Staged:** proposals were made first for the 30 check faces only; the
  other banded faces wait for the owner's check, since a failed check
  (section 6) would make their proposals moot.

First results (2026-10-30; three helpers, ten faces and one control each):

- All three controls were labelled as they must be (C23 of 1 and 2, the
  three zones within 0.1, no cue present). One control per helper is a
  weak test; it rejects gross misreadings only.
- The proposals: 25 of 30 faces banded; C23 is 1 on 19, 0 on 1, 2 on 2, 3 on 2
  and unknown on 6; of the 300 C24 items on banded faces, 85 present, 150
  absent and 65 unknown.
- C25 against C17 (section 3's control): where both are given (17 faces),
  the innermost C25 zone is coarse quartz or a cavity on 14 of the 17 faces
  exactly where `C17_core` is at least 0.05; 12 of 17 at 0.02, 13 at 0.1. The
  0.05 cut was picked after looking, and C25 shows quartz layers between
  bands that a core share doesn't, so this is no pass yet. Grade:
  speculative, to be read again after the owner's check.

### 9. Round 1 result and round 2 (choices 7l–7o, 2026-10-30)

The owner checked all 30 (blind 10 first). Agreement with the proposals,
scored as section 8 says: blind 97/188 items (banded 10/10, C23 8/9, C24
71/108, C25 8/61); all 302/461 (C23 22/25, C24 197/300, C25 53/106). More
than a third disagree on C24, C25 and pooled, blind and over the 30, so
round 1 meets section 6's fallback condition. That result stands as
scored. The owner kept the proposals' coarse sense of "even bands" on the
20 shown faces by choice (their note), so shown C25 (45/45) measures the
convention, not independent agreement. Read after scoring, so description
only: on 6 of the 8 blind faces with zones on both sides, the two agree on
84–94 % of the radius about where band material ends and quartz or a cavity
begins; the disagreement is mostly how finely the banded part is split (the
owner up to 17 zones, the proposals 2–5), which section 8 never fixed. The
owner judges C24's colour gradient easily muddled on still photos and the
faint striations the steadiest cue; they marked shadow unknown on 24 of 75
zones.

- **7l. Zone scale.** *(Recommended, taken)* A C25 zone covers at least
  0.05 of the line; fine alternations of bands and plain gaps within a
  stretch are one zone of bands ("even" or "curdled or mottled"). The fine
  sequence is what C6 and N(L) record. Alternatives: every visible layer a
  zone; two levels.
- **7m. Re-test.** *(Recommended, taken)* A fresh blind 10 from the banded
  faces nobody has checked, proposed under the new definitions; round 1's
  answers are kept as they are. Alternatives: re-tag the same 10 (not blind:
  the owner has seen them and knows the convention); no new blind round.
- **7n. C24.** *(Owner's choice)* Faint striations only, per third: 3 items
  per face. Gradient, shadow and iris are dropped from round 2 on.
- **7o. Scoring C25 along a drawn line.** *(Recommended, taken; the line is
  the owner's idea)* Each face gets one straight line from the rim (0) to the
  centre of its largest band set (1), with ticks every 0.1, shown alike to
  helpers and owner; shares are read off the ticks, and C24's thirds are
  thirds of it. A face's C25 agrees where the kinds match on at least
  two-thirds of the line. The computer drafts the line: the centre from
  where the band-following paths of `nodule.bands` converge, the direction
  from a spawned RNG stream per face. The owner checks each line before
  helpers see it and may move the centre or take the next direction only
  for a stated reason (crack or edge, misses the bands, waterline banding).
  This scoring applies from round 2 on, never to round 1. A line samples
  one path, so C25 describes that path, not the whole face. Alternatives:
  zone by zone, as round 1; the owner clicks each centre.

Drafted on the 30 checked photos (7e620bc), the centre was right on about
half by eye (13 from paths, about 6 good; 9 deepest points, about 6
sensible; 8 with no band spacing, about 4 outside the stone). The owner
chose to go ahead: on the line check they click the centre wherever the
draft is off, rather than the drafting being tuned first.

As built (round 2's rules, `nodule.zones`, fixed before any of its faces is
seen): round 2 rows carry `round = 2`; round 1's rows, scoring and report
are unchanged. A zone under 0.05 of the line is refused (7l). C25 is one
item per face, agreed where the kinds match on at least 2/3 of the line
(shares scaled to sum to 1). The blind 10 come from a fixed pool: the photos
whose view is a cut face or a slab in transmitted light, outside round 1's
check sample, shuffled by stream 3 of seed 6 (153 photos; the friend's 31
have no view labels and are not in it). A face the owner marks "no banded
face" at the line check gives way to the next in the pool; the line check
stores the owner's line beside the labels (rim, centre, its source, the
draw and one reason per draw passed over).

The line check (`nodule.linecheck`): drafts go to runs/zone-lines/ (photo at
1 MP as plain PNG, never committed); a page on this PC only shows each line
drawn exactly as helpers will see it, and the server, not the page, works
out the rim end from the face and refuses a centre off the stone. The
checked lines are then drawn for the helpers (`--ticked`). The three
control faces (section 8) get lines too, so helpers read them as they read
the real faces: each runs from a set's centre (the left set of the two) out
to its rim at a fixed angle (`zones.CONTROL_LINES`, an assumption: any angle
gives the same shares, as each set is an ellipse), so the zones control reads
0.5, 0.3 and 0.2 along it. A round 2 control answer is checked for its three
striation items.

Round 2 proposals (2026-10-30): three helpers, the blind 10 split 4/3/3 by
stream 4 of seed 6, one control each, every picture under a neutral name,
given the definitions above word for word. All three controls were labelled
as they must be. The proposals' contents are not summarised here until the
owner's check is saved, as all ten faces are blind. The check page serves
them with `python -m nodule.zonecheck --round2`: the ticked pictures, nothing
filled in, C24 as the three striation items, every save stored as round 2.

Round 2 result (2026-10-30, the owner's blind check of all ten, scored as
fixed above): 41/60 items (banded 10/10, C23 6/10, C24 17/30, C25 8/10;
median share of the line where C25 kinds match 0.74). More than a third
disagree on C23 (4 of 10) and C24 (13 of 30); C25 (2 of 10) and the whole
(19 of 60, one short of the cut) do not. Ten faces is few: 8/10 is consistent
with a true agreement rate from about 0.49 to 0.94 (Wilson 95 %). Read after
scoring, so description only:

- C25: where the two differ, the owner puts a cavity at the centre where the
  helpers saw coarse quartz or plain (two faces), and splits a thin curdled
  or mottled outer zone (0.06–0.25) from the even bands on four.
- C23: on all four disagreements the helpers count more centres than the
  owner (2 vs 1, 6 vs 1, 7 vs 3; 6 vs unknown).
- C24: the owner marks striations present on 23 of 30 thirds, the helpers on
  13 (8 unknown). Fine bands are nearly universal in agate (section
  "Context"), so a cue present almost everywhere may carry little.

Section 6's fallback is still weighed at the end of the re-tagging, as the
owner chose.

- **7p. The rest of round 2's pool.** *(Recommended, taken 2026-10-30)*
  Helpers propose only C25, along lines the owner checks
  (`linecheck --all`); C23 and the C24 items are stored as unknown unless the
  owner labels them. Only C25 held up blind, so proposing the other two would
  add labels known to be weak. These C25 labels stay proposals (graded by
  round 2's blind agreement, plausible), not checked. Alternatives: propose
  all of C23–C25 and let section 6 decide at the end; pause the re-tag for
  N2.

Fault and fix (2026-10-30). On the first pass the owner held out six faces
as "no banded face" because a clicked centre made the line vanish. Each was
a face with no rim judged "yes" and no band paths, so its face was
agatesim's working region, and on these photos that region was wrong: a
speck of background or a museum label (four faces), or a stone with half its
banding left out (two). Clicks off it were refused, but the page showed no
reason. Left in, the six would have been stored as unbanded (all are plainly
banded) and the blind 10 would have kept only faces the computer segments
well. The owner chose (over a rim click, or fixing agatesim's segmentation)
that the page shows the face's edge in cyan, says why a line is refused, and
lets the owner trace the stone's outline where the edge is wrong. The traced
outline is stored with the line (`outline`, [x, y] fractions) and becomes
the face for the rim end; with no centre clicked, the centre is its deepest
point. The direction is still drawn from the face's own stream. The six
records were cleared, so the blind 10 is again the pool's first ten (two of
them still to check). The other twelve lines stand as the owner accepted
them, though where a face was partly wrong their rim end stops at its edge.

On the second pass the owner traced dryhead's outline. Agate-creek turned
out to be a close-up whose banded face runs off every side of the photo, so
no rim is in view and no rim-to-centre line exists. The owner chose a third
outcome (over taking the photo's edge as the rim, or "no banded face"):
`rim_in_view = "no"`. The face is held out like an unbanded one and the
pool's next stands in, but it is not recorded as unbanded. The rule is
stated before the helpers see any face; it keeps close-ups out of round 2.
The blind 10 is now complete. Saving also finishes an outline still being
traced, and clicks on the photo's very edge are clamped into it.

Section 6's fallback, met by round 1, is weighed at the end of the
re-tagging with round 2 and the other two tests, as section 6's timing
reads (owner, 2026-10-30). Round 2's rules are fixed here before any of its
photos is seen, so it is no retry until a pass.

The rest of the pool (7p, 2026-10-31). The owner checked a line on all 153
faces of the pool: 109 lines (56 with a direction passed over, 41 with an
outline traced), 27 with no banded face and 17 with no rim in view. The 99
lines not in the blind 10 went to six helpers, split by stream 5 of seed 6
(16–17 faces each), one control each under a neutral name (each control
twice), with section 9's definitions word for word and C23 and the C24 items
left unknown. A control answer that leaves C23 unknown in round 2 is checked
on C25 and the striation items only (assumption: 7p proposes no C23); all six
were as they must be. The 99 C25 proposals are stored as proposals,
plausible by round 2's blind C25 agreement (8/10), not checked. Description
only: even bands appear on 98 of the 99 lines, plain stretches on 48, coarse
quartz on 41, curdled or mottled bands on 16 and a cavity on 14; helpers
marked 14 high confidence, 60 medium and 25 low.

- **7q. The rest of the re-tag.** *(Recommended, taken 2026-10-31)* In
  order: (1) the friend's 31 (agatesim ADR 0073), which had no view labels,
  are labelled as a second batch under ADR 0004's choice 9b, helpers
  proposing and the owner checking every one (`nodule.check --batch 2`);
  (2) the faces outside round 2's pool that are not exteriors
  (`zones.rest_faces`: the first batch's 15 broken surfaces and one unknown
  view, and the second batch's faces) get lines checked by the owner
  (`linecheck --rest`, "no banded face" being one click) and helper C25
  proposals as in 7p; (3) the 31 exteriors are stored as showing no banded
  face, after the owner looks over a contact sheet for any that do (a window
  or a chip); (4) section 6 is weighed. Round 1's check sample and round 2's
  pool are drawn from the first batch only (`labels.batch`), so the new
  labels leave both as they were. Alternatives: leave the friend's photos out
  of C23–C25; store the exteriors without the sheet.

Batch 2's proposals (2026-10-31): two helpers, split by stone (16 and 15),
the definitions of ADR 0004 section 1 and its C4, C5 and C22 rows. All 31
are proposed as polished cut faces in reflected light; the rim outline is
proposed as following the rim on 15. Stored as proposals until the owner's
check.

The owner checked all 31 (2026-10-31): every label stood except the finish,
sawn on all 31 where helpers proposed polished. The owner gave `C6_eye` on
all 31. All 31 stay cut faces, so all join the rest faces (47 with the first
batch's 16), whose lines were drafted in 141 s (peak 395 MiB).

## Alternatives considered

- **Keep C6 as a count and note the limit in prose.** Rejected: the record
  would still read as the stone's band number.
- **Drop band counts.** Loses the one band descriptor the owner can check by
  eye, and C10 needs the thickness series either way.
- **Estimate sub-limit bands from cues.** Nothing read supports a
  conversion; listed as a later candidate.
- **Impute the blank stretches**, as a varve study fills indistinct zones
  by resampling distinct ones [@arcusa2022, p. 413]. Rejected: that assumes
  the unseen resemble the seen, while bands below *L* are expected to be a
  different scale [@heaney1995], and agate has no independent clock to
  check against (reasoning).

## Risks

- **The limit is only as good as its measure.** A wrong *L* makes
  comparisons at a "common" limit uneven; choice 7a's measure needs its own
  check on synthetic faces of known blur.
- **Cues are subjective.** C24 rests on one observer's eye under unknown
  lighting; it informs where to scan, not the schema's claims.
- **Few faces.** C6 stands on 72 faces and `C6_eye` on 11, so the fallback
  test is coarse; the blind 10 are fewer still.
- **A face is one plane.** CT shows laminae present in the volume and absent
  from a plane [@jamba2025, Fig. 13], so bands that pinch out or curve away
  from the cut make a face count a lower bound for a second, geometric
  reason besides *L* (analogue, plausible).

## Consequences

- C23–C25 and their check are built (`nodule.zones`, `nodule.zonecheck`;
  section 8); `nodule.tags` doesn't read them until the owner has checked.
- `nodule.bands` gains *L*, `C6_blank` and N(L); `nodule.tags` stores C6
  and `C6_eye` as lower bounds with *L*; the check page gains C23–C25, the
  blind 10 and the owner's delayed recount. Timing is measured on 10 photos
  first (current re-tagging of 261 photos takes about 730 s; N(L) adds
  about one recount per step).
- ADR 0004's C6 entry is read through this ADR; its other descriptors are
  unchanged.
- All sources cited are in the catalogue. Choice 7e's search added five,
  at the level read: [@lorenzi2022] and [@winstrup2012] from their texts;
  [@jamba2025], [@arcusa2022] and [@vandenbulcke2019] at abstract level,
  having been seen only through a summarising fetch. Its report is in
  `notes/n1-reading/adr0006-search.md` (unreviewed agent output).
- Read in full after acceptance (2026-10-29), from the owner's downloads:
  [@jayaraman1953], [@andersen2006] and [@damci2016] (reports
  `notes/n1-reading/tier1-*.md`, unreviewed agent output). They change no
  decision or grade. They correct where the ½ ± ½ rule comes from (Andersen
  et al. take it from [@rasmussen2006]), back the lighting rule and the
  definition of *L* with published cases, widen the photo-limit arithmetic,
  and add one candidate (the pixel floor). Cue (c) stays speculative.
- Read in full next (2026-10-29), from the owner's downloads or fetched
  open access with the owner's agreement: [@rasmussen2006], [@raman1953],
  [@raman1955], [@graetsch1985], [@ojala2002], [@ojala2012],
  [@zolitschka2015] (its applications section skimmed), [@vandenbulcke2019]
  (the accepted manuscript, so pages are "ms p."), [@jamba2025] and
  [@arcusa2022] (reports `notes/n1-reading/tier1-*.md`, unreviewed agent
  output). They change no grade. They show the ½ ± ½ rule is an agreed
  convention, move the 25–75 % wording to [@andersen2006], find no source
  for a pixel rule or a count-against-resolution test, give the first
  density contrast between agate bands, add the nearest published case of
  cue (c), and set the two causes of cue (d) as options. The owner took
  choices 7f–7h from them. [@ojala2002] proved not to be the
  two-resolution count it was listed as.
- Still to read: Holten et al. 1998 for C11 (not yet downloaded); papers the
  reports suggest are in `notes/n1-reading/downloads.md`. The band code
  goes ahead under this ADR.
