# 6. Band counts as censored records, and what the eye adds

- Status: accepted (owner, 2026-10-29, with the recommendations in 7a–7e;
  the reading of 7e found no source that changes them)
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
  least 2 pixels, with blur and JPEG compression often 4 or more. So the
  finest couplet a good library photo can show is about 0.04–0.1 mm, two
  orders of magnitude above iris spacing. Library photos have no scale, so
  even this limit is known only in pixels.
- **So a band count belongs to the photo, not the stone.** C6 (machine,
  `nodule.bands`) and `C6_eye` (owner) count the bands coarser than the
  photo's limit. The same face photographed sharper gives more. ADR 0004
  already reads C6 as a lower bound where the outline cuts inside the rim;
  the same holds at every face for bands below the limit. C6 can also
  overcount, from noise or texture, which its range over the sweep shows
  [@weber2010].
- **Layer counting elsewhere meets the same limit.** In ice cores, "limited
  measurement resolution" can "decrease the number of layers that can be
  recognised", and the GICC05 count enters each uncertain layer as ½ ± ½
  year, so the uncertainty is an interval, not a point [@winstrup2012]. No
  source found (time-boxed search, 2026-10-29) measures how a count changes
  over a ladder of resolutions, or explains agate's shadow effect.

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
  checked on synthetic faces of known blur.
- **Bands finer than *L* are unknown, not zero.** No record may store or
  imply a count of them.
- **`C6_eye`** is stored the same way: "at least *n*", with *L* of the
  photo it was given on. The eye's own limit isn't measured; it is taken to
  be no finer than the photo's (assumption).
- **Counts are compared only at a common limit.** Two counts with different
  *L* are compared after the finer one is recounted at the coarser limit
  (section 2). Counts are also compared only within one lighting kind,
  since transmitted light can invert which bands look light (owner's note on
  agate3; ADR 0004 section 1 already records the lighting).

Negative controls: a synthetic face with fine bands below *L* drawn over
coarse ones must count only the coarse ones and report *L* above the fine
spacing; the same synthetic stone imaged at two resolutions must give a
higher-resolution count at least the lower-resolution one, and equal counts
once both are taken at the coarser *L*. Grade: plausible (the limit is
standard sampling; its use on agate photos is untested).

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
kept beside N(L). A varve study comparing CT with thin sections bears on
this and is to be read in full before it is cited for more than its
abstract [@jamba2025].

Negative controls: rings of one spacing give a flat N(L) that drops to 0
once *L* passes the spacing; two drawn scales give two steps; red noise must
give a curve whose rule agreement fails (ADR 0004, C6's control). Grade:
speculative (no source applies count-against-resolution to laminae; choice
7c).

### 3. New descriptors from the owner's notes (photo rung)

| # | Domain | Descriptor | Source | Negative control | Grade |
|---|---|---|---|---|---|
| C23 | B, D | Centre count: separate band centres in the face, by eye, cross-checked against C13 where it is measured | owner's notes; [@marichal2025] (via C13) | One concentric set gives 1; two drawn separate sets give 2 | plausible |
| C24 | B, E | Fine-banding evidence, per zone (outer, middle, inner): each cue present, absent or unknown: (a) colour gradient within a band, (b) faint striations at the limit, (c) shadow-like bands in radial sectors, (d) iris colours | owner's notes; [@heaney1995; @cartwright2026] for (d) | A synthetic face with no sub-limit bands must give no cue under the same rendering; a cue is never converted into a count | speculative |
| C25 | B, F | Zone sequence rim to centre: each zone's kind (even bands, curdled or mottled bands, colour change, coarse quartz, cavity) and its share of the radius | owner's notes; ADR 0004 C17 for the core | A face of one even banded zone gives one zone; C17's core and C25's quartz zone agree | plausible |

C24's cues are signs, not measurements. Cue (c) is the owner's observation;
no peer-reviewed source found explains the shadow effect, so it stays
speculative. Neighbouring agate bands can differ in translucency through
crystallite size alone (about 5 µm against 300 nm, by scattering)
[@lorenzi2022], which a shadow effect would need, but that source says
nothing of viewing angle, sectors or band spacing. The likeliest optics
source (Jayaraman 1953, with [@raman1955] and [@graetsch1985]) is on the
owner's download list. The owner's idea of reading
fine-band spacing from shadow banding is listed as a later candidate.

### 4. Who labels

Helper agents propose C23–C25 for the banded faces from the outline-check
pictures and, on the 30 checked faces, from the owner's notes; the owner
checks those 30 on the check page and the check overrides the proposal
(as ADR 0004, 9b; choice 7b). The pre-filled proposals may anchor the
owner's answers, as the first check may have been (agreement 28–30 of 30);
the check page shows C24 blank, so that one is answered unprompted.

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
  known.
- **Later, need a thin slab** (the owner has none): laser diffraction of
  iris banding, thin sections. CT may not separate bands that are all silica:
  an assumption, since no source read measures the density contrast between
  agate bands.

### 6. Fallback: measure the image, not the stone

Checked at the end of the re-tagging under this ADR. It is taken if any of:

- the machine count, at the owner's limit, disagrees with `C6_eye` on more
  than a third of the faces where both are given;
- the owner's check disagrees with the proposed C23–C25 on more than a third
  of the 30;
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

## Alternatives considered

- **Keep C6 as a count and note the limit in prose.** Rejected: the record
  would still read as the stone's band number.
- **Drop band counts.** Loses the one band descriptor the owner can check by
  eye, and C10 needs the thickness series either way.
- **Estimate sub-limit bands from cues.** Nothing read supports a
  conversion; listed as a later candidate.

## Risks

- **The limit is only as good as its measure.** A wrong *L* makes
  comparisons at a "common" limit uneven; choice 7a's measure needs its own
  check on synthetic faces of known blur.
- **Cues are subjective.** C24 rests on one observer's eye under unknown
  lighting; it informs where to scan, not the schema's claims.
- **Few faces.** C6 stands on 64 faces and `C6_eye` on 11, so the fallback
  test is coarse.

## Consequences

- `nodule.bands` gains *L* and N(L); `nodule.tags` stores C6 and `C6_eye` as
  lower bounds with *L*; the check page gains C23–C25. Timing is measured on
  10 photos first (current re-tagging takes about 630 s; N(L) adds about one
  recount per step).
- ADR 0004's C6 entry is read through this ADR; its other descriptors are
  unchanged.
- All sources cited are in the catalogue. Choice 7e's search added five,
  at the level read: [@lorenzi2022] and [@winstrup2012] from their texts;
  [@jamba2025], [@arcusa2022] and [@vandenbulcke2019] at abstract level,
  having been seen only through a summarising fetch. Its report is in
  `notes/n1-reading/adr0006-search.md` (unreviewed agent output).
- Owner downloads that could lift a grade: Jayaraman 1953
  (10.1007/BF03045262) for cue (c); Andersen et al. 2006
  (10.1016/j.quascirev.2006.08.002), the origin of the ½ ± ½ rule; Damci and
  Cagatay 2016 (10.1016/j.quaint.2015.05.007), a rule linking sampling
  resolution to the thinnest countable lamina.
