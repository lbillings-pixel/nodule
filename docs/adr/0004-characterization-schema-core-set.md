# 4. Characterization schema: a core set measurable from photos now

- Status: accepted (owner, 2026-10-27, with the recommendations in 9a–9c)
- Date: 2026-10-27
- Builds on: [ADR 0003](0003-scope-a-3d-specimen-simulator.md) sections 2
  and 5. Records the owner's section-sampling decision of 2026-10-26.

## Context

ADR 0003 asked for a descriptive schema over domains A–J of the
[scope brief](../scope.md): a label and, where feasible, measured descriptors
for each domain, with an observation status and a negative control for each
descriptor. The N1 reading pass (four topic reports and three full-text tier-1
reports, all sources catalogued) found:

- **Little is agate-specific.** No published measurements of agate exteriors,
  no band statistics beyond the radial widening of bands [@goldbaum2019] and
  Hurst exponents of grey profiles [@holten1997; @holten1998], and no
  published agate CT volume. The methods come from analogues: tree rings,
  varves and speleothems, stripe patterns, sediment grains, and crystal-size
  stereology. Each is plausible for agate, not yet shown.
- **A section does not determine the 3D object.** From single photos,
  particle numbers, size distributions of irregular shapes and connectivity
  cannot be measured at all [@cruzorive2017]. True band thickness and 3D
  volume fractions can be estimated only from cuts placed at random, and every
  2D-to-3D method read assumes such cuts [@morgan2006; @higgins2000]. With
  biased cuts they fail silently: the fit looks good and the answer is wrong.
  Even unbiased single sections scatter widely [@mock2005].
- **What the reference library can give.** agatesim's library has 300 open
  Commons photos, 230 usable (agatesim ADR 0066). None has a scale (agatesim
  ADR 0069), each specimen appears once, nearly all are polished cut faces
  (25 tagged rough, 21 polished), and where the cut was placed is unknown.
  So the descriptors usable now are those that need one face and no scale.

The owner chose (2026-10-27) a **core set** of about 20 descriptors measurable
from the library now, slots for descriptors that need more than a photo, and
a list of later candidates.

## Decision

### 1. Every observation is a record

A specimen (real or simulated) has one or more **views**. Each view records:

- **kind:** cut face, exterior, slab in transmitted light, broken surface;
- **finish:** sawn, polished, natural, tumbled, unknown;
- **lighting:** reflected, transmitted, unknown;
- **scale:** mm per pixel, or none;
- **section-sampling model** (section 2), for cut faces.

Each descriptor value records its **status**: observed (with the view it came
from), inferred (with the method and the model it assumes) or unknown. Lengths
measured on a section are labelled **apparent**. Values from projections
(exterior outlines) and from sections (cut-face outlines) are kept apart,
since they are different quantities [@blott2008]. A simulated specimen also
records its generator and grade (ADR 0003).

### 2. Section-sampling model (owner decided, 2026-10-26)

Photo sets are not random cuts: cutters look for the eye. So every set of cut
faces declares how its cuts were placed: **random**, **centre-seeking**,
**long-axis** or **unknown**. Synthetic cuts compared with a set are drawn the
same way. Inference that needs random cuts (3D volume fractions, sizes, 3D
shape from sections) is refused for any other model, and section tables for
shape inference are built per sampler, as [@higgins2000] builds them per cut
geometry. Rejected: treating photo statistics as descriptive only.

### 3. Observation ladder

Each descriptor names the lowest rung it needs:

1. **photo:** one face or one exterior view;
2. **slab pair:** two parallel faces a known distance apart;
3. **serial sections:** a slab series, resolving objects larger than the
   spacing;
4. **3D:** CT or serial grinding.

A weathered exterior gives partial true 3D shape [@higgins2000]. Where the
data stop short of a descriptor's rung, the value is **unknown**, never
estimated.

### 4. The core set (22 descriptors, photo rung, no scale needed)

Lengths are divided by the face's inner radius or by the median band spacing,
as agatesim ADR 0069 does. The same code measures photos and simulated cuts.
Five come from agatesim's specimen code (marked "agatesim") and are imported,
not copied. Grade is that of the claim "this measures what it says on agate
photos": *plausible* where the method is borrowed from an analogue,
*speculative* where no source was found.

| # | Domain | Descriptor | Source | Negative control | Grade |
|---|---|---|---|---|---|
| C1 | A | Outline aspect ratio b/a and regularity A/(πab) from an ellipse fit, per view kind | [@shea2010] | An ellipse gives regularity 1 at any aspect ratio; a lobed outline of the same aspect ratio gives < 1 | plausible |
| C2 | A | Riley circularity √(Di/Dc) | [@blott2008] | Square 0.841, octagon 0.961 exactly; the same shape at two resolutions gives the same value | plausible |
| C3 | A | Concavity irregularity I(2D) | [@blott2008] | Any convex outline gives exactly 0; 4-, 8- and 16-pointed stars rank in order | plausible |
| C4 | A | Exterior surface class: smooth, blistered, ridged, lobed, coarse-grained, broken (exterior views only) | [@breitkreuz2013] (lithophysae; no scheme exists for amygdale agates, so classes may be added) | A smooth ellipsoid render is classed smooth, with no ridges | plausible |
| C5 | A, G | Rind: absent or present, its kind (adhering host, clay or celadonite coat, spherulitic shell) and rim width relative to the face | [@jain2013; @goetze2024; @breitkreuz2013] | A tumbled stone with its coat removed gives "no rind"; adhering host rock is labelled host, not rind | plausible |
| C6 | B | Band count per profile, as a range over a parameter sweep, with the agreement of two counting rules | [@weber2010] | A uniform image gives 0; red noise gives counts that swing with the parameters | plausible |
| C7 | B | Thickness spread: CV of apparent band widths (agatesim) | agatesim ADR 0069 | Even bands give ≈ 0 | plausible |
| C8 | B | Rim-to-centre trend: slope of log apparent width against relative depth (agatesim) | [@goldbaum2019] | Bands with no trend give ≈ 0 | plausible |
| C9 | B | Spacing ratio: median ratio of successive widths (agatesim). A description, not evidence of a Liesegang mechanism [@goetze2020] | [@droz1999; @jahnke2008] | Equal spacing gives 1; a random renewal sequence must not settle on a ratio | plausible |
| C10 | B | Persistence: lag-1 autocorrelation of the thickness series | [@baker2008] | A shuffled copy of the series gives ≈ 0 | plausible |
| C11 | B, J | Grey-profile roughness: Hurst exponent over 2–10 band spacings (agatesim) | [@holten1997; @holten1998] | A periodic profile shows a crossover, not one power law | plausible |
| C12 | B, D | Continuity round the face: correlation of thickness series between profiles at different angles | [@baker2008; @marichal2025] | Independent series per profile give ≈ 0; perfect concentric bands give ≈ 1 | plausible |
| C13 | B, D | Concentricity and completeness: drift of successive band centres over the spacing; share of 360° present; count of wedging bands | [@marichal2025; @gillert2023] | Parallel stripes, or two separate concentric sets, fail or report two centres; random closed blobs give no nesting order | plausible |
| C14 | B, D | Defect density: band terminations and junctions per period² | [@murphy2015] | Perfect concentric bands give 0; thresholded noise of the same period gives a high density | plausible |
| C15 | B, I | Curvature: distribution of band-contour curvature times median spacing; count of sharp corners (fortification angles) | [@riecke2006] | Straight stripes give 0; a circle gives 1/r; a roughened circle gives 1/r after smoothing | plausible |
| C16 | I | Orientation order: structure-tensor coherence, Herman's order parameter, correlation length relative to face size; share of banded area with straight parallel bands | [@bigun1987] (method, citation only); [@murphy2015] | Isotropic noise gives coherence ≈ 0 and no profile is extracted; concentric rings give a parallel share ≈ 0, stripes ≈ 1 | plausible |
| C17 | D, G | Core and cavity area shares: coarse-quartz core, open cavity (core share from agatesim). Area share of this face only, not a volume fraction | [@cruzorive2017] | A face with no core gives 0; centre cuts of a phantom must not be reported as its volume fraction | plausible |
| C18 | C | Inclusion area share and profile count per face area, by class (moss, tube, stick, plume), with profile aspect ratios. Not converted to 3D number or size | [@higgins2000; @cruzorive2017] | Fortification-only faces give ≈ 0 (false-alarm rate) | plausible |
| C19 | D | Band topology: connected pieces and closed rings of the light and dark phases over a threshold sweep | [@riecke2006] | Concentric rings of known count give that count exactly; noise gives many small pieces | plausible |
| C20 | B, D | Eye fraction (per photo set): share of faces showing the innermost zone. Under random cuts it estimates the ratio of inner to outer mean width; under other models it is descriptive only | [@coleman1992] | Random cuts of a phantom with core radius 0.3 R show the core in ≈ 30%; a centre-seeking set (≈ 100%) is flagged non-random, not read as a large core | plausible |
| C21 | E | Colour: chroma spread, band-to-band colour contrast and number of colour classes, after lighting flattening. `effective`; lighting unknown | none (assumption) | A greyscale image gives chroma 0; a uniform tint changes mean hue but not contrast | speculative |
| C22 | H | Fracture traces: count per face, open or filled, and whether bands are offset across them | none (assumption) | An unfractured synthetic face gives 0 | speculative |

Composition (G) has no core descriptor: from photos it is always unknown
(ADR 0003, 6b). Texture (F) and hierarchy (J) need a scale or a second
resolution (section 5).

### 5. Slots that need more than a photo

These are part of the schema now, so a record says "unknown" rather than
leaving a gap. Asked of a single photo, each must return unknown: that is
their negative control against overclaiming.

| Descriptor | Needs | Source |
|---|---|---|
| Any size in mm | a scale in the photo | agatesim ADR 0069 |
| True band thickness and its distribution | 3D, or a known band normal; from sections only its moments [@coleman1993] | [@coleman1993; @cruzorive2017; @hildebrand1997] |
| Thickness against band slope (wall-lining or flat-lying growth) | a known "up" (slab pair) or 3D | [@batchelor2004] |
| Concentric-shell test: ρᵢ² − ρⱼ² the same in every cut | a cut plus exterior calipers, or a slab pair | [@coleman1992] |
| 3D form: long, intermediate and short axes, form class, true sphericity | exterior views in three orientations, or 3D | [@blott2008] |
| Volume fractions (core, cavity, moss) | random cuts or a slab series | [@cruzorive2017] |
| Inclusion number per volume and size distribution | slab pairs or 3D, and one shape habit | [@higgins2000; @morgan2006; @mock2005] |
| Connectivity of moss and tube networks | slab pairs, serial sections or 3D. Control: connected tubes and separate rods with the same section statistics must both give unknown from photos | [@cruzorive2017] |
| Band surfaces in 3D: sheet edges, branch lines | 3D | none yet |
| Oblique-slice test for any 2D-to-3D generator | 3D | [@kopf2007; @kench2021] |
| Absorption and scattering per mm, translucency | transmitted light through a slab of known thickness (capture protocol) | ADR 0003, 6c |
| Composition | a measurement or a mechanistic generator | ADR 0003, 6b |
| Grain, fibre texture, micro-bands (up to 1000 per mm [@goetze2020]) | a scale and micrographs | [@goetze2020] |
| CT records | voxel size, energy and filter, artefact corrections. Ring and beam-hardening artefacts mimic bands and rind; controls are a plain silica cylinder and an off-axis rescan | [@ketcham2001] |

### 6. Later candidates (listed, not adopted)

Wavelet and multitaper spectra of profiles [@torrence1998; @lantink2022];
radial chirp [@goldbaum2019] and width exponent [@droz1999]; nested band
hierarchies [@lantink2022]; boundary roughness [@murphy2015]; two-point and
lineal-path correlations [@jiao2007; @gommes2012]; Wadell roundness
[@blott2008]; spherical-harmonic spectra of exteriors [@jia2016; @sun2020];
abrasion state and rest points [@domokos2014; @novakszabo2018; @domokos2020];
ridge and star symmetry of thunder eggs [@ross1941]; cavity classes, fill
sequence and geopetal direction [@breitkreuz2013; @goetze2024]; Hurst exponent
of the thickness series; flags for sawn and polished surfaces (speculative, no
source found). Each joins the core set by amendment, with its negative
control.

### 7. Descriptive terms, trade names as aliases

Labels are descriptive. agatesim's form names stay as aliases, so old tags
carry over, but none of them is read as a cause:

| agatesim form | Descriptive term |
|---|---|
| fortification | concentric bands following the wall, with sharp corners |
| waterline | flat-lying parallel bands |
| fortification-waterline | both, in one face |
| eye | small closed rings around a point |
| quartz-geode | coarse-quartz core or open cavity |
| moss, tube, stick, plume | inclusion classes (C) |
| botryoidal | bands following rounded bulges |
| shadow, iris | optical effects, lighting-dependent (E) |
| seam | body shape (A), as in agatesim |

### 8. Re-tagging the library

- Code computes C1–C3, C6–C19 and C21 on the 230 usable photos, importing
  agatesim's profile code. Timing is measured on 10 photos first; agatesim's
  five statistics take about 5 minutes for the whole library.
- Existing agatesim tags carry over through section 7. New labels (view kind,
  finish, lighting, C4, C5, C22) are tagged per choice 9b.
- Tags live in this repository's `references/library-tags.toml`, keyed by
  agatesim library id. No image is copied.
- **Outline check (owner decided, 2026-10-28).** On 10 sampled photos,
  agatesim's segmentation, built to find a working region for band profiles,
  often missed the stone's rim: it cut inside the face, took in stand clips,
  or picked a patch of a cropped close-up or of a photo that is not a cut
  face. C1–C3 describe whatever outline they are given, so a value counts only
  after the outline has been seen to follow the rim. In the labelling batches
  (9b) each photo is shown with its outline drawn on and gets a yes/no
  "outline follows the rim"; "no" makes C1–C3 unknown. An outline touching
  the frame is unknown without a check, as [@shea2010] discard
  edge-bordering vesicles. Rejected: improving agatesim's segmentation first
  (worth doing later, in agatesim); C1–C3 only on photos taken to a protocol.
- **Band count and persistence (2026-10-28).** agatesim's profiles reach a few
  band spacings each way, too short to count a face's bands, so C6 and C10
  use new rim-to-centre profiles that follow the band normal
  (`nodule.bands`): two of [@weber2010]'s counting rules over a parameter
  sweep, and the lag-1 of detrended log thickness less a shuffled baseline
  [@baker2008]. Tests check drawn rings of known count and both controls (a
  flat profile counts 0; red noise swings with the parameters; a shuffled
  series gives ≈ 0). On 10 library photos the counts were low against the eye
  (median 1–6 per face) and wide over the sweep (one profile 3–16), and C10
  could be judged on only a few faces. C6 is a lower bound where the outline
  cuts inside the rim, so it shares C1–C3's outline check; on photos it
  stays plausible at best and is read as a range.

### 9. Choices (owner accepted the recommendations, 2026-10-27)

- **9a. Sampling model of the library.**
  - *(Recommended)* **Unknown, compared as centre-seeking.** Commons photos
    show chosen faces, and most show a closed centre (189 of 230 tagged
    fortification), which suggests centre-seeking cuts. Synthetic cuts compared with the library are
    then drawn centre-seeking, and volume fractions and sizes stay unknown.
  - **Random.** Unlocks volume fractions from the library, but most likely
    biased, and the failure would be silent.
  - **Unknown, no set-level comparison** until a set of known sampling exists.
- **9b. Who tags the new labels.**
  - *(Recommended)* Claude proposes labels in batches with helper agents
    (Commons photos only), and the owner checks a random sample of about 30.
  - The owner tags all of them.
  - Measured descriptors only for now, labels later.
- **9c. Core-set size.** All 22 as listed *(recommended)*, or a smaller first
  cut (C1–C3, C7–C11, C17, C20).

## Alternatives considered

- **The full schema at once** (about 60 candidates from the reading). Most
  need slab pairs or 3D and couldn't be tested now.
- **Photo statistics as description only** (no sampling model). Rejected by
  the owner: it blocks every comparison with random synthetic cuts.
- **Descriptors defined in 3D only**, with 2D as a projection. Cleaner, but
  nothing could be checked against the library until 3D data exist.

## Risks

- **Bias toward bands.** Most library photos are banded cut faces; exteriors
  (C4, C5) have about 25 photos, so domain A stays weakly validated.
- **Colour under unknown lighting.** C21 is effective only and may mostly
  measure the photographer.
- **One face per specimen.** Single-section scatter is large [@mock2005], so
  only set-level distributions are compared, never one photo against one cut.
- **Analogue methods.** Every plausible grade above rests on methods from
  other fields. Each is upgraded only by its negative control and, later, by
  slab pairs or 3D data.

## Consequences

- The record of section 1 and the descriptors become code in N2 alongside
  ADR 0005; their measurement side runs on 2D images and doesn't wait for the
  3D representation.
- All sources cited are already in the catalogue. Any change agatesim's code
  needs (for example, exposing per-profile thickness series) is made in
  agatesim, with its own tests, and the pin is moved.
- The roadmap's N1 ends with the re-tagged library.
