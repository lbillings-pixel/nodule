# 3. Scope: a 3D specimen simulator, describing before explaining

- Status: accepted (owner, 2026-10-26, with the recommendations in 6a–6d)
- Date: 2026-10-26
- Amends: [ADR 0002](0002-founding-nodule.md) decision 1 (the goal).
  Decisions 2–5 (graded claims, disputed science left open, agatesim as a
  library, shared practices) stand.

## Context

The owner's scope brief ([docs/scope.md](../scope.md)) sharpens and partly
reorders the plan that agatesim ADR 0072 handed to this project:

- **Goal order.** Image synthesis first, realistic specimens second,
  hypothesis testing third. ADR 0002's goal (how formation processes
  interact) is the third of these.
- **Description before explanation.** Ten characterization domains (A–J)
  replace the four taxonomy facets, with measurable descriptors wherever
  feasible. Observed form is kept apart from formation hypotheses: things
  that look alike need not share an origin.
- **An agate as a heterogeneous 3D object.** Complete 3D nodules, 2D
  sections and rendered images, from one representation.
- **Six subsystems,** organised by what the specimen is made of rather than
  by which process made it. Representation, generation, rendering and
  measurement are kept separate.
- **Observation status.** Each property is recorded as observed, inferred or
  unknown, and composition may stay unknown rather than being guessed from
  appearance.

Most of the brief agrees with decisions already made: graded claims, no
mechanism claimed from resemblance, sourced candidates only (lace, fire),
agatesim reused rather than copied. What's new is the goal order, the
breadth (exterior, texture, mineralogy, defects, anisotropy, multiscale), and
3D as the primary form.

## Decision

### 1. Goals, in order

1. **Image synthesis:** rendered images of cut and whole agates that a
   viewer takes for real.
2. **Realistic specimens:** complete 3D nodules whose exteriors, cuts and
   measured statistics match real specimens.
3. **Hypothesis testing:** the coupled-process exploration of ADR 0002, run
   inside the same specimen representation.

Graded claims (ADR 0002) still apply to every statement about formation.
Goals 1 and 2 are judged by a separate **realism score** against real data
and never earn a mechanism grade: a procedural specimen that looks right is
evidence only that it looks right.

### 2. Describe before explaining: a ten-domain schema (ADR 0004)

The schema covers domains A–J of the brief. Its terms are descriptive only.
For example, "flat-lying parallel bands" is a descriptor, while "settled" or
"gravitational" are hypotheses. Each value carries:

- a **qualitative label** and, where feasible, **quantitative descriptors**
  measured by code;
- an **observation status:** observed (with modality: exterior photo, cut
  face, transmitted light, CT, microscopy), inferred (with method) or
  unknown;
- for simulated specimens, the **generator** that made it and its grade.

agatesim ADR 0072's four facets fold in: banding style → B, body shape → A,
internal structures → C, D, H, optical effects → E. Seams stay in scope as a
body shape under A. Every descriptor gets a negative control: a pattern
lacking the property, which it must score as absent.

### 3. Six subsystems, four separate concerns

| Subsystem | Starts from agatesim | New |
|---|---|---|
| Geometry | `geometry`, `shapes`, `cut_shapes`, `volume` (3D runs, cuts) | lobed and indented exteriors, rind, cut vs natural surface |
| Pattern generation | front growth (H1), Liesegang, settling models | procedural bands (statistical, not mechanistic) |
| Feature generation | dendrite engines, crystal growth, nucleation, tubes, sticks | fractures, pores, healed cracks |
| Material model | `fields`, pigments and palette (ADR 0050), `optics` | per-region composition with "unknown", texture, physical optical coefficients |
| Spatial relationships | — | nesting, enclosure, connectivity, intersection rules |
| Characterization and validation | specimen analysis, criteria, classifier, feature maps, reference library | descriptors for A–J, realism score |

Across them, four concerns are kept separate:

- **Representation:** one specimen object in world coordinates (mm).
- **Generators:** procedural or mechanistic, and interchangeable. Each
  writes to the representation and records what it did.
- **Renderer:** turns the representation into images under a stated
  lighting, viewpoint, thickness and finish.
- **Measurement:** the same descriptor code runs on real and simulated
  images or cuts.

### 4. 3D first, with an implicit representation (detail in ADR 0005)

Recommendation: shapes (exterior, band surfaces, cavities, fractures) are
defined as functions of world position, evaluated at whatever resolution a
cut or render needs. Material values attach to regions and fields. A voxel
grid is used only where a mechanistic generator produces one.

Why: a full grid at photo resolution is beyond the GPU. A 6 cm nodule at
0.05 mm per cell is 1200³ ≈ 1.73 × 10⁹ cells, which is 6.9 GB per float32
field, and a specimen needs many fields. Evaluating on demand lets a cut be
made at full resolution without building the volume. It also keeps
randomness resolution-independent (noise in world coordinates), as our rules
require. To be measured in N2.

### 5. Validation that respects 2D → 3D

A cut does not determine the 3D object, so comparisons are made between
**distributions** of descriptors, like with like: random cuts of simulated
specimens against cuts of real ones, and exteriors against exteriors. One
known bias is that bands cut obliquely look wider (t / sin α, with α the
angle between the cut and the band surface). Comparing like with like
handles it without correcting the real data. Methods for estimating 3D
properties from sections (stereology) are to be sourced before use.

### 6. Choices (owner accepted the recommendations, 2026-10-26)

- **6a. Explicit or statistical.**
  - *explicit:* anything larger than a few output pixels whose shape or
    connectivity matters: the exterior, band surfaces, cavities, large
    inclusions, fractures;
  - *statistical:* anything below that: fibre texture, grain, fine pores,
    fine moss. These are carried as per-region distributions, with noise in
    world coordinates.

  The boundary moves with the output resolution, so a close-up makes more
  things explicit.
- **6b. Physical or visual.**
  - *physical units:* geometry (mm) and optical properties (absorption and
    scattering per mm, refractive index). Domain E's dependence on
    thickness, lighting and finish needs these even for goal 1.
  - *visual (labelled `effective`):* colour patterns and procedural
    fields.
  - *composition:* physical only where a mechanistic generator or measured
    data supplies it. Otherwise it stays "unknown", with an appearance-only
    material.
- **6c. Data, in priority order.** Each new external source is asked first,
  per agatesim ADR 0063's tiers.
  1. Re-tag the existing 300-photo reference library on the ten domains. No
     new data needed.
  2. Exterior photos of whole nodules, ideally paired with a cut face of the
     same specimen (domain A, natural vs cut).
  3. The owner's own specimens under a capture protocol: dry polished face,
     diffuse light, a scale, several angles, and transmitted light through a
     thin slab (domain E). These stay on the PC.
  4. Published measurements: band-by-band chemistry, band thickness
     distributions, crystallite size and moganite (domains F, G). Some are
     already in the catalogue.
  5. 3D data: open micro-CT volumes of agate, if any exist, or a sawn slab
     series of one nodule. This is rare but the only direct test of 3D
     structure.

  Items 3 and 5 need the owner's time and specimens (and, for slabs, a saw).
  They are asked about separately when the data track reaches them.
- **6d. What moves.** agatesim ADR 0072's shared-chemistry, coupled-process
  model and its 2D trial of three couplings move from N2 to the hypothesis
  milestone (N5). They become mechanistic generators in the same
  representation. Nothing from ADR 0072 is dropped.

## Alternatives considered

- **Keep ADR 0002's order** (mechanism first, a 2D chemistry trial next).
  This conflicts with the brief's goal order and would build the
  representation around one process family, which the guiding principle
  warns against.
- **2D first, 3D later.** Cheaper to start, but the brief makes 3D the
  primary form, and retrofitting 3D onto a 2D representation is what
  agatesim's `Fields` made costly.
- **A voxel grid as the only representation.** It is simple, but see the
  memory figures in 4.

## Risks

- **Breadth.** Ten domains could stall the project. Each milestone takes a
  first cut of a few domains (roadmap), not all ten in depth.
- **Realism mistaken for explanation.** This is guarded by a separate
  realism score, grades on formation claims only, and generators recorded
  per specimen.
- **Sparse 3D and exterior data.** Most open photos are polished cut faces.
  Exterior shape and true 3D structure may stay weakly validated.
- **Literature gaps.** The catalogue's 67 sources cover formation,
  mineralogy and optics. It has nothing yet on exterior morphology and
  weathering, quantitative band descriptors, 3D imaging of agate, or
  stereology. N1 starts with a reading pass on these, and the owner is asked
  for anything blocked.

## Consequences

- The roadmap is reorganised around the goal order (N1–N5 plus a data
  track).
- ADR 0004 (characterization schema) and ADR 0005 (representation and
  subsystem interfaces) follow before code.
- The README and ADR 0002 now point to this ADR for the goal order.
