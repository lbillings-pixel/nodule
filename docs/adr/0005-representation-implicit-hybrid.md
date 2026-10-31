# 5. Representation: an implicit specimen with coarse world-fixed grids

- Status: accepted (owner, 2026-10-31: option C, fractures cut and offset,
  band coordinate as recommended in section 3, and the recommendations in
  section 5)
- Date: 2026-10-31
- Builds on: [ADR 0003](0003-scope-a-3d-specimen-simulator.md) sections 3–4
  (six subsystems, implicit representation recommended). Its consumers:
  [ADR 0006](0006-band-counts-as-censored-records.md) section 10 (the
  render-and-compare fallback needs N2's renderer) and the N2 vertical slice
  in the [roadmap](../roadmap.md).

## Context

ADR 0003 recommended that shapes be functions of world position, because a
full grid at photo resolution is too large: a 6 cm nodule at 0.05 mm is
1200³ ≈ 1.73 × 10⁹ cells, 6.9 GB per float32 field. This ADR settles the
representation and the interfaces between the six subsystems before N2's
code.

**What exists in agatesim (checked in its source).** The 3D side is already
implicit. A `Solid` answers `contains`, `level`, `wall_distance`, `support`
and `section` at any points. A model is prepared once into an evaluator,
points → fields (agatesim ADR 21), and cuts evaluate it in the cut plane
(agatesim ADR 62). The 3D growth model (agatesim ADR 23,
`volume/growth.py`) is a grid-free deposition-time model. Stages (lining,
spherulites, crystals, settling) each give an arrival depth at any point,
and bands are periods of formation time read at that depth. Grids appear only
as outputs, as world-fixed lattices for H2 and dendrites (agatesim ADRs 25,
27), and as the smooth-noise lattice `SmoothNoise3`, read by trilinear
interpolation. Missing in 3D: fractures, noise-displaced exteriors,
band-limited noise and 3D light transport.

**What the reading says** (22 sources read in full, catalogued 2026-10-31).
All of it is from analogues, structural geology and computer graphics, and
none is evidence about how agate forms (ADR 0002).

- **Implicit fields keep layers from crossing by construction.** Explicit
  surfaces need pairwise checks, sealing and repair [@caumon2009], and
  struggle with closed, lobed shapes [@wellmann2018]. Nested agate bands are
  the "layers must not cross" case. *Verified* as the sources' statements;
  *plausible* for agate.
- **Hybrids are the norm.** GeoChron maps each point to a deposition
  coordinate on demand and keeps properties in that coordinate
  [@mallet2004]. Wellmann and Caumon separate a coarse geometry basis from a
  fine property basis [@wellmann2018]. Procedural texture pioneers stored
  only low-frequency tables and computed the detail [@peachey1985;
  @perlin1985; @dorsey1999]. Distance fields are cached adaptively, not stored
  at full resolution [@frisken2000]. Sparse grids [@museth2013] give little
  saving on a nodule banded throughout (helper estimate ≈ 7.2 GB, an
  inference).
- **Fractures can be functions of position.** A crack is two signed
  distances: φ to the sheet and ψ to its front. The crack is where φ = 0 and
  ψ < 0, and the mesh form is "not necessary" [@moes2002, p. 2552]. A grid is
  needed only to grow cracks [@gravouil2002]. Fault offsets come from a fault
  frame and a slip profile that falls to zero at the tip line
  [@laurent2013]. LoopStructural applies faults as operators in reverse time
  order [@grose2021]. Implicit potential-field faults use a one-sided drift
  step [@calcagno2008; @lajaunie1997], which GemPy 1.0 smoothed
  [@delavarga2019]. A fault may end inside a model only where its offset is
  zero [@caumon2009].
- **Noise added to a field can fold it.** Perturbing a distance field makes
  irregular bodies [@henrion2010], but correlated noise can create spurious
  closed surfaces [@wellmann2018, PDF p. 49]. Sphere tracing needs a
  Lipschitz bound per shape, and loose bounds cost steps [@hart1996]. Filtered
  noise lets previews drop octaves safely [@lagae2010]. Function
  representation gives the algebra [@pasko1995].
- **Real agate bands lose their sharp corners toward the centre.** "As the
  centre is approached, the sharp edges in the banding are commonly lost"
  [@moxon2013, p. 228]. *Verified* as an observation.

**Measured on this PC** (CPU; a 5-sphere `SphereUnion` nodule, 49 × 43 × 43 mm;
10⁶ points uniform in its bounding box):

| Operation | Time | Peak memory |
|---|---|---|
| exact `wall_distance`, 10⁶ points | 21.8 s | 117 MB |
| `SmoothNoise3` evaluation, 10⁶ points | 0.09 s | 56 MB |
| world grid at 0.5 mm: 0.7 M cells | — | 3 MB per float32 field |
| distance transform on it (lower bound for a variable-speed solve) | 0.07 s | 37 MB |
| lookup on it, 10⁶ points: trilinear / tricubic | 0.06 / 0.29 s | 4 / 10 MB |
| world grid at 0.25 mm: 5.9 M cells | — | 24 MB per float32 field |
| distance transform on it | 0.63 s | 289 MB |
| lookup on it, 10⁶ points: trilinear / tricubic | 0.17 / 1.08 s | 4 / 51 MB |

Exact wall distance is the slow part. As implemented, one 1000 × 1000 cut
takes tens of seconds.

## Decision

### 1. Option C: implicit specimen, coarse world-fixed grids where needed

The specimen is an evaluator: world points (mm) → fields. Geometry, bands,
inclusions and fractures are functions of position, evaluated at whatever
resolution a cut or render asks for. Coarse grids at fixed world positions
are allowed for two purposes only:

- **caching smooth coordinates** (wall distance now; a growth-time field
  later) when evaluation is too slow;
- **state of coupled processes** (agatesim ADR 72 decision 5) in later
  milestones.

Grids never hold banded values: bands are always computed from a smooth
coordinate at the point asked for. Grid spacing is an *effective* parameter,
0.5 mm to start, 0.25 mm if the checks in section 4 require it. Grids are
read by tricubic interpolation, because trilinear interpolation makes the
gradient, and so band thickness, jump at every cell (*plausible*,
[inference]; checked in section 4). Near the wall a cached coordinate is
seeded from exact distances, so bands don't come out stair-stepped.

Option A (no grids at all) leaves coupled processes nowhere to keep state.
Option B (a fine grid as the primary state) costs 6.9 GB per field.

### 2. Fractures cut and offset, as reverse-time operators

For the vertical slice: one flat fracture that cuts the bands and offsets
them by slip. No mesh and no grid.

- **The sheet** is two signed functions [@moes2002]. φ is the signed
  distance to the plane, positive on the moving side (Nodule's own
  convention; the paper's printed sign is the reverse). ψ is the signed
  distance to the tip line within the plane, negative inside the fracture.
- **The offset** follows [@laurent2013]. For a flat fault the fault frame
  (φ, a coordinate along the slip, and one across it) has exact formulas.
  The slip size is v_max times a profile that falls to zero at the tip line
  and with distance from the plane. v_max and the profile shapes are
  *effective* until calibrated against offsets measured on the owner's
  stones.
- **It is applied backward.** To evaluate a point, step it back along the
  slip to where it sat before the fracture, then evaluate the unfractured
  model there. This is LoopStructural's reverse-time order [@grose2021].
- **Order of events.** Band noise and cached coordinates live in
  pre-fracture coordinates, so offsets carry the bands with them
  [@wellmann2018, PDF p. 48]. A cache read after the backward step never
  interpolates across the fracture (*plausible*, [inference]; compare the
  special processing near faults that adaptive grids need [@caumon2009]).
- **The fracture itself** is reported as a field (distance to the sheet,
  masked by the tip line) so the renderer and descriptors can show it.
  Aperture is zero in the slice. This is an **assumption**: no source read
  covers a crack that opens and fills with later quartz. Opening and fill
  come later, as a younger body cutting the bands (*speculative*, per
  [@calcagno2008]; [@lei2017] for fracture sets).
- **Rules** (spatial relationships): a fracture ends on the rim, on another
  fracture, or at a tip where its offset is zero [@caumon2009].

Fracture operators are made in agatesim, with its own tests (CLAUDE.md).

### 3. Band coordinate: growth time, starting from wall distance with a capped wobble

What is decided: **bands are a record laid down over growth time, read at
each point's growth time t(x)** [@mallet2004]. The band pattern is a 1D
profile in t. Geometry only supplies t, so an environmental record
(agatesim ADR 14's forcing) stays separate from shape. t is *effective* (order and
apparent rate) unless calibrated.

**First implementation (the vertical slice): agatesim's growth model as it
is.** The lining front advances at one rate everywhere (exact inward
offsets of the wall), plus a wobble. The wobble is world-coordinate noise
whose slope is capped below 1 mm per mm. Where the added term's slope
reaches 1 the band coordinate stops increasing inward. Bands then fold into
islands or come out in the wrong order: [@henrion2010]'s detached fragments,
[@wellmann2018]'s spurious closed surfaces. Below 1 they cannot fold
(*verified* as mathematics, away from the medial surface where wall
distance itself has a kink). The cap applies to the whole added term,
including agatesim's taper near the wall. A cap s limits a band's local
thickness to between 1/(1 + s) and 1/(1 − s) times its average: 0.67× to 2×
for s = 0.5.

**Planned upgrade (not in the slice): a growth-rate field.** The lining rate
varies with position, and optionally with time. t is solved as an arrival
time on a coarse grid (section 1) and read back by tricubic interpolation.
For any positive rate it cannot fold, it allows any thickness ratio, and
each band can thin and thicken in its own pattern. An assumed rate rule (for
example, slower in tight corners) could round corners off toward the centre
as [@moxon2013] observes. That rule would be procedural, not evidence for a
mechanism. Cost: setup of about a second or more (the distance transform is
a measured lower bound; the variable-speed solve is not yet measured), then
lookups of 0.06–0.3 s per 10⁶ points at 0.5 mm (measured).

**What triggers the upgrade:** the owner's blind real-or-synthetic test
(ADR 0006 section 10) or the render-and-compare statistics show either
problem: corners staying sharp to the centre, or bands too uniform around
the nodule.

### 4. Checks, each with its negative control

| Check | Rejects (negative control) |
|---|---|
| Band coordinate increases away from the nearest growth surface (the wall or an inclusion such as a tube) everywhere except at medial surfaces (no folds, no islands)¹ | a wobble whose slope exceeds 1 |
| Fracture offset is zero at the tip line | a fracture with constant slip ending inside the nodule |
| Fracture map is one-to-one (no rock moved onto other rock) | a slip profile steep enough to overlap; sufficient condition τ·‖∇v‖ < 1 per step (*plausible*, [inference]) |
| Band thickness has no steps at grid-cell spacing | the same coordinate read by trilinear interpolation |
| A preview and a full-resolution cut agree where their pixels coincide | noise defined on the pixel grid instead of in world coordinates |

¹ Amended 2026-10-31 (owner). The first wording measured "inward" from the
wall alone. Bands also grow outward from each tube, so in the N2 slice that
wording reported about 200 folds in 1600 point pairs that are not folds
(`nodule.specimen_checks`). Each new inclusion type that bands grow from
joins the growth surface.

### 5. Smaller choices (recommendations the owner took)

- **Noise:** band-limited noise [@lagae2010] is added in agatesim, so
  previews drop octaves without aliasing. Thresholded (sharp-edged) bands
  defeat filtering, so antialiasing belongs to the renderer.
- **Caching:** cache only smooth coordinates, never banded values
  [@frisken2000]. The first cache is wall distance on the 0.5 mm grid.
  Filling it costs about 15 s, scaled from the measured rate and not yet
  measured. Build it only if the slice's cuts or renders are too slow.
- **Sphere tracing:** each `Solid` reports a Lipschitz bound [@hart1996], a
  change made in agatesim. Fractures narrower than the tracing tolerance
  would vanish, so the fracture field is checked separately ([inference]).
- **Renderer:** its own ADR. The reading points to volumetric path tracing
  through the slab, because agate bands are about as thin as light's
  scattering length and cross the cut face [@donner2005; @jensen2001]
  (*plausible*). Its cost is measured before it is chosen.

### 6. Subsystem interfaces

| Subsystem (ADR 0003) | Interface | Where |
|---|---|---|
| Geometry | `Solid`: contains, level, wall distance, support, section, plus a Lipschitz bound (new) | agatesim |
| Pattern generation | growth stages → t(x); band record = 1D profile in t → fields | agatesim `volume/growth.py`; growth-rate field later |
| Feature generation | inclusions as pieces of the growth solid (tubes, dendrites, sticks: existing); fractures as events with a backward map and a sheet field (new) | agatesim |
| Material model | `Fields` from t, stage and impurity (existing) | agatesim |
| Spatial relationships | events in time order; the rules in section 2; the checks in section 4 | Nodule, as criteria |
| Characterization and validation | cuts through the evaluator; the same descriptors on cuts and on library photos | Nodule |

The specimen is one object: an exterior `Solid`, an ordered list of
generators and events, their spawned RNG streams and any coarse grids. It
evaluates to fields at any points.

## Alternatives considered

- **A, pure implicit:** no place for coupled-process state (agatesim ADR 72
  decision 5).
- **B, fine grid as primary state:** 6.9 GB per float32 field at 0.05 mm, and
  grids smear contrasts at their resolution [@wellmann2018, PDF p. 26].
- **Explicit triangulated fractures:** exact, sharp tips, but
  fracture–fracture contacts are "the most consequential step" and need
  sealing [@caumon2009]. Kept as a fallback for curved or connected fracture
  networks.
- **GemPy-style drift faults:** bend the layers rather than cut them in
  GemPy 1.0's implementation [@delavarga2019].
- **Band coordinate from a potential field fitted to traced bands**
  [@lajaunie1997; @calcagno2008]: ties shapes to specimens, but one field
  cannot vary band thicknesses independently, and it needs about a minute
  per 3000 data points.
- **Growth-rate field first:** better fidelity is possible, but it adds
  several days to the slice before the pipeline exists. It is planned as
  the upgrade (section 3).

## Risks

- **Exact wall distance is slow** (21.8 s per 10⁶ points, measured). Renders
  sample many points per pixel. Mitigation: the section 5 cache; faster
  distance code in agatesim.
- **Corners stay sharp to the centre** in the first implementation, against
  [@moxon2013]. Mitigation: the section 3 upgrade, triggered by the owner's
  test.
- **The medial surface:** wall distance has a kink where fronts from
  opposite walls meet. Bands meet there with a crease. It is not known
  whether real agates show the same crease.
- **Slip only:** no aperture or fill in the slice (assumption, section 2).
- **Unmeasured costs:** the variable-speed solve, the wall-distance cache
  fill and rendering. Each is measured before use, and anything over about
  30 s or 2 GB is flagged.
- **Everything here is analogue.** A representation that reproduces a photo
  is one member of a family of fits, and is not evidence for a formation
  mechanism (ADR 0002).

## Consequences

- **agatesim changes, each with its own tests:**
  - a Lipschitz bound per `Solid`;
  - a slope-capped wobble plus the fold check;
  - fracture events (backward map plus sheet field);
  - band-limited noise;
  - a wall-distance cache if needed.

  Each change moves Nodule's agatesim pin.
- **The N2 vertical slice proceeds:** a lobed exterior, wall-parallel bands
  from the growth model, one inclusion type, one flat fracture with offset,
  cut anywhere. Time and memory are measured. Then the renderer ADR.
- **The 22 sources** read for this ADR are in `references/catalogue.toml`.
  Seven further references flagged in [@wellmann2018] are deferred to later
  milestones.
