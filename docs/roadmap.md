# Roadmap

Milestones end with a check-in: results, images, and the next decisions. The
order follows the goals in [ADR 0003](adr/0003-scope-a-3d-specimen-simulator.md):
image synthesis, then realistic specimens, then hypothesis testing.

| Milestone | Scope | Status |
|---|---|---|
| **N0: Foundation** | Repository, CI, the founding decision ([ADR 0002](adr/0002-founding-nodule.md)), agatesim as a pinned library, the literature catalogue carried over | done |
| **N1: Scope and characterization schema** | The owner's [scope brief](scope.md) as ADR 0003. A reading pass on the gaps (exterior morphology and weathering, quantitative band descriptors, 3D imaging of agate, stereology), done. [ADR 0004](adr/0004-characterization-schema-core-set.md), accepted: a core set of 22 descriptors measurable from photos now, slots for those needing slab pairs or 3D, a section-sampling model per photo set. [ADR 0006](adr/0006-band-counts-as-censored-records.md), accepted: band counts as lower bounds with the photo's resolving limit, three descriptors from the owner's notes, a render-and-compare fallback, taken 2026-10-31 (section 10). The reference library re-tagged on the domains | in progress |
| **N2: Representation and a 3D vertical slice** | ADR 0005: the specimen representation and the six subsystems' interfaces. One simple nodule end to end: a lobed exterior, wall-parallel bands, one inclusion type, one fracture; complete in 3D, cut anywhere, rendered; descriptors measured on its cuts and on library photos by the same code, each with a negative control. Time and memory measured | planned |
| **N3: Image synthesis** | Procedural generators for exterior, banding, inclusions, defects and colour (domains A, B, C, E, H); optics in physical units, so appearance depends on lighting, thickness and finish; a realism score against the library (descriptors, classifier, feature maps) | planned |
| **N4: Realistic specimens** | Architecture, texture, composition, anisotropy and multiscale structure (D, F, G, I, J); whole nodules whose random cuts and exteriors match real distributions; cut and polished surfaces vs natural ones | planned |
| **N5: Hypothesis testing** | agatesim ADR 0072's coupled-process programme as mechanistic generators in the same representation: shared chemistry, two banding drivers, fabric per cell, ageing, optics from fabric; graded claims; mixtures of forms against the library | planned |
| **Data track** (alongside) | Reference data in ADR 0003's priority order: library re-tagging, exterior photos paired with cuts, the owner's specimens under a capture protocol, published measurements, 3D data; each new external source asked first | ongoing |
