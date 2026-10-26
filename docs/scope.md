# Scope: Agate Nodule Simulator

The owner's scope brief (2026-10-26), kept as written apart from formatting.
[ADR 0003](adr/0003-scope-a-3d-specimen-simulator.md) records how it changes
the plan.

## Objectives

Develop a computational simulator for agate nodules that can represent,
generate, and characterize their external morphology, internal structures,
material properties, and spatial organization in three dimensions.

The simulator should treat an agate as a heterogeneous 3D physical object, not
merely as a 2D image or a collection of decorative bands.

The previous project agatesim is a progenitor of this project, Nodule, and the
findings, lessons, and tools from agatesim should be used/carried forward on
Nodule when feasible and as need be.

## 1. Core physical characterization domains

**A. External morphology**
- Overall 3D shape, dimensions, volume, and aspect ratio
- Surface curvature, lobes, protrusions, indentations, and asymmetry
- Surface roughness, weathering, fractures, and other exterior features
- Distinction between original nodule morphology and shapes introduced by
  cutting or polishing

**B. Banding and layered structures**
- Band thickness, spacing, curvature, orientation, and continuity
- Concentricity, nesting, symmetry, and regularity
- Branching, merging, termination, and interruption of bands
- Spatial variation in band geometry and relationships between adjacent bands

**C. Internal inclusions and features**
- Moss, dendrites, plumes, rods, sticks, needles, tubes, and other inclusions
- Cavities, crystal-lined voids, and other internal features
- Feature size, shape, orientation, density, distribution, and connectivity
- Spatial relationships between inclusions, bands, and surrounding material

**D. Internal architecture and topology**
- Relationships among bands, inclusions, cavities, and distinct material
  regions
- Nested structures, connected networks, branching, intersections, and
  enclosure
- Continuity and discontinuity of internal structures
- Spatial organization that cannot be captured by describing individual
  features independently

**E. Color and optical properties**
- Spatial distribution of colors, gradients, mottling, and color boundaries
- Translucency, opacity, light transmission, scattering, and internal
  reflections
- Spatial variation in optical properties across bands and inclusions
- Dependence of observed appearance on lighting, viewing angle, specimen
  thickness, and surface finish

**F. Texture and microstructure**
- Fine- versus coarse-grained appearance
- Fibrous, crystalline, granular, or other textural characteristics
- Variation in texture across bands and material regions
- Separation of macroscopic features from microscopic structures

**G. Mineralogical and chemical heterogeneity**
- Distinct mineral phases and compositional regions
- Spatial variation in mineralogy, impurities, and chemical composition
- Relationships between composition and observed color, texture, or optical
  properties
- Ability to represent unknown or uncertain material composition without
  assuming it can be inferred from appearance alone

**H. Discontinuities and defects**
- Open and healed fractures, cracks, pores, cavities, and missing material
- Geometry, orientation, size, and connectivity of discontinuities
- Interactions between discontinuities and banding or inclusions

**I. Orientation and anisotropy**
- Preferred orientations of bands, inclusions, fibers, and fractures
- Direction-dependent geometric or material properties
- Differences between isotropic and directionally organized structures

**J. Multiscale organization**
- Hierarchy of features at nodule, region, band, inclusion, and
  microstructural scales
- Relationships between patterns at different scales
- Repeated motifs, scale-dependent variability, and changes in structural
  complexity with scale

## 2. Quantitative characterization

Where feasible, supplement qualitative labels with measurable descriptors,
including:
- Geometric dimensions and shape descriptors
- Band-thickness and spacing distributions
- Curvature and orientation distributions
- Inclusion density, size distributions, and spatial correlations
- Color statistics and spatial color variation
- Surface roughness and internal feature density
- Connectivity, branching, and other topological descriptors
- Measures of regularity, variability, anisotropy, and structural complexity

Keep observed morphology separate from hypotheses about formation mechanisms.
Similar-looking structures need not share the same origin.

## 3. Observation and validation

Support characterization from multiple observational modalities, where
feasible:
- Exterior and cross-sectional photographs
- Multiple viewing angles and controlled lighting
- Polished or naturally broken surfaces
- 3D imaging or volumetric reconstruction
- Microscopy and mineralogical analysis, where available

Record which properties are directly observed, which are inferred, and which
remain unknown. Account for the fact that a 2D section or photograph does not
uniquely determine the underlying 3D structure.

## 4. Recommended initial architecture

Organize the simulator around six core subsystems:
- **Geometry:** external shape and 3D spatial domain.
- **Pattern generation:** bands, layers, and other spatially organized
  structures.
- **Feature generation:** inclusions, cavities, fractures, and other internal
  features.
- **Material model:** spatially varying composition, texture, color, and
  optical properties.
- **Spatial relationships:** nesting, connectivity, intersections, anisotropy,
  and multiscale organization.
- **Characterization and validation:** quantitative descriptors,
  visualization, and comparison with real specimens.

Keep the physical representation, procedural generation algorithms,
rendering, and analytical measurements conceptually separate.

## 5. Scope definition, questions and answers

- The findings, lessons, and tools from agatesim should help sharpen scope.
- The goals are image synthesis, realistic specimen generation, and
  scientific hypothesis testing, in that order.
- The simulator should be capable of generating complete 3D nodules, 2D
  cross-sections, and rendered images.
- We need to decide which structures must be explicitly modeled, and which
  can be approximated statistically.
- We need to consider whether and how to expand our reference catalogs of
  real-world data for calibration and validation, and which sorts of data
  sets to prioritize.
- We need to decide which properties must be physically meaningful, and which
  are primarily visual.

## Guiding principle

Prioritize a modular, multiscale 3D representation in which morphology,
internal architecture, material heterogeneity, and statistical organization
are distinct but interacting components. Avoid prematurely committing to a
single generative mechanism or treating visual resemblance as proof of
physical plausibility.
