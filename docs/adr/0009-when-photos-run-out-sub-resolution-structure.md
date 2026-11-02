# 9. When the photos run out: structure finer than they resolve

- Status: proposed (the owner agreed with the outline on 2026-11-02; this
  text awaits their acceptance)
- Date: 2026-11-02
- Builds on: [ADR 0006](0006-band-counts-as-censored-records.md) section 10
  (band realism by set-level render-and-compare), [ADR 0008](0008-set-level-render-and-compare.md)
  (the compare, and the stone's verdict after amendment 7) and
  [ADR 0002](0002-founding-nodule.md) (graded claims).

## Context

- **The owner's concern (2026-11-02).** The evidence from agatesim and
  nodule suggests that banding too fine or faint for the available photos
  is a main stumbling block. At some point the noisy photos will hold no
  more signal about it, and further work on that line would be wasted. The
  project needs a plan for that point before it reaches it.
- **Most of an agate's banding lies below the photos.** Iris striations
  0.1–5 µm apart are "present in virtually all agates", and defects
  oscillate at about 0.2 µm, 700 µm and cm [@heaney1995]. Ordered two-unit
  zoning runs from submicron to about 3 µm [@frondel1985]; periodic
  striations are usually 1–3 µm [@frondel1978]. Zones with iris banding
  exist in nearly every agate, often hidden by pigment [@cartwright2026].
  agatesim ADR 0065 judges a photo fit for calibration at a resolving
  limit of 0.2 mm or finer (borderline to 0.4 mm), and the library's photos
  carry no scale. So only the coarsest level of the band hierarchy is
  resolved (*verified* for the sources' numbers; that this holds for the
  library as a whole is *plausible*).
- **The photos still hold some signal about it.** The stone is outside the
  photos on spectrum alone (ADR 0008, the stone's verdict). Its fine-side
  slope (C26_slope 3.24 against the photos' 95th percentile of 2.93) says
  it has less power just above the resolving limit than the photos do. Fine
  sub-bands are the likely missing source (*plausible*; diagnosis in
  runs/stone-2026-11-02/diagnosis.md). The photos cannot show the
  sub-bands, but they show their combined effect at the limit.

## Decision

1. **Two kinds of photo signal.** *Resolved structure* (bands at or above
   a photo's resolving limit) is measured directly, as now. *Unresolved
   structure* is read only through its combined visible effect: rendered,
   given the photo's blur and noise, and compared at set level (ADR 0006
   section 10). The photos never fix its form beyond that effect.

2. **A stopping rule for each unresolved property** (micro-band spacing,
   sub-band contrast, and the like). Render two settings that span its
   plausible range, from the literature where it gives one. Give both the
   photos' looks, and run the set-level spread test of one setting's cuts
   against the other's, with power by cuts up to the library's photo count
   (97 reflected). If the two are told apart in fewer than 50 % of
   resamples (`effective`), the photos cannot choose between them and the
   property **stops** being studied from photos.
   - Null control: two seeds of one setting must not be told apart.
   - Negative control: a pair that differs in resolved structure (band
     period halved) must be told apart.
   - Cost: the CPU stand-in where it shows the property (about 10 s a cut,
     0.5 GB); otherwise the GPU render (about 100 s a cut). Each run is
     announced with its cost first.

3. **After a stop.** The property is set from published measurements
   (`physical` where a source measured it, cited) or by the simplest
   choice (`effective`). Either way it is labelled *not constrained by
   photos* beside its parameter. No more photo work goes into it without
   new data.

4. **No tuning to the test.** A parameter set from photos is chosen on a
   fixed random half of a stratum's photos and checked on the other half.
   It is reported as calibration, never as evidence that the stone is
   realistic.

5. **Data that resolve more** (the data track; each needs the owner's
   approval, and the specimen-photo rule of CLAUDE.md applies), cheapest
   first:
   - a flatbed scan of a polished, dry slab at 4,800 dpi (about 5 µm a
     pixel);
   - laser diffraction through a thin slab, which reads iris spacing
     directly;
   - open-licence micrographs from papers (agatesim's image-tier policy);
   - thin sections; CT last (all-silica bands may not separate).

6. **What the limit means for each goal** (docs/roadmap.md):
   - **N3, image synthesis.** Unresolved structure matters only through
     what a viewer sees, and the owner's blind real-or-synthetic test is
     the final judge. The goal survives the limit.
   - **N4, realistic specimens.** Each specimen states a resolution floor.
     Structure finer than the floor is procedural and *speculative*, so
     close-up views do not overclaim.
   - **N5, hypothesis testing.** Mechanism claims rest only on resolved
     features or on new data, never on structure the model supplied below
     the floor (ADR 0002).

## Alternatives considered

- **Keep extracting from photos until results stall**, with no rule.
  Rejected: it risks fitting noise, and it gives no point at which to stop.
- **Stop using photos for fine structure now.** Rejected: the spectrum
  slope shows that signal remains.
- **Wait for higher-resolution data before N3.** Rejected: it blocks goals
  1 and 2, which the limit does not block.

## Consequences

- The first candidates for the stopping rule are the stone's missing fine
  power (C26_slope) and the micro-band settings agatesim already has
  (`micro_band_um`, `micro_band_variability`, `iris_fraction`). The band
  regularity being tested now (`period_variability`, `alternation`) is
  resolved structure (about 67 bands over a growth run of up to 30 mm, so
  bands of a few tenths of a mm on average), so the photos
  can still constrain it, under point 4.
- Grades: the sources' numbers are *verified* (read in full, except
  [@cartwright2026], read in sections). That the combined effect is enough
  for N3 is *plausible*. The 50 % threshold and the stopping rule are
  *speculative*, reasoned here, not taken from a source; their controls
  are their only checks.
