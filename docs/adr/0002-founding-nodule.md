# 2. Founding Nodule: coupled processes in one nodule, with graded claims

- Status: accepted (owner, 2026-10-26)
- Date: 2026-10-26
- Builds on: agatesim ADR 0072 (the full reasoning and the literature passes
  behind this project)

## Context

agatesim reproduces thirteen classic agate forms one at a time, each from one
mechanism or a fixed chain of stages. The owner's review (agatesim ADR 0072)
found four problems with that frame: real specimens mix several forms that
interact; processes run one after another instead of together; pigments sit
apart from the chemistry that shapes growth; and the list of forms mixes four
kinds of phenomenon. Sixteen papers read in full for that review showed that
the main questions (what drives banding, what the silica precursor is, what
twists the fibres) are disputed in the literature itself.

The owner decided that agatesim is complete as a first project, and that the
lessons go into a new, separate project: Nodule.

## Decision

1. **Goal.** Explore how formation processes interact within a single nodule
   or seam, and which combinations of conditions give the mixtures of forms
   seen in real specimens. "Explore", not "show": the project does not claim
   to prove how agate forms.
2. **Graded claims.** Every result is labelled:
   - *verified:* passes stated criteria with every process sourced and
     nothing judged imposed (as agatesim ADR 0035);
   - *plausible:* every process has a source, but the couplings between them
     are assumed and stated;
   - *speculative:* a process or coupling without a source.
   Procedural (rule-based) results are never evidence for a mechanism.
3. **Disputed science stays open.** Where sources disagree, the model offers
   the alternatives as options (two banding drivers; gel or solution) rather
   than choosing one.
4. **Built on agatesim.** Nodule depends on agatesim as a library, pinned to a
   commit, and reuses its geometry, solvers, rendering and optics, 3D cuts,
   criteria, specimen analysis, classifier and feature maps, progress
   reporting and the reference catalogue's format. agatesim keeps receiving
   fixes; Nodule doesn't copy its code.
5. **Same practices as agatesim:** ADRs, a literature catalogue with honest
   `checked` levels, reproducible runs, resolution-independent randomness,
   tests including negative controls, and the owner's specimen photos never
   leaving the PC.

## Consequences

- The design choices listed in agatesim ADR 0072 (a faceted taxonomy, one
  shared chemistry, fabric per cell, an ageing step, optics from fabric,
  horizontal bands as batches, the reference library as yardstick) are this
  project's starting brief. Each becomes its own ADR here before code: the
  taxonomy first (ADR 0003), then the architecture with a small 2D trial.
- `references/catalogue.toml` starts as a copy of agatesim's catalogue at
  commit 0870a9d and grows here.
