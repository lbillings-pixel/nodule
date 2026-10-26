# 1. Record architecture decisions

- Status: accepted
- Date: 2026-10-26
- Carried over from agatesim's ADR 0001

## Context

People reading this project later, including future collaborators, will need to
know why it is built the way it is, not only how.

## Decision

Each significant, hard-to-reverse decision is recorded as a short numbered
Architecture Decision Record (ADR) in `docs/adr/`. Each ADR gives the context, the
decision and its consequences. An accepted ADR is never edited to change its
meaning. A later ADR replaces it and marks the old one `superseded by N`.

agatesim's ADRs (github.com/lbillings-pixel/agatesim, `docs/adr/`) are cited as
"agatesim ADR N"; this project's own as "ADR N".

## Consequences

- Decisions can be traced, and reversing one is a deliberate act.
- Changes to the taxonomy, the claim grades, the tech stack or the model
  architecture need an ADR.
