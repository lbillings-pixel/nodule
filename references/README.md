# References

## Literature: `catalogue.toml`

Every literature source the project relies on, with:

- **how far it has been checked:** citation, abstract, sections, or full text.
  The docs never claim more than that.
- **its licence.** Only material whose licence allows it (for example CC BY) is
  stored in this repository. Everything else is cited and linked.

Docs cite entries as `[@id]`. The test suite checks that every citation
resolves, using agatesim's catalogue format (`agatesim.references`).

The catalogue started as a copy of agatesim's at commit 0870a9d (2026-10-26)
and grows here; page references in its notes are the journals' unless a note
says otherwise.

## Data held in reserve: `hematite-querry1985.toml`

Hematite's optical constants n, k (ordinary and extraordinary ray, 0.30-1.00 µm)
from Querry 1985, copied unchanged from the public-domain refractiveindex.info
database. No code reads it: it is the data to start from if the carnelian red is
reopened (ADR 0019 section 14).
