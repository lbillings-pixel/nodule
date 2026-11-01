"""The renderer: a polished cut face by GPU Monte Carlo (ADR 0007; N2 step 4).

Modules:
- `tgrid`: the cached growth-time grid under the face, filled from agatesim (CPU).
- `scene`: pack the specimen and grid into the GPU scene (`pack` CPU, `upload` GPU).
- `camera`: orthographic camera + standard lights; preview = same code, larger pixels.
- `slab`: forward slab Monte Carlo, the independent reference.
- `reference`: exact and published values for the checks.

Grades (ADR 0007 section 5): checks against analytic and published values are
*verified*; rendered images are procedural and never evidence for how agate
forms or what its bands are made of (ADR 0002).
"""
