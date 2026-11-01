"""The cached growth-time grid under a cut face (ADR 0007 section 1, choice 7c).

The grid lies in RESTORED (pre-fracture) coordinates, in the cut plane's frame:
point = origin + s u + t v - w n, with w the depth below the face (w > 0 inside
the slab). It holds agatesim's smooth growth time `Fields.t_smooth` (agatesim
ADR 0076 section 6), NaN outside the stone and in tubes. The renderer maps a
present point back through the fractures and reads t here by trilinear
interpolation; the stone and tube edges are tested exactly (scene.cuh), so the
NaN cells are filled outward first (`fill_nan`) to define t up to those edges.

Axes sit at integer multiples of the spacing in plane coordinates (as
`Specimen.window`), so grids at different spacings share nodes. Known error
(notes/n2-proto/RESULTS.md, 0.2 mm): |dt| p50 0.0016, p99 0.037, max 0.21
band periods; the tail is band corners where fronts from different wall lobes
meet, rounded within about one cell [inference].

Cost (n2-slice, 0.2 mm, 10 mm deep): 4.6 M points, 18 MB float32, fill
~9 s (~2 s per million points, CPU). 0.1 mm is ~36 M points, ~75 s: over the
30 s flag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from agatesim import progress
from agatesim.grid import FloatArray
from agatesim.volume.slicing import Plane
from numpy.typing import NDArray

from nodule.render.record import BandRecord
from nodule.specimen import Specimen

CHUNK_POINTS = 500_000  # points per evaluation call; bounds peak memory
FILL_SECONDS_PER_MPOINT = 2.1  # measured 2026-11-01 (n2-slice, RTX 4070 PC, CPU fill)

F32 = NDArray[np.float32]


def grid_axes(
    specimen: Specimen, plane: Plane, spacing_mm: float, depth_mm: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """(s, t, w) node coordinates (mm): the cut window padded by total slip + h
    (`Specimen.window`), and depths from -pad to depth + pad."""
    if not (spacing_mm > 0 and depth_mm > 0):
        raise ValueError("spacing_mm and depth_mm must be positive")
    s, t = specimen.window(plane, spacing_mm)
    pad = specimen.max_slip_mm + spacing_mm
    k0, k1 = math.floor(-pad / spacing_mm), math.ceil((depth_mm + pad) / spacing_mm)
    return s, t, np.arange(k0, k1 + 1) * spacing_mm


def fill_cost(
    specimen: Specimen, plane: Plane, spacing_mm: float, depth_mm: float
) -> tuple[int, float, float]:
    """(points, estimated fill seconds, float32 megabytes) before filling, to flag
    fills over ~30 s (CLAUDE.md)."""
    s, t, w = grid_axes(specimen, plane, spacing_mm, depth_mm)
    n = len(s) * len(t) * len(w)
    return n, n / 1e6 * FILL_SECONDS_PER_MPOINT, n * 4 / 1e6


def to_world(plane: Plane, s: FloatArray, t: FloatArray, w: FloatArray) -> FloatArray:
    """World points (n, 3) at plane coordinates s, t and depth w (each (n,))."""
    return (
        plane.origin
        + np.asarray(s)[:, None] * plane.u
        + np.asarray(t)[:, None] * plane.v
        - np.asarray(w)[:, None] * plane.normal
    )


def to_plane(plane: Plane, points: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Inverse of `to_world`: (s, t, w) of world points (n, 3)."""
    rel = points - plane.origin
    return rel @ plane.u, rel @ plane.v, -(rel @ plane.normal)


@dataclass(frozen=True)
class GrowthGrid:
    plane: Plane
    s: FloatArray  # (ns,) mm along plane.u
    t: FloatArray  # (nt,) mm along plane.v
    w: FloatArray  # (nw,) mm depth below the face
    values: F32  # (nw, nt, ns) t_smooth; NaN outside the stone and in tubes
    bands: BandRecord | None = None  # what fills each growth step, from the same nodes

    @property
    def spacing_mm(self) -> float:
        return float(self.s[1] - self.s[0])

    @classmethod
    def fill(
        cls, specimen: Specimen, plane: Plane, spacing_mm: float, depth_mm: float
    ) -> GrowthGrid:
        """Evaluate agatesim at every node (CPU; ~2 s per million points), keeping
        what fills each growth step (`bands`) for the materials table."""
        s, t, w = grid_axes(specimen, plane, spacing_mm, depth_mm)
        shape = (len(w), len(t), len(s))
        n = math.prod(shape)
        out = np.empty(n, np.float32)
        bands = BandRecord.empty()
        for a in range(0, n, CHUNK_POINTS):
            progress.report(a / n, "filling the growth-time grid")
            k, j, i = np.unravel_index(np.arange(a, min(a + CHUNK_POINTS, n)), shape)
            p = to_world(plane, s[i], t[j], w[k])
            inside = specimen.volume.solid.contains(p)
            f = specimen.volume.evaluate(p, inside)
            if f.t_smooth is None:
                raise ValueError("the model gives no smooth growth time (agatesim ADR 0076 s. 6)")
            out[a : a + len(p)] = f.t_smooth
            bands = bands.merge(BandRecord.of(f, inside))
        progress.report(1.0, "filling the growth-time grid")
        return cls(plane=plane, s=s, t=t, w=w, values=out.reshape(shape), bands=bands)

    def filled(self, passes: int = 3, linear: bool = True) -> F32:
        return fill_nan(self.values, passes=passes, linear=linear)

    def lookup(self, restored: FloatArray, values: F32 | None = None) -> FloatArray:
        """Trilinear t at RESTORED world points (n, 3) on the CPU; NaN where any
        corner is NaN or the point is off the grid. A check on the GPU lookup."""
        g = self.values if values is None else values
        ps, pt, pw = to_plane(self.plane, restored)
        h = self.spacing_mm
        f = np.stack([(pw - self.w[0]) / h, (pt - self.t[0]) / h, (ps - self.s[0]) / h])
        i0 = np.floor(f).astype(np.intp)
        a = f - i0
        out = np.zeros(len(restored))
        off = np.zeros(len(restored), dtype=bool)
        for axis, size in enumerate(g.shape):
            off |= (i0[axis] < 0) | (i0[axis] >= size - 1)
        i0 = np.where(off, 0, i0)
        for dk in (0, 1):
            for dj in (0, 1):
                for di in (0, 1):
                    wt = (
                        (a[0] if dk else 1 - a[0])
                        * (a[1] if dj else 1 - a[1])
                        * (a[2] if di else 1 - a[2])
                    )
                    out += wt * g[i0[0] + dk, i0[1] + dj, i0[2] + di]
        return np.where(off, np.nan, out)


def _shift(g: F32, axis: int, k: int) -> F32:
    """g shifted so out[i] = g[i - k] along axis, NaN where that falls off the grid."""
    r = np.roll(g, k, axis=axis)
    idx: list[slice] = [slice(None)] * 3
    idx[axis] = slice(0, k) if k > 0 else slice(k, None)
    r[tuple(idx)] = np.nan
    return r


def fill_nan(grid: F32, passes: int = 3, linear: bool = True) -> F32:
    """Extend valid t into NaN cells, `passes` cell layers deep; NaN left after
    that becomes 0. Each NaN cell takes the mean over the 6 axis directions of the
    linear extrapolation 2 v1 - v2 from its two nearest cells in that direction
    (v1 alone where v2 is missing). Exact for linear t, so the error near an edge
    is ~h^2. `linear=False` (plain averaging) is the negative control: edge error
    max 0.41 band periods against 0.086 (RESULTS.md part 2b)."""
    g = grid.astype(np.float32, copy=True)
    for _ in range(passes):
        bad = np.isnan(g)
        if not bad.any():
            break
        tot = np.zeros_like(g)
        cnt = np.zeros_like(g)
        for axis in range(3):
            for sh in (1, -1):
                v1, v2 = _shift(g, axis, sh), _shift(g, axis, 2 * sh)
                est = v1 if not linear else np.where(np.isnan(v2), v1, 2 * v1 - v2)
                ok = ~np.isnan(est)
                tot[ok] += est[ok]
                cnt[ok] += 1
        upd = bad & (cnt > 0)
        g[upd] = tot[upd] / cnt[upd]
    return np.nan_to_num(g, nan=0.0)
