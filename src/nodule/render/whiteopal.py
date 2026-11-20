"""White opal by place (ADR 0019 s.13): white plumes, and white rims round plumes and moss.

Reading (s.13): white plumes are an opal phase [@kile2002, p. 260], "opaque to
translucent white" in hand samples and tan to pale brown in thin section [@kile2002,
p. 258] (*plausible*); green moss is "commonly mantled" with milky-white opal-CT
[@zhang2026, p. 3] (*verified*). How white, how wide and how often are the owner's picks
from the probe sheets (2026-11-20), all `effective`.

The camera mixes one white row into a stone sample with the node's white share as
probability (`Scene.white`, camera.cu matid). The row: air-filled pores (an assumption)
in opal (n 1.45, one Del Norte sample's 1.444 [@kile2002, p. 258]), radii PORE_RADII_UM
in equal volumes (an assumption: sub-micron, so bluish in front light and tan backlit,
the reading's description; the owner's pick), scaled to a transport coefficient mu_s
(1 - g) at 550 nm; its absorption is the matrix's (`OpticsTable.matrix_mua`).

The share comes from the layer bake's own sub-points (`layerbake.Baker`: each node's share
of its SUB^3 sub-points inside a plume or moss strand, its cover). White plumes: the
share is the cover, and the bake leaves the plume pigment out, so a thin frond or a
plume's edge fades instead of dropping out at the grid. White rims: 1 - cover, within
RIM_MM of the nodes at least half covered, with a one-node ramp past RIM_MM so a flat
face's rim is RIM_MM wide on average at any grid spacing (the probe's rim at 0.1 mm,
softened; `rim_ramp`). Rims lie on stone nodes only (none in tubes, hollows or the host);
the pigment stays.

Limits: the share is trilinear on the bake's grid, so detail under the spacing blurs; a
rim follows nodes at least half covered, so fronds thinner than that take none; the
white row keeps the layers' absorption at its place (a halo's lost colour included).
Appearance only (ADR 0002).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from nodule._scipy import distance_inside
from nodule.render.mie import pore_part

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]
WhiteKind = Literal["none", "plumes", "rims"]

N_OPAL = 1.45  # `physical`: one Del Norte sample, 1.444 [@kile2002, p. 258]
N_PORE = 1.0  # air-filled pores: an assumption (water pores scatter far less, s.13)
PORE_RADII_UM = (0.07, 0.10, 0.14)  # `effective`: sub-micron, the owner's bluish kind
RIM_MM = 0.3  # `effective`: the owner's pick as shown; [@zhang2026] Fig. 2e by eye 0.2-0.7
CHUNK_ROWS = 64  # rows of the grid's t axis per distance-transform chunk (memory)


def white_optics(transport: float, wavelengths_nm: F64) -> tuple[F64, F64]:
    """mu_s (per mm) and g per wavelength of the white row: PORE_RADII_UM pores in opal,
    equal volume per radius, scaled so mu_s (1 - g) at 550 nm is `transport` per mm."""
    if not transport > 0:
        raise ValueError("white opal needs a positive transport coefficient")
    wl = np.asarray(wavelengths_nm, np.float64)
    if not np.any(np.isclose(wl, 550.0)):
        raise ValueError("the channels must include 550 nm, where the strength is set")
    mus, gm = np.zeros(len(wl)), np.zeros(len(wl))
    for r in PORE_RADII_UM:
        m, g = pore_part(r, 1.0 / len(PORE_RADII_UM), N_PORE, wl, N_OPAL)
        mus += m
        gm += m * g
    g = gm / mus
    k = int(np.argmin(np.abs(wl - 550.0)))
    return mus * transport / (mus[k] * (1.0 - g[k])), g


def white_share(
    kind: WhiteKind, cover: F32 | None, values: F32, spacing_mm: float, rim_mm: float = RIM_MM
) -> F32 | None:
    """Each node's white share (0..1) on the bake's grid: `cover` the bake's inclusion
    cover (`GrowthGrid.cover`), `values` the grid's growth time (NaN off the stone, in
    tubes and hollows). None without white, without a cover, or with no white node."""
    if kind == "none" or cover is None:
        return None
    if cover.shape != values.shape:
        raise ValueError("the cover must lie on the grid's nodes")
    stone = np.isfinite(values)
    if kind == "plumes":
        share = np.where(stone, cover, 0.0).astype(np.float32)
    elif kind == "rims":
        near = rim_ramp(cover >= 0.5, rim_mm / spacing_mm)
        share = np.where(stone, near * (1.0 - cover), 0.0).astype(np.float32)
    else:
        raise ValueError(f"no white opal of kind {kind!r}")
    return share if share.any() else None


def rim_ramp(seed: NDArray[np.bool_], reach: float) -> F32:
    """clip(reach + 1 - d, 0, 1), d each node's distance (nodes) to the nearest seed
    node: 1 out to `reach`, falling to 0 over the next node. A plume's face lies on
    average half a node past its last seed node (one at least half covered), so with the
    (1 - cover) of the nodes it crosses (`white_share`) a flat face's rim is `reach` nodes
    wide on average at any spacing. Worked in chunks of CHUNK_ROWS rows of axis 1 inside
    the seeds' padded box, each chunk padded past the ramp, which keeps it exact (a seed
    outside a chunk lies beyond the ramp)."""
    out = np.zeros(seed.shape, np.float32)
    if not seed.any():
        return out
    pad = math.ceil(reach + 1.0) + 1
    box: list[tuple[int, int]] = []
    for axis in range(3):
        on = np.flatnonzero(seed.any(axis=tuple(a for a in range(3) if a != axis)))
        box.append((max(int(on[0]) - pad, 0), min(int(on[-1]) + pad + 1, seed.shape[axis])))
    (w0, w1), (t0, t1), (s0, s1) = box
    for a in range(t0, t1, CHUNK_ROWS):
        b = min(a + CHUNK_ROWS, t1)
        lo, hi = max(a - pad, t0), min(b + pad, t1)
        sub = seed[w0:w1, lo:hi, s0:s1]
        if not sub.any():
            continue
        d = distance_inside(~sub)[:, a - lo : b - lo]
        out[w0:w1, a:b, s0:s1] = np.clip(reach + 1.0 - d, 0.0, 1.0)
    return out
