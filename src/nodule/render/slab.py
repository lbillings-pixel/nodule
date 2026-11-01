"""Forward Monte Carlo through a slab: the renderer's independent reference (GPU).

A normal pencil beam enters a slab 0 < z < d (d = inf: half-space) through a
smooth face; returns the diffuse reflectance Rd (specular excluded) and total
transmittance Tt per channel, with standard errors. Checked against MCML
[@wang1995a] Tables 1-2 and exact values in tests/test_render_gpu.py; the
camera kernel is checked against it by reciprocity.
"""

# pyright: basic

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nodule.render import _gpu

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class SlabResult:
    rd: F64  # (3,) diffuse reflectance, specular excluded
    rd_se: F64
    tt: F64  # (3,) total transmittance
    tt_se: F64
    events: float  # tentative collisions per packet
    seconds: float


def slab(
    packets: int,
    *,
    depth_mm: float,
    n_in: float,
    g: float,
    mua: ArrayLike,
    mus: ArrayLike,
    n_out: float = 1.0,
    mubar: float | None = None,
    layers: ArrayLike | None = None,
    layer_mm: float = 1.0,
    analog: bool = False,
    seed: int = 1,
    block: int = 256,
) -> SlabResult:
    """mua, mus: (materials, 3) per mm. `layers`: material id per layer of
    `layer_mm` from the face (the last repeats); none = material 0 throughout.
    `analog` = unweighted delta tracking, the negative control for mubar < mu_t."""
    cp = _gpu.cupy()
    a = cp.asarray(np.atleast_2d(np.asarray(mua, np.float32)))
    s = cp.asarray(np.atleast_2d(np.asarray(mus, np.float32)))
    mb = float((a + s).max()) if mubar is None else mubar
    if layers is None:
        lay, nl = cp.zeros(1, cp.uint8), 0
    else:
        lay = cp.asarray(np.asarray(layers, np.uint8))
        nl = len(lay)
    out = cp.empty((packets, 7), cp.float32)
    k = _gpu.kernel("slab", "slab")
    _gpu.synchronize()
    t0 = time.perf_counter()
    k(((packets + block - 1) // block,), (block,),
      (np.int64(packets), np.uint32(seed), np.float32(depth_mm), np.float32(n_in),
       np.float32(n_out), np.float32(g), a, s, np.float32(mb), np.int32(analog),
       lay, np.int32(nl), np.float32(layer_mm), out))  # fmt: skip
    _gpu.synchronize()
    sec = time.perf_counter() - t0
    o = out.get().astype(np.float64)
    m, se = o.mean(0), o.std(0, ddof=1) / math.sqrt(packets)
    return SlabResult(rd=m[0:3], rd_se=se[0:3], tt=m[3:6], tt_se=se[3:6],
                      events=float(m[6]), seconds=sec)  # fmt: skip
