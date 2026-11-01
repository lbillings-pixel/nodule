"""Pack a specimen and its growth-time grid into the GPU `struct Scene` (scene.cuh).

`pack` is pure numpy (runs in CI); `upload` needs CuPy. The kernel handles only
what the n2-slice specimen uses, and `pack` refuses anything else rather than
render it wrong: a sphere-union solid, BELL fracture profiles, at most
MAX_FRACTURES fractures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from agatesim.volume.fractures import BELL
from agatesim.volume.solids import SphereUnion
from numpy.typing import NDArray

from nodule.render.tgrid import GrowthGrid
from nodule.specimen import Specimen

MAX_FRACTURES = 4
F32 = NDArray[np.float32]

SCENE_DTYPE = np.dtype(
    [("o", "<f4", 3), ("u", "<f4", 3), ("v", "<f4", 3), ("n", "<f4", 3),
     ("s0", "<f4"), ("t0", "<f4"), ("w0", "<f4"), ("h", "<f4"),
     ("ns", "<i4"), ("nt", "<i4"), ("nw", "<i4"),
     ("nsph", "<i4"), ("ntube", "<i4"), ("nfrac", "<i4"),
     ("sph", "<u8"), ("tube", "<u8"), ("frac", "<u8"), ("grid", "<u8")]
)  # fmt: skip
assert SCENE_DTYPE.itemsize == 120  # the CUDA struct's size, pointers 8-aligned


@dataclass(frozen=True)
class SceneArrays:
    """Host-side contents of `struct Scene`; `grid` is the NaN-filled t grid."""

    header: NDArray[np.void]  # 0-d SCENE_DTYPE; pointer fields set by `upload`
    spheres: F32  # (nsph, 4) centre, radius
    tubes: F32  # (ntube, 7) ends a, b, radius
    fractures: F32  # (nfrac, 17) centre, frame rows, slip, radii, drag, steps
    grid: F32  # (nw, nt, ns)


def pack(specimen: Specimen, grid: GrowthGrid, filled: F32 | None = None) -> SceneArrays:
    solid = specimen.volume.solid
    if not isinstance(solid, SphereUnion):
        raise TypeError(f"the GPU scene handles a sphere-union solid, not {type(solid).__name__}")
    rows: list[list[float]] = []
    for f in specimen.fractures:
        if f.along is not BELL or f.across is not BELL:
            raise ValueError("the GPU scene assumes BELL fracture profiles")
        rows.append([*f.centre, *f.frame.ravel(), f.slip_mm, f.radius_along_mm,
                     f.radius_across_mm, f.drag_mm, f.step_count])  # fmt: skip
    if len(rows) > MAX_FRACTURES:
        raise ValueError(f"at most {MAX_FRACTURES} fractures")
    plane = grid.plane
    h = np.zeros((), SCENE_DTYPE)
    h["o"], h["u"], h["v"], h["n"] = plane.origin, plane.u, plane.v, plane.normal
    h["s0"], h["t0"], h["w0"], h["h"] = grid.s[0], grid.t[0], grid.w[0], grid.spacing_mm
    h["ns"], h["nt"], h["nw"] = len(grid.s), len(grid.t), len(grid.w)
    spheres = np.c_[solid.centres, solid.radii].astype(np.float32)
    tubes = np.asarray(specimen.volume.setting.tubes, np.float32).reshape(-1, 7)
    fractures = np.asarray(rows, np.float32).reshape(-1, 17)
    h["nsph"], h["ntube"], h["nfrac"] = len(spheres), len(tubes), len(fractures)
    return SceneArrays(
        header=h,
        spheres=spheres,
        tubes=tubes,
        fractures=fractures,
        grid=grid.filled() if filled is None else filled,
    )


def homogeneous() -> SceneArrays:
    """An empty scene with the face at z = 0, normal +z: for flag 4 (homogeneous
    material 0), the analytic checks."""
    h = np.zeros((), SCENE_DTYPE)
    h["u"], h["v"], h["n"] = (1, 0, 0), (0, 1, 0), (0, 0, 1)
    h["h"], h["ns"], h["nt"], h["nw"] = 1.0, 2, 2, 2
    z = np.zeros
    return SceneArrays(h, z((0, 4), np.float32), z((0, 7), np.float32),
                       z((0, 17), np.float32), z((2, 2, 2), np.float32))  # fmt: skip


@dataclass
class DeviceScene:
    """The packed scene on the GPU; keeps its device arrays alive."""

    struct: Any  # cupy uint8 array holding the struct bytes
    arrays: dict[str, Any]


def upload(scene: SceneArrays) -> DeviceScene:
    from nodule.render import _gpu

    cp = _gpu.cupy()
    arrays: dict[str, Any] = {}
    for key in ("spheres", "tubes", "fractures", "grid"):
        a: F32 = getattr(scene, key)
        arrays[key] = cp.asarray(a if a.size else np.zeros(1, np.float32), cp.float32)
    h = scene.header.copy()
    for field, key in (("sph", "spheres"), ("tube", "tubes"), ("frac", "fractures"),
                       ("grid", "grid")):  # fmt: skip
        h[field] = arrays[key].data.ptr
    struct = cp.asarray(np.frombuffer(h.tobytes(), np.uint8))
    return DeviceScene(struct=struct, arrays=arrays)
