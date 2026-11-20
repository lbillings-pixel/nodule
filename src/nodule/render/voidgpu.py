"""The hollow as air on the GPU (ADR 0019 s.6.12, step 4): `struct Void` (cuda/voidbox.cuh).

`pack_box` and the analytic shapes are pure numpy; `upload` needs CuPy. kind 1 is a stone's
void box (`nodule.render.voidbox`); kinds 2 (a flat air layer) and 3 (an air sphere) are
analytic shapes for the camera's checks. The wall test runs the same code for all three.
"""

# pyright: basic

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from agatesim.volume.crystal_gpu import shape_numbers
from numpy.typing import NDArray

from nodule.render.voidbox import VoidBox

VOID_DTYPE = np.dtype(
    [("kind", "<i4"), ("flags", "<i4"), ("nx", "<i4"), ("ny", "<i4"), ("nz", "<i4"),
     ("ax", "<i4"), ("ay", "<i4"), ("az", "<i4"), ("pnx", "<i4"), ("pny", "<i4"),
     ("pnz", "<i4"), ("faces", "<i4"), ("h", "<f4"), ("cot_a", "<f4"), ("elong", "<f4"),
     ("last", "<f4"), ("p", "<f4", (8,)), ("ids", "<u8"), ("nf", "<u8"), ("phi", "<u8"),
     ("rows", "<u8")]
)  # fmt: skip
assert VOID_DTYPE.itemsize == 128  # the CUDA struct's size

GATE, CHAL, SHELL = 1, 2, 4  # node flag bits (voidbox.cuh)
NO_WALL_N2 = 1  # Void.flags: control, the n^2 left out at the wall
# Void.flags, control for the shell's exit below the slab (camera.cu): the bottom rule's turn
# made where the path ended, outside the shell, as before the 2026-11-19 fix (the next short
# flight then meets the rule again)
SHELL_STAY = 2
LEGS_IGNORE_AIR = 4  # Void.flags, control for the small lamp: shadow legs pass the hollow
ANALYTIC_STEP_MM = 0.05  # march step for the analytic shapes (effective, checks only)


@dataclass(frozen=True)
class VoidArrays:
    """Host-side contents of `struct Void`; arrays absent for the analytic shapes."""

    header: NDArray[np.void]
    ids: NDArray[np.int32] | None = None
    nf: NDArray[np.uint8] | None = None
    phi: NDArray[np.float32] | None = None
    rows: NDArray[np.float32] | None = None

    @property
    def kind(self) -> int:
        return int(self.header["kind"])

    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.ids, self.nf, self.phi, self.rows) if a is not None)


def _header(kind: int, flags: int = 0) -> NDArray[np.void]:
    h = np.zeros((), VOID_DTYPE)
    h["kind"], h["flags"] = kind, flags
    return h


def layer(top_mm: float, bottom_mm: float, *, flags: int = 0) -> VoidArrays:
    """Air between depths top < bottom below the face (top 0: open at the face)."""
    if not top_mm < bottom_mm:
        raise ValueError("the layer's top must lie above its bottom")
    h = _header(2, flags)
    h["p"][:3] = (top_mm, bottom_mm, 0.0)
    h["p"][7] = ANALYTIC_STEP_MM
    return VoidArrays(h)


def sphere(centre: tuple[float, float, float], radius_mm: float, *, flags: int = 0) -> VoidArrays:
    """An air sphere (world coordinates)."""
    h = _header(3, flags)
    h["p"][:4] = (*centre, radius_mm)
    h["p"][7] = ANALYTIC_STEP_MM
    return VoidArrays(h)


def facet(
    point: tuple[float, float, float], normal: tuple[float, float, float], *, flags: int = 0
) -> VoidArrays:
    """Air below the face on the side of a plane (world coordinates) that its normal points
    to: a flat crystal facet open at the face, for the small lamp's glint check."""
    m = np.asarray(normal, np.float64)
    h = _header(4, flags)
    h["p"][:6] = (*point, *(m / np.linalg.norm(m)))
    h["p"][7] = ANALYTIC_STEP_MM
    return VoidArrays(h)


def none() -> VoidArrays:
    """No air anywhere (kind 0): the void kernel must then give the plain kernel's image."""
    return VoidArrays(_header(0))


def pack_box(box: VoidBox, *, flags: int = 0) -> VoidArrays:
    """A stone's void box: ids, node flags (gate, chalcedony near phi's surface, shell), phi
    and the crystals' rows (agatesim's `packed`), float32 as the GPU reads them."""
    if box.shell is None:
        raise ValueError("the box holds no shell (built before step 4)")
    nf = box.gate.astype(np.uint8) * GATE | box.shell.astype(np.uint8) * SHELL
    if box.chal is not None:
        nf |= box.chal.astype(np.uint8) * CHAL
    h = _header(1, flags)
    h["nx"], h["ny"], h["nz"] = box.shape
    h["ax"], h["ay"], h["az"] = (int(v) for v in box.a)
    h["h"] = box.h
    # agatesim ADR 0092 adds the face count (an agatesim before it gives two numbers and
    # round crystals: drop the length test when the pin moves)
    nums = shape_numbers(box.wall)
    h["cot_a"], h["elong"], h["last"] = nums[0], nums[1], box.wall.last_reach
    h["faces"] = nums[2] if len(nums) > 2 else 0  # pyright: ignore[reportGeneralTypeIssues]
    if box.phi is not None:
        h["pnx"], h["pny"], h["pnz"] = box.phi.shape
    return VoidArrays(
        header=h,
        ids=np.ascontiguousarray(box.ids, np.int32),
        nf=np.ascontiguousarray(nf, np.uint8),
        phi=None if box.phi is None else np.ascontiguousarray(box.phi, np.float32),
        rows=np.ascontiguousarray(box.wall.packed(), np.float32),
    )


@dataclass
class DeviceVoid:
    """The packed void on the GPU; keeps its device arrays alive."""

    struct: Any
    arrays: dict[str, Any]
    kind: int


def upload(v: VoidArrays) -> DeviceVoid:
    from nodule.render import _gpu

    cp = _gpu.cupy()
    arrays: dict[str, Any] = {}
    h = v.header.copy()
    for key in ("ids", "nf", "phi", "rows"):
        a = getattr(v, key)
        if a is not None:
            arrays[key] = cp.asarray(a)
            h[key] = arrays[key].data.ptr
    return DeviceVoid(cp.asarray(np.frombuffer(h.tobytes(), np.uint8)), arrays, v.kind)


def air_points(scene: Any, void: DeviceVoid, points: NDArray[np.floating]) -> tuple[Any, Any, Any]:
    """The GPU's air rule at world points: (air (n,) bool, values (n, 2): level, phi,
    within the shell (n,) bool)."""
    from nodule.render import _gpu

    cp = _gpu.cupy()
    p = cp.asarray(np.ascontiguousarray(points, np.float32))
    n = len(points)
    out, vals = cp.zeros(n, cp.uint8), cp.zeros((n, 2), cp.float32)
    k = _gpu.kernel("camera_void", "air_points")
    k(((n + 255) // 256,), (256,), (np.int64(n), p, scene.struct, void.struct, out, vals))
    o = out.get()
    return (o & 1).astype(bool), vals.get(), (o & 2).astype(bool)


def turn_points(
    dirs: NDArray[np.floating], normals: NDArray[np.floating], n1: float, n2: float, xi: float
) -> NDArray[np.float32]:
    """The wall's Fresnel turn on the GPU: (n, 4) new direction and 1 where it refracted."""
    from nodule.render import _gpu

    cp = _gpu.cupy()
    d = cp.asarray(np.ascontiguousarray(dirs, np.float32))
    m = cp.asarray(np.ascontiguousarray(normals, np.float32))
    n = len(dirs)
    out = cp.zeros((n, 4), cp.float32)
    k = _gpu.kernel("camera_void", "turn_points")
    k(((n + 255) // 256,), (256,),
      (np.int64(n), d, m, np.float32(n1), np.float32(n2), np.float32(xi), out))  # fmt: skip
    return out.get()
