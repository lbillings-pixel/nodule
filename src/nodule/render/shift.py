"""Materials that vary along the bands, for the GPU camera (ADR 0016).

agatesim ADR 0078 gives each band pigment lenses and a translucency drift, worked
out from a point's place and growth time alone (`MaterialVariation.shift`). The
camera works the lenses out at each sample (cuda/shift.cuh), since 0.1 mm lenses
are thinner than the growth-time grid (ADR 0016 16a A). The drift, which has no
structure finer than a band, is baked once per cut on that grid (`bake_drift`,
ADR 0016 s.9, the owner's choice) and read by trilinear lookup; the per-sample
drift stays as the reference (`Rendering.drift = "exact"`). This module finds the
stone's own fields (`variation_of`, agatesim's documented route), packs them into
the GPU `struct Shift` (`pack`, pure numpy) and uploads them with the material
table's per-row shift columns (`upload`, CuPy).

The pack reads agatesim's `SheetNoise` and `BandNoise` internals, a copy kept
honest by tests/test_render_gpu.py, which compares the kernel with agatesim's
own `material_shift` at random points. Every lens and drift setting is
`effective` (agatesim ADR 0078); how the renderer applies them (16b) is in
materials.py. Renders are procedural, never evidence for how lenses form.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
from agatesim.models.growth import TIME_STEPS, ProceduralGrowth
from agatesim.noise import CELLS_PER_WAVELENGTH, BandNoise
from agatesim.volume.features import (
    DRIFT_RANGE,
    LENS_EDGE,
    MaterialVariation,
    SheetNoise,
    wall_features,
)
from numpy.typing import NDArray

from nodule.render.camera import Materials
from nodule.render.tgrid import GrowthGrid, to_world
from nodule.specimen import Specimen

F32 = NDArray[np.float32]

SHIFT_DTYPE = np.dtype(
    [("depth", "<u8"), ("sheets", "<u8", 2), ("cj", "<u8", 2), ("coef", "<u8", 2),
     ("band", "<u8"), ("dgrid", "<u8"), ("M", "<f4", (2, 9)), ("nsheet", "<i4", 2), ("n", "<i4", 2),
     ("on", "<i4", 2), ("steps", "<i4"), ("tr0", "<i4"), ("ntr", "<i4"),
     ("strength", "<f4"), ("threshold", "<f4"), ("drift", "<f4")]
)  # fmt: skip
assert SHIFT_DTYPE.itemsize == 192  # the CUDA struct's size, pointers 8-aligned
# the kernel's copies of agatesim's constants (shift.cuh)
assert (LENS_EDGE, DRIFT_RANGE) == (0.5, 2.0)


def variation_of(specimen: Specimen) -> MaterialVariation | None:
    """The stone's own lenses and drift: agatesim rebuilds its features from the run's
    seed (`wall_features(model, setting, default_rng(seed))`, its documented route)."""
    config = specimen.volume.config
    if not isinstance(config.model, ProceduralGrowth):
        return None
    rng = np.random.default_rng(config.seed)
    return wall_features(config.model, specimen.volume.setting, rng).variation


@dataclass(frozen=True)
class SheetArrays:
    """One SheetNoise as the kernel reads it."""

    sheets: F32  # (count,) sheet depths, mm
    cj: F32  # (count, 3) index constant per sheet: offset @ R.T / cell + tile offset, mod n
    M: F32  # (3, 3) rotation / cell
    coef: F32  # (n, n, n) spline coefficients

    @classmethod
    def of(cls, sheet: SheetNoise) -> SheetArrays:
        noise: BandNoise = sheet._noise  # pyright: ignore[reportPrivateUsage]
        if noise.dim != 3 or len(noise.wavelengths_mm) != 1 or noise.weights[0] != 1.0:
            raise ValueError("the GPU copy handles one-octave 3D sheet noise only")
        coef = noise._full[0]  # pyright: ignore[reportPrivateUsage]
        rot = np.asarray(noise._rotations[0], np.float64)  # pyright: ignore[reportPrivateUsage]
        off0 = np.asarray(noise._offsets[0], np.float64)  # pyright: ignore[reportPrivateUsage]
        n = coef.shape[0]
        if n & (n - 1) or coef.shape != (n, n, n):
            raise ValueError("the GPU copy needs a cubic tile of a power-of-two size")
        cell = float(noise.wavelengths_mm[0]) / CELLS_PER_WAVELENGTH
        offsets = np.asarray(sheet._offsets, np.float64)  # pyright: ignore[reportPrivateUsage]
        # folded in float64 and wrapped, so float32 indices stay small (a point's own part
        # is at most a few hundred cells); the tile is periodic, so the wrap changes nothing
        cj = np.mod(offsets @ rot.T / cell + off0, n)
        return cls(
            sheets=np.asarray(sheet._sheets, np.float32),  # pyright: ignore[reportPrivateUsage]
            cj=cj.astype(np.float32),
            M=(rot / cell).astype(np.float32),
            coef=np.ascontiguousarray(coef, np.float32),
        )


@dataclass(frozen=True)
class ShiftArrays:
    """Host-side contents of `struct Shift`, before the material table's columns."""

    depth: F32  # (TIME_STEPS + 1,)
    lens: SheetArrays | None
    drift: SheetArrays | None
    strength: float
    threshold: float
    drift_size: float

    @property
    def max_gain(self) -> float:
        return 1.0 + self.strength if self.lens is not None else 1.0

    @classmethod
    def of(cls, v: MaterialVariation) -> ShiftArrays:
        if len(v.depth_mm) != TIME_STEPS + 1:
            raise ValueError("depth table must have one entry per step edge")
        return cls(
            depth=np.asarray(v.depth_mm, np.float32),
            lens=SheetArrays.of(v.lens) if v.lens is not None else None,
            drift=SheetArrays.of(v.drift) if v.drift is not None else None,
            strength=float(v.lens_strength),
            threshold=float(v.lens_threshold),
            drift_size=float(v.drift_size),
        )


@dataclass
class DeviceShift:
    """The packed shift on the GPU, for one material table; keeps its arrays alive."""

    struct: Any  # cupy uint8 array holding the struct bytes
    arrays: dict[str, Any]
    materials: Materials | None  # the table it was built for; None for `off()`
    max_gain: float = 1.0
    drift_size: float = 1.0  # bounds each band's reachable translucency (majorant); 1: any
    grid_shape: tuple[int, ...] | None = None  # the baked drift's grid; None: per sample


def upload(
    shift: ShiftArrays, materials: Materials | None, drift_grid: F32 | None = None
) -> DeviceShift:
    """The shift and the per-row columns of `materials` (from OpticsTable.materials with
    this shift) on the GPU. Without materials only the fields are there, for the
    `shift_points` check; the camera refuses such a shift. With `drift_grid` (`bake_drift`
    on the cut's scene), the camera reads the drift from it instead of working it out."""
    from nodule.render import _gpu

    if materials is not None and materials.along is None:
        raise ValueError("the material table has no shift columns (OpticsTable.materials)")
    if materials is not None and (materials.tr_rows[1] > 0) != (shift.drift is not None):
        raise ValueError("the material table's drift rows do not match the shift")
    cp = _gpu.cupy()
    arrays: dict[str, Any] = {"depth": cp.asarray(shift.depth)}
    h = np.zeros((), SHIFT_DTYPE)
    h["depth"] = arrays["depth"].data.ptr
    if drift_grid is not None:
        if shift.drift is None:
            raise ValueError("a baked drift grid needs a shift with drift")
        arrays["dgrid"] = cp.asarray(np.ascontiguousarray(drift_grid, np.float32))
        h["dgrid"] = arrays["dgrid"].data.ptr
    if materials is not None:
        arrays["band"] = cp.asarray(np.ascontiguousarray(materials.along, np.float32))
        h["band"] = arrays["band"].data.ptr
        h["tr0"], h["ntr"] = materials.tr_rows
    for f, sheet in enumerate((shift.lens, shift.drift)):
        if sheet is None:
            continue
        for key in ("sheets", "cj", "coef"):
            arrays[f"{key}{f}"] = cp.asarray(np.ascontiguousarray(getattr(sheet, key)))
            h[key][f] = arrays[f"{key}{f}"].data.ptr
        h["M"][f] = sheet.M.ravel()
        h["nsheet"][f], h["n"][f], h["on"][f] = len(sheet.sheets), sheet.coef.shape[0], 1
    h["steps"] = len(shift.depth) - 1
    h["strength"], h["threshold"], h["drift"] = shift.strength, shift.threshold, shift.drift_size
    struct = cp.asarray(np.frombuffer(h.tobytes(), np.uint8))
    shape = tuple(drift_grid.shape) if drift_grid is not None else None
    return DeviceShift(struct=struct, arrays=arrays, materials=materials,
                       max_gain=shift.max_gain, drift_size=shift.drift_size,
                       grid_shape=shape)  # fmt: skip


BAKE_CHUNK = 1 << 21  # nodes per kernel launch (~60 MB of device scratch)


def bake_drift(shift: ShiftArrays, grid: GrowthGrid, t: F32) -> F32:
    """The drift's translucency change at every node of a cut's growth-time grid
    (restored coordinates), from its filled growth time `t` (`SceneArrays.grid`): the
    tested GPU copy (`shift_points`) at the nodes, for trilinear lookup (ADR 0016 s.9).
    Its error against the per-sample drift is measured in tests/test_shift_gpu.py."""
    from nodule.render import _gpu

    if shift.drift is None:
        raise ValueError("the shift has no drift to bake")
    shape = (len(grid.w), len(grid.t), len(grid.s))
    if t.shape != shape:
        raise ValueError(f"growth time {t.shape} does not match the grid {shape}")
    cp = _gpu.cupy()
    dev = upload(dataclasses.replace(shift, lens=None), None)  # the drift alone
    k = _gpu.kernel("shift", "shift_points")
    n = math.prod(shape)
    out = np.empty(n, np.float32)
    flat = t.ravel()
    for a in range(0, n, BAKE_CHUNK):
        kk, j, i = np.unravel_index(np.arange(a, min(a + BAKE_CHUNK, n)), shape)
        p = cp.asarray(to_world(grid.plane, grid.s[i], grid.t[j], grid.w[kk]), cp.float32)
        m = len(i)
        ts = cp.asarray(flat[a : a + m], cp.float32)
        gain, change = cp.empty(m, cp.float32), cp.empty(m, cp.float32)
        k(((m + 255) // 256,), (256,), (np.int64(m), p, ts, dev.struct, gain, change))
        out[a : a + m] = change.get()
    return out.reshape(shape)


@cache
def off() -> DeviceShift:
    """No shift: a struct of zeros (null pointers), which the kernel reads as off."""
    from nodule.render import _gpu

    cp = _gpu.cupy()
    struct = cp.asarray(np.zeros(SHIFT_DTYPE.itemsize, np.uint8))
    return DeviceShift(struct=struct, arrays={}, materials=None)
