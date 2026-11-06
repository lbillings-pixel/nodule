"""CuPy access and kernel compilation for nodule.render.

CuPy comes through agatesim's `gpu` extra (`uv sync --extra gpu`, then
`uv run --no-sync ...`). CI has no GPU: `available()` is False there and the
GPU tests skip. CuPy has no type stubs, hence the basic checking mode.
"""

# pyright: basic

from __future__ import annotations

from functools import cache
from importlib import resources
from typing import Any

SOURCES = {
    "slab": ("common.cuh", "slab.cu"),
    "scene": ("common.cuh", "scene.cuh"),
    "shift": ("common.cuh", "scene.cuh", "shift.cuh"),
    "camera": ("common.cuh", "scene.cuh", "shift.cuh", "camera.cu"),
}
# fast-math is safe in the camera kernel (checked against analytic values in
# tests/test_render_gpu.py); the shift module is the camera's shift code alone, compiled
# the same way so its test checks what renders use. The reference slab kernel stays
# exact IEEE.
#
# The camera kernel's registers are capped (ADR 0016 s.15): a kernel holds its peak count on
# every path, so the lens code, which most samples never run, slowed every sample. Registers
# only: the image is bit-identical. 128 is `effective`, measured on the RTX 4070, on a plateau
# (96 to 128) above a sharp slowdown at 80; on another GPU it may not be the best value.
FAST = ("--use_fast_math",)
CAMERA_REGISTERS = 128
OPTIONS: dict[str, tuple[str, ...]] = {
    "camera": (*FAST, f"--maxrregcount={CAMERA_REGISTERS}"),
    "shift": FAST,
}


@cache
def available() -> bool:
    try:
        import cupy  # type: ignore[import-not-found]

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def cupy() -> Any:
    if not available():
        raise RuntimeError("nodule.render needs a CUDA GPU and CuPy (uv sync --extra gpu)")
    import cupy  # type: ignore[import-not-found]

    return cupy


def source(name: str) -> str:
    folder = resources.files("nodule.render") / "cuda"
    return "\n".join((folder / f).read_text(encoding="utf-8") for f in SOURCES[name])


@cache
def kernel(module: str, function: str) -> Any:
    """Compile (once per process; CuPy also caches on disk) and return a kernel."""
    cp = cupy()
    mod = cp.RawModule(code=source(module), options=OPTIONS.get(module, ()))
    return mod.get_function(function)


def synchronize() -> None:
    cupy().cuda.Device().synchronize()
