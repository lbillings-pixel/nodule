"""Render the polished cut face: orthographic camera + a standard light (GPU).

Physics in cuda/camera.cu (header comment). One engine, two speeds (ADR 0007
section 1): a preview is the same call at k times larger pixels and k^2 times
more samples per pixel, so it has the same expected image as the full render
box-averaged k x k (`View.coarser`; checked in tests/test_render_gpu.py).

Materials are an optical table indexed 0 host, 1 tube core, 2.. band materials,
with band material = profile[floor(t / dt)]. Every coefficient is `effective`
until ADR 0007 section 3 maps agatesim fields to it; n = 1.54 is `physical`
(bulk chalcedony 1.53-1.539 [@lorenzi2022; @white1961]). Output is linear
radiance; `reflectance_factor` divides by a white Lambertian standard under the
same light (section 2) and `srgb` encodes it with no tone curve.
"""

# pyright: basic

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import IntFlag
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nodule.render import _gpu
from nodule.render.scene import DeviceScene
from nodule.render.tgrid import GrowthGrid

N_CHALCEDONY = 1.54  # physical: bulk chalcedony 1.53-1.539 [@lorenzi2022; @white1961]
F32 = NDArray[np.float32]
F64 = NDArray[np.float64]


class Flag(IntFlag):
    """Kernel switches: negative controls and check modes (cuda/camera.cu)."""

    NO_BACKWARD_MAP = 1
    NO_RATIO_TRACKING = 2
    HOMOGENEOUS = 4  # material 0 everywhere, no geometry: the analytic checks
    NO_EXIT_N2 = 8
    NO_MIRROR = 16
    SCATTERED_ONLY = 32  # dome: drop light that escapes without a real collision


@dataclass(frozen=True)
class Materials:
    mua: F32  # (m, 3) absorption per mm at 450, 550, 650 nm
    mus: F32  # (m, 3) scattering per mm
    g: F32  # (m,) Henyey-Greenstein asymmetry
    profile: NDArray[np.uint16]  # band material (from 0) per growth-time step
    dt: float  # growth-time step of `profile`

    @classmethod
    def of(cls, mua: ArrayLike, mus: ArrayLike, g: ArrayLike,
           profile: ArrayLike = (0,), dt: float = 1.0) -> Materials:  # fmt: skip
        m = cls(
            mua=np.atleast_2d(np.asarray(mua, np.float32)),
            mus=np.atleast_2d(np.asarray(mus, np.float32)),
            g=np.atleast_1d(np.asarray(g, np.float32)),
            profile=np.atleast_1d(np.asarray(profile, np.uint16)),
            dt=dt,
        )
        if m.mua.shape != m.mus.shape or m.mua.shape[1] != 3 or len(m.g) != len(m.mua):
            raise ValueError("mua, mus must be (m, 3) and g (m,)")
        if (m.mua < 0).any() or (m.mus < 0).any():
            raise ValueError("coefficients must be non-negative")
        if len(m.g) > 1 and int(m.profile.max()) + 2 >= len(m.g):
            raise ValueError("profile names a band material beyond the table (2 + id)")
        return m


@dataclass(frozen=True)
class Light:
    kind: Literal["directional", "dome"]
    theta_deg: float = 45.0  # directional: polar angle from the face normal
    phi_deg: float = 90.0  # directional: azimuth from plane.u toward plane.v
    irradiance: float = 1.0  # directional: E, perpendicular to the beam
    radiance: float = 1.0  # dome: Ld

    def white(self) -> float:
        """Radiance from a white Lambertian standard under this light."""
        if self.kind == "dome":
            return self.radiance
        return self.irradiance * math.cos(math.radians(self.theta_deg)) / math.pi


@dataclass(frozen=True)
class View:
    """Pixel (i, j) covers [s0 + i p, s0 + (i + 1) p) x [t0 + j p, ...), row j up."""

    s0: float
    t0: float
    pixel_mm: float
    nx: int
    ny: int

    @classmethod
    def covering(cls, grid: GrowthGrid, pixel_mm: float) -> View:
        s0, t0 = float(grid.s[0]), float(grid.t[0])
        nx = int((float(grid.s[-1]) - s0) / pixel_mm)
        ny = int((float(grid.t[-1]) - t0) / pixel_mm)
        return cls(s0, t0, pixel_mm, nx, ny)

    def coarser(self, k: int) -> View:
        """Same origin, pixels k times larger (whole coarse pixels only)."""
        return View(self.s0, self.t0, self.pixel_mm * k, self.nx // k, self.ny // k)


@dataclass(frozen=True)
class Rendered:
    image: F64  # (ny, nx, 3) linear radiance, row 0 at t0 (bottom)
    seconds: float
    paths: int
    events: float  # tentative collisions per path
    real: float  # real collisions per path

    @property
    def mpaths_per_s(self) -> float:
        return self.paths / self.seconds / 1e6


CAM_DTYPE = np.dtype(
    [("s0", "<f4"), ("t0", "<f4"), ("pix", "<f4"), ("nx", "<i4"), ("ny", "<i4"),
     ("spp", "<i4"), ("seed", "<u4"), ("n_in", "<f4"), ("D", "<f4"), ("light", "<i4"),
     ("lth", "<f4"), ("lph", "<f4"), ("E", "<f4"), ("Ld", "<f4"), ("mubar", "<f4"),
     ("nmat", "<i4"), ("nprof", "<i4"), ("dt", "<f4"), ("order", "<i4"), ("flags", "<i4")]
)  # fmt: skip


def render(
    scene: DeviceScene,
    view: View,
    materials: Materials,
    light: Light,
    *,
    spp: int,
    n_in: float = N_CHALCEDONY,
    depth_mm: float = math.inf,
    mubar: float | None = None,
    order: int = 0,
    flags: int = 0,
    seed: int = 1,
    block: int = 128,
) -> Rendered:
    """Linear radiance per pixel, including the dome's face specular F(0) Ld.

    mubar: global majorant (default max mu_t). order > 0 keeps only that real-
    collision order (single scattering check)."""
    cp = _gpu.cupy()
    mua, mus = cp.asarray(materials.mua), cp.asarray(materials.mus)
    gm, prof = cp.asarray(materials.g), cp.asarray(materials.profile)
    mb = float((materials.mua + materials.mus).max()) if mubar is None else mubar
    paths = view.nx * view.ny * spp
    if paths >= 2**63:
        raise ValueError("too many paths")
    c = np.zeros((), CAM_DTYPE)
    vals = dict(s0=view.s0, t0=view.t0, pix=view.pixel_mm, nx=view.nx, ny=view.ny, spp=spp,
                seed=seed, n_in=n_in, D=depth_mm, light=0 if light.kind == "directional" else 1,
                lth=math.radians(light.theta_deg), lph=math.radians(light.phi_deg),
                E=light.irradiance, Ld=light.radiance, mubar=mb, nmat=len(materials.g),
                nprof=len(materials.profile), dt=materials.dt, order=order,
                flags=int(flags))  # fmt: skip
    for key, v in vals.items():
        c[key] = v
    cam = cp.asarray(np.frombuffer(c.tobytes(), np.uint8))
    img = cp.zeros((view.ny, view.nx, 3), cp.float32)
    stats = cp.zeros(2, cp.uint64)
    k = _gpu.kernel("camera", "render")
    _gpu.synchronize()
    t0 = time.perf_counter()
    k(((paths + block - 1) // block,), (block,),
      (scene.struct, cam, prof, mua, mus, gm, img, stats))  # fmt: skip
    _gpu.synchronize()
    sec = time.perf_counter() - t0
    out = img.get().astype(np.float64)
    if light.kind == "dome":
        out += ((n_in - 1) / (n_in + 1)) ** 2 * light.radiance  # face specular seen by the camera
    st = stats.get()
    return Rendered(image=out, seconds=sec, paths=paths, events=float(st[0]) / paths,
                    real=float(st[1]) / paths)  # fmt: skip


def reflectance_factor(image: F64, light: Light) -> F64:
    """Radiance relative to a white Lambertian standard under the same light."""
    return image / light.white()


def srgb(x: F64, exposure: float = 1.0) -> NDArray[np.uint8]:
    """sRGB-encode linear values x * exposure (clipped to [0, 1]); rows flipped so
    +v is up in the stored image, and channels reversed from the renderer's
    450/550/650 nm order to R, G, B. Each channel is taken as the matching sRGB
    primary, an approximation (three wavelengths, no colour-matching functions).
    No tone curve (ADR 0007 section 2)."""
    y = np.clip(x * exposure, 0.0, 1.0)
    y = np.where(y <= 0.0031308, 12.92 * y, 1.055 * y ** (1 / 2.4) - 0.055)
    return (y[::-1, :, ::-1] * 255 + 0.5).astype(np.uint8)
