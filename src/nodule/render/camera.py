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

# ADR 0007 amendment 4: forward-peak cut for renders under the L2 disc lamp (effective). Dome
# and other diffuse-light renders show no sparkle and use the full phase function (0).
L2_CUT_DEG = 4.0
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
    NO_PHASE_WEIGHT = 64  # skip the per-channel phase-function weight (control)
    NEE_ONLY = 128  # disc: light sampling alone, weight 1 (a check)
    PHASE_ONLY = 256  # disc: phase sampling alone, weight 1 (a check)
    NO_MIS_WEIGHT = 512  # disc: both strategies at weight 1 (control: light counted twice)
    UNTRUNCATED_DRAWS = 1024  # draw from the whole lobes despite cut_deg (control)


@dataclass(frozen=True)
class Materials:
    mua: F32  # (m, 3) absorption per mm at 450, 550, 650 nm
    mus: F32  # (m, 3) scattering per mm
    g: F32  # (m, 3) Henyey-Greenstein asymmetry of lobe a (the grain part)
    profile: NDArray[np.uint16]  # band material (from 0) per growth-time step
    dt: float  # growth-time step of `profile`
    f: F32  # (m, 3) share of mus in lobe b (pores and other near-isotropic scatterers)
    gb: F32  # (m, 3) asymmetry of lobe b
    # Forward-peak truncation (ADR 0007 amendment 4): scattering by less than cut_deg is
    # counted as no scattering. `effective`; 0 renders the full phase function.
    cut_deg: float = 0.0

    @classmethod
    def of(cls, mua: ArrayLike, mus: ArrayLike, g: ArrayLike,
           profile: ArrayLike = (0,), dt: float = 1.0,
           f: ArrayLike | None = None, gb: ArrayLike | None = None,
           cut_deg: float = 0.0) -> Materials:  # fmt: skip
        """g, f, gb: (m,) for one value per material or (m, 3) per channel. Without f the
        phase function is one HG lobe (f = 0). mus, f, g, gb describe the full phase
        function; `cut_deg` truncates it (see `kept`)."""
        mua32 = np.atleast_2d(np.asarray(mua, np.float32))
        mus32 = np.atleast_2d(np.asarray(mus, np.float32))
        if mua32.shape != mus32.shape or mua32.shape[1] != 3:
            raise ValueError("mua, mus must be (m, 3)")

        def per_channel(v: ArrayLike | None, name: str) -> F32:
            a = np.zeros(len(mua32), np.float32) if v is None else np.asarray(v, np.float32)
            if a.ndim <= 1:
                a = np.atleast_1d(a)[:, None].repeat(3, 1)
            if a.shape != mua32.shape:
                raise ValueError(f"{name} must be (m,) or (m, 3) like mua")
            return np.ascontiguousarray(a)

        m = cls(
            mua=mua32, mus=mus32, g=per_channel(g, "g"),
            profile=np.atleast_1d(np.asarray(profile, np.uint16)), dt=dt,
            f=per_channel(f, "f"), gb=per_channel(gb, "gb"), cut_deg=float(cut_deg),
        )  # fmt: skip
        if (m.mua < 0).any() or (m.mus < 0).any():
            raise ValueError("coefficients must be non-negative")
        if (m.f < 0).any() or (m.f > 1).any():
            raise ValueError("lobe share f must lie in [0, 1]")
        if (np.abs(m.g) >= 1).any() or (np.abs(m.gb) >= 1).any():
            raise ValueError("asymmetry must lie in (-1, 1)")
        if len(m.g) > 1 and int(m.profile.max()) + 2 >= len(m.g):
            raise ValueError("profile names a band material beyond the table (2 + id)")
        if not 0.0 <= m.cut_deg < 90.0:
            raise ValueError("cut_deg must lie in [0, 90)")
        return m

    @property
    def omc_cut(self) -> float:
        """1 - cos(cut_deg), accurate for small angles."""
        return 2.0 * math.sin(math.radians(self.cut_deg) / 2.0) ** 2

    def kept(self) -> tuple[F64, F64, F64, F64]:
        """(mus, f, x_a, x_b) after truncation: x = share of each lobe scattering by more
        than cut_deg; mus keeps only those shares, f is lobe b's part of what is kept. The
        kernel draws each lobe within its kept part and divides its density by x."""
        xa, xb = hg_kept(self.g, self.omc_cut), hg_kept(self.gb, self.omc_cut)
        f64 = self.f.astype(np.float64)
        share = (1 - f64) * xa + f64 * xb
        fk = np.divide(f64 * xb, share, out=np.zeros_like(share), where=share > 0)
        return self.mus * share, fk, xa, xb

    def phase_table(self) -> F32:
        """(m, 3 channels, {g_a, f, g_b, x_a, x_b, omc_cut}) as the kernel reads it
        (common.cuh phase_pdf_omc), f and x after truncation."""
        _, fk, xa, xb = self.kept()
        cut = np.full_like(xa, self.omc_cut)
        return np.ascontiguousarray(np.stack([self.g, fk, self.gb, xa, xb, cut], -1), np.float32)

    def phase(self, c: ArrayLike) -> F32:
        """Phase density per steradian, (m, 3) per scattering cosine c (scalar), of the
        kept (truncated) phase function: 0 within cut_deg of forward."""
        omc = 1.0 - float(np.asarray(c))
        _, fk, xa, xb = self.kept()
        if self.cut_deg > 0 and omc <= self.omc_cut:
            return np.zeros_like(self.g)
        return ((1 - fk) * hg_density(self.g, omc) / xa
                + fk * hg_density(self.gb, omc) / xb).astype(np.float32)  # fmt: skip


def hg_density(g: ArrayLike, omc: ArrayLike) -> F64:
    """Henyey-Greenstein density per steradian at 1 - cos = omc."""
    g64 = np.asarray(g, np.float64)
    om = np.asarray(omc, np.float64)
    return (1 - g64**2) / (4 * math.pi * ((1 - g64) ** 2 + 2 * g64 * om) ** 1.5)


def hg_kept(g: ArrayLike, omc_cut: float) -> F64:
    """Share of a Henyey-Greenstein lobe scattering by more than the cut, P(1 - cos >
    omc_cut), from the inverse of the sampling map in common.cuh hg (Wang 1995 Eq. 3.22):
    xi = ((1 - g^2) / sqrt(1 + g^2 - 2 g cos) - (1 - g)) / 2g; isotropic 1 - omc_cut / 2."""
    g64 = np.asarray(g, np.float64)
    safe = np.where(np.abs(g64) < 1e-9, 0.5, g64)  # placeholder where the isotropic form is used
    d = (1 - safe) ** 2 + 2 * safe * omc_cut
    x = ((1 - safe**2) / np.sqrt(d) - (1 - safe)) / (2 * safe)
    return np.clip(np.where(np.abs(g64) < 1e-9, 1 - omc_cut / 2, x), 1e-12, 1.0)


LIGHT_KIND = {"directional": 0, "dome": 1, "disc": 2}


@dataclass(frozen=True)
class Light:
    """ADR 0007 7b: L1 = `Light("dome")`, L2 = `Light("disc")` (45 deg, radius 2.5 deg,
    `effective`). A disc has uniform radiance over a cone of half-angle `radius_deg`,
    scaled so the irradiance on a surface facing it is `irradiance`; radius -> 0 is the
    directional light."""

    kind: Literal["directional", "dome", "disc"]
    theta_deg: float = 45.0  # directional, disc: polar angle of the centre from the normal
    phi_deg: float = 90.0  # directional, disc: azimuth from plane.u toward plane.v
    irradiance: float = 1.0  # directional, disc: E on a surface facing the light
    radiance: float = 1.0  # dome: Ld
    radius_deg: float = 2.5  # disc: angular radius, effective (ADR 0007 7b)

    def __post_init__(self) -> None:
        if self.kind == "disc":
            if not 0.0 < self.radius_deg <= 90.0:
                raise ValueError("disc radius must lie in (0, 90] degrees")
            if self.theta_deg + self.radius_deg > 90.0 + 1e-9:
                raise ValueError("the disc must lie above the horizon")

    @property
    def one_minus_cos(self) -> float:
        """Disc: 1 - cos(radius), written to stay accurate for small radii."""
        return 2.0 * math.sin(math.radians(self.radius_deg) / 2.0) ** 2

    @property
    def disc_radiance(self) -> float:
        """Disc: Lr = E / (pi sin^2 radius)."""
        return self.irradiance / (math.pi * math.sin(math.radians(self.radius_deg)) ** 2)

    def white(self) -> float:
        """Radiance from a white Lambertian standard under this light. For a disc,
        the integral of cos over a cone is exactly cos(centre) times its value facing
        the cone, so this is E cos(theta) / pi as for the directional light."""
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
     ("nmat", "<i4"), ("nprof", "<i4"), ("dt", "<f4"), ("order", "<i4"), ("flags", "<i4"),
     ("lomc", "<f4")]
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
    """Linear radiance per pixel, including the face's specular F(0) L of a dome, or
    of a disc that contains the normal.

    mubar: global majorant (default max mu_t). order > 0 keeps only that real-
    collision order (single scattering check)."""
    cp = _gpu.cupy()
    kept_mus = materials.kept()[0].astype(np.float32)
    mua, mus = cp.asarray(materials.mua), cp.asarray(kept_mus)
    ph, prof = cp.asarray(materials.phase_table()), cp.asarray(materials.profile)
    mb = float((materials.mua + kept_mus).max()) if mubar is None else mubar
    paths = view.nx * view.ny * spp
    if paths >= 2**63:
        raise ValueError("too many paths")
    c = np.zeros((), CAM_DTYPE)
    vals = dict(s0=view.s0, t0=view.t0, pix=view.pixel_mm, nx=view.nx, ny=view.ny, spp=spp,
                seed=seed, n_in=n_in, D=depth_mm, light=LIGHT_KIND[light.kind],
                lth=math.radians(light.theta_deg), lph=math.radians(light.phi_deg),
                E=light.irradiance, Ld=light.radiance, mubar=mb, nmat=len(materials.g),
                nprof=len(materials.profile), dt=materials.dt, order=order,
                flags=int(flags), lomc=light.one_minus_cos)  # fmt: skip
    for key, v in vals.items():
        c[key] = v
    cam = cp.asarray(np.frombuffer(c.tobytes(), np.uint8))
    img = cp.zeros((view.ny, view.nx, 3), cp.float32)
    stats = cp.zeros(2, cp.uint64)
    k = _gpu.kernel("camera", "render")
    _gpu.synchronize()
    t0 = time.perf_counter()
    k(((paths + block - 1) // block,), (block,),
      (scene.struct, cam, prof, mua, mus, ph, img, stats))  # fmt: skip
    _gpu.synchronize()
    sec = time.perf_counter() - t0
    out = img.get().astype(np.float64)
    f0 = ((n_in - 1) / (n_in + 1)) ** 2  # the face's specular reflectance seen by the camera
    if light.kind == "dome":
        out += f0 * light.radiance
    elif light.kind == "disc" and light.theta_deg <= light.radius_deg:  # the normal is in the disc
        out += f0 * light.disc_radiance
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
