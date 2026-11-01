"""nodule.render on the GPU: the ADR 0007 section 4 checks, each with its control.

Skipped without a CUDA GPU (CI). Run locally with `uv run --no-sync pytest`.
Seeds are fixed and Philox streams are keyed per packet, so each result is
reproducible: a |z| < 3 pass here stays a pass. Numbers from the prototype are
in notes/n2-proto/RESULTS.md. Units mm.
"""

# pyright: basic

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from agatesim.models.growth import ProceduralGrowth
from agatesim.volume.config import SliceConfig
from agatesim.volume.fractures import restore
from agatesim.volume.growth import segment_distance

from nodule.render import _gpu, reference, scene, tgrid
from nodule.render.camera import Flag, Light, Materials, View, render
from nodule.render.reference import T1, T2
from nodule.render.slab import slab
from nodule.specimen import Specimen, load_preset

pytestmark = pytest.mark.skipif(not _gpu.available(), reason="needs a CUDA GPU and CuPy")

N = 1_000_000
CUT = SliceConfig(tilt_deg=90.0, azimuth_deg=270.0, offset_mm=0.0)
H, DEPTH = 0.2, 3.0  # the 10 mm prototype grid takes ~9 s to fill; 3 mm ~5 s


def z(a: float, sa: float, b: float, sb: float = 0.0) -> float:
    return (a - b) / math.hypot(sa, sb)


def t1(**kw):
    return slab(N, depth_mm=T1["d"], n_in=T1["n"], g=T1["g"], mua=[T1["mua"]] * 3,
                mus=[T1["mus"]] * 3, **kw)  # fmt: skip


# --- forward slab against MCML and exact values -------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 4])
def test_t1_slab_any_majorant(k: int) -> None:
    r = t1(mubar=10.0 * k, seed=k)
    assert abs(z(r.rd[0], r.rd_se[0], T1["rd"])) < 3
    assert abs(z(r.tt[0], r.tt_se[0], T1["tt"])) < 3


def test_spectral_tracking_matches_one_channel_runs() -> None:
    coeffs = [(1.0, 9.0), (0.2, 4.0), (3.0, 15.0)]
    spec = slab(N, depth_mm=0.2, n_in=1.0, g=0.75, mua=[c[0] for c in coeffs],
                mus=[c[1] for c in coeffs], seed=7)  # fmt: skip
    assert abs(z(spec.rd[0], spec.rd_se[0], T1["rd"])) < 3
    for c, (a, s) in enumerate(coeffs):
        mono = slab(N, depth_mm=0.2, n_in=1.0, g=0.75, mua=[a] * 3, mus=[s] * 3, seed=11 + c)
        assert abs(z(spec.rd[c], spec.rd_se[c], mono.rd[0], mono.rd_se[0])) < 3
        assert abs(z(spec.tt[c], spec.tt_se[c], mono.tt[0], mono.tt_se[0])) < 3


@pytest.mark.parametrize("k", [1, 3])
def test_t2_halfspace_with_specular(k: int) -> None:
    r = slab(N, depth_mm=math.inf, n_in=T2["n"], g=T2["g"], mua=[T2["mua"]] * 3,
             mus=[T2["mus"]] * 3, mubar=10.0 * k, seed=20 + k)  # fmt: skip
    assert abs(z(r.rd[0] + 0.04, r.rd_se[0], T2["rd_with_specular"])) < 3


def test_matched_isotropic_halfspace_exact_h() -> None:
    r = slab(N, depth_mm=math.inf, n_in=1.0, g=0.0, mua=[1.0] * 3, mus=[9.0] * 3, seed=50)
    assert abs(z(r.rd[0], r.rd_se[0], reference.halfspace_albedo_iso(0.9))) < 3


def test_furnace_conserves_energy() -> None:
    r = slab(N, depth_mm=0.5, n_in=1.54, g=0.75, mua=[0.0] * 3, mus=[9.0] * 3, mubar=20.0,
             seed=52)  # fmt: skip
    assert r.rd[0] + r.tt[0] + reference.fresnel(1.0, 1.0, 1.54) == pytest.approx(1.0, abs=1e-5)


# pure absorber, layers 0.05 mm of mua 20 / 0.5 per mm
PHANTOM: dict[str, Any] = dict(depth_mm=0.2, n_in=1.0, g=0.0, mua=[[20.0] * 3, [0.5] * 3],
                               mus=[[0.0] * 3] * 2, layers=[0, 1, 0, 1], layer_mm=0.05)  # fmt: skip
PHANTOM_TT = math.exp(-(2 * 0.05 * 20 + 2 * 0.05 * 0.5))


@pytest.mark.parametrize("mubar", [None, 40.0])
def test_band_phantom(mubar: float | None) -> None:
    r = slab(N, mubar=mubar, seed=30, **PHANTOM)
    assert abs(z(r.tt[0], r.tt_se[0], PHANTOM_TT)) < 3


def test_weighted_tracking_survives_a_non_bounding_majorant() -> None:
    r = slab(N, mubar=10.0, seed=42, **PHANTOM)  # half the largest mu_t
    assert abs(z(r.tt[0], r.tt_se[0], PHANTOM_TT)) < 3


def test_control_analog_tracking_fails_with_a_non_bounding_majorant() -> None:
    r = slab(N, mubar=10.0, analog=True, seed=40, **PHANTOM)
    assert abs(z(r.tt[0], r.tt_se[0], PHANTOM_TT)) > 10  # prototype z 464
    r = t1(mubar=5.0, analog=True, seed=41)
    assert abs(z(r.tt[0], r.tt_se[0], T1["tt"])) > 10  # prototype z 628


# --- the specimen on the GPU ------------------------------------------------------------------


@pytest.fixture(scope="module")
def specimen_scene():
    config = load_preset("n2-slice")
    sp = Specimen.build(config)
    grid = tgrid.GrowthGrid.fill(sp, sp.plane(CUT), H, DEPTH)
    dev = scene.upload(scene.pack(sp, grid))
    model = config.volume.model
    assert isinstance(model, ProceduralGrowth)
    return sp, grid, dev, model.band_period


def reference_classes(sp: Specimen, pts: np.ndarray):
    f, ins = sp.evaluate(pts)
    back = restore(pts, sp.fractures)
    tube = np.zeros(len(pts), bool)
    for row in sp.volume.setting.tubes:
        tube |= segment_distance(back, row[:3], row[3:6]) <= row[6]
    return np.where(~ins, 0, np.where(tube, 1, 2)), f.t_smooth, back


def gpu_classes(dev, pts: np.ndarray, mode: int):
    cp = _gpu.cupy()
    n = len(pts)
    cls, tt = cp.empty(n, cp.int32), cp.empty(n, cp.float32)
    k = _gpu.kernel("scene", "classify_points")
    k(((n + 255) // 256,), (256,),
      (np.int64(n), cp.asarray(pts, cp.float32), dev.struct, np.int32(mode), cls, tt))  # fmt: skip
    return cls.get(), tt.get().astype(np.float64)


@pytest.fixture(scope="module")
def check_points(specimen_scene):
    sp, grid, _, _ = specimen_scene
    rng = np.random.default_rng(7)
    n = 200_000
    ps, pt = rng.uniform(grid.s[0], grid.s[-1], n), rng.uniform(grid.t[0], grid.t[-1], n)
    pts = tgrid.to_world(grid.plane, ps, pt, rng.uniform(0, DEPTH, n))
    return pts, *reference_classes(sp, pts)


def test_classify_matches_specimen_evaluate(specimen_scene, check_points) -> None:
    _, _, dev, period = specimen_scene
    pts, ref, t_exact, _ = check_points
    assert (ref == 1).any()  # tubes are sampled
    assert (ref == 0).any()  # and the outside
    cls, tt = gpu_classes(dev, pts, 0)
    assert np.count_nonzero(cls != ref) == 0  # prototype: 0 of 400k
    both = ref == 2
    err = np.abs(tt[both] - t_exact[both]) / period
    p50, p99, top = np.percentile(err, [50, 99, 100])
    assert p50 < 0.005  # prototype .0016
    assert p99 < 0.06  # .037
    assert top < 0.3  # .21


def test_control_no_backward_map_fails(specimen_scene, check_points) -> None:
    _, _, dev, period = specimen_scene
    pts, ref, t_exact, _ = check_points
    cls, tt = gpu_classes(dev, pts, 1)
    both = (cls == 2) & (ref == 2)
    err = np.abs(tt[both] - t_exact[both]) / period
    assert np.percentile(err, 99) > 0.5  # prototype p99 1.67


def test_control_grid_only_edges_fail(specimen_scene, check_points) -> None:
    """Stone or not from the nearest raw grid cell instead of the exact test."""
    _, grid, _, _ = specimen_scene
    _, ref, _, back = check_points
    ps, pt, pw = tgrid.to_plane(grid.plane, back)
    idx = [np.clip(np.rint((p - a[0]) / H).astype(int), 0, len(a) - 1)
           for p, a in ((pw, grid.w), (pt, grid.t), (ps, grid.s))]  # fmt: skip
    stone = ~np.isnan(grid.values[idx[0], idx[1], idx[2]])
    assert np.mean(stone != (ref == 2)) > 0.001  # prototype 0.29%


# --- camera and lights on homogeneous media -------------------------------------------------


@pytest.fixture(scope="module")
def flat():
    return scene.upload(scene.homogeneous())


FLAT_VIEW = View(0.0, 0.0, 0.1, 64, 64)


def mean_px(dev, light: Light, *, spp: int, seed: int, mua: float, mus: float, g: float,
            flags: int = 0, **kw):  # fmt: skip
    m = Materials.of([[mua] * 3], [[mus] * 3], [g])
    im = render(dev, FLAT_VIEW, m, light, spp=spp, seed=seed, flags=Flag.HOMOGENEOUS | flags,
                **kw).image[..., 0].ravel()  # fmt: skip
    return float(im.mean()), float(im.std(ddof=1) / math.sqrt(im.size))


DOME = Light("dome")


@pytest.mark.parametrize(("depth", "mua", "mus", "g"),
                         [(2.0, 0.1, 10.0, 0.9), (math.inf, 0.05, 2.0, 0.97)])  # fmt: skip
def test_dome_matches_forward_albedo_by_reciprocity(flat, depth, mua, mus, g) -> None:
    m, se = mean_px(flat, DOME, spp=256, seed=11, mua=mua, mus=mus, g=g, depth_mm=depth)
    f = slab(2 * N, depth_mm=depth, n_in=1.54, g=g, mua=[mua] * 3, mus=[mus] * 3, seed=12)
    rsp = reference.fresnel(1.0, 1.0, 1.54)
    assert abs(z(m, se, f.rd[0] + rsp, f.rd_se[0])) < 3


def test_dome_matches_exact_h(flat) -> None:
    m, se = mean_px(flat, DOME, spp=256, seed=13, mua=1.0, mus=9.0, g=0.0, n_in=1.0)
    assert abs(z(m, se, reference.halfspace_albedo_iso(0.9))) < 3


def test_control_no_exit_n2_fails(flat) -> None:
    m, _ = mean_px(flat, DOME, spp=64, seed=11, mua=0.1, mus=10.0, g=0.9, depth_mm=2.0,
                   flags=Flag.NO_EXIT_N2)  # fmt: skip
    f = slab(N, depth_mm=2.0, n_in=1.54, g=0.9, mua=[0.1] * 3, mus=[10.0] * 3, seed=12)
    assert m > 1.5 * (f.rd[0] + 0.045)  # prototype .561 vs .263


@pytest.mark.parametrize(
    ("mua", "mus", "g", "theta"), [(0.5, 2.0, 0.8, 45.0), (0.2, 3.0, 0.0, 30.0)]
)
def test_single_scattering_closed_form(flat, mua, mus, g, theta) -> None:
    light = Light("directional", theta_deg=theta)
    m, se = mean_px(flat, light, spp=256, seed=21, mua=mua, mus=mus, g=g, order=1)
    assert abs(z(m, se, reference.single_scatter_halfspace(1.54, mua, mus, g, theta))) < 3


def test_control_no_ratio_tracking_fails(flat) -> None:
    light = Light("directional", theta_deg=45.0)
    m, _ = mean_px(flat, light, spp=64, seed=21, mua=0.5, mus=2.0, g=0.8, order=1,
                    flags=Flag.NO_RATIO_TRACKING)  # fmt: skip
    assert m > 1.5 * reference.single_scatter_halfspace(1.54, 0.5, 2.0, 0.8, 45.0)  # proto 2.1x


# --- two-lobe phase function per channel (ADR 0007 7a C) -----------------------------------


def mean_rgb(dev, m: Materials, light: Light, *, spp: int, seed: int, flags: int = 0, **kw):
    im = render(dev, FLAT_VIEW, m, light, spp=spp, seed=seed, flags=Flag.HOMOGENEOUS | flags,
                **kw).image.reshape(-1, 3)  # fmt: skip
    return im.mean(0), im.std(0, ddof=1) / math.sqrt(len(im))


def ss_mix(f: float, ga: float, gb: float) -> float:
    return sum(w * reference.single_scatter_halfspace(1.54, 0.3, 3.0, g, 40.0)
               for w, g in ((1 - f, ga), (f, gb)))  # fmt: skip


def test_two_lobe_single_scattering_is_the_mixture(flat) -> None:
    m = Materials.of([[0.3] * 3], [[3.0] * 3], [0.9], f=[0.3], gb=[0.0])
    mu, se = mean_rgb(flat, m, Light("directional", theta_deg=40.0), spp=256, seed=41, order=1)
    want = ss_mix(0.3, 0.9, 0.0)
    assert abs(z(mu[0], se[0], want)) < 3  # measured -1.5
    # control: one lobe with the mean cosine (the merge the two lobes replace) is far off
    assert abs(z(ss_mix(0.0, 0.63, 0.0), se[0], want)) > 100  # measured -517


LOBES: dict[str, Any] = dict(
    mua=[[0.2, 0.1, 0.05]], mus=[[8.0, 4.0, 12.0]], g=[[0.95, 0.8, 0.99]],
    f=[[0.5, 0.2, 0.05]], gb=[[0.0, 0.3, 0.0]],
)  # fmt: skip


def one_channel(c: int) -> Materials:
    """Material whose three channels all carry channel c's coefficients and lobes."""
    one: dict[str, Any] = {k: [[v[0][c]] * 3] for k, v in LOBES.items()}
    return Materials.of(**one)


@pytest.fixture(scope="module")
def lobe_singles(flat):
    return [mean_rgb(flat, one_channel(c), DOME, spp=1024, seed=52 + c, depth_mm=1.0)
            for c in range(3)]  # fmt: skip


@pytest.mark.parametrize(
    ("flags", "lo", "hi"), [(0, 0.0, 3.0), (Flag.NO_PHASE_WEIGHT, 20.0, math.inf)]
)
def test_per_channel_lobes_match_one_channel_runs(flat, lobe_singles, flags, lo, hi) -> None:
    """Three channels with different lobes share one path; each must equal its own
    single-channel run. Control: without the p_c / p_mix weight the channels drift
    (prototype of this check: |z| 26-130)."""
    mu, se = mean_rgb(flat, Materials.of(**LOBES), DOME, spp=1024, seed=51, flags=flags,
                      depth_mm=1.0)  # fmt: skip
    for c, (m1, s1) in enumerate(lobe_singles):
        assert lo <= abs(z(mu[c], se[c], m1[0], s1[0])) < hi, c


THIN: dict[str, Any] = dict(depth_mm=2.0, mua=0.05, mus=0.5, g=0.9)


def hemisphere_integral(flat, flags: int) -> tuple[float, float]:
    """NEE from 16 directional lights over the sky (Gauss-Legendre in cos theta)."""
    x, wq = np.polynomial.legendre.leggauss(16)
    mu, wq = 0.5 * (x + 1), 0.5 * wq
    vals, ses = [], []
    for i, m_ in enumerate(mu):
        light = Light("directional", theta_deg=math.degrees(math.acos(m_)))
        v, s_ = mean_px(flat, light, spp=256, seed=100 + i, flags=flags, **THIN)
        vals.append(v)
        ses.append(s_)
    return (2 * math.pi * float(np.dot(wq, vals)),
            2 * math.pi * math.sqrt(float(np.dot(wq**2, np.square(ses)))))  # fmt: skip


@pytest.fixture(scope="module")
def dome_diffuse(flat):
    m, se = mean_px(flat, DOME, spp=1024, seed=31, flags=Flag.SCATTERED_ONLY, **THIN)
    return m - reference.fresnel(1.0, 1.0, 1.54), se


def test_directional_over_the_sky_matches_the_dome(flat, dome_diffuse) -> None:
    tot, se = hemisphere_integral(flat, 0)
    assert abs(z(tot, se, *dome_diffuse)) < 3


def test_control_no_mirror_term_fails(flat, dome_diffuse) -> None:
    tot, se = hemisphere_integral(flat, Flag.NO_MIRROR)
    assert abs(z(tot, se, *dome_diffuse)) > 10  # prototype z -59


# --- disc lamp, L2 (ADR 0007 7b): light and phase sampling under MIS ------------------------

L2 = Light("disc")  # 45 deg, radius 2.5 deg


def disc_rgb(dev, light: Light, *, seed: int, flags: int = 0):
    return mean_rgb(dev, Materials.of(**LOBES), light, spp=1024, seed=seed, flags=flags,
                    depth_mm=1.0)  # fmt: skip


@pytest.fixture(scope="module")
def disc_mis(flat):
    return disc_rgb(flat, L2, seed=61)


@pytest.mark.parametrize(("flags", "seed"), [(Flag.NEE_ONLY, 62), (Flag.PHASE_ONLY, 63)])
def test_disc_strategies_agree(flat, disc_mis, flags, seed) -> None:
    """Light sampling alone, phase sampling alone and their MIS blend estimate the same
    image (sharp two-lobe material, all three channels; measured |z| <= 1.4)."""
    mu, se = disc_rgb(flat, L2, seed=seed, flags=flags)
    for c in range(3):
        assert abs(z(mu[c], se[c], disc_mis[0][c], disc_mis[1][c])) < 3, c


def test_control_disc_without_mis_weights_fails(flat, disc_mis) -> None:
    mu, se = disc_rgb(flat, L2, seed=64, flags=Flag.NO_MIS_WEIGHT)
    zs = [z(mu[c], se[c], disc_mis[0][c], disc_mis[1][c]) for c in range(3)]
    assert max(zs) > 10  # measured 22, 17, 6


ONE: dict[str, Any] = dict(mua=0.1, mus=10.0, g=0.9)


@pytest.mark.parametrize("depth", [2.0, math.inf])
@pytest.mark.parametrize("flags", [0, Flag.NEE_ONLY, Flag.PHASE_ONLY])
def test_hemisphere_disc_is_the_dome(flat, depth, flags) -> None:
    """A 90 deg disc at the zenith is the dome with Ld = E / pi: every angle of the light
    sampling, refraction Jacobian and lamp-hit test is exercised (measured |z| <= 1.7)."""
    dome = mean_px(flat, Light("dome", radiance=1 / math.pi), spp=256, seed=65, depth_mm=depth,
                   **ONE)  # fmt: skip
    sky = Light("disc", theta_deg=0.0, radius_deg=90.0)
    m, se = mean_px(flat, sky, spp=256, seed=66, depth_mm=depth, flags=flags, **ONE)
    assert abs(z(m, se, *dome)) < 3


def test_control_part_sky_disc_is_not_the_dome(flat) -> None:
    dome = mean_px(flat, Light("dome", radiance=1 / math.pi), spp=256, seed=65, depth_mm=2.0,
                   **ONE)  # fmt: skip
    m, se = mean_px(flat, Light("disc", theta_deg=0.0, radius_deg=60.0), spp=256, seed=66,
                    depth_mm=2.0, **ONE)  # fmt: skip
    assert abs(z(m, se, *dome)) > 10  # measured 56


@pytest.mark.parametrize(("theta", "lo", "hi"), [(45.0, 0.0, 3.0), (50.0, 10.0, math.inf)])
def test_small_disc_is_the_directional_light(flat, theta, lo, hi) -> None:
    """A 0.25 deg disc matches the directional light at 45 deg (measured z 0.7).
    Control: the disc moved to 50 deg does not (measured -24)."""
    kw: dict[str, Any] = dict(spp=256, depth_mm=2.0, mua=0.2, mus=3.0, g=0.8)
    want = mean_px(flat, Light("directional", theta_deg=45.0), seed=67, **kw)
    m, se = mean_px(flat, Light("disc", theta_deg=theta, radius_deg=0.25), seed=68, **kw)
    assert lo <= abs(z(m, se, *want)) < hi


# --- preview against full render on the specimen -------------------------------------------

# EFFECTIVE stand-ins for plumbing only, not agate optics (ADR 0007 section 3 will set them):
# host, tube core, then band materials clear, milky, pigment.
STAND_IN = Materials.of(
    mua=[[0.6, 1.0, 1.6], [2.5, 0.8, 1.8], [0.01] * 3, [0.01, 0.01, 0.02], [0.2, 0.8, 2.0]],
    mus=[[15] * 3, [10] * 3, [1.4, 1.7, 2.3], [20] * 3, [5] * 3],
    g=[0.8, 0.8, 0.97, 0.7, 0.9],
    profile=np.random.default_rng(5).choice(3, size=400, p=[0.55, 0.35, 0.10]),
    dt=0.0075,
)


def test_preview_matches_the_full_render(specimen_scene) -> None:
    _, grid, dev, _ = specimen_scene
    k, spp = 2, 128
    full_view = View.covering(grid, 0.4)
    preview = full_view.coarser(k)
    kw: dict[str, Any] = dict(depth_mm=DEPTH)
    full = render(dev, full_view, STAND_IN, DOME, spp=spp, seed=1, **kw).image
    pv = [render(dev, preview, STAND_IN, DOME, spp=spp * k * k, seed=s, **kw).image
          for s in (2, 3)]  # fmt: skip
    ny, nx = preview.ny, preview.nx
    boxed = full[: ny * k, : nx * k].reshape(ny, k, nx, k, 3).mean((1, 3))
    noise = np.std(pv[0] - pv[1]) / math.sqrt(2)
    ratio = np.sqrt(np.mean((pv[0] - boxed) ** 2)) / noise
    assert 1.2 < ratio < 1.7  # equal noise on both sides: 1.41 expected (prototype 1.43)
    # control: the window shifted 1 mm must not match
    shifted = View(preview.s0 + 1.0, preview.t0, preview.pixel_mm, nx, ny)
    other = render(dev, shifted, STAND_IN, DOME, spp=spp * k * k, seed=2, **kw).image
    assert np.sqrt(np.mean((other - boxed) ** 2)) / noise > 3  # prototype 6.7
