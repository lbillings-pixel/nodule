"""nodule.render.record and nodule.render.materials (CPU; these run in CI)."""

# pyright: basic

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from agatesim.fields import Fields, Phase, Pigment
from agatesim.volume.config import SliceConfig

from nodule.render import tgrid
from nodule.render.camera import Materials
from nodule.render.materials import (
    GOETHITE_K,
    GOETHITE_N,
    HEMATITE_NK,
    STEPS,
    Optics,
    OpticsTable,
    rayleigh_absorption,
)
from nodule.render.record import BandRecord
from nodule.specimen import Specimen, load_preset

CUT = SliceConfig(tilt_deg=90.0, azimuth_deg=270.0, offset_mm=0.0)


def fields(t_dep, phase=None, impurity=None, translucency=None, pigment=None) -> Fields:
    """Synthetic fields at points; band properties a function of the step by default."""
    t = np.asarray(t_dep, np.float32)
    k = np.floor(np.nan_to_num(t) * STEPS)
    ph = np.full(t.shape, Phase.CHALCEDONY, np.uint8) if phase is None else np.asarray(phase)
    return Fields(
        phase=ph.astype(np.uint8),
        band=k.astype(np.int32),
        impurity=(k % 7 / 10 if impurity is None else np.asarray(impurity)).astype(np.float32),
        translucency=(k % 5 / 4 if translucency is None else np.asarray(translucency)).astype(
            np.float32
        ),
        t_dep=t,
        grain=np.full(t.shape, -1, np.int32),
        micro_spacing_um=np.full(t.shape, np.nan, np.float32),
        pigment=None if pigment is None else np.asarray(pigment, np.uint8),
        t_smooth=t,
    )


def steps_mid(k) -> np.ndarray:
    return (np.asarray(k) + 0.5) / STEPS


# --- pigment arithmetic: verified against the reading notes ------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [("go2022", (1.39, 0.56, 0.21)), ("bedidi1993", (0.40, 0.22, 0.18))],
)
def test_hematite_rayleigh_matches_the_notes(name, expected) -> None:
    # notes/n2-reading/agate-optics2.md: mu_a per phi = 1e-4 (Rayleigh, n_m 1.54)
    n, k = HEMATITE_NK[name]
    assert np.allclose(rayleigh_absorption(n, k, 1e-4), expected, atol=0.006)


def test_goethite_rayleigh_matches_the_notes_in_the_red() -> None:
    # Notes: 0.15 / 0.12 at 550 / 650 nm with Bedidi's k (450 is the effective edge here).
    mua = rayleigh_absorption(GOETHITE_N, GOETHITE_K, 1e-4)
    assert np.allclose(mua[1:], (0.15, 0.12), atol=0.006)


def test_rayleigh_absorption_vanishes_without_k() -> None:
    # Control: a non-absorbing particle (k = 0) absorbs nothing.
    assert np.allclose(rayleigh_absorption((3.0, 3.0, 3.0), (0, 0, 0), 1e-3), 0.0)


# --- the record ---------------------------------------------------------------------------


def test_record_reads_each_step_once() -> None:
    k = np.array([3, 3, 3, 10, 10, 250])
    t = steps_mid(k) + np.array([-0.4, 0, 0.4, 0, 0.1, 0]) / STEPS
    rec = BandRecord.of(fields(t), np.ones(len(t), bool))
    assert np.flatnonzero(rec.seen).tolist() == [3, 10, 250]
    assert rec.impurity[10] == pytest.approx(10 % 7 / 10)
    assert rec.pigment[3] == Pigment.IRON  # agatesim: no pigment means iron throughout


def test_record_refuses_properties_that_vary_within_a_step() -> None:
    # Negative control: a field that is not a band field must be rejected.
    t = steps_mid([5, 5])
    with pytest.raises(ValueError, match="impurity differs"):
        BandRecord.of(fields(t, impurity=[0.1, 0.2]), np.ones(2, bool))


def test_record_refuses_voids_and_dendrites_but_passes_tubes() -> None:
    inside = np.ones(2, bool)
    void = fields([steps_mid(1), np.nan], phase=[Phase.CHALCEDONY, Phase.VOID])
    with pytest.raises(ValueError, match="no void material"):
        BandRecord.of(void, inside)
    dendrite = fields(steps_mid([1, 2]), phase=[Phase.CHALCEDONY, Phase.INCLUSION])
    with pytest.raises(ValueError, match="dendrites"):
        BandRecord.of(dendrite, inside)
    tube = fields([steps_mid(1), np.nan], phase=[Phase.CHALCEDONY, Phase.INCLUSION])
    assert BandRecord.of(tube, inside).seen.sum() == 1
    outside = fields([steps_mid(1), np.nan], phase=[Phase.CHALCEDONY, Phase.HOST])
    assert BandRecord.of(outside, np.array([True, False])).seen.sum() == 1


def test_merge_combines_and_refuses_disagreement() -> None:
    a = BandRecord.of(fields(steps_mid([1, 2])), np.ones(2, bool))
    b = BandRecord.of(fields(steps_mid([2, 9])), np.ones(2, bool))
    assert np.flatnonzero(a.merge(b).seen).tolist() == [1, 2, 9]
    c = BandRecord.of(fields(steps_mid([2]), impurity=[0.99]), np.ones(1, bool))
    with pytest.raises(ValueError, match="disagree on impurity"):
        a.merge(c)


def test_nearest_seen() -> None:
    rec = BandRecord.of(fields(steps_mid([10, 20])), np.ones(2, bool))
    near = rec.nearest_seen()
    assert near[[0, 10, 14, 15, 16, 20, STEPS - 1]].tolist() == [10, 10, 10, 10, 20, 20, 20]


# --- the table ----------------------------------------------------------------------------


def test_table_responds_to_the_fields_as_stated() -> None:
    tab = OpticsTable()
    chal, iron = Phase.CHALCEDONY, Pigment.IRON
    lo, hi = tab.band(chal, iron, 0.1, 0.5), tab.band(chal, iron, 0.8, 0.5)
    assert (hi[0] > lo[0]).all()  # more stain absorbs more
    red = tab.band(chal, Pigment.HEMATITE, 1.0, 0.5).mua
    assert red[0] > red[1] > red[2]  # absorbs blue most: red
    yellow = tab.band(chal, Pigment.GOETHITE, 1.0, 0.5).mua
    assert yellow[0] > 2 * yellow[1]  # the effective blue edge makes it yellow
    with pytest.raises(ValueError, match="VOID"):
        tab.band(Phase.VOID, iron, 0.0, 0.0)


def test_iron_floor_clears_iron_oxides_below_it_only() -> None:
    """ADR 0014 4l (option A): below the floor iron-oxide pigments absorb nothing; at and
    above it they load as without a floor. Off (0) by default."""
    off, on = OpticsTable(), OpticsTable(iron_floor=0.28)
    chal, qtz = Phase.CHALCEDONY, Phase.QUARTZ
    assert off.iron_floor == 0.0
    assert (off.band(chal, Pigment.IRON, 0.05, 0.5).mua > off.matrix_mua).all()  # off: tinted
    for p in (Pigment.IRON, Pigment.HEMATITE, Pigment.GOETHITE):
        np.testing.assert_array_equal(on.band(chal, p, 0.27, 0.5).mua, on.matrix_mua)
        for x in (0.28, 0.6):  # at the floor and above: unchanged
            np.testing.assert_array_equal(
                on.band(chal, p, x, 0.5).mua, off.band(chal, p, x, 0.5).mua
            )
    np.testing.assert_array_equal(on.band(qtz, Pigment.IRON, 0.1, 1.0).mua, on.matrix_mua)


def test_iron_floor_negative_control_leaves_the_rest_alone() -> None:
    """Negative control: the floor must not touch pigments that are not iron oxides, nor
    any band's scattering."""
    off, on = OpticsTable(), OpticsTable(iron_floor=0.28)
    chal = Phase.CHALCEDONY
    for p in (Pigment.CELADONITE, Pigment.MANGANESE, Pigment.VIOLET):
        lo = on.band(chal, p, 0.1, 0.5).mua
        assert (lo > on.matrix_mua).all()
        np.testing.assert_array_equal(lo, off.band(chal, p, 0.1, 0.5).mua)
    for p in (Pigment.IRON, Pigment.BLUE, Pigment.WHITE):
        a, b = on.band(chal, p, 0.1, 0.3), off.band(chal, p, 0.1, 0.3)
        for name in ("mus", "g", "f", "gb"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_scattering_parts_follow_translucency_and_pigment() -> None:
    """ADR 0007 7a C: grains in lobe a, pores (and BLUE) in lobe b."""
    tab = OpticsTable()
    chal, iron = Phase.CHALCEDONY, Pigment.IRON
    assert tab.grain_radius(1.0) == pytest.approx(tab.grain_r_clear_um)
    assert tab.grain_radius(0.0) == pytest.approx(tab.grain_r_milky_um)
    clear, milky = tab.band(chal, iron, 0.1, 1.0), tab.band(chal, iron, 0.1, 0.0)
    assert (clear.f == 0).all()  # no pores at the clear end
    assert (clear.mus < 1.0).all()  # 0.15 um grains: 0.28-0.69 per mm
    assert (milky.mus > 5 * clear.mus).all()
    assert (milky.g > 0.996).all()  # coarse grains scatter almost straight ahead
    assert (milky.f > 0).all()
    assert (milky.gb < 0.65).all()  # pores: near-isotropic lobe (water: 0.62 in blue, ADR 0010)
    # transport mus' = mus_a (1 - g) + mus_b (1 - gb): pores dominate in milky bands
    grain_t = milky.mus * (1 - milky.f) * (1 - milky.g)
    pore_t = milky.mus * milky.f * (1 - milky.gb)
    assert (pore_t > 10 * grain_t).all()
    white = [tab.band(chal, Pigment.WHITE, x, 1.0).f for x in (0.0, 0.5, 1.0)]
    assert (white[0] == 0).all()
    assert (white[1] < white[2]).all()
    blue = tab.band(chal, Pigment.BLUE, 1.0, 1.0)
    assert blue.mus[0] > blue.mus[2]
    assert (blue.gb == 0).all()
    assert (blue.f > 0.5).all()
    quartz = tab.band(Phase.QUARTZ, iron, 0.0, 0.0)
    assert (quartz.f == 0).all()
    assert (quartz.g == tab.quartz_g).all()


def test_materials_validation_catches_a_profile_beyond_the_table() -> None:
    with pytest.raises(ValueError, match="beyond the table"):
        Materials.of(np.zeros((3, 3)), np.zeros((3, 3)), np.zeros(3), profile=[0, 1])


@pytest.fixture(scope="module")
def specimen_grid() -> tuple[Specimen, tgrid.GrowthGrid]:
    sp = Specimen.build(load_preset("n2-slice"))
    return sp, tgrid.GrowthGrid.fill(sp, sp.plane(CUT), 0.5, 2.0)  # ~0.2 s


def test_profile_reproduces_the_specimen_at_random_points(specimen_grid) -> None:
    """The camera's lookup (2 + profile[floor(t / dt)]) gives each point the optics of
    its own agatesim fields. Control: the profile one step out of register fails."""
    sp, grid = specimen_grid
    assert grid.bands is not None
    tab = OpticsTable()
    m = tab.materials(grid.bands)
    rng = np.random.default_rng(7)
    n = 4000
    pts = tgrid.to_world(
        grid.plane,
        rng.uniform(grid.s[0], grid.s[-1], n),
        rng.uniform(grid.t[0], grid.t[-1], n),
        rng.uniform(0.0, 2.0, n),
    )
    inside = sp.volume.solid.contains(pts)
    f = sp.volume.evaluate(pts, inside)
    assert f.t_smooth is not None
    ok = inside & np.isfinite(f.t_smooth)
    assert ok.sum() > 1000
    k = np.floor(f.t_smooth[ok].astype(np.float32) / np.float32(m.dt)).astype(int)
    pig = np.full(ok.sum(), Pigment.IRON) if f.pigment is None else f.pigment[ok]
    want = np.array([np.r_[*r] for r in (
        tab.band(int(p), int(g), float(x), float(tr))
        for p, g, x, tr in zip(f.phase[ok], pig, f.impurity[ok], f.translucency[ok], strict=True)
    )])  # fmt: skip
    table = np.c_[m.mua, m.mus, m.g, m.f, m.gb]

    def agree(shift: int) -> float:
        got = table[2 + m.profile[np.clip(k + shift, 0, STEPS - 1)]]
        return float(np.all(np.isclose(got, want, rtol=1e-5), axis=1).mean())

    assert agree(0) > 0.999
    assert agree(1) < 0.9  # measured 0.82: a band spans several steps
    assert agree(3) < 0.7  # measured 0.47


def test_tuning_step_1_values_and_the_air_water_equivalence() -> None:
    """ADR 0010: step 1's values, and 1.75% water-filled pores scattering like the
    tested 0.3% air (reduced mu_s of the pore part within 3.5% per channel). Control:
    1.75% air must not match."""
    tab = replace(OpticsTable(), pore_gamma=1.0, pore_milky=0.0175)  # step 1 (ADR 0012)
    assert (tab.pore_n, tab.pore_white, tab.phi_iron_oxide) == (1.333, 0.011667, 9e-4)

    def pore_transport(t: OpticsTable):
        o = t.scattering(Pigment.IRON, 0.5, 0.0)
        return o.mus * o.f * (1 - o.gb)

    air = pore_transport(replace(tab, pore_n=1.0, pore_milky=3e-3))
    assert (abs(pore_transport(tab) / air - 1) < 0.035).all()
    assert (pore_transport(replace(tab, pore_n=1.0)) > 3 * air).all()


def _reduced(o: Optics) -> np.ndarray:
    return o.mus * ((1 - o.f) * (1 - o.g) + o.f * (1 - o.gb))


def test_tuning_step_2_pores_fade_as_a_fourth_power() -> None:
    """ADR 0012 12a A: gamma 4, pore_milky rescaled so the n2-slice record's median band
    (t 0.264) keeps step 1's 1.29%. A clear band (t 0.806) then scatters 9x less than
    under step 1, down near the grains-only floor (the probe's 0.99 -> 0.11 per mm at
    550 nm). Control: step 1's values (gamma 1, 1.75%) must keep the old optics, and
    fail the clear-band bound."""
    tab = OpticsTable()
    assert tab.pore_gamma == 4.0
    assert abs(tab.pore_milky - 0.043894) < 1e-6
    step1 = replace(tab, pore_gamma=1.0, pore_milky=0.0175)
    med, clear = 0.264, 0.806
    assert np.allclose(_reduced(tab.scattering(Pigment.IRON, 0.1, med)),
                       _reduced(step1.scattering(Pigment.IRON, 0.1, med)), rtol=1e-9)  # fmt: skip
    floor = _reduced(replace(tab, pore_milky=0.0).scattering(Pigment.IRON, 0.1, clear))
    now = _reduced(tab.scattering(Pigment.IRON, 0.1, clear))
    old = _reduced(step1.scattering(Pigment.IRON, 0.1, clear))
    assert np.allclose(old[1], 0.99, atol=0.01)
    assert np.allclose(now[1], 0.11, atol=0.01)
    assert (now < 1.2 * floor).all()
    assert not (old < 1.2 * floor).any()  # control: step 1 is far above the floor
