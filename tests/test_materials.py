"""nodule.render.record and nodule.render.materials (CPU; these run in CI)."""

# pyright: basic

from __future__ import annotations

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
    clear, milky = tab.band(chal, iron, 0.1, 1.0), tab.band(chal, iron, 0.1, 0.0)
    assert clear[1][0] == pytest.approx(tab.mus_clear)
    assert milky[1][0] == pytest.approx(tab.mus_milky)
    red = tab.band(chal, Pigment.HEMATITE, 1.0, 0.5)[0]
    assert red[0] > red[1] > red[2]  # absorbs blue most: red
    yellow = tab.band(chal, Pigment.GOETHITE, 1.0, 0.5)[0]
    assert yellow[0] > 2 * yellow[1]  # the effective blue edge makes it yellow
    blue = tab.band(chal, Pigment.BLUE, 1.0, 0.5)
    assert blue[1][0] > blue[1][2]
    assert blue[2] < tab.g_band
    with pytest.raises(ValueError, match="VOID"):
        tab.band(Phase.VOID, iron, 0.0, 0.0)


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
    want = np.array([np.r_[r[0], r[1], r[2]] for r in (
        tab.band(int(p), int(g), float(x), float(tr))
        for p, g, x, tr in zip(f.phase[ok], pig, f.impurity[ok], f.translucency[ok], strict=True)
    )])  # fmt: skip
    table = np.c_[m.mua, m.mus, m.g]

    def agree(shift: int) -> float:
        got = table[2 + m.profile[np.clip(k + shift, 0, STEPS - 1)]]
        return float(np.all(np.isclose(got, want, rtol=1e-5), axis=1).mean())

    assert agree(0) > 0.999
    assert agree(1) < 0.9  # measured 0.82: a band spans several steps
    assert agree(3) < 0.7  # measured 0.47
