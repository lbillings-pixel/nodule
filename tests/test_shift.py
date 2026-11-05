# pyright: basic

"""Materials that vary along the bands, host side (ADR 0016): the record, the stone's own
fields, the material table and the majorant. The GPU copy is checked in test_shift_gpu.py."""

from __future__ import annotations

import dataclasses
from statistics import NormalDist
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from agatesim.fields import Phase, Pigment
from agatesim.models.growth import ProceduralGrowth
from agatesim.volume.features import wall_features

from nodule.render.camera import majorant
from nodule.render.materials import TR_NODES, OpticsTable
from nodule.render.record import STEPS, BandRecord
from nodule.render.shift import DeviceShift, ShiftArrays, variation_of
from nodule.specimen import Specimen, SpecimenConfig, load_preset


@pytest.fixture(scope="module")
def sample(look: Specimen) -> tuple[np.ndarray, Any]:
    rng = np.random.default_rng(1)
    lo, hi = look.volume.solid.bounds()
    p = rng.uniform(lo, hi, (30_000, 3))
    p = p[look.volume.solid.contains(p)]
    return p, look.volume.evaluate(p, np.ones(len(p), bool))


def banded(f: Any) -> np.ndarray:
    return (f.phase == Phase.CHALCEDONY) & (f.band >= 0) & np.isfinite(f.t_smooth)


def test_rebuilt_variation_is_the_stones_own(look, sample) -> None:
    p, f = sample
    b = banded(f)
    v = variation_of(look)
    assert v is not None
    gain, change = v.shift(p[b], f.t_smooth[b].astype(np.float64))
    assert (gain > 1.01).mean() > 0.1  # lenses are sampled
    imp = np.clip(f.band_impurity[b] * gain, 0, 1)
    tr = np.clip(f.band_translucency[b] + change, 0, 1)
    # measured 2.8e-5 and 1.1e-6: agatesim's t is float64, Fields stores it as float32
    assert np.abs(imp - f.impurity[b]).max() < 1e-3
    assert np.abs(tr - f.translucency[b]).max() < 1e-5
    # control: another seed's fields are not the stone's
    model = look.volume.config.model
    assert isinstance(model, ProceduralGrowth)
    rng = np.random.default_rng(look.volume.config.seed + 1)
    other = wall_features(model, look.volume.setting, rng).variation
    assert other is not None
    g2, _ = other.shift(p[b], f.t_smooth[b].astype(np.float64))
    assert np.abs(np.clip(f.band_impurity[b] * g2, 0, 1) - f.impurity[b]).max() > 0.1


def test_record_reads_band_values_and_marks_banded_chalcedony(sample) -> None:
    _, f = sample
    rec = BandRecord.of(f, np.ones(len(f.phase), bool))
    filled = np.isfinite(f.t_dep)
    k = np.clip(np.floor(np.where(filled, f.t_dep, 0) * STEPS).astype(int), 0, STEPS - 1)
    b = banded(f)
    assert rec.shifts[k[b]].all()
    assert not rec.shifts[k[~b & filled]].any()
    assert rec.impurity[k[b]] == pytest.approx(f.band_impurity[b])
    # control: the varied values differ within a step, so reading them is refused
    plain = dataclasses.replace(f, band_impurity=None, band_translucency=None)
    with pytest.raises(ValueError, match="differs within a growth step"):
        BandRecord.of(plain, np.ones(len(f.phase), bool))


def test_preset_varies_materials_along_bands() -> None:
    """ADR 0016 s.8: n2-slice carries set 4 (lens cover 0.1, drift 0.1)."""
    v = variation_of(Specimen.build(load_preset("n2-slice")))
    assert v is not None
    assert v.lens_strength == 1.5
    assert v.drift_size == pytest.approx(0.1)
    assert v.lens_threshold == pytest.approx(NormalDist().inv_cdf(0.9))  # cover 0.1


def test_without_variation_nothing_is_marked(plain_config: SpecimenConfig) -> None:
    sp = Specimen.build(plain_config)
    assert variation_of(sp) is None
    rng = np.random.default_rng(2)
    lo, hi = sp.volume.solid.bounds()
    p = rng.uniform(lo, hi, (5_000, 3))
    inside = sp.volume.solid.contains(p)
    rec = BandRecord.of(sp.volume.evaluate(p, inside), inside)
    assert rec.seen.any()
    assert not rec.shifts.any()


@pytest.fixture(scope="module")
def tables(look, sample):
    _, f = sample
    rec = BandRecord.of(f, np.ones(len(f.phase), bool))
    v = variation_of(look)
    assert v is not None
    host = ShiftArrays.of(v)
    tab = OpticsTable()
    return rec, host, tab, tab.materials(rec), tab.materials(rec, host)


def test_material_table_with_shift(tables) -> None:
    rec, _, tab, plain, m = tables
    assert plain.along is None
    assert plain.tr_rows == (0, 0)
    n = len(plain.mua)
    assert m.tr_rows == (n, TR_NODES)
    assert len(m.mua) == n + TR_NODES
    for name in ("mua", "mus", "g", "f", "gb"):  # band rows as before
        assert np.array_equal(getattr(m, name)[:n], getattr(plain, name))
    assert np.array_equal(m.profile, plain.profile)
    assert m.along is not None
    takes = m.along[:, 0] >= 0
    assert not takes[:2].any()
    assert not takes[n:].any()
    # 16b A: a row's absorption is matrix + its pigment at gain 1
    assert m.mua[takes] == pytest.approx(tab.matrix_mua + m.along[takes, 1:], abs=1e-6)
    seen = np.flatnonzero(rec.seen & rec.shifts)
    rows = 2 + m.profile[seen]
    assert m.along[rows, 0] == pytest.approx(rec.translucency[seen])
    # below the iron floor a row has no pigment, so no lens shows
    low = rec.impurity[seen] < tab.iron_floor
    assert low.any()
    assert (m.along[rows[low], 1:] == 0).all()
    assert (m.along[rows[~low], 1:] > 0).all()


def test_drift_table_matches_the_band_optics() -> None:
    """Nearest-node error of the drift's table, against OpticsTable.band (ADR 0016 s.3)."""
    tab = OpticsTable()

    def error(near: float, x: float) -> float:
        """Largest relative error in mus and in 1 - g (what carries light), any channel."""
        exact = tab.band(Phase.CHALCEDONY, Pigment.IRON, 0.3, x)
        got = tab.scattering(Pigment.IRON, 0.0, near)
        return float(max(np.abs(got.mus / exact.mus - 1).max(),
                         np.abs((1 - got.g) / (1 - exact.g) - 1).max(),
                         np.abs(got.f - exact.f).max()))  # fmt: skip

    nodes = np.linspace(0, 1, TR_NODES)
    a = np.random.default_rng(3).uniform(0, 1, 40)
    worst = max(error(float(nodes[np.argmin(np.abs(nodes - x))]), float(x)) for x in a)
    assert worst < 0.01  # within 1% of the exact translucency's optics
    # control: a 33-node table fails half a node (1/64) from its nearest node
    assert error(0.5, 0.5 + 1 / 64) > 0.01


def test_other_pigments_are_refused_when_they_take_the_shift(tables) -> None:
    rec, _, tab, _, _ = tables
    stub: Any = SimpleNamespace(drift=None)
    k = int(np.flatnonzero(rec.seen & rec.shifts)[0])
    pig = rec.pigment.copy()
    pig[k] = Pigment.CELADONITE
    green = dataclasses.replace(rec, pigment=pig)
    with pytest.raises(ValueError, match="iron-stained chalcedony only"):
        tab.materials(green, stub)
    tab.materials(rec, stub)  # control: the iron record is drawn
    still = rec.shifts.copy()
    still[k] = False
    tab.materials(
        dataclasses.replace(green, shifts=still), stub
    )  # and a green band that doesn't shift


def reached(m, kept, gain: float, d: float) -> float:
    """Brute force: the largest mu_t the kernel's own float32 pick reaches, over changes
    in [-d, d] (camera.cu: tr = clip(band + change), nearest of the drift table's rows)."""
    takes = m.along[:, 0] >= 0
    first, count = m.tr_rows
    change = np.linspace(-d, d, 20001, dtype=np.float32)
    top = 0.0
    for i in np.flatnonzero(takes):
        tr = np.clip(np.float32(m.along[i, 0]) + change, np.float32(0), np.float32(1))
        idx = np.minimum((tr * np.float32(count - 1) + np.float32(0.5)).astype(np.int32),
                         count - 1)  # fmt: skip
        mut = m.mua[i] + (gain - 1) * m.along[i, 1:] + kept[first + np.unique(idx)]
        top = max(top, float(mut.max()))
    return top


def test_majorant_covers_the_largest_lens(tables) -> None:
    _, host, _, plain, m = tables
    kept_plain, kept = plain.kept()[0].astype(np.float32), m.kept()[0].astype(np.float32)
    d = host.drift_size
    dev = DeviceShift(struct=None, arrays={}, materials=m, max_gain=host.max_gain,
                      drift_size=d)  # fmt: skip
    assert host.max_gain == 2.5
    assert 0 < d < 0.5
    assert m.along is not None
    takes = m.along[:, 0] >= 0
    first, count = m.tr_rows
    mb = majorant(m, kept, dev)
    assert mb >= reached(m, kept, host.max_gain, d)  # covers every sample the kernel draws
    # ADR 0016 s.11: tighter than the whole-table bound (drift_size 1), which it equals
    # when the drift can reach every row
    loose = dataclasses.replace(dev, drift_size=1.0)
    whole = (m.mua + 1.5 * m.along[:, 1:])[takes].max(0) + kept[first : first + count].max(0)
    assert majorant(m, kept, loose) >= float(whole.max())
    assert mb < majorant(m, kept, loose)
    assert majorant(m, kept, loose) == pytest.approx(float(whole.max()) * (1 + 1e-5), rel=1e-6)
    # negative control: a bound that ignores the drift (size 0) misses samples it reaches
    assert majorant(m, kept, dataclasses.replace(dev, drift_size=0.0)) < reached(
        m, kept, host.max_gain, d
    )
    assert mb > majorant(plain, kept_plain, None)
    # nothing takes the shift: exactly the plain majorant, so the plain image
    none = dataclasses.replace(
        m, along=np.where(takes[:, None], [-1, 0, 0, 0], m.along).astype(np.float32)
    )
    assert majorant(none, kept, dev) == majorant(plain, kept_plain, None)
