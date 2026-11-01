"""nodule.render.fakes: the obvious fakes of ADR 0008 s.4 (CPU; these run in CI)."""

# pyright: basic

from __future__ import annotations

import numpy as np
import pytest
from agatesim.fields import Phase, Pigment
from agatesim.volume.config import SliceConfig

from nodule.render import fakes, tgrid
from nodule.render.record import STEPS, BandRecord
from nodule.specimen import Specimen, load_preset


def record(widths, translucency, phase=None, impurity=None) -> BandRecord:
    """A record of runs: run i is `widths[i]` steps of its values; the rest of
    STEPS is a final open run (translucency 0.5)."""
    n = len(widths)
    ph = [Phase.CHALCEDONY] * n if phase is None else phase
    imp = [0.1 * (i % 3) for i in range(n)] if impurity is None else impurity
    rep = np.repeat(np.arange(n), widths)
    pad = STEPS - len(rep)
    assert pad > 0
    return BandRecord(
        seen=np.ones(STEPS, bool),
        phase=np.r_[np.asarray(ph)[rep], [Phase.CHALCEDONY] * pad].astype(np.uint8),
        pigment=np.full(STEPS, Pigment.IRON, np.uint8),
        impurity=np.r_[np.asarray(imp)[rep], [0.0] * pad].astype(np.float32),
        translucency=np.r_[np.asarray(translucency)[rep], [0.5] * pad].astype(np.float32),
    )


def alternating(n: int) -> list[float]:
    return [0.8 if i % 2 == 0 else 0.2 for i in range(n)]


# --- runs ---------------------------------------------------------------------------------


def test_runs_fix_the_last_run_and_quartz() -> None:
    w = [3, 5, 2, 6, 4]
    ph = [Phase.CHALCEDONY, Phase.CHALCEDONY, Phase.QUARTZ, Phase.CHALCEDONY, Phase.CHALCEDONY]
    r = fakes.runs(record(w, alternating(5), phase=ph))
    assert r.width.tolist() == [*w, STEPS - sum(w)]
    assert r.movable.tolist() == [True, True, False, True, True, False]
    assert [s.tolist() for s in r.segments()] == [[0, 1], [3, 4]]


def test_runs_fill_unseen_steps_from_their_neighbours() -> None:
    rec = record([4, 4], [0.8, 0.2])
    holes = rec.seen.copy()
    holes[[1, 5]] = False
    gappy = BandRecord(holes, rec.phase, rec.pigment, rec.impurity, rec.translucency)
    assert fakes.runs(gappy).width.tolist() == fakes.runs(rec).width.tolist()


# --- F2 and F3: widths --------------------------------------------------------------------


def test_remapped_trades_thick_and_thin_and_keeps_the_order() -> None:
    w = [2, 4, 6, 8, 3]
    rec = record(w, alternating(5))
    r = fakes.runs(fakes.remapped(rec))
    assert r.width[:5].tolist() == [8, 4, 3, 2, 6]
    d0, d1 = fakes.dense(rec), fakes.dense(fakes.remapped(rec))
    assert np.allclose(d1["translucency"][r.start[:5]], alternating(5))
    assert np.allclose(d1["impurity"][r.start[:5]], d0["impurity"][fakes.runs(rec).start[:5]])
    assert np.array_equal(d1["translucency"][sum(w) :], d0["translucency"][sum(w) :])


def test_remapped_leaves_equal_widths_alone() -> None:
    # Control: with nothing thick or thin, there is nothing to trade.
    rec = record([5] * 6, alternating(6))
    assert np.array_equal(fakes.remapped(rec).translucency, rec.translucency)


def test_regular_makes_widths_equal_to_the_step() -> None:
    w = [2, 9, 3, 7, 4, 6]
    r = fakes.runs(fakes.regular(record(w, alternating(6))))
    assert r.width[:6].sum() == sum(w)
    assert r.width[:6].max() - r.width[:6].min() <= 1


def test_regular_leaves_a_regular_record_alone() -> None:
    # Control: already regular.
    rec = record([4] * 8, alternating(8))
    assert np.array_equal(fakes.regular(rec).translucency, rec.translucency)


def test_widths_stay_inside_their_segment() -> None:
    w = [2, 8, 5, 3, 9]
    ph = [Phase.CHALCEDONY, Phase.CHALCEDONY, Phase.QUARTZ, Phase.CHALCEDONY, Phase.CHALCEDONY]
    rec = record(w, alternating(5), phase=ph)
    for fake in (fakes.remapped, fakes.regular):
        out = fake(rec)
        assert np.array_equal(out.phase, rec.phase)  # the quartz run has not moved
        r = fakes.runs(out)
        assert r.width[0] + r.width[1] == 10
        assert r.width[3] + r.width[4] == 12


# --- F1: smoothing ------------------------------------------------------------------------


def test_smoothed_removes_a_sub_band_and_keeps_the_bands() -> None:
    # Bands 8 steps wide, one with a 1-step sub-band inside.
    w = [8, 3, 1, 4, 8, 8, 8]
    tr = [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]  # runs 1-3 are one dark band with a bright line
    rec = record(w, tr)
    out = fakes.dense(fakes.smoothed(rec))["translucency"]
    sub = out[11] - 0.5 * (out[10] + out[12])
    assert sub < 0.25 * 0.8  # the sub-band keeps under a quarter of its contrast
    band = out[4] - out[28]  # the middle of a bright and of a dark 8-step band
    assert band > 0.5 * 0.8


def test_smoothed_spares_wide_bands_when_the_record_has_no_fine_structure() -> None:
    # Control: equal bands and nothing finer. sigma = 24 / 2 pi = 3.8 steps, so
    # each edge leaks 0.5 erfc(6 / 3.8 sqrt 2) ~ 6 % into a middle 6 steps
    # away: two inner neighbours keep ~77 % of their contrast. Sharp edges go.
    rec = record([12] * 6, alternating(6))
    out = fakes.dense(fakes.smoothed(rec))["translucency"]
    assert out[30] - out[18] > 0.7 * 0.6
    assert 0.3 < out[12] < 0.7  # an edge is now a ramp


def test_smoothed_keeps_categories_and_fixed_runs() -> None:
    w = [3, 5, 4, 6]
    ph = [Phase.CHALCEDONY, Phase.CHALCEDONY, Phase.QUARTZ, Phase.CHALCEDONY]
    rec = record(w, alternating(4), phase=ph)
    out = fakes.smoothed(rec)
    assert np.array_equal(out.phase, rec.phase)
    assert np.array_equal(out.pigment, rec.pigment)
    assert np.array_equal(out.translucency[8:12], rec.translucency[8:12])  # quartz
    assert np.array_equal(out.translucency[18:], rec.translucency[18:])  # open last run
    assert np.array_equal(out.translucency[12:18], rec.translucency[12:18])  # lone run


# --- the stone ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stone() -> tuple[Specimen, BandRecord]:
    sp = Specimen.build(load_preset("n2-slice"))
    return sp, fakes.stone_record(sp, np.random.default_rng(3), points=200_000)  # ~0.3 s


def test_stone_record_agrees_with_a_cut_and_covers_it(stone) -> None:
    sp, rec = stone
    grid = tgrid.GrowthGrid.fill(
        sp, sp.plane(SliceConfig(tilt_deg=90.0, azimuth_deg=270.0, offset_mm=0.0)), 0.5, 2.0
    )
    assert grid.bands is not None
    rec.merge(grid.bands)  # raises if they disagree at a step
    cut_runs = fakes.runs(grid.bands).width
    assert len(fakes.runs(rec).width) >= len(cut_runs) > 20


def test_fakes_of_the_stone_are_complete_records(stone) -> None:
    _, rec = stone
    outs = [fakes.smoothed(rec), fakes.remapped(rec), fakes.regular(rec)]
    outs.append(fakes.phase_randomised(rec, np.random.default_rng(8)))
    for out in outs:
        assert out.seen.all()
        assert ((out.translucency >= 0) & (out.translucency <= 1)).all()


# --- F4: phase randomisation (diagnostic) -------------------------------------------------


def varied(seed: int, n: int = 30) -> BandRecord:
    g = np.random.default_rng(seed)
    tr = np.where(np.arange(n) % 2 == 0, g.uniform(0.6, 0.9, n), g.uniform(0.1, 0.4, n))
    return record(g.integers(3, 9, n), tr, impurity=1.0 - tr)  # impurity tied to translucency


def segment(rec: BandRecord, name: str) -> np.ndarray:
    r = fakes.runs(rec)
    seg = r.segments()[0]
    return fakes.dense(rec)[name][r.start[seg[0]] : r.start[seg[-1]] + r.width[seg[-1]]]


def spectrum_gap(x: np.ndarray, y: np.ndarray) -> float:
    a, b = np.abs(np.fft.rfft(x))[1:], np.abs(np.fft.rfft(y))[1:]
    return float(np.sqrt(((a - b) ** 2).sum() / (a**2).sum()))


def test_phase_randomised_keeps_values_and_spectrum_and_loses_the_runs() -> None:
    rec = varied(0)
    out = fakes.phase_randomised(rec, np.random.default_rng(1))
    x, y = segment(rec, "translucency"), segment(out, "translucency")
    assert np.array_equal(np.sort(x), np.sort(y))  # exactly its own values
    shuffled = np.random.default_rng(2).permutation(x)  # control: values kept, spectrum lost
    assert spectrum_gap(x, y) < 0.25
    assert spectrum_gap(x, y) < 0.5 * spectrum_gap(x, shuffled)
    assert (np.diff(y) != 0).sum() > 2 * (np.diff(x) != 0).sum()  # runs and edges gone
    assert np.array_equal(out.translucency[len(x) :], rec.translucency[len(x) :])


def test_phase_randomised_shares_phases_between_fields() -> None:
    # impurity = 1 - translucency; one phase shift per frequency keeps the tie.
    out = fakes.phase_randomised(varied(3), np.random.default_rng(4))
    corr = np.corrcoef(segment(out, "translucency"), segment(out, "impurity"))[0, 1]
    assert corr < -0.8


def test_phase_randomised_follows_its_rng() -> None:
    rec = varied(5)
    a, b, c = (fakes.phase_randomised(rec, np.random.default_rng(s)) for s in (6, 6, 7))
    assert np.array_equal(a.translucency, b.translucency)
    assert not np.array_equal(a.translucency, c.translucency)


def test_end_jump_is_zero_for_a_periodic_series_and_large_for_a_ramp() -> None:
    assert fakes.end_jump(np.array([0.2, 0.8, 0.2, 0.8, 0.2])) == 0.0
    assert fakes.end_jump(np.linspace(0, 1, 20)) > 0.5
