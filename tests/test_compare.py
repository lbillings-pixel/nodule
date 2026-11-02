"""Set-level compare (ADR 0008 sections 3-4), on synthetic descriptor tables,
with negative and null controls."""

from __future__ import annotations

import numpy as np
import pytest

from nodule.compare import (
    BLOCKS,
    COMPARED,
    block_presence,
    block_scores,
    blocks,
    crossing_shift,
    crossing_shifts,
    distances,
    excluded,
    held_out_rate,
    knn_scores,
    library_table,
    per_descriptor,
    power_by_cuts,
    power_by_size,
    rejection_rate,
    robust_scale,
    spread_test,
    table,
)

D = len(COMPARED)


PARTS = blocks()


def _faces(n: int, rng: np.random.Generator, shift: float = 0.0, missing: float = 0.4):
    """Faces drawn from one population, each block missing with chance
    `missing`, as where a gate fails (whole blocks, as in the library)."""
    x = rng.normal(shift, 1.0, size=(n, D))
    for _, cols in PARTS:
        x[np.ix_(rng.random(n) < missing, cols)] = np.nan
    return x


def test_compared_values_are_the_25_of_adr_0006() -> None:
    assert D == 25
    assert "C6_count" not in COMPARED
    assert "C17_core" not in COMPARED
    assert "C16_corr_censored" not in COMPARED


def test_table_reads_unknowns_and_drops_empty_rows() -> None:
    rows = [{"C7_spread": 0.5, "C13_sets": 2, "C8_trend": "unknown"}, {"C6_count": 3.0}]
    t = table(rows, ["a", "b"])
    assert t.ids == ("a",)
    j = COMPARED.index("C13_sets")
    assert t.values[0, j] == 2.0
    assert np.isnan(t.values[0, COMPARED.index("C8_trend")])


def test_distance_uses_shared_columns_only() -> None:
    a = np.array([[0.0, 1.0, np.nan]])
    b = np.array([[2.0, np.nan, 5.0], [np.nan, np.nan, 1.0]])
    d = distances(a, b)
    assert d[0, 0] == pytest.approx(4.0)  # only column 0 shared
    assert np.isnan(d[0, 1])  # nothing shared


def test_unscalable_descriptor_is_left_out() -> None:
    p = np.array([[1.0, 3.0], [1.0, 5.0], [1.0, 7.0]])
    _centre, width = robust_scale(p)
    assert np.isnan(width[0])
    assert width[1] == pytest.approx(2.0)


def test_knn_needs_k_neighbours() -> None:
    d = np.array([[1.0, 2.0, np.nan], [np.nan, np.nan, 3.0]])
    s = knn_scores(d, k=2)
    assert s[0] == pytest.approx(1.5)
    assert np.isnan(s[1])


def test_same_population_lies_within_and_shifted_is_rejected() -> None:
    rng = np.random.default_rng(1)
    photos = _faces(100, rng)
    same = _faces(100, rng)
    fake = _faces(100, rng, shift=2.0)  # negative control
    assert spread_test(same, photos).within
    assert not spread_test(fake, photos).within
    g1, g2 = np.random.default_rng(2).spawn(2)
    assert rejection_rate(fake, photos, g1, resamples=50) >= 0.8
    assert rejection_rate(same, photos, g2, resamples=50) <= 0.2


def test_bootstrap_duplicates_do_not_score_against_themselves() -> None:
    rng = np.random.default_rng(3)
    photos = _faces(60, rng)
    doubled = np.concatenate([photos, photos])
    origin = np.concatenate([np.arange(60), np.arange(60)])
    cols = PARTS[0][1]
    same = origin[:, None] == origin[None, :]
    honest = block_scores(doubled[:, cols], doubled[:, cols], same, k=1)
    leaky = block_scores(doubled[:, cols], doubled[:, cols], np.eye(120, dtype=bool), k=1)
    has = np.isfinite(honest)
    assert np.all(leaky[has] < 1e-9)  # each copy finds its twin at 0
    assert np.all(honest[has] > 1e-6)


def test_null_n1_held_out_photos_fail_about_five_percent() -> None:
    rate = held_out_rate(_faces(200, np.random.default_rng(4)))
    assert 0.01 <= rate <= 0.11


def test_rates_settle_on_the_verdict_line_as_photos_grow() -> None:
    """The verdict is a line (median cut vs the photos' 95th percentile), so
    more photos make the rate settle, toward 1 for a fake beyond the line and
    0 for one inside it; at few photos the noisy threshold adds spurious
    fails (ADR 0008 amendment 1)."""
    rng = np.random.default_rng(5)
    photos = _faces(120, rng)
    far = _faces(80, rng, shift=2.0)
    near = _faces(80, rng, shift=0.6)  # inside the line
    g1, g2 = np.random.default_rng(6).spawn(2)
    p_far = power_by_size(far, photos, g1, sizes=(15, 60), resamples=40)
    p_near = power_by_size(near, photos, g2, sizes=(15, 60), resamples=40)
    assert list(p_far) == [15, 60, 120]
    assert p_far[120] >= 0.8
    assert p_near[120] <= 0.05 < p_near[15]


def test_power_by_cuts_rejects_a_far_fake_with_few_cuts_but_not_a_near_one() -> None:
    """The fakes' pilot (ADR 0008 amendment 6): a fake well beyond the line is
    rejected already at 10 cuts; the negative control, one inside the line,
    is not rejected once all its cuts are drawn."""
    rng = np.random.default_rng(11)
    photos = _faces(120, rng)
    far = _faces(40, rng, shift=2.0)
    near = _faces(40, rng, shift=0.6)
    g1, g2 = np.random.default_rng(12).spawn(2)
    p_far = power_by_cuts(far, photos, g1, sizes=(10, 20, 60), resamples=40)
    p_near = power_by_cuts(near, photos, g2, sizes=(10, 20), resamples=40)
    assert list(p_far) == [10, 20, 40]  # 60 is above the cuts' count
    assert min(p_far.values()) >= 0.8
    assert p_near[40] <= 0.1


def test_per_descriptor_marks_the_shifted_value() -> None:
    rng = np.random.default_rng(7)
    photos = _faces(100, rng, missing=0.0)
    cuts = _faces(100, rng, missing=0.0)
    cuts[:, 3] += 4.0
    singles = per_descriptor(cuts, photos)
    assert singles[3].percentile > 99
    assert singles[3].outside_share > 0.9
    others = [s.outside_share for i, s in enumerate(singles) if i != 3]
    assert max(others) < 0.2


def test_library_strata_load() -> None:
    reflected = library_table("reflected")
    assert 90 <= len(reflected) <= 172
    polished = library_table("reflected", finishes=("polished",))
    assert 0 < len(polished) < len(reflected)


def test_crossing_shift_finds_the_line() -> None:
    rng = np.random.default_rng(8)
    photos = _faces(100, rng)
    cuts = _faces(100, rng)
    shift = crossing_shift(cuts, photos)
    assert 0.0 < shift < 5.0
    _, width = robust_scale(photos)
    step = np.where(np.isfinite(width), width, 0.0)
    assert not spread_test(cuts + shift * step, photos).within
    assert spread_test(cuts + 0.95 * shift * step, photos).within


def test_crossing_shift_controls() -> None:
    """Negative controls: cuts already outside cross at 0; a descriptor the
    cuts lack can never move them (the compare is blind there)."""
    rng = np.random.default_rng(9)
    photos = _faces(80, rng)
    far = _faces(60, rng, shift=2.0)
    assert crossing_shift(far, photos) == 0.0
    cuts = _faces(60, rng)
    cuts[:, 0] = np.nan
    assert crossing_shift(cuts, photos, column=0) == float("inf")


def test_one_descriptor_needs_a_larger_shift_than_all() -> None:
    rng = np.random.default_rng(10)
    photos = _faces(80, rng, missing=0.0)
    cuts = _faces(60, rng, missing=0.0)
    shifts = crossing_shifts(cuts, photos)
    assert set(shifts) == {*COMPARED, "all"}
    together = min(shifts["all"])
    assert all(min(shifts[n]) > together for n in COMPARED)


def test_an_odd_value_cannot_hide_beside_photos_that_lack_it() -> None:
    """ADR 0008 amendment 2. Most photos lack the rings block; cuts far off
    there alone must still fail, and the crossing shift is finite. Before the
    blocks, such cuts scored beside the photos without rings and passed."""
    rng = np.random.default_rng(11)
    photos = _faces(100, rng, missing=0.1)
    rings = dict(PARTS)["rings"]
    photos[np.ix_(rng.random(100) < 0.7, rings)] = np.nan
    cuts = _faces(60, rng, missing=0.0)
    assert spread_test(cuts, photos).within
    j = COMPARED.index("C13_depth")
    assert crossing_shift(cuts, photos, column=j) < 10.0
    odd = cuts.copy()
    odd[:, rings] += 5.0
    assert not spread_test(odd, photos).within


def test_more_blocks_are_not_less_typical_by_chance() -> None:
    """Cuts with every block, against photos missing many, from the same
    population: the share outside stays near 5 %."""
    rng = np.random.default_rng(12)
    photos = _faces(150, rng, missing=0.5)
    cuts = _faces(150, rng, missing=0.0)
    s = spread_test(cuts, photos)
    assert s.within
    assert s.outside_share <= 0.15


def test_block_presence_counts_rows_without_values() -> None:
    rows = [{"C7_spread": 0.5}, {"C16_banded": 0.9, "C7_spread": 0.6}, {"C6_count": 3.0}]
    p = block_presence(table(rows))
    assert set(p) == set(BLOCKS)
    assert p["shape"] == pytest.approx(2 / 3)
    assert p["image"] == pytest.approx(1 / 3)


def test_owner_exclusions_are_left_out() -> None:
    """ADR 0008 amendment 3: the six listed photos leave the reflected
    stratum; every listed id is a real reflected cut face."""
    out = excluded()
    assert len(out) == 6
    kept = library_table("reflected")
    every = library_table("reflected", leave_out=frozenset())
    assert out <= set(every.ids)
    assert not out & set(kept.ids)
    assert len(every) - len(kept) == 6
