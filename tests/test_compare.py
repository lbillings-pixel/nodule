"""Set-level compare (ADR 0008 sections 3-4), on synthetic descriptor tables,
with negative and null controls."""

from __future__ import annotations

import numpy as np
import pytest

from nodule.compare import (
    COMPARED,
    crossing_shift,
    crossing_shifts,
    distances,
    held_out_rate,
    knn_scores,
    library_table,
    per_descriptor,
    power_by_size,
    rejection_rate,
    robust_scale,
    spread_test,
    table,
)

D = len(COMPARED)


def _faces(n: int, rng: np.random.Generator, shift: float = 0.0, missing: float = 0.4):
    """Faces drawn from one population, with a share of values missing as
    where a gate fails."""
    x = rng.normal(shift, 1.0, size=(n, D))
    x[rng.random((n, D)) < missing] = np.nan
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
    fake = _faces(100, rng, shift=1.5)  # negative control
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
    honest = spread_test(photos[:1], doubled, origin=origin)
    leaky = spread_test(photos[:1], doubled)  # each copy finds its twin at 0
    assert np.nanmin(leaky.photo_scores) < np.nanmin(honest.photo_scores)


def test_null_n1_held_out_photos_fail_about_five_percent() -> None:
    rate = held_out_rate(_faces(200, np.random.default_rng(4)))
    assert 0.01 <= rate <= 0.11


def test_rates_settle_on_the_verdict_line_as_photos_grow() -> None:
    """The verdict is a line (median cut vs the photos' 95th percentile), so
    more photos make the rate settle, toward 1 for a fake beyond the line and
    0 for one inside it; at few photos the noisy threshold adds spurious
    fails (2026-11-01 finding, ADR 0008 section 4)."""
    rng = np.random.default_rng(5)
    photos = _faces(120, rng)
    far = _faces(80, rng, shift=1.2)
    near = _faces(80, rng, shift=0.6)  # inside the line
    g1, g2 = np.random.default_rng(6).spawn(2)
    p_far = power_by_size(far, photos, g1, sizes=(15, 60), resamples=40)
    p_near = power_by_size(near, photos, g2, sizes=(15, 60), resamples=40)
    assert list(p_far) == [15, 60, 120]
    assert p_far[120] >= 0.8
    assert p_near[120] <= 0.05 < p_near[15]


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
