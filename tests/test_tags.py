"""Library tag records (ADR 0004, section 8): what is carried over, what is
imported from agatesim's cache, and that nothing is left as a gap."""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
from agatesim.library import Label
from agatesim.shapes import VERSION
from numpy.typing import NDArray

from nodule import rims, tags
from nodule.bands import COUPLET_SPACINGS, Bands, Rung
from nodule.concentric import Concentric
from nodule.curvature import Curvature
from nodule.defects import Defects
from nodule.orientation import Order
from nodule.outline import Outline

LABEL = Label(verdict="forms", forms=["fortification"], features=["quartz core"])
SHAPE = Outline(aspect=0.8, regularity=0.97, circularity=0.9, irregularity=0.1, concavities=1)
BAND = Bands(
    14.0,
    12.0,
    15.0,
    0.0714,
    None,
    20,
    300,
    limit_px=14.67,
    limit_share=0.0544,
    blank=0.0,
    ladder=(Rung(14.67, 14.0, 12.0, 15.0, 0.0714), Rung(42.72, 0.0, 0.0, 0.0, 0.0)),
    continuity=0.91234,
    continuity_near=0.95,
)
FACE = tags.Face(
    Concentric(sets=1, depth=9, drift=0.04321, complete=1.0, wedges=0),
    Order(0.8, 0.7, 6.5, False, 0.9, banded=0.95),
    Curvature(0.21, 0.6, 2, banded=0.95),
    Defects(0.012, 3, 1),
)
YES = tags.rim_tags(None, rims.Verdict(True, True, "helper", "held"))
NOISE = 1.5  # grey levels of the drawn faces' pixel noise, as the gate would report


def drawn(
    kind: str, n: int = 400, period: float = 24.0
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """A round face n px across, concentric rings of couplet `period` px or
    even ("even"), with pixel noise; the mask at half size, as a working-size
    outline is."""
    y, x = np.mgrid[:n, :n].astype(np.float64)
    r = np.hypot(y - (n - 1) / 2, x - (n - 1) / 2)
    lum = 128 + 40 * np.cos(2 * np.pi * r / period) if kind == "rings" else np.full((n, n), 128.0)
    lum = lum + np.random.default_rng(1).normal(0, NOISE, lum.shape)
    return lum.astype(np.float32), (r < 0.45 * n)[::2, ::2]


def entry(level: str = "fit", size: int = 1234, version: int = VERSION) -> dict[str, object]:
    return {
        "key": f"{version}:{size}:maps",
        "level": level,
        "spread": 0.61234,
        "trend": -1.0,
        "ratio": 0.9,
        "hurst": None,
        "core": 0,
    }


def test_imported_statistics_are_read_when_current_and_gated() -> None:
    got = tags.imported(entry(), 1234)
    assert got == {
        "C7_spread": 0.6123,
        "C8_trend": -1.0,
        "C9_ratio": 0.9,
        "C11_hurst": tags.UNKNOWN,  # agatesim could not judge it
        "C17_core": 0.0,
    }


def test_a_stale_or_ungated_cache_entry_gives_unknown() -> None:
    for stale in (tags.imported(entry(size=999), 1234), tags.imported(entry(version=0), 1234)):
        assert set(stale.values()) == {tags.UNKNOWN}
    assert set(tags.imported(entry(level="not fit"), 1234).values()) == {tags.UNKNOWN}
    assert set(tags.imported(None, 1234).values()) == {tags.UNKNOWN}


def test_a_record_has_every_field_and_no_gap_when_nothing_is_measured() -> None:
    full = tags.record(LABEL, SHAPE, entry(), 1234, BAND, YES, FACE)
    empty = tags.record(LABEL, None, None, 1234)
    assert full.keys() == empty.keys()
    assert full["forms"] == ["fortification"]
    assert full["C1_aspect"] == 0.8
    assert full["C6_count"] == 14.0
    assert full["C6_profiles"] == 20
    assert full["C6_L_px"] == 14.67
    assert full["N_L_px"] == [14.67, 42.72]
    assert full["N_count"] == [14.0, 0.0]
    assert full["C10_persistence"] == tags.UNKNOWN  # too few widths to judge
    assert full["C12_corr"] == 0.9123
    assert full["C13_sets"] == 1
    assert full["C13_drift"] == 0.0432
    assert full["C14_ends"] == 3
    assert full["C15_corners"] == 2
    assert full["C16_corr_censored"] is False
    assert full["C16_banded"] == 0.95
    names = {*tags.FROM_CONTINUITY, *tags.FROM_RINGS, *tags.FROM_ORDER}
    assert names | {*tags.FROM_CURVE, *tags.FROM_DEFECTS} <= full.keys()
    assert all(empty[k] == tags.UNKNOWN for k in empty if k.startswith("C"))
    assert empty["view_kind"] == empty["finish"] == empty["lighting"] == tags.UNKNOWN
    assert empty["scale"] == "none"
    assert empty["rim"] == empty["outline_check"] == tags.UNKNOWN


def test_outline_bound_values_need_a_yes_on_that_outline() -> None:
    """Negative controls (agatesim ADR 0075): an unjudged outline, one judged
    "no", or none at all keeps C1-C3, C6, C12 and C13 unknown; band shape and
    C14-C16 still count."""
    no = rims.Verdict(True, False, "owner", "held")
    cases = {
        "unjudged": tags.rim_tags(None, None),
        "judged no": tags.rim_tags(None, no),
        "declined": tags.rim_tags("cut-off", None),
    }
    for case, rim in cases.items():
        rec = tags.record(LABEL, SHAPE, entry(), 1234, BAND, rim, FACE)
        assert all(rec[k] == tags.UNKNOWN for k in tags.OUTLINE_BOUND), case
        assert rec["C12_corr"] == rec["C13_sets"] == tags.UNKNOWN, case
        assert rec["C7_spread"] == 0.6123, case
        assert rec["C14_density"] == 0.012, case
        assert rec["C15_curv"] == 0.21, case
        assert rec["C16_coherence"] == 0.8, case
    assert cases["judged no"]["outline_check"] == "no"
    assert cases["judged no"]["outline_by"] == "owner"
    assert cases["declined"]["rim"] == "none: cut-off"
    assert YES["outline_check"] == "yes"


def test_a_ringed_face_measured_with_a_working_size_mask_is_one_set() -> None:
    """`face_measure` takes the mask at working size, as `measure_photo` hands
    it over: rings drawn at full size read as one complete concentric set."""
    lum, mask = drawn("rings")
    got = tags.face_values(None, tags.face_measure(lum, mask, 0.5, NOISE, 24.0 / COUPLET_SPACINGS))
    assert got["C13_sets"] == 1
    assert isinstance(got["C13_complete"], float)
    assert got["C13_complete"] > 0.9
    assert isinstance(got["C16_coherence"], float)
    assert got["C16_coherence"] > 0.8
    assert got["C12_corr"] == tags.UNKNOWN  # no band given: C12 comes with C6


def test_negative_control_a_bandless_face_gives_no_set_and_no_defect_density() -> None:
    """An even face with pixel noise has no band edges: no concentric set and
    C14 "unknown", not a density read off noise."""
    lum, mask = drawn("even")
    got = tags.face_values(None, tags.face_measure(lum, mask, 0.5, NOISE, 12.0))
    assert got["C13_sets"] in (0, tags.UNKNOWN)
    assert got["C14_density"] == tags.UNKNOWN


def test_the_tag_file_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "tags.toml"
    tags.write({"b-photo": tags.record(LABEL, SHAPE, entry(), 1234, rim=YES)}, path)
    doc = tomllib.loads(path.read_text("utf-8"))
    assert doc["meta"]["sampling"] == tags.SAMPLING
    assert doc["meta"]["rim_outlines_used"] == 1
    assert "2 of 20" in doc["meta"]["rim_outlines_wrong"]
    assert doc["images"]["b-photo"]["C2_circularity"] == 0.9
