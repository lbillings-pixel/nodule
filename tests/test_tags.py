"""Library tag records (ADR 0004, section 8): what is carried over, what is
imported from agatesim's cache, and that nothing is left as a gap."""

from __future__ import annotations

import tomllib
from pathlib import Path

from agatesim.library import Label
from agatesim.shapes import VERSION

from nodule import tags
from nodule.outline import Outline

LABEL = Label(verdict="forms", forms=["fortification"], features=["quartz core"])
SHAPE = Outline(aspect=0.8, regularity=0.97, circularity=0.9, irregularity=0.1, concavities=1)


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
    full = tags.record(LABEL, SHAPE, entry(), 1234)
    empty = tags.record(LABEL, None, None, 1234)
    assert full.keys() == empty.keys()
    assert full["forms"] == ["fortification"]
    assert full["C1_aspect"] == 0.8
    assert all(empty[k] == tags.UNKNOWN for k in empty if k.startswith("C"))
    assert empty["view_kind"] == empty["finish"] == empty["lighting"] == tags.UNKNOWN
    assert empty["scale"] == "none"


def test_the_tag_file_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "tags.toml"
    tags.write({"b-photo": tags.record(LABEL, SHAPE, entry(), 1234)}, path)
    doc = tomllib.loads(path.read_text("utf-8"))
    assert doc["meta"]["sampling"] == tags.SAMPLING
    assert doc["images"]["b-photo"]["C2_circularity"] == 0.9
