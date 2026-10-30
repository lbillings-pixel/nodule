"""C23-C25 labels, their control faces and the owner's check (ADR 0006, sections 3 and 4)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from nodule import labels, zones
from nodule._scipy import hole_free_patches
from nodule.check import ThreadingHTTPServer, handler
from nodule.zonecheck import ZoneChecker

ABSENT = dict.fromkeys(zones.CUE_FIELDS, "absent")
BANDED: dict[str, Any] = {
    "banded": "yes",
    "C23_centres": 1,
    **ABSENT,
    "C25_zones": [{"kind": "even bands", "share": 0.7}, {"kind": "coarse quartz", "share": 0.3}],
}
PROPOSED = BANDED | {"confidence": "medium", "note": "one set round a quartz core"}
NOT_BANDED: dict[str, Any] = {
    "banded": "no",
    "C23_centres": "unknown",
    **dict.fromkeys(zones.CUE_FIELDS, "unknown"),
    "C25_zones": [],
}


def test_valid_labels_have_no_problems() -> None:
    assert zones.problems(PROPOSED) == []
    assert zones.problems(BANDED | {"blind": True}, checked=True) == []
    assert zones.problems(NOT_BANDED | {"confidence": "high"}) == []
    no_zones: list[dict[str, Any]] = []  # unknown zones
    assert zones.problems(BANDED | {"C25_zones": no_zones}, checked=True) == []


@pytest.mark.parametrize(
    ("change", "says"),
    [
        ({"banded": "maybe"}, "banded"),
        ({"C23_centres": -1}, "C23_centres"),
        ({"C23_centres": 1.5}, "C23_centres"),
        ({"C24_outer_iris": "yes"}, "C24_outer_iris"),
        ({"C25_zones": [{"kind": "agate", "share": 1.0}]}, "not one of"),
        ({"C25_zones": [{"kind": "even bands", "share": 0.6}]}, "sum to 0.60"),
        ({"C25_zones": [{"kind": "cavity", "share": 0.5}] * 2}, "same kind"),
        ({"C25_zones": [{"kind": "cavity"}]}, "needs a kind and a share"),
        ({"C6": 3}, "not a label"),
    ],
)
def test_invalid_labels_are_refused(change: dict[str, Any], says: str) -> None:
    found = zones.problems(PROPOSED | change)
    assert any(says in p for p in found), found


def test_a_face_that_is_not_banded_leaves_c23_to_c25_unknown() -> None:
    assert zones.problems(NOT_BANDED | {"C23_centres": 1}, checked=True)
    assert zones.problems(NOT_BANDED | {"C24_inner_gradient": "present"}, checked=True)


def test_each_side_gives_only_its_own_fields() -> None:
    assert zones.problems(BANDED | {"confidence": "high"}, checked=True)
    assert zones.problems(PROPOSED | {"blind": False})


def test_items_score_each_cue_and_zone_position() -> None:
    same = zones.items(BANDED, BANDED)
    assert all(a for _, a in same)
    assert [k for k, _ in same].count("C24") == 12
    assert [k for k, _ in same].count("C25") == 2
    one_zone = BANDED | {"C25_zones": [{"kind": "even bands", "share": 1.0}]}
    c25 = [a for k, a in zones.items(BANDED, one_zone) if k == "C25"]
    assert c25 == [True, False]  # the missing quartz zone disagrees
    assert zones.items(BANDED, NOT_BANDED) == [("banded", False)]
    assert zones.items(NOT_BANDED, NOT_BANDED) == [("banded", True)]


def test_the_report_flags_the_fallback_on_the_blind_faces() -> None:
    wrong = NOT_BANDED | {"blind": True}
    images = {f"b{i}": {"proposed": PROPOSED, "checked": wrong} for i in range(10)}
    images |= {f"s{i}": {"proposed": PROPOSED, "checked": BANDED} for i in range(20)}
    text = zones.report(images)
    assert "blind  0/10 items" in text
    assert "FALLBACK: more than a third of the blind whole items" in text
    assert "of the all whole items" not in text  # 320 of 330 agree overall


def test_one_part_failing_trips_the_fallback_though_the_whole_agrees() -> None:
    other = BANDED | {"C23_centres": 2}  # C23 wrong on every face, all else right
    images = {f"s{i}": {"proposed": PROPOSED, "checked": other} for i in range(30)}
    text = zones.report(images)
    assert "FALLBACK: more than a third of the all C23 items" in text
    assert "whole items" not in text  # 1 of 16 items per face disagree
    assert "C24 items" not in text


def test_blind_and_recount_faces_are_fixed_and_separate_streams() -> None:
    sample = [f"p{i:02d}" for i in range(30)]
    blind = zones.blind_sample(sample)
    assert len(blind) == zones.BLIND
    assert set(blind) <= set(sample)
    assert blind == zones.blind_sample(list(reversed(sample)))
    assert blind != labels.check_sample(sample, zones.BLIND)  # not the check sample's stream
    views: dict[str, dict[str, labels.Row]] = {
        k: {"checked": {"C6_eye": 5 if i % 3 else "unknown"}} for i, k in enumerate(sample)
    }
    recount = zones.recount_sample(views)
    assert len(recount) == zones.RECOUNT
    assert all(isinstance(views[k]["checked"]["C6_eye"], int) for k in recount)


def test_the_labels_file_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "zones.toml"
    assert zones.add_proposals({"a": PROPOSED}, path) == []
    assert zones.load(path) == {"a": {"proposed": PROPOSED}}
    assert zones.add_proposals({"b": PROPOSED | {"banded": "?"}}, path)
    assert "b" not in zones.load(path)  # nothing written when any is wrong


def centres(picture: np.ndarray) -> int:
    """Band centres of a control: milky bands are rings, except at a centre,
    where the innermost one is a disc, the only milky patch with no hole."""
    grey = picture.astype(float).mean(axis=2)
    return hole_free_patches(grey > (zones.MILKY.mean() + zones.CLEAR.mean()) / 2)


def test_the_controls_have_the_centres_they_claim() -> None:
    assert centres(zones.render_control("one-set", 256)) == 1
    assert centres(zones.render_control("two-sets", 256)) == 2


def test_the_controls_are_resolution_independent() -> None:
    for name in zones.CONTROLS:
        fine = zones.render_control(name, 512).astype(float)
        coarse = zones.render_control(name, 128).astype(float)
        binned = fine.reshape(128, 4, 128, 4, 3).mean(axis=(1, 3))
        assert np.median(np.abs(binned - coarse)) < 8, name


def test_the_controls_hold_no_bands_below_the_limit() -> None:
    # C24's negative control: band couplets are many pixels wide at the size drawn.
    radius_px = 0.5 * 1024 / 2  # the narrowest set (two-sets) spans 0.5 of the half-width
    assert radius_px / zones.COUPLETS > 8


def test_control_answers_are_checked() -> None:
    right = BANDED | {"C25_zones": [{"kind": "even bands", "share": 1.0}]}
    assert zones.control_problems("one-set", right) == []
    assert zones.control_problems("two-sets", right) == ["two-sets: C23 1, not 2"]
    cue = right | {"C24_outer_striations": "present"}
    assert "a cue on a face without fine bands" in zones.control_problems("one-set", cue)[0]
    assert zones.control_problems("zones", right)  # misses the quartz and cavity zones


# The check page.

SAMPLE = [f"p{i:02d}" for i in range(30)]


def serve(tmp_path: Path, **kw: Any) -> tuple[str, ZoneChecker, ThreadingHTTPServer]:
    views, path, pictures = tmp_path / "views.toml", tmp_path / "zones.toml", tmp_path / "pictures"
    pictures.mkdir(parents=True)
    for k in SAMPLE:
        (pictures / f"{k}.png").write_bytes(b"png")
    labels.save({k: {"checked": {"C6_eye": 4}} for k in SAMPLE}, views)
    with_cue = PROPOSED | {"C24_outer_iris": "present"}  # must never reach the page
    zones.save({k: {"proposed": with_cue} for k in SAMPLE}, path)
    checker = ZoneChecker(path, pictures, views, **kw)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(checker))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}", checker, server


def post(url: str, body: object) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_blind_faces_come_first_hide_proposals_and_must_be_done_first(tmp_path: Path) -> None:
    base, checker, server = serve(tmp_path)
    try:
        rows = checker.rows()
        assert [r["blind"] for r in rows] == [True] * 10 + [False] * 20
        assert all(r["proposed"] is None for r in rows[:10])
        shown = rows[10]["proposed"]
        assert isinstance(shown, dict)
        assert not set(zones.CUE_FIELDS) & set(cast("dict[str, Any]", shown))
        with urllib.request.urlopen(base + "/") as reply:
            assert (
                "present" not in reply.read().decode().split("const DATA")[1].split("cue_values")[0]
            )
        status, got = post(base + "/save", {"id": rows[10]["id"], "values": BANDED})
        assert status == 400
        assert "blind faces first" in got["problems"][0]
        for r in rows[:10]:
            assert post(base + "/save", {"id": r["id"], "values": BANDED | {"note": ""}})[0] == 200
        assert post(base + "/save", {"id": rows[10]["id"], "values": BANDED})[0] == 200
        saved = zones.load(checker.path)
        assert saved[str(rows[0]["id"])]["checked"] == BANDED | {"blind": True}
        assert saved[str(rows[10]["id"])]["checked"]["blind"] is False
        status, _ = post(base + "/save", {"id": rows[11]["id"], "values": BANDED | {"blind": True}})
        assert zones.load(checker.path)[str(rows[11]["id"])]["checked"]["blind"] is False
    finally:
        server.shutdown()
        server.server_close()


def test_the_recount_opens_after_a_week_and_hides_the_first_count(tmp_path: Path) -> None:
    base, checker, server = serve(tmp_path, recount=True, today=date(2026, 11, 1))
    try:
        assert len(checker.sample) == zones.RECOUNT
        assert all(set(r) == {"id", "recount"} for r in checker.rows())
        status, got = post(base + "/save", {"id": checker.sample[0], "values": {"C6_eye": 7}})
        assert status == 400
        assert "opens on 2026-11-05" in got["problems"][0]
    finally:
        server.shutdown()
        server.server_close()
    base, checker, server = serve(tmp_path / "later", recount=True, today=zones.RECOUNT_FROM)
    try:
        assert post(base + "/save", {"id": checker.sample[0], "values": {"C6_eye": 7.5}})[0] == 400
        assert post(base + "/save", {"id": checker.sample[0], "values": {"C6_eye": 7}})[0] == 200
        got = zones.load(checker.path)[checker.sample[0]]["recount"]
        assert got == {"C6_eye": 7, "date": "2026-11-05"}
    finally:
        server.shutdown()
        server.server_close()
