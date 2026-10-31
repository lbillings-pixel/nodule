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


ROUND2: dict[str, Any] = {
    "round": 2,
    "banded": "yes",
    "C23_centres": 1,
    **dict.fromkeys(zones.ROUND2_CUE_FIELDS, "absent"),
    "C25_zones": [{"kind": "even bands", "share": 0.7}, {"kind": "coarse quartz", "share": 0.3}],
}


def test_round_2_rows_have_their_own_fields() -> None:
    assert zones.problems(ROUND2 | {"confidence": "low"}) == []
    assert zones.problems(ROUND2 | {"blind": True}, checked=True) == []
    # Round 1 cues are refused in round 2, and round 1 rows still need theirs.
    assert any("not a label" in p for p in zones.problems(ROUND2 | ABSENT, checked=True))
    assert zones.problems(BANDED | {"round": 1}, checked=True)
    assert zones.problems(ROUND2 | {"round": True}, checked=True)
    assert zones.problems(BANDED, checked=True) == []


def test_round_2_zones_cover_at_least_a_twentieth_of_the_line() -> None:
    thin = [
        {"kind": "even bands", "share": 0.96},
        {"kind": "plain: no visible bands", "share": 0.04},
    ]
    assert any("choice 7l" in p for p in zones.problems(ROUND2 | {"C25_zones": thin}, True))
    assert zones.problems(BANDED | {"C25_zones": thin}, checked=True) == []  # round 1 unchanged


def test_line_match_measures_the_shared_stretch() -> None:
    a = [{"kind": "even bands", "share": 0.7}, {"kind": "coarse quartz", "share": 0.3}]
    b = [{"kind": "even bands", "share": 0.5}, {"kind": "coarse quartz", "share": 0.5}]
    assert zones.line_match(a, a) == pytest.approx(1.0)
    assert zones.line_match(a, b) == pytest.approx(0.8)
    # Shares off 1 within SHARE_SUM are scaled, not left short.
    assert zones.line_match(a, [{"kind": "even bands", "share": 0.98}]) == pytest.approx(0.7)
    assert zones.line_match(a, []) == 0.0


def test_round_2_scores_c25_once_along_the_line() -> None:
    same = zones.items(ROUND2, ROUND2)
    assert [k for k, _ in same] == ["banded", "C23", "C24", "C24", "C24", "C25"]
    # Negative control: zones in the right order but the split far off (0.7 vs
    # 0.2 of the line in bands) match on only half the line, so disagree.
    off = ROUND2 | {
        "C25_zones": [{"kind": "even bands", "share": 0.2}, {"kind": "coarse quartz", "share": 0.8}]
    }
    assert zones.items(ROUND2, off)[-1] == ("C25", False)
    near = ROUND2 | {
        "C25_zones": [{"kind": "even bands", "share": 0.5}, {"kind": "coarse quartz", "share": 0.5}]
    }
    assert zones.items(ROUND2, near)[-1] == ("C25", True)
    with pytest.raises(ValueError, match="round 1"):
        zones.items(BANDED, ROUND2)


def test_the_rounds_are_reported_and_tested_apart() -> None:
    images = {f"r1-{i}": {"proposed": PROPOSED, "checked": BANDED} for i in range(30)}
    first = zones.report(images)
    assert "round 2" not in first
    wrong = ROUND2 | {"C23_centres": 3, "blind": True}
    images |= {f"r2-{i}": {"proposed": ROUND2, "checked": wrong} for i in range(10)}
    text = zones.report(images)
    assert text.startswith(first)  # round 1's result stands as scored
    assert "round 2 (ADR 0006, s.9)" in text
    assert "blind  50/60 items" in text
    assert "FALLBACK: more than a third of the blind C23 items" in text.split("round 2")[1]
    assert "share of the line where kinds match: median 1.00, n 10" in text


def test_round_2_draws_a_fixed_pool_outside_round_1() -> None:
    views: dict[str, dict[str, labels.Row]] = {
        f"p{i:03d}": {"proposed": {"view_kind": "exterior" if i % 5 == 0 else "cut face"}}
        for i in range(120)
    }
    first = set(labels.check_sample(views))
    images: dict[str, dict[str, Any]] = {k: {"proposed": PROPOSED} for k in first}
    pool = zones.round2_pool(views, images)
    assert not first & set(pool)
    assert all(views[k]["proposed"]["view_kind"] == "cut face" for k in pool)
    assert pool == zones.round2_pool(dict(reversed(views.items())), images)
    picked = zones.round2_sample(views, images)
    assert picked == pool[: zones.BLIND]
    # A face the owner marks not banded at the line check gives way to the next.
    images[picked[0]] = {"line": {"banded": "no"}}
    assert zones.round2_sample(views, images) == pool[1 : zones.BLIND + 1]
    # Round 2 answers keep a face in the pool; round 1 labels take it out.
    images[pool[1]] = {"proposed": ROUND2}
    assert pool[1] in zones.round2_pool(views, images)
    images[pool[2]] = {"proposed": PROPOSED}
    assert pool[2] not in zones.round2_pool(views, images)


@pytest.mark.parametrize(
    ("line", "says"),
    [
        ({"banded": "no", "note": "a broken surface"}, None),
        ({"rim": [0.1, 0.5], "centre": [0.5, 0.5], "source": "paths", "draw": 0}, None),
        (
            {
                "rim": [0.1, 0.5],
                "centre": [0.5, 0.5],
                "source": "owner",
                "draw": 1,
                "reasons": ["crack or edge"],
            },
            None,
        ),
        ({"banded": "yes"}, "only no"),
        ({"rim": [1.2, 0.5], "centre": [0.5, 0.5], "source": "paths", "draw": 0}, "rim"),
        ({"rim": [0.1, 0.5], "centre": [0.5], "source": "paths", "draw": 0}, "centre"),
        ({"rim": [0.1, 0.5], "centre": [0.5, 0.5], "source": "guess", "draw": 0}, "source"),
        ({"rim": [0.1, 0.5], "centre": [0.5, 0.5], "source": "paths", "draw": 2}, "one per draw"),
        (
            {
                "rim": [0.1, 0.5],
                "centre": [0.5, 0.5],
                "source": "paths",
                "draw": 1,
                "reasons": ["prettier"],
            },
            "each one of",
        ),
    ],
)
def test_checked_lines_are_validated(line: dict[str, Any], says: str | None) -> None:
    found = zones.line_problems(line)
    assert found == [] if says is None else any(says in p for p in found), found


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


def test_round_2_control_answers_are_checked() -> None:
    right = ROUND2 | {"C25_zones": [{"kind": "even bands", "share": 1.0}]}
    assert zones.control_problems("one-set", right) == []
    cue = right | {"C24_inner_striations": "present"}
    assert "a cue on a face without fine bands" in zones.control_problems("one-set", cue)[0]


def along(picture: np.ndarray, name: str, t: float) -> np.ndarray:
    """The control picture's colour at `t` along its line (0 rim, 1 centre)."""
    (x0, y0), (x1, y1) = zones.control_line(name)
    h, w = picture.shape[:2]
    x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
    return picture[int(y * h), int(x * w)].astype(float)


def test_the_control_lines_run_from_the_rim_with_exact_shares() -> None:
    for name in zones.CONTROLS:
        picture = zones.render_control(name, 1024)
        assert np.array_equal(along(picture, name, -0.02), zones.BACKGROUND), name
        assert not np.array_equal(along(picture, name, 0.02), zones.BACKGROUND), name
    # Negative control on each side of every boundary: bands are bluish (B >= R),
    # quartz warm (R > B), the cavity dark.
    picture = zones.render_control("zones", 1024)
    (r45, _, b45), (r55, _, b55) = along(picture, "zones", 0.45), along(picture, "zones", 0.55)
    assert r45 <= b45
    assert r55 > b55
    assert along(picture, "zones", 0.75).mean() > 100 > along(picture, "zones", 0.85).mean()


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


def test_round_2_serves_its_blind_10_with_striations_only(tmp_path: Path) -> None:
    views_path, path = tmp_path / "views.toml", tmp_path / "zones.toml"
    views: dict[str, dict[str, labels.Row]] = {
        f"p{i:03d}": {"proposed": {"view_kind": "cut face"}} for i in range(120)
    }
    labels.save(views, views_path)
    picked = zones.round2_sample(views, {})
    proposed = ROUND2 | {"confidence": "medium"}
    # The last face has no proposal yet, so it is not served.
    zones.save({k: {"proposed": proposed} for k in picked[:-1]}, path)
    checker = ZoneChecker(path, tmp_path, views_path, round2=True)
    assert checker.sample == picked[:-1]
    assert all(r["blind"] and r["proposed"] is None for r in checker.rows())
    assert checker.data()["cues"] == {"striations": zones.CUES["striations"]}
    answer = {k: v for k, v in ROUND2.items() if k != "round"}
    assert checker.save(picked[0], answer) == []
    assert zones.load(path)[picked[0]]["checked"] == ROUND2 | {"blind": True}
    # Negative control: a round 1 answer (twelve cues, zone under 0.05) is refused.
    old = BANDED | {
        "C25_zones": [{"kind": "even bands", "share": 0.97}, {"kind": "cavity", "share": 0.03}]
    }
    assert checker.save(picked[1], old)


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
