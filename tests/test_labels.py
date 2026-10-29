"""Labels judged by eye (ADR 0004, choice 9b): what a valid row is, that the
owner's check wins over the proposal, and that an outline judged off the rim
makes the outline-bound values unknown."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from agatesim.library import Label
from agatesim.shapes import VERSION

from nodule import check, labels, rims, tags
from nodule.bands import Bands
from nodule.outline import Outline

LABEL = Label(verdict="forms", forms=["fortification"], features=["quartz core"])
SHAPE = Outline(aspect=0.8, regularity=0.97, circularity=0.9, irregularity=0.1, concavities=1)
BAND = Bands(14.0, 12.0, 15.0, 0.0714, None, 20, 300)
ENTRY: dict[str, object] = {
    "key": f"{VERSION}:1234:maps",
    "level": "fit",
    "spread": 0.61234,
    "trend": -1.0,
    "ratio": 0.9,
    "hurst": None,
    "core": 0,
}

CUT_FACE: labels.Row = {
    "view_kind": "cut face",
    "finish": "polished",
    "lighting": "reflected",
    "C4_surface": "unknown",
    "C5_rind": "adhering host",
    "C5_width": 0.03,
    "C22_count": 0,
    "C22_fill": "none",
    "C22_offset": "unknown",
    "outline": "yes",
}
PROPOSED = CUT_FACE | {"confidence": "medium", "note": "clear rim"}


def test_a_valid_proposal_and_check_have_no_problems() -> None:
    assert labels.problems(PROPOSED) == []
    assert labels.problems(CUT_FACE | {"C6_eye": 12, "note": "ok"}, checked=True) == []
    assert labels.problems(CUT_FACE, checked=True) == []  # the eye count is optional


@pytest.mark.parametrize(
    ("change", "says"),
    [
        ({"view_kind": "slice"}, "view_kind"),
        ({"outline": "unknown"}, "outline"),  # the check must decide
        ({"C5_width": 0.9}, "C5_width"),
        ({"C22_count": 2.5}, "C22_count"),
        ({"C22_count": True}, "C22_count"),
        ({"C4_surface": "ridged"}, "not an exterior"),
        ({"C5_rind": "absent"}, "no rind"),
        ({"C22_count": 0, "C22_fill": "open"}, "no fractures"),
        ({"C22_count": 3}, "count is not 0"),
        ({"colour": "red"}, "not a label"),
    ],
)
def test_invalid_or_inconsistent_labels_are_refused(change: labels.Row, says: str) -> None:
    found = labels.problems(PROPOSED | change)
    assert found
    assert any(says in p for p in found)


def test_each_side_gives_only_its_own_fields() -> None:
    assert labels.problems(CUT_FACE) != []  # a proposal states its confidence
    assert labels.problems(PROPOSED | {"C6_eye": 3}) != []  # only the owner counts by eye
    assert labels.problems(CUT_FACE | {"C6_eye": -1}, checked=True) != []


def test_the_owners_check_wins_over_the_proposal() -> None:
    owner = CUT_FACE | {"finish": "sawn"}
    assert labels.chosen({"proposed": PROPOSED, "checked": owner}) == (owner, "owner, checked")
    assert labels.chosen({"proposed": PROPOSED}) == (PROPOSED, "Claude, proposed")
    assert labels.chosen(None) == ({}, labels.UNKNOWN)


def test_agreement_counts_only_checked_photos() -> None:
    images = {
        "a": {"proposed": PROPOSED, "checked": CUT_FACE | {"finish": "sawn"}},
        "b": {"proposed": PROPOSED, "checked": dict(CUT_FACE)},
        "c": {"proposed": PROPOSED},
    }
    got = labels.agreement(images)
    assert got["finish"] == (1, 2)
    assert got["view_kind"] == (2, 2)


def test_the_check_sample_is_fixed_and_bounded() -> None:
    ids = [f"photo-{i:03d}" for i in range(230)]
    assert labels.check_sample(ids) == labels.check_sample(reversed(ids))
    assert len(labels.check_sample(ids)) == labels.CHECK_SAMPLE
    assert labels.check_sample(ids[:5]) == sorted(ids[:5])


def test_labels_fill_the_record_but_their_old_outline_check_is_left_out() -> None:
    """The labels' outline check judged `segment()`'s outline; the rim
    outline's own verdict (agatesim ADR 0075) decides instead."""
    yes = tags.rim_tags(None, rims.Verdict(True, True, "helper", "held"))
    rec = tags.record(LABEL, SHAPE, ENTRY, 1234, BAND, yes)
    on_rim = tags.labelled(rec, {"proposed": PROPOSED})
    assert on_rim.keys() == rec.keys()
    assert on_rim["view_kind"] == "cut face"
    assert on_rim["outline_check"] == "yes"
    assert on_rim["labels_by"] == "Claude, proposed"
    assert on_rim["C1_aspect"] == 0.8
    assert on_rim["C6_count"] == 14.0
    old_no = tags.labelled(rec, {"proposed": PROPOSED | {"outline": "no"}})
    assert old_no["outline_check"] == "yes"
    assert old_no["C1_aspect"] == 0.8
    unlabelled = tags.labelled(rec, None)
    assert unlabelled["view_kind"] == tags.UNKNOWN
    assert unlabelled["C1_aspect"] == 0.8


def test_the_labels_file_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "labels.toml"
    labels.save(
        {"b": {"proposed": PROPOSED}, "a": {"proposed": PROPOSED, "checked": CUT_FACE}}, path
    )
    assert labels.load(path) == {
        "a": {"proposed": PROPOSED, "checked": CUT_FACE},
        "b": {"proposed": PROPOSED},
    }


def serve(tmp_path: Path) -> tuple[str, check.Checker, check.ThreadingHTTPServer]:
    path, pictures = tmp_path / "labels.toml", tmp_path / "pictures"
    pictures.mkdir()
    (pictures / "a.png").write_bytes(b"png")
    (tmp_path / "secret.png").write_bytes(b"no")
    labels.save({"a": {"proposed": PROPOSED}}, path)
    checker = check.Checker(path, pictures)
    server = check.ThreadingHTTPServer(("127.0.0.1", 0), check.handler(checker))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}", checker, server


def post(url: str, body: object) -> tuple[int, dict[str, object]]:
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_the_check_page_saves_valid_checks_only(tmp_path: Path) -> None:
    base, checker, server = serve(tmp_path)
    try:
        with urllib.request.urlopen(base + "/") as reply:
            assert '"id": "a"' in reply.read().decode()
        with urllib.request.urlopen(base + "/img/a.png") as reply:
            assert reply.read() == b"png"
        for name in ("/img/../secret.png", "/img/secret.png", "/labels.toml"):
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(base + name)
        status, got = post(base + "/save", {"id": "a", "values": CUT_FACE | {"lighting": "sun"}})
        assert status == 400
        assert got["problems"]
        status, _ = post(base + "/save", {"id": "elsewhere", "values": CUT_FACE})
        assert status == 400
        assert "checked" not in labels.load(checker.path)["a"]
        status, _ = post(
            base + "/save", {"id": "a", "values": CUT_FACE | {"C6_eye": 9, "note": ""}}
        )
        assert status == 200
        assert labels.load(checker.path)["a"]["checked"] == CUT_FACE | {"C6_eye": 9}
    finally:
        server.shutdown()
        server.server_close()
