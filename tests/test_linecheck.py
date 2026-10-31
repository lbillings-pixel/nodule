"""The owner's check of round 2's drawn lines (ADR 0006, section 9, choice 7o)."""

from __future__ import annotations

import io
import json
import math
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from nodule import labels, linecheck, zones
from nodule.check import ThreadingHTTPServer, handler

SIZE, K = 160, 4  # picture px; full px per face px


def disc() -> np.ndarray:
    """A round face filling most of the picture, at 1/K size."""
    n = SIZE // K
    i, j = np.mgrid[:n, :n]
    return (i - n / 2 + 0.5) ** 2 + (j - n / 2 + 0.5) ** 2 < (0.4 * n) ** 2


def test_the_line_is_drawn_with_ticks() -> None:
    img = Image.new("RGB", (400, 400), (90, 90, 90))
    out = np.asarray(linecheck.ticked(img, (0.1, 0.5), (0.9, 0.5)))
    yellow = (out[..., 0] > 200) & (out[..., 1] > 200) & (out[..., 2] < 80)
    assert yellow[200, 50:350].all()  # along the line
    assert yellow[190:210, 200].all()  # the tick at 0.5 crosses it
    assert not yellow[100:150].any()  # nothing away from the line and its labels
    assert np.asarray(img).max() == 90  # the picture itself is untouched


def test_the_controls_get_their_lines(tmp_path: Path) -> None:
    written = linecheck.write_ticked_controls(tmp_path)
    assert [p.stem for p in written] == list(zones.CONTROLS)
    for path in written:
        with Image.open(path) as img:
            out = np.asarray(img)
        assert ((out[..., 0] > 200) & (out[..., 1] > 200) & (out[..., 2] < 80)).any(), path


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[str, linecheck.LineChecker]]:
    views, path, pictures = tmp_path / "views.toml", tmp_path / "zones.toml", tmp_path / "lines"
    labels.save({f"p{i:03d}": {"proposed": {"view_kind": "cut face"}} for i in range(60)}, views)
    pool = zones.round2_pool(labels.load(views), {})
    pictures.mkdir()
    drafts: dict[str, Any] = {}
    for k in pool[:12]:
        Image.new("RGB", (SIZE, SIZE), (90, 90, 90)).save(pictures / f"{k}.png")
        np.savez_compressed(pictures / f"{k}.npz", face=disc(), k=float(K), shape=(SIZE, SIZE))
        drafts[k] = {"rim": [0.9, 0.5], "centre": [0.5, 0.5], "source": "paths"}
    (pictures / linecheck.DRAFTS).write_text(json.dumps(drafts), "utf-8")
    checker = linecheck.LineChecker(path, pictures, views)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(checker))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", checker
    server.shutdown()
    server.server_close()


def post(url: str, body: object) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_the_server_draws_the_line_asked_for(served: tuple[str, linecheck.LineChecker]) -> None:
    base, checker = served
    first = checker.sample[0]
    with urllib.request.urlopen(f"{base}/img/{first}.png?draw=1&x=0.4&y=0.5") as reply:
        assert Image.open(io.BytesIO(reply.read())).size == (SIZE, SIZE)
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(f"{base}/img/p999.png")  # not a drafted face


def test_lines_are_worked_out_and_stored_by_the_server(
    served: tuple[str, linecheck.LineChecker],
) -> None:
    base, checker = served
    a, b, c = checker.sample[:3]
    assert post(base + "/save", {"id": a, "values": {"centre": None, "draw": 0}})[0] == 200
    line = zones.load(checker.path)[a]["line"]
    assert line["source"] == "paths"
    assert line["draw"] == 0
    reach = math.hypot(line["rim"][0] - 0.5, line["rim"][1] - 0.5)
    assert 0.37 < reach < 0.41  # the rim end lies on the disc's edge, 0.4 out
    # The owner's centre, and a direction passed over for a stated reason.
    values = {"centre": [0.45, 0.55], "draw": 1, "reasons": ["crack or edge"], "note": "x"}
    assert post(base + "/save", {"id": b, "values": values})[0] == 200
    line = zones.load(checker.path)[b]["line"]
    assert line["source"] == "owner"
    assert line["centre"] == [0.45, 0.55]
    assert line["reasons"] == ["crack or edge"]
    assert line["note"] == "x"
    assert zones.line_problems(line) == []
    # Negative controls: a centre off the stone, and a direction passed over with no reason.
    status, got = post(base + "/save", {"id": c, "values": {"centre": [0.02, 0.02], "draw": 0}})
    assert status == 400
    assert "inside the stone" in got["problems"][0]
    status, got = post(base + "/save", {"id": c, "values": {"centre": None, "draw": 1}})
    assert status == 400
    assert "one per draw" in got["problems"][0]
    assert c not in zones.load(checker.path)


def test_a_face_with_no_bands_gives_way_to_the_next(
    served: tuple[str, linecheck.LineChecker],
) -> None:
    base, checker = served
    before = zones.round2_sample(checker.views, checker.images)
    assert post(base + "/save", {"id": before[0], "values": {"banded": "no"}})[0] == 200
    after = zones.round2_sample(checker.views, zones.load(checker.path))
    assert after == checker.sample[1 : zones.BLIND + 1]
    assert zones.load(checker.path)[before[0]]["line"] == {"banded": "no"}


def test_a_traced_outline_replaces_a_wrong_face(
    served: tuple[str, linecheck.LineChecker],
) -> None:
    base, checker = served
    a, b = checker.sample[:2]
    corner = [[0.04, 0.04], [0.26, 0.04], [0.26, 0.26], [0.04, 0.26]]  # off the computer's disc
    q = "draw=0&x=0.15&y=0.15"
    o = "o=" + ",".join(str(v) for p in corner for v in p)
    # Negative control: off the computer's face, the click is refused and the page told why.
    with urllib.request.urlopen(f"{base}/problems/{a}?{q}") as reply:
        assert "cyan edge" in json.loads(reply.read())["problems"][0]
    with urllib.request.urlopen(f"{base}/problems/{a}?{q}&{o}") as reply:
        assert json.loads(reply.read())["problems"] == []
    values: dict[str, Any] = {"centre": [0.15, 0.15], "draw": 0, "outline": corner}
    assert post(base + "/save", {"id": a, "values": values})[0] == 200
    line = zones.load(checker.path)[a]["line"]
    assert line["outline"] == corner
    assert zones.line_problems(line) == []
    x, y = line["rim"]
    assert 0.03 < x < 0.27  # the rim end lies on the traced square
    assert 0.03 < y < 0.27
    assert min(x - 0.04, 0.26 - x, y - 0.04, 0.26 - y) < 0.02
    # With no centre clicked, the centre is the deepest point of the outline.
    assert post(base + "/save", {"id": b, "values": {"draw": 0, "outline": corner}})[0] == 200
    line = zones.load(checker.path)[b]["line"]
    assert line["source"] == "deepest point"
    assert all(abs(v - 0.15) < 0.02 for v in line["centre"])
    # Negative controls: an outline of two points; a centre outside the traced outline.
    status, got = post(base + "/save", {"id": b, "values": {"draw": 0, "outline": corner[:2]}})
    assert status == 400
    assert "at least 3" in got["problems"][0]
    values = {"centre": [0.5, 0.5], "draw": 0, "outline": corner}
    status, got = post(base + "/save", {"id": b, "values": values})
    assert status == 400
    assert "traced outline" in got["problems"][0]


def test_the_page_picture_shows_the_face_edge(
    served: tuple[str, linecheck.LineChecker],
) -> None:
    base, checker = served
    first = checker.sample[0]

    def cyan(query: str) -> np.ndarray:
        with urllib.request.urlopen(f"{base}/img/{first}.png?{query}") as reply:
            out = np.asarray(Image.open(io.BytesIO(reply.read())).convert("RGB")).astype(int)
        return (abs(out - linecheck.CYAN) < 30).all(axis=-1)

    edge = cyan("draw=0")
    assert edge[SIZE // 2, 10:25].any()  # the disc's edge, 0.4 out from the middle
    assert not edge[SIZE // 2, 40:120].any()  # nothing inside the face
    assert not edge[:8, :8].any()  # nor outside it
    assert cyan("draw=0&x=0.02&y=0.02")[SIZE // 2, 10:25].any()  # shown even with no line
    tracing = cyan("draw=0&trace=1&o=0.1,0.1,0.9,0.1")
    assert tracing[16, 40:120].all()  # the outline so far
    assert not tracing[SIZE // 2, 10:25].any()  # and not the computer's edge


def test_a_face_with_no_rim_in_view_gives_way_too(
    served: tuple[str, linecheck.LineChecker],
) -> None:
    base, checker = served
    before = zones.round2_sample(checker.views, checker.images)
    assert post(base + "/save", {"id": before[1], "values": {"rim_in_view": "no"}})[0] == 200
    after = zones.round2_sample(checker.views, zones.load(checker.path))
    assert after == [before[0], *checker.sample[2 : zones.BLIND + 1]]
    stored = zones.load(checker.path)[before[1]]["line"]
    assert stored == {"rim_in_view": "no"}  # held out, but not stored as unbanded
    assert "banded" not in stored
    # Negative controls: only "no" is stored, and a held-out face carries no line.
    assert zones.line_problems({"rim_in_view": "yes"}) != []
    assert zones.line_problems({"rim_in_view": "no", "rim": [0.1, 0.1]}) != []
    assert not zones.held_out({"rim": [0.1, 0.1], "centre": [0.5, 0.5]})
