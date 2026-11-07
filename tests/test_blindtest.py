"""The blind test's crops, pairs, page and score (ADR 0008 sections 5 and 5a). CPU only."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from nodule import blindtest as b
from nodule import check

PLAN = b.Plan(4, 2, {"F1": 2}, obvious_from=9, seed=1)


def disc(n: int = 400, radius: int = 180) -> b.Mask:
    y, x = np.mgrid[:n, :n]
    return (y - n / 2) ** 2 + (x - n / 2) ** 2 <= radius**2


def cuts_and_photos(fill: int = 200) -> tuple[list[b.Cut], dict[str, tuple[b.Rgb, b.Mask]]]:
    face = disc()
    made = np.full((400, 400, 3), fill, np.uint8)
    cuts = [b.Cut("stone", f"stone-{k:03d}", f"p{k}", made, face) for k in range(4)]
    cuts += [b.Cut("F1", f"F1-{k:03d}", f"p{k}", made, face) for k in range(2)]
    photos = {f"p{k}": (np.full((400, 400, 3), 40, np.uint8), face) for k in range(4)}
    return cuts, photos


def test_decision_rule_of_section_5_is_the_exact_binomial() -> None:
    assert b.binomial_tail(113, 200) < 0.05 < b.binomial_tail(112, 200)


def test_trial_rule_is_far_past_chance_and_a_coin_rarely_reaches_it() -> None:
    assert b.binomial_tail(b.TRIAL.obvious_from or 0, 16) < 0.01


def test_largest_square_of_known_masks() -> None:
    mask = np.zeros((50, 60), bool)
    assert b.largest_square(mask) == 0
    mask[10:30, 5:45] = True
    assert b.largest_square(mask) == 20


def test_largest_square_is_stopped_by_a_hole() -> None:
    """Control: the same block with a hole in it holds only a smaller square."""
    mask = np.ones((40, 40), bool)
    assert b.largest_square(mask) == 40
    mask[20, 20] = False
    assert b.largest_square(mask) == 20


def test_crops_stay_inside_the_face() -> None:
    """Pixels outside the face are marked; no crop may hold one."""
    face = disc()
    rgb = np.where(face[..., None], 100, 255).astype(np.uint8).repeat(3, axis=2)
    size = b.largest_square(face)
    rng = np.random.default_rng(0)
    for _ in range(40):
        out = b.crop(rgb, face, size, rng)
        assert out.shape == (size, size, 3)
        assert (out == 100).all()


def test_a_crop_that_cannot_fit_is_refused() -> None:
    face = disc()
    rgb = np.zeros((*face.shape, 3), np.uint8)
    with pytest.raises(ValueError, match="fits"):
        b.crop(rgb, face, b.largest_square(face) + 1, np.random.default_rng(0))


def test_faces_too_small_are_refused() -> None:
    with pytest.raises(ValueError, match="share only"):
        b.crop_size(b.MIN_CROP_PX - 1, np.random.default_rng(0))


def test_pulled_in_keeps_off_the_edge() -> None:
    mask = np.zeros((30, 30), bool)
    mask[5:25, 5:25] = True
    assert b.largest_square(b.pulled_in(mask, 2)) == 16
    assert b.pulled_in(np.ones((10, 10), bool), 2).sum() == 36  # the frame is an edge too


def pairs(tmp_path: Path) -> list[dict[str, Any]]:
    cuts, photos = cuts_and_photos()
    return b.build_pairs(cuts, photos, PLAN, tmp_path, np.random.default_rng(0))


def test_pairs_have_equal_sizes_balanced_sides_and_the_cut_on_its_keyed_side(
    tmp_path: Path,
) -> None:
    key = pairs(tmp_path)
    assert len(key) == 10
    for k in key:
        folder = tmp_path / "pairs"
        with (
            Image.open(folder / f"{k['trial']}-a.png") as a,
            Image.open(folder / f"{k['trial']}-b.png") as other,
        ):
            assert a.size == other.size == (k["size"], k["size"])
        made = np.asarray(Image.open(tmp_path / "pairs" / f"{k['trial']}-{k['synthetic']}.png"))
        assert (made == 200).all()  # the fixture's cut, not its photo
    main = [k["synthetic"] for k in key if k["run"] == "stone"]
    catch = [k["synthetic"] for k in key if k["run"] != "stone"]
    assert main.count("a") == main.count("b") == 4
    assert catch.count("a") == catch.count("b") == 1


def test_pair_files_carry_no_metadata_or_answer(tmp_path: Path) -> None:
    pairs(tmp_path)
    for path in (tmp_path / "pairs").iterdir():
        with Image.open(path) as img:
            assert img.info == {}
            assert not img.getexif()
        assert path.stem.split("-")[1] in b.SIDES
        assert "stone" not in path.name
        assert "F1" not in path.name


def test_score_perfect_judge() -> None:
    key = [{"trial": f"t{i}", "synthetic": b.SIDES[i % 2], "run": "stone"} for i in range(10)]
    answers = {k["trial"]: {"pick": k["synthetic"], "seconds": 3.0} for k in key}
    s = b.score(key, answers, PLAN)
    assert (s.main_correct, s.main_trials) == (10, 10)
    assert s.obvious is True
    assert s.p_value == pytest.approx(1 / 1024)


def test_score_rejects_a_judge_with_no_ability() -> None:
    """Negative control: always picking one side on balanced sides is chance."""
    key = [{"trial": f"t{i}", "synthetic": b.SIDES[i % 2], "run": "stone"} for i in range(10)]
    answers = {k["trial"]: {"pick": "a", "seconds": 3.0} for k in key}
    s = b.score(key, answers, PLAN)
    assert s.main_correct == 5
    assert s.obvious is False
    assert s.p_value > 0.5
    assert s.interval[0] < 0.5 < s.interval[1]


def test_score_keeps_catch_pairs_out_of_the_main_count() -> None:
    key = [
        {"trial": "t1", "synthetic": "a", "run": "stone"},
        {"trial": "t2", "synthetic": "a", "run": "F1"},
    ]
    answers = {"t1": {"pick": "b", "seconds": 1.0}, "t2": {"pick": "a", "seconds": 1.0}}
    s = b.score(key, answers, PLAN)
    assert (s.main_correct, s.main_trials, s.catch_correct, s.catch_trials) == (0, 1, 1, 1)
    assert s.by_run == {"stone": (0, 1), "F1": (1, 1)}


def test_score_splits_main_pairs_by_render_noise() -> None:
    key = [
        {"trial": "t1", "synthetic": "a", "run": "stone", "matched_noise": True},
        {"trial": "t2", "synthetic": "a", "run": "stone", "matched_noise": False},
        {"trial": "t3", "synthetic": "a", "run": "F1", "matched_noise": False},
    ]
    answers = {t: {"pick": "a", "seconds": 1.0} for t in ("t1", "t2", "t3")}
    s = b.score(key, answers, PLAN)
    assert s.by_noise == {"matched": (1, 1), "noisier": (1, 1)}


def test_no_score_until_every_pair_is_answered() -> None:
    key = [{"trial": "t1", "synthetic": "a", "run": "stone"}]
    with pytest.raises(ValueError, match="1 left"):
        b.score(key, {}, PLAN)


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[str, b.BlindTest]]:
    pairs(tmp_path)
    test = b.BlindTest(tmp_path)
    server = check.ThreadingHTTPServer(("127.0.0.1", 0), check.handler(test))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", test
    finally:
        server.shutdown()
        server.server_close()


def post(base: str, trial: str, values: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    body = json.dumps({"id": trial, "values": values}).encode()
    req = urllib.request.Request(base + "/save", data=body, method="POST")
    try:
        with urllib.request.urlopen(req) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_the_page_is_not_told_the_answers(served: tuple[str, b.BlindTest]) -> None:
    base, test = served
    with urllib.request.urlopen(base + "/") as reply:
        page = reply.read().decode()
    assert '"trials"' in page
    for word in ('synthetic":', "stone", 'photo":', "F1"):
        assert word not in page
    assert test.data() == {"trials": test.trials, "answered": []}
    with urllib.request.urlopen(f"{base}/img/{test.trials[0]}-a.png") as reply:
        assert reply.read(4) == b"\x89PNG"


def test_the_key_cannot_be_fetched(served: tuple[str, b.BlindTest]) -> None:
    base, test = served
    assert (test.out / "key.json").exists()
    for path in ("/key.json", "/img/key.png", "/img/../key.json", "/img/t001-c.png"):
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + path)


def test_an_answer_is_saved_once_and_survives_a_restart(served: tuple[str, b.BlindTest]) -> None:
    base, test = served
    trial = test.trials[0]
    status, reply = post(base, trial, {"pick": "a", "note": "too clean", "seconds": 2.04})
    assert status == 200
    assert reply["ok"]
    status, reply = post(base, trial, {"pick": "b", "note": "", "seconds": 1.0})
    assert status == 400
    assert "already answered" in reply["problems"][0]
    again = b.BlindTest(test.out)
    assert again.answers[trial] == {"pick": "a", "note": "too clean", "seconds": 2.0}
    assert again.data()["answered"] == [trial]


def test_bad_answers_are_refused(served: tuple[str, b.BlindTest]) -> None:
    base, test = served
    trial = test.trials[0]
    assert post(base, "nope", {"pick": "a", "seconds": 1.0})[0] == 400
    assert post(base, trial, {"pick": "c", "seconds": 1.0})[0] == 400
    assert post(base, trial, {"pick": "a", "seconds": "x"})[0] == 400
    assert post(base, trial, {"pick": "a", "note": 3, "seconds": 1.0})[0] == 400
    assert test.answers == {}
