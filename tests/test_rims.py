"""Rim-outline verdicts (agatesim ADR 0075): a verdict applies only to the very
outline it judged, the owner's answer wins, and a doubtful merge keeps "no"."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from nodule import rims


def disc(h: int = 600, w: int = 900, r: float = 200.0, cx: float = 450.0) -> np.ndarray:
    y, x = np.mgrid[:h, :w]
    return (x - cx) ** 2 + (y - 300) ** 2 <= r * r


def saved(mask: np.ndarray) -> np.ndarray:
    """As agatesim's scripts/rim_check.py saves a mask."""
    small = Image.fromarray(mask)
    small.thumbnail((rims.MASK_PX, rims.MASK_PX), Image.Resampling.NEAREST)
    return np.asarray(small, dtype=bool)


def test_fingerprint_matches_the_saved_mask():
    m = disc()
    assert rims.fingerprint(m) == rims.reduced_fingerprint(saved(m))


def test_a_changed_outline_finds_no_verdict():
    """Negative control: a verdict must not carry over to another outline."""
    judged = disc()
    verdicts = {"stone": {rims.fingerprint(judged): rims.Verdict(True, True, "helper", "held")}}
    assert rims.verdict(verdicts, "stone", judged) is not None
    assert rims.verdict(verdicts, "stone", disc(cx=470.0)) is None
    assert rims.verdict(verdicts, "other", judged) is None
    assert rims.verdict(verdicts, "stone", np.zeros_like(judged)) is None


def write_run(
    root: Path, run: str, method: str, masks: dict[str, np.ndarray], reasons: dict[str, str | None]
) -> None:
    (root / run).mkdir(parents=True)
    results = {k: {"reason": v} for k, v in reasons.items()}
    (root / run / "results.json").write_text(json.dumps({"method": method, "results": results}))
    np.savez_compressed(root / run / "masks.npz", **{k: saved(v) for k, v in masks.items()})  # pyright: ignore[reportArgumentType]


def answers(root: Path, name: str, run: str, rows: dict[str, tuple[str, str]]) -> None:
    data: dict[str, object] = {"_run": run}
    data |= {k: {"rim_in_frame": a, "outline_follows": b, "note": ""} for k, (a, b) in rows.items()}
    (root / name).write_text(json.dumps(data))


def test_gather_ties_answers_to_outlines(tmp_path: Path):
    a, b, empty = disc(), disc(cx=430.0), np.zeros((600, 900), dtype=bool)
    write_run(
        tmp_path, "held", "C", {"a": a, "b": b, "c": empty}, {"a": None, "b": None, "c": "cut-off"}
    )
    write_run(tmp_path, "tune-x", "C", {"a": a}, {"a": None})
    write_run(tmp_path, "tune-b", "B", {"b": a}, {"b": None})
    answers(
        tmp_path,
        "judge-held.json",
        "held",
        {"a": ("yes", "yes"), "b": ("yes", "yes"), "c": ("no", "no")},
    )
    answers(tmp_path, "judge-tune-x.json", "tune-x", {"a": ("yes", "no")})
    answers(tmp_path, "judge-tune-b.json", "tune-b", {"b": ("yes", "no")})
    answers(tmp_path, "owner-held.json", "held", {"b": ("yes", "no")})
    got = rims.gather(tmp_path)
    assert set(got) == {"a", "b"}  # no outline on c; method B's run is not read
    (va,) = got["a"]
    assert (va["follows"], va["by"]) == (False, "helper")  # two helper runs disagree: "no"
    (vb,) = got["b"]
    assert (vb["mask"], vb["follows"], vb["by"]) == (rims.fingerprint(b), False, "owner")


def test_gather_reads_several_folders_and_the_owner_still_wins(tmp_path: Path):
    a = disc()
    there, here = tmp_path / "agatesim", tmp_path / "nodule"
    write_run(there, "held", "C", {"a": a}, {"a": None})
    write_run(here, "own", "C", {"a": a}, {"a": None})
    answers(there, "owner-held.json", "held", {"a": ("yes", "yes")})
    answers(here, "judge-own.json", "own", {"a": ("yes", "no")})
    (va,) = rims.gather(there, here)["a"]
    assert (va["follows"], va["by"]) == (True, "owner")


def test_unjudged_outlines_are_listed(tmp_path: Path):
    path = tmp_path / "tags.toml"
    rows = {
        "a": 'rim = "found"\noutline_check = "unknown"',
        "b": 'rim = "found"\noutline_check = "yes"',
        "c": 'rim = "none: cut-off"\noutline_check = "unknown"',
    }
    path.write_text("".join(f"[images.{k}]\n{v}\n" for k, v in rows.items()))
    assert rims.unjudged(path) == ["a"]


def test_write_and_load_round_trip(tmp_path: Path):
    m = disc()
    rows = {
        "a": [
            {
                "mask": rims.fingerprint(m),
                "rim_in_frame": True,
                "follows": True,
                "by": "owner",
                "run": "held",
            }
        ]
    }
    path = tmp_path / "v.toml"
    rims.write(rows, path)
    v = rims.verdict(rims.load(path), "a", m)
    assert v is not None
    assert v.passed
    assert v.by == "owner"
    assert rims.load(tmp_path / "missing.toml") == {}


def test_error_share_interval():
    low, high = rims.wilson(2, 20)
    assert 0.02 < low < 0.04
    assert 0.29 < high < 0.31
    assert "2 of 20" in rims.error_note()
