"""Outline-check pictures (ADR 0004, section 8): the outline and each counted
concavity are drawn, and a region running off the frame is flagged."""

from __future__ import annotations

import numpy as np

from nodule import overlay, rims


def notched_disc(size: int = 200) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size] - size / 2
    disc = np.hypot(x, y) < 70
    disc[90:110, 140:] = False  # one deep notch on the right
    return disc


def test_outline_and_concavity_are_drawn() -> None:
    mask = notched_disc()
    img = np.asarray(overlay.draw(np.zeros((*mask.shape, 3), np.uint8), mask))
    assert (img == overlay.GREEN).all(axis=-1).sum() > 100
    red = (img == overlay.RED).all(axis=-1)
    assert red[:, 120:].any()  # the notch's line lies on the right
    assert not red[:, :90].any()


def test_the_caption_counts_and_flags_a_region_off_the_frame() -> None:
    assert overlay.caption(notched_disc(), 1) == "1 concavities"
    mask = np.zeros((100, 100), bool)
    mask[20:80, 50:] = True
    assert "touches frame" in overlay.caption(mask, 0)


def test_no_region_says_so() -> None:
    assert overlay.caption(np.zeros((50, 50), bool), 0) == "no outline found"
    img = overlay.draw(np.zeros((50, 50, 3), np.uint8), np.zeros((50, 50), bool))
    assert img.size == (50, 50)


def test_the_note_gives_the_outline_verdict() -> None:
    assert overlay.judgment("cut-off", None) == "no outline: cut-off"
    assert overlay.judgment(None, None) == "unjudged"
    no = rims.Verdict(True, False, "owner", "held")
    assert overlay.judgment(None, no) == "judged no (owner)"
