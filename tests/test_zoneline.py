"""The drafted rim-to-centre line (ADR 0006, section 9, choice 7o)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from agatesim.specimen.quality import luminance

from nodule import bands, zoneline, zones


def synthetic(name: str, size: int) -> tuple[np.ndarray, np.ndarray, float]:
    """A control face (`zones.CONTROLS`), or one of parallel bands or plain
    material in the same outline: luminance, face mask, band spacing (px)."""
    if name in zones.CONTROLS:
        rgb = zones.render_control(name, size)
    else:
        x, y = np.meshgrid(*[(np.arange(size) + 0.5) / size * 2 - 1] * 2)
        if name == "parallel":
            band = 0.5 + 0.5 * np.cos(2 * np.pi * zones.COUPLETS * y / 0.7)
        else:  # plain, with faint noise
            band = 0.5 + 0.02 * np.random.default_rng(1).standard_normal((size, size))
        rgb = zones.CLEAR + (zones.MILKY - zones.CLEAR) * band[..., None]
        rgb[np.hypot(x / 0.85, y / 0.7) > 1] = zones.BACKGROUND
        rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    mask = np.any(rgb != zones.BACKGROUND.astype(np.uint8), axis=-1)
    spacing = 0.85 * (size / 2) / zones.COUPLETS
    return luminance(rgb).astype(np.float32), mask, spacing


def drafted(name: str, size: int = 384) -> tuple[zoneline.Draft, zoneline.Line]:
    lum, mask, spacing = synthetic(name, size)
    traced = bands.trace(lum, mask, 1.0, spacing)
    line, draft = zoneline.draw_line(traced, spacing, name, mask.shape)
    return draft, line


@pytest.mark.parametrize("name", ["one-set", "zones"])
def test_centre_of_one_set_even_past_a_quartz_core(name: str) -> None:
    draft, line = drafted(name)
    assert draft.centre is not None
    assert line.source == "paths"
    assert line.centre == pytest.approx((0.5, 0.5), abs=0.01)


def test_two_sets_give_the_centre_of_one() -> None:
    draft, line = drafted("two-sets")
    assert draft.centre is not None
    x, y = line.centre
    # The sets' centres sit at x = (1 ± 0.42) / 2 of the width, mid-height.
    assert min(abs(x - 0.29), abs(x - 0.71)) < 0.02
    assert y == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize("name", ["parallel", "plain"])
def test_no_draft_without_a_centre(name: str) -> None:
    """Negative controls: parallel bands meet nowhere; a plain face has no
    band paths. The deepest point stands in, marked as such."""
    draft, line = drafted(name)
    assert draft.centre is None
    assert line.source == "deepest point"


def test_line_runs_from_the_outline_to_the_centre() -> None:
    lum, mask, spacing = synthetic("one-set", 384)
    traced = bands.trace(lum, mask, 1.0, spacing)
    line, _ = zoneline.draw_line(traced, spacing, "one-set", mask.shape)
    x, y = line.rim
    # On the outline: the control's ellipse has semi-axes 0.85 and 0.7.
    d = math.hypot((2 * x - 1) / 0.85, (2 * y - 1) / 0.7)
    assert d == pytest.approx(1.0, abs=0.02)


def test_same_line_at_any_resolution() -> None:
    small, large = drafted("one-set", 256)[1], drafted("one-set", 512)[1]
    assert small.centre == pytest.approx(large.centre, abs=0.01)
    assert small.rim == pytest.approx(large.rim, abs=0.01)


def test_direction_is_fixed_per_face_and_draw() -> None:
    a = zoneline.direction("face-a")
    assert a == zoneline.direction("face-a")
    assert a != zoneline.direction("face-b")
    assert a != zoneline.direction("face-a", draw=1)
    assert zoneline.direction("face-a", draw=1) == zoneline.direction("face-a", draw=1)
    assert 0 <= a < 2 * math.pi


def test_an_owner_centre_is_kept() -> None:
    lum, mask, spacing = synthetic("one-set", 384)
    traced = bands.trace(lum, mask, 1.0, spacing)
    line, _ = zoneline.draw_line(traced, spacing, "x", mask.shape, centre=(150.0, 200.0))
    assert line.source == "owner"
    assert line.centre == pytest.approx((200.5 / 384, 150.5 / 384))
