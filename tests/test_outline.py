"""Outline shape C1-C3 (ADR 0004) on drawn shapes with known answers, and the
negative controls the ADR names for each."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from nodule import outline


def polygon_mask(vertices: NDArray[np.float64], size: int) -> NDArray[np.bool_]:
    """Pixels whose centres lie inside a polygon (even-odd rule)."""
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    inside = np.zeros((size, size), dtype=bool)
    for (x0, y0), (x1, y1) in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        if y0 == y1:
            continue
        crosses = (ys >= min(y0, y1)) & (ys < max(y0, y1))
        x_at = x0 + (ys - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (xs < x_at)
    return inside


def radial(radius: NDArray[np.float64], size: int, stretch: float = 1.0, turn: float = 0.0):
    """Polygon of radius r(θ) around the image centre, squeezed by `stretch`
    along y, then turned by `turn` radians."""
    theta = np.linspace(0.0, 2.0 * math.pi, len(radius), endpoint=False)
    x, y = radius * np.cos(theta), stretch * radius * np.sin(theta)
    c, s = math.cos(turn), math.sin(turn)
    pts = np.stack([c * x - s * y, s * x + c * y], axis=1) + size / 2.0
    return polygon_mask(pts, size)


def _unturned(size: int, turn: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pixel-centre coordinates about the image centre, turned back by `turn`."""
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64) - size / 2.0
    c, s = math.cos(turn), math.sin(turn)
    return c * xs + s * ys, -s * xs + c * ys


def ellipse(size: int, a: float, b: float, turn: float = 0.0) -> NDArray[np.bool_]:
    x, y = _unturned(size, turn)
    return (x / a) ** 2 + (y / b) ** 2 <= 1.0


def regular(n: int, size: int, r: float, turn: float = 0.0) -> NDArray[np.bool_]:
    return radial(np.full(n, r), size, 1.0, turn)


def star(points: int, size: int, r: float, inner: float = 0.5) -> NDArray[np.bool_]:
    radius = np.tile([r, inner * r], points)
    return radial(radius, size)


def lobed(size: int, r: float, stretch: float) -> NDArray[np.bool_]:
    """Six rounded lobes, squeezed by `stretch` along y."""
    x, y = _unturned(size, 0.0)
    y = y / stretch
    return np.hypot(x, y) <= r * (1.0 + 0.25 * np.cos(6.0 * np.arctan2(y, x)))


def measured(mask: NDArray[np.bool_]) -> outline.Outline:
    result = outline.measure(mask)
    assert result is not None
    return result


@pytest.mark.parametrize(("b", "turn"), [(150.0, 0.0), (75.0, 0.5), (40.0, 1.2)])
def test_an_ellipse_is_perfectly_regular_at_any_aspect_ratio(b: float, turn: float) -> None:
    got = measured(ellipse(400, 150.0, b, turn))
    assert got.aspect == pytest.approx(b / 150.0, abs=0.01)
    assert got.regularity == pytest.approx(1.0, abs=0.01)


def test_a_lobed_outline_of_the_same_aspect_ratio_is_less_regular() -> None:
    smooth = measured(ellipse(400, 150.0, 90.0))
    lobes = measured(lobed(400, 120.0, 0.6))
    assert lobes.aspect == pytest.approx(smooth.aspect, abs=0.05)
    assert lobes.regularity < smooth.regularity - 0.03


@pytest.mark.parametrize(("n", "expected"), [(4, 0.841), (8, 0.961)])
@pytest.mark.parametrize("size", [120, 480])
def test_riley_circularity_matches_blott_and_pye(n: int, expected: float, size: int) -> None:
    got = measured(regular(n, size, 0.4 * size, turn=math.pi / n))
    assert got.circularity == pytest.approx(expected, abs=0.01)


def test_a_circle_has_circularity_one() -> None:
    assert measured(ellipse(400, 150.0, 150.0)).circularity == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize(
    "mask",
    [
        ellipse(400, 150.0, 150.0),
        ellipse(400, 150.0, 50.0, 0.7),
        regular(4, 400, 150.0, 0.3),
        regular(8, 400, 150.0),
        regular(3, 400, 150.0, 0.1),
    ],
)
def test_any_convex_outline_has_no_irregularity(mask: NDArray[np.bool_]) -> None:
    got = measured(mask)
    assert got.irregularity == 0.0
    assert got.concavities == 0


@pytest.mark.parametrize("points", [4, 8, 16])
def test_a_star_has_one_concavity_per_point_of_known_depth(points: int) -> None:
    got = measured(star(points, 480, 200.0))
    # Each concavity's tip is at half the outer radius; the hull there is the
    # chord between two points, at r·cos(π/n).
    expected = points * (1.0 - 0.5 / math.cos(math.pi / points))
    assert got.concavities == points
    assert got.irregularity == pytest.approx(expected, rel=0.03)


def test_stars_rank_in_order_of_their_points() -> None:
    values = [measured(star(n, 480, 200.0)).irregularity for n in (4, 8, 16)]
    assert values == sorted(values)
    assert values[0] < values[1] < values[2]


def test_the_same_shape_at_two_resolutions_gives_the_same_values() -> None:
    small = measured(lobed(200, 60.0, 0.7))
    large = measured(lobed(800, 240.0, 0.7))
    for name in ("aspect", "regularity", "circularity"):
        assert getattr(small, name) == pytest.approx(getattr(large, name), abs=0.02), name
    # A sum over concavities: within 3% of itself.
    assert small.irregularity == pytest.approx(large.irregularity, rel=0.03)
    assert small.concavities == large.concavities == 6


def test_holes_are_filled_and_only_the_largest_region_counts() -> None:
    ring = ellipse(400, 150.0, 150.0) & ~ellipse(400, 60.0, 60.0)
    ring[5:15, 5:15] = True  # a speck apart from the face
    got = measured(ring)
    assert got.circularity == pytest.approx(1.0, abs=0.01)
    assert got.irregularity == 0.0


def test_the_enclosing_circle_holds_every_point_and_is_tight() -> None:
    pts = np.random.default_rng(3).normal(size=(500, 2))
    centre, r = outline.enclosing_circle(pts)
    dist = np.linalg.norm(pts - centre, axis=1)
    assert dist.max() <= r + 1e-9
    assert np.sum(dist > r - 1e-9) >= 2  # held by at least two points on its rim


def test_a_face_cut_off_by_the_frame_gives_none() -> None:
    # The same disc, whole and then running off the image's edge.
    assert outline.measure(ellipse(400, 150.0, 150.0)) is not None
    assert outline.measure(ellipse(400, 150.0, 150.0)[:, 120:]) is None


def test_too_small_a_region_gives_none() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:7, 5:7] = True
    assert outline.measure(mask) is None
