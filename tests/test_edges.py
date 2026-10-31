"""The band edge map (`nodule.edges`)."""

from __future__ import annotations

import numpy as np
import pytest

from nodule.edges import EDGE_PX, edge_face


def disc(n: int, kind: str, spacing: float = 16.0, noise: float = 1.5):
    y, x = np.mgrid[:n, :n].astype(np.float64)
    c = (n - 1) / 2
    r = np.hypot(y - c, x - c)
    mask = r < 0.45 * n
    if kind == "rings":
        lum = 128 + 40 * np.cos(2 * np.pi * r / spacing)
    elif kind == "step":
        lum = np.where(x < c, 100.0, 160.0)
    else:
        lum = np.full((n, n), 128.0)
    lum = lum + np.random.default_rng(0).normal(0, noise, lum.shape)
    lum[~mask] = 20
    return lum.astype(np.float32), mask


@pytest.mark.parametrize("spacing", [16.0, 32.0])
def test_concentric_rings_give_two_edges_per_couplet_where_cos_crosses_zero(
    spacing: float,
) -> None:
    lum, mask = disc(400 if spacing == 16 else 800, "rings", spacing)
    ef = edge_face(lum, mask, 1.0, spacing / 2)
    assert ef.spacing == EDGE_PX
    h, w = ef.edges.shape
    row = ef.edges[h // 2, w // 2 :]
    found = np.flatnonzero(row)
    # Zero crossings of cos(2 pi r / spacing): r = (k / 2 + 1 / 4) spacings.
    expected = (np.arange(0, 2 * 0.45 * 400 / 16) / 2 + 0.25) * EDGE_PX
    inner = found[found < 0.8 * found.max()]
    assert inner.size >= 10
    gaps = np.abs(inner[:, None] - (expected[None, :] + (w // 2 - (w - 1) / 2))).min(axis=1)
    assert np.all(gaps <= 1.5)
    assert np.allclose(np.diff(inner), EDGE_PX / 2, atol=1.5)


def test_a_step_gives_a_straight_edge_at_the_step() -> None:
    # Pixel noise on the flat ground keeps edges too (module docstring): only
    # the step's own edge is checked.
    lum, mask = disc(400, "step")
    ef = edge_face(lum, mask, 1.0, 8.0)
    h, w = ef.edges.shape
    middle = (w - 1) / 2
    for row in ef.edges[h // 4 : 3 * h // 4]:
        cols = np.flatnonzero(row)
        assert np.abs(cols - middle).min() <= 1.0


def test_negative_control_an_even_face_without_noise_has_no_edges() -> None:
    lum, mask = disc(400, "even", noise=0.0)
    assert not edge_face(lum, mask, 1.0, 8.0).edges.any()
