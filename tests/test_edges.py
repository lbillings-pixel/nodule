"""The band edge map (`nodule.edges`)."""

from __future__ import annotations

import numpy as np
import pytest

from nodule.edges import EDGE_PX, edge_face, noise_gradient

NOISE = 1.5  # grey levels of the test faces' pixel noise, as the gate would report


def disc(n: int, kind: str, spacing: float = 16.0, noise: float = NOISE, amp: float = 40.0):
    y, x = np.mgrid[:n, :n].astype(np.float64)
    c = (n - 1) / 2
    r = np.hypot(y - c, x - c)
    mask = r < 0.45 * n
    if kind == "rings":
        lum = 128 + amp * np.cos(2 * np.pi * r / spacing)
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
    ef = edge_face(lum, mask, 1.0, NOISE, spacing / 2)
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
    # The noise gate leaves no edges on the flat ground either side.
    lum, mask = disc(400, "step")
    ef = edge_face(lum, mask, 1.0, NOISE, 8.0)
    h, w = ef.edges.shape
    middle = (w - 1) / 2
    for row in ef.edges[h // 4 : 3 * h // 4]:
        cols = np.flatnonzero(row)
        assert np.abs(cols - middle).min() <= 1.0
    cols = np.nonzero(ef.edges)[1]
    assert np.abs(cols - middle).max() <= 1.5


def test_negative_control_an_even_face_without_noise_has_no_edges() -> None:
    lum, mask = disc(400, "even", noise=0.0)
    assert not edge_face(lum, mask, 1.0, 0.0, 8.0).edges.any()


@pytest.mark.parametrize(
    ("n", "spacing", "noise"), [(400, 16.0, 1.5), (800, 32.0, 1.5), (400, 16.0, 5.0)]
)
def test_negative_control_an_even_face_with_pixel_noise_has_no_edges(
    n: int, spacing: float, noise: float
) -> None:
    lum, mask = disc(n, "even", spacing, noise)
    # Without the gate, the noise is all short edges (module docstring).
    assert edge_face(lum, mask, 1.0, 0.0, spacing / 2).edges.sum() > 1000
    assert not edge_face(lum, mask, 1.0, noise, spacing / 2).edges.any()


def test_faint_rings_twice_the_noise_keep_their_edges() -> None:
    lum, mask = disc(400, "rings", noise=NOISE, amp=2 * NOISE)
    gated = edge_face(lum, mask, 1.0, NOISE, 8.0).edges
    bare = edge_face(lum, mask, 1.0, 0.0, 8.0).edges
    clean = edge_face(*disc(400, "rings", noise=0.0, amp=2 * NOISE), 1.0, 0.0, 8.0).edges
    assert gated.sum() >= 0.9 * clean.sum()
    assert gated.sum() <= bare.sum()


def test_the_noise_floor_follows_the_noise_and_the_block_size() -> None:
    # Twice the px per spacing averages blocks of twice the side: half the noise.
    one = noise_gradient(1.0, 16.0)
    assert noise_gradient(3.0, 16.0) == pytest.approx(3 * one)
    assert noise_gradient(1.0, 32.0) == pytest.approx(one / 2, rel=0.05)
    assert noise_gradient(0.0, 16.0) == 0.0
