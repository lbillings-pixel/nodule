"""Typed wrappers around the scipy functions nodule uses that agatesim's own
wrappers (`agatesim._scipy`) do not cover.

scipy ships incomplete type information, so calls are isolated here and the
rest of the package stays under strict type checking.
"""
# pyright: basic

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.spatial import ConvexHull, KDTree


def hull_2d(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The convex hull of 2D points: its vertices (k, 2) in counter-clockwise
    order, and its edges as lines n·p + c ≤ 0 inside, rows (nx, ny, c) with unit
    outward normals n."""
    hull = ConvexHull(points)
    vertices = np.asarray(points[hull.vertices], dtype=np.float64)
    return vertices, np.asarray(hull.equations, dtype=np.float64)


def nearest(points: NDArray[np.float64], queries: NDArray[np.float64]) -> NDArray[np.intp]:
    """Index of the nearest of `points` (n, 2) to each of `queries` (m, 2)."""
    _, index = KDTree(points).query(queries)
    return np.asarray(index, dtype=np.intp)


def hole_free_patches(mask: NDArray[np.bool_]) -> int:
    """How many connected patches of `mask` enclose no hole (discs, not rings)."""
    found, n = cast("tuple[NDArray[np.int32], int]", ndimage.label(mask))
    patches = [found == i for i in range(1, n + 1)]
    return sum(
        int(cast("NDArray[np.bool_]", ndimage.binary_fill_holes(p)).sum() == p.sum())
        for p in patches
    )
