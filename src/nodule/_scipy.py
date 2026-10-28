"""Typed wrappers around the scipy functions nodule uses that agatesim's own
wrappers (`agatesim._scipy`) do not cover.

scipy ships incomplete type information, so calls are isolated here and the
rest of the package stays under strict type checking.
"""
# pyright: basic

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull


def hull_2d(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The convex hull of 2D points: its vertices (k, 2) in counter-clockwise
    order, and its edges as lines n·p + c ≤ 0 inside, rows (nx, ny, c) with unit
    outward normals n."""
    hull = ConvexHull(points)
    vertices = np.asarray(points[hull.vertices], dtype=np.float64)
    return vertices, np.asarray(hull.equations, dtype=np.float64)
