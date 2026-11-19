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
from scipy import ndimage, special
from scipy.spatial import ConvexHull, KDTree


def hull_2d(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The convex hull of 2D points: its vertices (k, 2) in counter-clockwise
    order, and its edges as lines n·p + c ≤ 0 inside, rows (nx, ny, c) with unit
    outward normals n."""
    hull = ConvexHull(points)
    vertices = np.asarray(points[hull.vertices], dtype=np.float64)
    return vertices, np.asarray(hull.equations, dtype=np.float64)


def hull_planes(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """The faces of the convex hull of points (n, d) as planes n·p + c ≤ 0
    inside, rows (n_1 .. n_d, c) with unit outward normals n."""
    return np.asarray(ConvexHull(points).equations, dtype=np.float64)


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


def label_patches(mask: NDArray[np.bool_]) -> tuple[NDArray[np.int32], int]:
    """4-connected patches of a 2D mask, numbered 1..n (0 outside them)."""
    return cast("tuple[NDArray[np.int32], int]", ndimage.label(mask))


def largest_patch_size(mask: NDArray[np.bool_]) -> int:
    """Pixels in the largest 4-connected patch of a 2D mask; 0 if none."""
    found, n = cast("tuple[NDArray[np.int32], int]", ndimage.label(mask))
    return int(np.bincount(found.ravel())[1:].max()) if n else 0


def riccati_bessel(
    n: NDArray[np.int_], x: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """x j_n(x) and x y_n(x) for orders n at a real argument (the Mie series)."""
    j = np.asarray(special.spherical_jn(n, x), dtype=np.float64)
    y = np.asarray(special.spherical_yn(n, x), dtype=np.float64)
    return x * j, x * y


def spherical_bessel(n: NDArray[np.int_], z: complex, derivative: bool = False,
                     kind: str = "j") -> NDArray[np.complex128]:  # fmt: skip
    """j_n(z) or y_n(z) (or their derivatives) at a complex argument; for tests."""
    f = special.spherical_jn if kind == "j" else special.spherical_yn
    return np.asarray(f(n, z, derivative=derivative), dtype=np.complex128)


def normal_cdf(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """The standard normal distribution function, elementwise (NaN stays NaN)."""
    return np.asarray(special.ndtr(z), dtype=np.float64)


def blur_noise_factor(sigma: float) -> float:
    """How much a 2D Gaussian blur of `sigma` px (`ndimage.gaussian_filter`, as
    agatesim's `gaussian_rgb`) scales the standard deviation of white pixel noise:
    the sum of squares of its 1D kernel (separable, so the 2D factor is that squared,
    square-rooted). 1 for no blur."""
    if not sigma > 0:
        return 1.0
    delta = np.zeros(2 * int(4 * sigma + 0.5) + 3)
    delta[len(delta) // 2] = 1.0
    k = ndimage.gaussian_filter1d(delta, sigma)
    return float(np.sum(k * k))


def distance_inside(mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Each pixel's distance to the nearest pixel outside `mask`; 0 outside it."""
    return cast(NDArray[np.float64], ndimage.distance_transform_edt(mask))


def largest_component_3d(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """The largest face-connected component of a 3D mask (all False if empty)."""
    found, n = cast("tuple[NDArray[np.int32], int]", ndimage.label(mask))
    if n <= 1:
        return mask.copy()
    sizes = np.bincount(found.ravel())
    sizes[0] = 0
    return found == int(np.argmax(sizes))


def nearest_within(
    points: NDArray[np.float64], queries: NDArray[np.float64], bound: float
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Distance to, and index of, the nearest of `points` for each query; queries
    with none within `bound` get distance inf and index len(points)."""
    dist, index = KDTree(points).query(queries, distance_upper_bound=bound)
    return np.asarray(dist, dtype=np.float64), np.asarray(index, dtype=np.intp)
