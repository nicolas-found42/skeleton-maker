# SPDX-License-Identifier: MIT
"""Shared camera-intrinsic validation and depth backprojection."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class IntrinsicsError(ValueError):
    """Camera intrinsics cannot define a stable pinhole projection."""


def validate_intrinsics(value: ArrayLike, where: str) -> NDArray[np.float64]:
    """Return normalized finite invertible intrinsics or a pointed validation error."""
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise IntrinsicsError(f"{where} must be a finite 3x3 matrix") from exc
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise IntrinsicsError(f"{where} must be a finite 3x3 matrix")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise IntrinsicsError(f"{where} focal lengths must be positive")
    if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-9, rtol=0.0):
        raise IntrinsicsError(f"{where} last row must be [0, 0, 1]")
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > 1e12:
        raise IntrinsicsError(f"{where} must be invertible and well-conditioned")
    return matrix


def backproject(
    intrinsics: ArrayLike, u: ArrayLike, v: ArrayLike, depth: ArrayLike
) -> NDArray[np.float64]:
    """Backproject pixel coordinates, preserving the depth value as camera-space Z."""
    matrix = validate_intrinsics(intrinsics, "depth intrinsics")
    u_values, v_values, z_values = np.broadcast_arrays(u, v, depth)
    pixels = np.stack([u_values.reshape(-1), v_values.reshape(-1), np.ones(u_values.size)], axis=0)
    rays = np.linalg.solve(matrix, pixels)
    if not np.isfinite(rays).all() or np.any(np.abs(rays[2]) <= 1e-12):
        raise IntrinsicsError("depth intrinsics produce an invalid camera ray")
    rays /= rays[2]
    points = (rays * z_values.reshape(1, -1)).T
    if not np.isfinite(points).all():
        raise IntrinsicsError("depth intrinsics produce non-finite camera points")
    return points.reshape((*u_values.shape, 3))
