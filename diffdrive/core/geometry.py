"""Planar geometry helpers."""

from __future__ import annotations

import math
from typing import overload

import numpy as np

from diffdrive.core.types import Array


@overload
def wrap_angle(a: float) -> float: ...
@overload
def wrap_angle(a: Array) -> Array: ...
def wrap_angle(a):
    """Wrap an angle (or array of angles) to ``[-π, π)``."""
    if isinstance(a, np.ndarray):
        return (a + np.pi) % (2.0 * np.pi) - np.pi
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rotation(theta: float) -> Array:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def world_to_body(pose: Array, points: Array) -> Array:
    """Express world points ``(..., 2)`` in the body frame of ``pose = [x, y, θ]``."""
    d = np.asarray(points, dtype=float) - pose[:2]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], axis=-1)


def body_to_world(pose: Array, points: Array) -> Array:
    """Map body-frame points ``(..., 2)`` of ``pose`` into the world frame."""
    p = np.asarray(points, dtype=float)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.stack(
        [pose[0] + c * p[..., 0] - s * p[..., 1], pose[1] + s * p[..., 0] + c * p[..., 1]], axis=-1
    )


def body_frame_error(pose: Array, ref: Array) -> Array:
    """Tracking error ``[e_x, e_y, e_θ]`` of ``ref`` expressed in the robot body frame.

    Convention (docs/differential_drive_control_benchmark.md §1.4)::

        [e_x, e_y] = R(θ)ᵀ (p_r - p),   e_θ = wrap(θ_r - θ)

    ``e_x > 0`` means the reference is ahead, ``e_y > 0`` means it is to the left.
    Works on single poses ``(3,)`` or batches ``(N, 3)``.
    """
    pose = np.asarray(pose, dtype=float)
    ref = np.asarray(ref, dtype=float)
    dx = ref[..., 0] - pose[..., 0]
    dy = ref[..., 1] - pose[..., 1]
    c, s = np.cos(pose[..., 2]), np.sin(pose[..., 2])
    return np.stack(
        [c * dx + s * dy, -s * dx + c * dy, wrap_angle(ref[..., 2] - pose[..., 2])], axis=-1
    )


def compose_body_offset(pose: Array, dx: float, dy: float, dtheta: float) -> Array:
    """Apply an offset expressed in the body frame of ``pose``."""
    x, y = body_to_world(pose, np.array([dx, dy]))
    return np.array([x, y, wrap_angle(pose[2] + dtheta)])
