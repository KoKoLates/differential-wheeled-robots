"""Geometric paths parameterised by arclength, and monotonic progress tracking."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from diffdrive.core.geometry import wrap_angle
from diffdrive.core.types import Array

DEFAULT_DS = 0.005  # resampling resolution [m]


class Path:
    """A planar curve ``p(s) = [x(s), y(s)]`` sampled on a uniform arclength grid.

    Stores heading (unwrapped, so it can be interpolated) and signed curvature
    ``κ = dθ/ds``. A path carries no timing; see :class:`Trajectory` for that.
    """

    def __init__(self, s: Array, xy: Array, theta: Array, kappa: Array) -> None:
        self.s = np.asarray(s, dtype=float)
        self.xy = np.asarray(xy, dtype=float)
        self.theta = np.asarray(theta, dtype=float)  # unwrapped
        self.kappa = np.asarray(kappa, dtype=float)
        if not (len(self.s) == len(self.xy) == len(self.theta) == len(self.kappa)):
            raise ValueError("path arrays must have equal length")
        if len(self.s) < 2 or np.any(np.diff(self.s) <= 0):
            raise ValueError("path arclength grid must be strictly increasing with >= 2 samples")

    # ----------------------------------------------------------- constructors
    @classmethod
    def from_points(cls, points: Array, ds: float = DEFAULT_DS) -> Path:
        """Build from a *dense* polyline (consecutive duplicates are removed)."""
        pts = np.asarray(points, dtype=float)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep = np.concatenate([[True], seg > 1e-12])
        pts = pts[keep]
        if len(pts) < 2:
            raise ValueError("path needs at least two distinct points")
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        length = float(cum[-1])
        n = max(math.ceil(length / ds) + 1, 3)
        s = np.linspace(0.0, length, n)
        x = np.interp(s, cum, pts[:, 0])
        y = np.interp(s, cum, pts[:, 1])
        dx = np.gradient(x, s, edge_order=2)
        dy = np.gradient(y, s, edge_order=2)
        theta = np.unwrap(np.arctan2(dy, dx))
        kappa = np.gradient(theta, s, edge_order=2)
        return cls(s, np.column_stack([x, y]), theta, kappa)

    @classmethod
    def from_function(
        cls,
        f: Callable[[Array], tuple[Array, Array]],
        u0: float,
        u1: float,
        n: int = 40_001,
        ds: float = DEFAULT_DS,
    ) -> Path:
        """Sample a parametric curve ``u ↦ (x(u), y(u))`` densely, then reparameterise."""
        u = np.linspace(u0, u1, n)
        x, y = f(u)
        return cls.from_points(np.column_stack([x, y]), ds=ds)

    # ---------------------------------------------------------------- queries
    @property
    def length(self) -> float:
        return float(self.s[-1])

    @property
    def start_pose(self) -> Array:
        return np.array([self.xy[0, 0], self.xy[0, 1], wrap_angle(float(self.theta[0]))])

    def sample(self, s: Array | float) -> tuple[Array, Array, Array]:
        """Return ``(xy (n,2), theta_unwrapped (n,), kappa (n,))`` at arclengths ``s``.

        ``s`` is clamped to ``[0, length]``.
        """
        s = np.clip(np.atleast_1d(np.asarray(s, dtype=float)), 0.0, self.length)
        x = np.interp(s, self.s, self.xy[:, 0])
        y = np.interp(s, self.s, self.xy[:, 1])
        th = np.interp(s, self.s, self.theta)
        k = np.interp(s, self.s, self.kappa)
        return np.column_stack([x, y]), th, k

    def project(self, point: Array, s_lo: float = 0.0, s_hi: float | None = None) -> float:
        """Arclength of the point on the path closest to ``point`` within ``[s_lo, s_hi]``."""
        s_hi = self.length if s_hi is None else s_hi
        lo = int(np.searchsorted(self.s, max(s_lo, 0.0), side="left"))
        hi = int(np.searchsorted(self.s, min(s_hi, self.length), side="right"))
        lo = min(max(lo - 1, 0), len(self.s) - 2)
        hi = max(hi, lo + 2)
        seg_a = self.xy[lo : hi - 1]
        seg_b = self.xy[lo + 1 : hi]
        ab = seg_b - seg_a
        ap = np.asarray(point, dtype=float)[:2] - seg_a
        denom = np.einsum("ij,ij->i", ab, ab)
        t = np.clip(np.einsum("ij,ij->i", ap, ab) / np.where(denom > 0, denom, 1.0), 0.0, 1.0)
        closest = seg_a + t[:, None] * ab
        d2 = np.sum((closest - point[:2]) ** 2, axis=1)
        i = int(np.argmin(d2))
        s = self.s[lo + i] + t[i] * (self.s[lo + i + 1] - self.s[lo + i])
        return float(np.clip(s, max(s_lo, 0.0), min(s_hi, self.length)))

    def signed_lateral_error(self, point: Array, s: float) -> float:
        """Signed distance of ``point`` from the path at ``s`` (positive = left of path)."""
        xy, th, _ = self.sample(s)
        d = np.asarray(point[:2]) - xy[0]
        return float(-math.sin(th[0]) * d[0] + math.cos(th[0]) * d[1])

    # ---------------------------------------------------------- transforms
    def transformed(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> Path:
        """Apply a rigid transform (rotate by ``theta`` about the origin, then translate)."""
        c, s_ = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s_], [s_, c]])
        xy = self.xy @ R.T + np.array([x, y])
        return Path(self.s.copy(), xy, self.theta + theta, self.kappa.copy())

    def to_arrays(self) -> dict[str, Array]:
        return {"s": self.s, "xy": self.xy, "theta": self.theta, "kappa": self.kappa}

    @classmethod
    def from_arrays(cls, d: dict[str, Array]) -> Path:
        return cls(d["s"], d["xy"], d["theta"], d["kappa"])


class ProgressTracker:
    """Monotonic arclength progress along a path.

    Each update searches only a forward window ``[s, s + max_advance]`` (plus an
    optional small backtrack), so the estimate cannot jump across a self-
    intersection such as the centre of a figure-eight (doc §2.2, §5.2).
    """

    def __init__(self, path: Path, backtrack: float = 0.0, min_window: float = 0.5) -> None:
        self.path = path
        self.backtrack = backtrack
        self.min_window = min_window
        self.s = 0.0

    def reset(self, s0: float = 0.0) -> None:
        self.s = float(s0)

    def update(self, point: Array, max_advance: float = 0.0) -> float:
        window = max(self.min_window, max_advance)
        s_new = self.path.project(point, self.s - self.backtrack, self.s + window)
        self.s = max(self.s, s_new)
        return self.s
