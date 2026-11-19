"""Time-indexed reference trajectories built from a path plus a speed profile."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from diffdrive.core.geometry import wrap_angle
from diffdrive.core.limits import Limits
from diffdrive.core.types import Array, ReferenceWindow
from diffdrive.references.path import Path


@dataclass(frozen=True)
class SpeedProfile:
    """Rules for assigning a speed to every point of a path.

    The resulting profile respects: cruise speed, ``limits.v_max``, a curvature
    cap so that ``|ω_r| = v|κ| ≤ omega_fraction · ω_max`` (leaving control
    authority for feedback), optional lateral acceleration, and longitudinal
    accel/decel (forward–backward pass). ``stops`` are path fractions where the
    reference comes to rest for ``stop_duration`` seconds (start–stop scenario).
    """

    cruise: float = 0.5
    accel: float = 0.5
    decel: float | None = None
    start_speed: float = 0.0
    end_speed: float = 0.0
    lateral_accel: float | None = None
    omega_fraction: float = 0.8
    stops: Sequence[float] = field(default_factory=tuple)
    stop_duration: float = 0.0


def compute_speed(path: Path, profile: SpeedProfile, limits: Limits) -> Array:
    """Speed ``v(s)`` on ``path.s`` satisfying the profile's constraints."""
    if profile.cruise <= 0 or profile.accel <= 0:
        raise ValueError("cruise speed and accel must be positive")
    v_cap = min(profile.cruise, limits.v_max)
    k = np.abs(path.kappa)
    with np.errstate(divide="ignore"):
        cap = np.full_like(k, v_cap)
        w_cap = profile.omega_fraction * limits.omega_max
        cap = np.minimum(cap, np.where(k > 1e-9, w_cap / k, np.inf))
        if profile.lateral_accel is not None:
            cap = np.minimum(cap, np.where(k > 1e-9, np.sqrt(profile.lateral_accel / k), np.inf))
    if limits.accel_max is not None:
        acc = min(profile.accel, limits.accel_max)
    else:
        acc = profile.accel
    dec = acc if profile.decel is None else profile.decel

    v = cap.copy()
    v[0] = min(v[0], profile.start_speed)
    v[-1] = min(v[-1], profile.end_speed)
    for frac in profile.stops:
        v[int(np.clip(round(frac * (len(v) - 1)), 0, len(v) - 1))] = 0.0
    ds = np.diff(path.s)
    for i in range(1, len(v)):  # forward pass (acceleration)
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * acc * ds[i - 1]))
    for i in range(len(v) - 2, -1, -1):  # backward pass (deceleration)
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * dec * ds[i]))
    return v


class Trajectory:
    """A timed reference ``q_r(t), u_r(t)`` with its underlying :class:`Path`.

    Internally stores a grid ``(t_i, s_i, v_i)``; repeated ``s`` values with
    increasing ``t`` encode dwell (stops). Everything else (pose, ω = vκ) is
    derived from the path, so ``q_r`` and ``u_r`` are consistent by construction.
    """

    def __init__(self, path: Path, t: Array, s: Array, v: Array) -> None:
        self.path = path
        self.t = np.asarray(t, dtype=float)
        self.s = np.asarray(s, dtype=float)
        self.v = np.asarray(v, dtype=float)
        if np.any(np.diff(self.t) <= 0):
            raise ValueError("trajectory time grid must be strictly increasing")

    @classmethod
    def from_path(cls, path: Path, profile: SpeedProfile, limits: Limits) -> Trajectory:
        v = compute_speed(path, profile, limits)
        ds = np.diff(path.s)
        vsum = v[1:] + v[:-1]
        if np.any(vsum <= 0):
            raise ValueError("speed profile has a zero-speed segment; cannot time-parameterise")
        dt = 2.0 * ds / vsum
        t = np.concatenate([[0.0], np.cumsum(dt)])
        s = path.s.copy()
        if profile.stops and profile.stop_duration > 0:
            stop_idx = np.where(
                (v == 0.0) & (np.arange(len(v)) > 0) & (np.arange(len(v)) < len(v) - 1)
            )[0]
            for j, i in enumerate(sorted(stop_idx)):
                i = i + j  # account for inserted samples
                t = np.insert(t, i + 1, t[i] + profile.stop_duration)
                t[i + 2 :] += profile.stop_duration
                s = np.insert(s, i + 1, s[i])
                v = np.insert(v, i + 1, 0.0)
        return cls(path, t, s, v)

    # -------------------------------------------------------------- queries
    @property
    def duration(self) -> float:
        return float(self.t[-1])

    @property
    def length(self) -> float:
        return self.path.length

    def _from_s(self, s: Array, v: Array) -> tuple[Array, Array, Array]:
        xy, th, k = self.path.sample(s)
        poses = np.column_stack([xy, wrap_angle(th)])
        controls = np.column_stack([v, v * k])
        return poses, controls, k

    def _s_v_at(self, times: Array) -> tuple[Array, Array]:
        """Arclength and speed at ``times`` assuming constant acceleration per grid cell.

        The grid timing ``Δt = 2Δs/(v_i + v_{i+1})`` is exactly the constant-
        acceleration time, so this makes ``ds/dt == v`` hold exactly (plain linear
        interpolation of ``s(t)`` would not, e.g. when starting from rest).
        After the end the final pose is held with zero velocity.
        """
        times = np.atleast_1d(np.asarray(times, dtype=float))
        i = np.clip(np.searchsorted(self.t, times, side="right") - 1, 0, len(self.t) - 2)
        tau = np.clip(times - self.t[i], 0.0, None)
        cell = self.t[i + 1] - self.t[i]
        a = (self.v[i + 1] - self.v[i]) / cell
        tau = np.minimum(tau, cell)
        s = self.s[i] + self.v[i] * tau + 0.5 * a * tau**2
        v = self.v[i] + a * tau
        after = times >= self.t[-1]
        s = np.where(after, self.s[-1], s)
        v = np.where(after, 0.0, v)
        before = times < self.t[0]
        s = np.where(before, self.s[0], s)
        v = np.where(before, 0.0, v)
        return s, v

    def sample(self, times: Array | float) -> tuple[Array, Array, Array]:
        """Reference at ``times``: ``(poses (n,3), controls (n,2), s (n,))``."""
        s, v = self._s_v_at(np.atleast_1d(np.asarray(times, dtype=float)))
        poses, controls, _ = self._from_s(s, v)
        return poses, controls, s

    def time_at(self, s: Array | float) -> Array:
        """First time the reference reaches arclength ``s`` (well-defined at stops)."""
        s = np.clip(np.atleast_1d(np.asarray(s, dtype=float)), self.s[0], self.s[-1])
        j = np.clip(np.searchsorted(self.s, s, side="left"), 1, len(self.s) - 1)
        s0, s1 = self.s[j - 1], self.s[j]
        frac = np.where(s1 > s0, (s - s0) / np.where(s1 > s0, s1 - s0, 1.0), 0.0)
        return self.t[j - 1] + frac * (self.t[j] - self.t[j - 1])

    def speed_at(self, s: Array | float) -> Array:
        return np.interp(np.atleast_1d(np.asarray(s, dtype=float)), self.s, self.v)

    # -------------------------------------------------------------- windows
    def window_time(self, t0: float, horizon: int, dt: float) -> ReferenceWindow:
        times = t0 + dt * np.arange(horizon + 1)
        s, v = self._s_v_at(times)
        poses, controls, k = self._from_s(s, v)
        return ReferenceWindow(
            mode="time",
            times=times,
            s=s,
            poses=poses,
            controls=controls,
            curvature=k,
            path_length=self.length,
            finished=bool(t0 >= self.duration),
        )

    def window_path(self, s0: float, horizon: int, ds: float) -> ReferenceWindow:
        s = np.minimum(s0 + ds * np.arange(horizon + 1), self.length)
        v = self.speed_at(s)
        poses, controls, k = self._from_s(s, v)
        return ReferenceWindow(
            mode="path",
            times=self.time_at(s),
            s=s,
            poses=poses,
            controls=controls,
            curvature=k,
            path_length=self.length,
            finished=bool(s0 >= self.length - 1e-6),
        )

    # ---------------------------------------------------------- persistence
    def to_arrays(self) -> dict[str, Array]:
        d = {f"path_{k}": v for k, v in self.path.to_arrays().items()}
        d.update({"t": self.t, "s": self.s, "v": self.v})
        return d

    @classmethod
    def from_arrays(cls, d: dict[str, Array]) -> Trajectory:
        path = Path.from_arrays({k[5:]: v for k, v in d.items() if k.startswith("path_")})
        return cls(path, d["t"], d["s"], d["v"])


@dataclass(frozen=True)
class FeasibilityReport:
    max_v: float
    min_v: float
    max_abs_omega: float
    max_accel: float
    max_alpha: float
    consistency_error: float  # max |finite-difference pose rate - f(q_r, u_r)|
    violations: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.violations


def check_feasibility(traj: Trajectory, limits: Limits, dt: float = 0.01) -> FeasibilityReport:
    """Verify the reference respects the command limits and is kinematically consistent."""
    times = np.arange(0.0, traj.duration, dt)
    poses, u, _ = traj.sample(times)
    acc = np.abs(np.diff(u[:, 0])) / dt
    alpha = np.abs(np.diff(u[:, 1])) / dt
    th = np.unwrap(poses[:, 2])
    rate = np.column_stack([np.diff(poses[:, 0]), np.diff(poses[:, 1]), np.diff(th)]) / dt
    mid_th = 0.5 * (th[1:] + th[:-1])
    mid_u = 0.5 * (u[1:] + u[:-1])
    model = np.column_stack(
        [mid_u[:, 0] * np.cos(mid_th), mid_u[:, 0] * np.sin(mid_th), mid_u[:, 1]]
    )
    cons = float(np.max(np.abs(rate - model))) if len(rate) else 0.0
    tol = 1e-6
    viol = []
    if u[:, 0].max() > limits.v_max + tol:
        viol.append(f"v_r exceeds v_max ({u[:, 0].max():.3f} > {limits.v_max})")
    if u[:, 0].min() < limits.v_min - tol:
        viol.append(f"v_r below v_min ({u[:, 0].min():.3f} < {limits.v_min})")
    if np.abs(u[:, 1]).max() > limits.omega_max + tol:
        viol.append(f"|ω_r| exceeds ω_max ({np.abs(u[:, 1]).max():.3f} > {limits.omega_max})")
    return FeasibilityReport(
        max_v=float(u[:, 0].max()),
        min_v=float(u[:, 0].min()),
        max_abs_omega=float(np.abs(u[:, 1]).max()),
        max_accel=float(acc.max()) if len(acc) else 0.0,
        max_alpha=float(alpha.max()) if len(alpha) else 0.0,
        consistency_error=cons,
        violations=tuple(viol),
    )
