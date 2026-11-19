"""Data types shared by every layer.

Conventions (see docs/architecture.md):
    * SI units: metres, seconds, radians.
    * World frame is right-handed, x forward / y left, heading θ measured CCW from +x.
    * Pose arrays are ``[x, y, θ]`` and control arrays are ``[v, ω]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    theta: float

    def as_array(self) -> Array:
        return np.array([self.x, self.y, self.theta], dtype=float)

    @classmethod
    def from_array(cls, a: Array) -> Pose2D:
        return cls(float(a[0]), float(a[1]), float(a[2]))


@dataclass(frozen=True)
class Control:
    v: float
    omega: float

    def as_array(self) -> Array:
        return np.array([self.v, self.omega], dtype=float)

    @classmethod
    def from_array(cls, a: Array) -> Control:
        return cls(float(a[0]), float(a[1]))

    @classmethod
    def zero(cls) -> Control:
        return cls(0.0, 0.0)


@dataclass(frozen=True)
class Observation:
    """What a controller is allowed to know at a control tick.

    ``pose`` is a *measurement* (possibly noisy, biased and delayed); it is never
    the simulator's ground truth unless the scenario's sensor is ideal.

    Attributes:
        pose: estimated pose.
        t: current simulation time (when the controller is running).
        t_sample: time at which ``pose`` was sampled; ``t - t_sample`` is its age.
        last_command: the last command this controller sent (after the limiter).
            This is always legitimately known by the controller; it is *not* the
            applied velocity, which may differ under latency or actuator lag.
    """

    pose: Pose2D
    t: float
    t_sample: float
    last_command: Control

    @property
    def age(self) -> float:
        return self.t - self.t_sample


RefMode = Literal["time", "path"]


@dataclass(frozen=True)
class RefRequest:
    """How much reference preview a controller asks for, and how it is indexed.

    * ``mode="time"``: samples at ``t, t+dt, ..., t+H·dt`` of the timed trajectory
      (LQR / MPC / iLQR style trajectory tracking).
    * ``mode="path"``: samples start at the estimated path progress ``s₀`` and are
      spaced by ``ds`` metres (Pure Pursuit / path following). Timing is free.

    ``dt``/``ds`` default to the control period / 5 cm. The request is logged so
    that preview budgets are visible when comparing controllers.
    """

    mode: RefMode = "time"
    horizon: int = 0
    dt: float | None = None
    ds: float | None = None


@dataclass(frozen=True)
class ReferenceWindow:
    """A slice of the reference handed to a controller.

    Shapes: ``times``/``s`` are ``(H+1,)``, ``poses`` ``(H+1, 3)``, ``controls``
    ``(H+1, 2)``. Index 0 is "now" (time mode) or the current progress (path
    mode). ``controls`` is the feasible feed-forward ``[v_r, ω_r]``. ``curvature``
    is the path curvature at each sample.
    """

    mode: RefMode
    times: Array
    s: Array
    poses: Array
    controls: Array
    curvature: Array
    path_length: float
    finished: bool = False

    @property
    def horizon(self) -> int:
        return len(self.times) - 1

    @property
    def pose(self) -> Pose2D:
        return Pose2D.from_array(self.poses[0])

    @property
    def control(self) -> Control:
        return Control.from_array(self.controls[0])


class Status(str, Enum):
    OK = "ok"
    FALLBACK = "fallback"  # solver failed, but the controller produced a safe fallback
    FAILED = "failed"  # no usable command; the runner will still apply what was returned


@dataclass
class ControlResult:
    """What a controller returns.

    ``predicted_states`` ``(N+1, 3)`` and ``predicted_controls`` ``(N, 2)`` are
    optional; if provided they are logged and drawn by the visualiser.
    ``diagnostics`` holds scalar extras (iterations, cost, ...). Solve time is
    measured by the runner, so controllers do not need to time themselves.
    """

    command: Control
    status: Status = Status.OK
    predicted_states: Array | None = None
    predicted_controls: Array | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)
