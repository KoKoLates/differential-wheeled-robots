import math
import numpy as np

from .base import Control

from core import Waypoint
from dataclasses import dataclass, field
from scipy.linalg import solve_discrete_are


@dataclass
class LQRParams:
    Q: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 0.5]))
    R: np.ndarray = field(default_factory=lambda: np.diag([0.1, 0.1]))
    v_ref: float  = 2.0
    v_max: float  = 10.0
    w_max: float  = math.pi
    threshold: float   = 15.0
    heading_tol: float = 0.1


class LQR(Control):
    def __init__(self, params: LQRParams = LQRParams()):
        super().__init__()
        self.params = params
        self.last_heading: float = float('inf')
        self.K: np.ndarray = None

    def update(
        self, 
        state: np.ndarray, 
        waypoints: list[Waypoint], 
        index: int,
        dt: float = 0.5
    ) -> tuple[float, float, int]:
        target: Waypoint = waypoints[index]

        if self.K is None or \
            abs(self._wrap_angle(target.heading - self.last_heading)) > self.params.heading_tol:
            self.K = self._compute_gain(target.heading, dt)
            self.last_heading = target.heading

        x_ref = np.array((target.x, target.y, target.heading)).reshape(3, 1)
        du = -self.K @ (state - x_ref)
        
        ctrl_v = float(np.clip(self.params.v_ref + du[0, 0], -self.params.v_max, self.params.v_max))
        ctrl_w = float(np.clip(du[1, 0], -self.params.w_max, self.params.w_max))

        pose = tuple(state[:2].flatten())
        d = self._distance(pose, (target.x, target.y))
        new_index = index + 1 if d < self.params.threshold else index

        return ctrl_v, ctrl_w, new_index

    def reset(self) -> None:
        self.last_heading = float('inf')
        self.K = None

    def _compute_gain(self, heading: float, dt: float) -> np.ndarray:
        A: np.ndarray = np.array([
            [0, 0, -self.params.v_ref * np.sin(heading)],
            [0, 0, -self.params.v_ref * np.cos(heading)],
            [0, 0, 0]
        ])
        B: np.ndarray = np.array([
            [np.cos(heading), 0], [-np.sin(heading), 0], [0, 1]
        ])

        Ad = np.eye(3) + A * dt
        Bd = B * dt

        P = solve_discrete_are(Ad, Bd, self.params.Q, self.params.R)
        return np.linalg.inv(self.params.R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
