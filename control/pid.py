import math
import numpy as np

from .base import Control

from core import Waypoint
from dataclasses import dataclass, field

@dataclass
class PIDGains:
    Kp: float = 0.1
    Ki: float = 0.0
    Kd: float = 0.1


@dataclass
class PIDParams:
    v: PIDGains = field(default_factory=PIDGains)
    w: PIDGains = field(default_factory=PIDGains)
    
    v_max: float = 10
    w_max: float = math.pi
    threshold: float = 0.1


class PID(Control):
    def __init__(self, params: PIDParams = PIDParams()) -> None:
        super().__init__()
        self.params = params
        self.reset()

    def update(
        self,
        state: np.ndarray,
        waypoints: list[Waypoint],
        index: int
    ) -> tuple[float, float, int]:
        target = (waypoints[index].x, waypoints[index].y)
        pose = tuple(state[:2].flatten())
        head = float(state[2, 0])

        err_v = self._distance(pose, target)
        err_w = self._wrap_angle(-self._bearing(pose, target) - head)

        self.hist_err_v += err_v
        self.hist_err_w += err_w

        ctrl_v = (
            self.params.v.Kp * err_v
            + self.params.v.Ki * self.hist_err_v
            + self.params.v.Kd * (err_v - self.prev_err_v)
        )
        ctrl_w = (
            self.params.w.Kp * err_w
            + self.params.w.Ki * self.hist_err_w
            + self.params.w.Kd * (err_w - self.prev_err_w)
        )

        self.prev_err_v, self.prev_err_w = err_v, err_w

        ctrl_v = float(np.clip(ctrl_v, -self.params.v_max, self.params.v_max))
        ctrl_w = float(np.clip(ctrl_w, -self.params.w_max, self.params.w_max))

        new_index = index + 1 if err_v < self.params.threshold else index
        return ctrl_v, ctrl_w, new_index

    def reset(self) -> None:
        self.prev_err_v: float = 0.0
        self.prev_err_w: float = 0.0
        self.hist_err_v: float = 0.0
        self.hist_err_w: float = 0.0
