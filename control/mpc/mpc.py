import math
import numpy as np

from base import Control
from core import Waypoint

from .model import Model
from dataclasses import dataclass
from scipy.optimize import minimize, OptimizeResult


@dataclass
class MPCParams:
    horizon: int = 5
    Q: np.ndarray = np.diag([1.0, 1.0])
    R: np.ndarray = np.diag([0.01, 0.01])
    Rd: np.ndarray = np.diag([0.01, 1.0])

    v_max: float = 10
    w_max: float = math.pi
    threshold: float = 15


class MPC(Control):
    def __init__(self, params: MPCParams = MPCParams()):
        super().__init__()
        self.params = params

    def update(
        self,
        state: np.ndarray,
        waypoints: list[Waypoint],
        index: int,
        dt: float = 0.5
    ) -> tuple[float, float, int]:
        bounds: list = [
            (-self.params.v_max, self.params.v_max),
            (-self.params.w_max, self.params.w_max)
        ] * self.params.horizon

        target = (waypoints[index].x, waypoints[index].y)
        pose = tuple(state[:2].flatten())
        
        ctrl: OptimizeResult = minimize(
            self._cost,
            np.zeros((self.params.horizon, 2)),
            args=(state, np.array(target), dt),
            method="SLSQP",
            bounds=bounds
        )
        
        ctrl_v, ctrl_w = ctrl.x[0], ctrl.x[1]
        new_index = index + 1 if self._distance(pose, target) < self.params.threshold else index

        return ctrl_v, ctrl_w, new_index

    def _cost(
        self,
        x0: np.ndarray,
        state: np.ndarray,
        target: np.ndarray,
        dt: float
    ) -> float:
        u_k = x0.reshape(self.params.horizon, 2)
        x_k = Model.rollout(state, u_k, dt)

        # input cost
        cost = np.sum(self.params.R @ (u_k**2))

        # input difference cost
        diff = np.diff(u_k, axis=1)
        cost += np.sum(self.params.Rd @ (diff**2))

        # state cost
        err = target[:, np.newaxis] - x_k[:,:-1]
        cost += np.sum(self.params.Q @ (err**2))

        return float(cost)
