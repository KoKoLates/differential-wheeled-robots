import math
import numpy as np

from control.base import Control
from core import Waypoint

from control.mpc import Model
from dataclasses import dataclass
from scipy.optimize import minimize, OptimizeResult


@dataclass
class MPCParams:
    horizon: int = 10
    Q: np.ndarray = np.diag([10.0, 5.0])
    R: np.ndarray = np.diag([0.5, 0.5])
    Rd: np.ndarray = np.diag([0.5, 1.0])

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
            (0, self.params.v_max),
            (-self.params.w_max, self.params.w_max)
        ] * self.params.horizon

        target = (waypoints[index].x, waypoints[index].y)
        pose = tuple(state[:2].flatten())
        
        ctrl: OptimizeResult = minimize(
            fun=self._cost,
            x0=np.tile([1.0, 0.0], self.params.horizon),
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

        r  = np.diag(self.params.R)
        rd = np.diag(self.params.Rd)
        q  = np.diag(self.params.Q)

        cost = np.sum((u_k ** 2) * r)

        diff = np.diff(u_k, axis=0)
        cost += np.sum((diff ** 2) * rd)

        err = target[:, np.newaxis] - x_k[:2, :-1]
        cost += np.sum((err.T ** 2) * q)

        return float(cost)
    
    def reset(self) -> None:
        pass
