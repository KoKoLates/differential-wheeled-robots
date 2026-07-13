import math
import numpy as np

from core import Waypoint
from control.base import Control
from control.mpc.model import Model
from control.mpc.lie import pose_error

from dataclasses import dataclass, field
from scipy.optimize import minimize, OptimizeResult


@dataclass
class LieMPCParams:
    horizon: int   = 10
    Q: np.ndarray  = field(default_factory=lambda: np.diag([10.0, 5.0, 1.0]))
    R: np.ndarray  = field(default_factory=lambda: np.diag([0.5, 0.5]))
    Rd: np.ndarray = field(default_factory=lambda: np.diag([0.5, 1.0]))

    v_max: float = 10
    w_max: float = math.pi
    threshold: float = 15


class LieMPC(Control):
    """ Same single-shooting nonlinear MPC as `MPC` -- same solver, same
    rollout, same effort/smoothness terms. The only real difference is
    `_cost`'s tracking term: instead of a raw `(x, y)` Euclidean difference
    against a moving position target, it uses `pose_error` (see `lie.py`)
    against the target's full `(x, y, heading)`, so heading and position
    error are computed consistently in one Lie-group operation instead of
    two unrelated ones. """

    def __init__(self, params: LieMPCParams = LieMPCParams()) -> None:
        super().__init__()
        self.params = params

    def update(
        self,
        state: np.ndarray,
        waypoints: list[Waypoint],
        index: int,
        dt: float = 0.5,
    ) -> tuple[float, float, int]:
        """ Line-for-line the same shape as `MPC.update` -- the only thing
        that's actually new in this class is `_cost` below. """
        target: Waypoint = waypoints[index]
        pose = tuple(state[:2].flatten())

        ## TODO 1: build `bounds` the same way `MPC.update` does, from
        ##   `self.params.v_max`/`w_max` and `self.params.horizon`.
        ## TODO 2: call `minimize(fun=self._cost, x0=np.tile([1.0, 0.0],
        ##   self.params.horizon), args=(state, target, dt), method="SLSQP",
        ##   bounds=bounds)`.
        ## TODO 3: `ctrl_v, ctrl_w = ctrl.x[0], ctrl.x[1]`.
        ## TODO 4: advance `index` past `self.params.threshold`, same
        ##   distance check as `MPC.update` (against `(target.x, target.y)`).
        ...

    def _cost(
        self,
        x0: np.ndarray,
        state: np.ndarray,
        target: Waypoint,
        dt: float,
    ) -> float:
        """ Same structure as `MPC._cost` (control effort + smoothness +
        tracking) -- only the tracking term changes. """
        u_k = x0.reshape(self.params.horizon, 2)
        x_k = Model.rollout(state, u_k, dt)

        ## TODO 1: control effort + smoothness terms -- copy these straight
        ##   from `MPC._cost`, they don't change:
        ##   `cost = np.sum((u_k ** 2) * np.diag(self.params.R))`
        ##   `cost += np.sum((np.diff(u_k, axis=0) ** 2) * np.diag(self.params.Rd))`
        ## TODO 2: tracking term -- for each predicted state `x_k[:, i]`
        ##   (i.e. each column except the last), compute
        ##   `e = pose_error((x_k[0,i], x_k[1,i], x_k[2,i]),
        ##                   (target.x, target.y, target.heading))`
        ##   and accumulate `e @ self.params.Q @ e` (note `Q` is now 3x3,
        ##   over x/y/theta, since the error already carries heading).
        ...

    def reset(self) -> None:
        pass
