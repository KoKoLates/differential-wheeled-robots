import math
import numpy as np

from .base import Control

from core import Waypoint
from dataclasses import dataclass


@dataclass
class PurePursuitParams:
    L: float     = 30.0
    v_ref: float = 2.0
    w_max: float = math.pi


class PurePursuit(Control):
    def __init__(self, params: PurePursuitParams = PurePursuitParams()) -> None:
        super().__init__()
        self.params = params

    def update(
        self,
        state: np.ndarray,
        waypoints: list[Waypoint],
        index: int,
        dt: float = 0.5
    ) -> tuple[float, float, int]:
        pose = tuple(state[:2].flatten())
        head = float(state[2, 0])

        target_idx = self._find_lookahead_index(pose, waypoints, index)
        target = (waypoints[target_idx].x, waypoints[target_idx].y)

        alpha = self._wrap_angle(-self._bearing(pose, target) - head)

        ctrl_v = self.params.v_ref
        ctrl_w = np.clip(
            2 * ctrl_v * math.sin(alpha) / self.params.L,
            -self.params.w_max, self.params.w_max
        )

        return ctrl_v, ctrl_w, target_idx
    
    def reset(self) -> None:
        pass

    def _find_lookahead_index(
        self,
        pose: tuple[float, float],
        waypoints: list[Waypoint],
        index: int
    ) -> int:
        for i in range(index, len(waypoints)):
            d = self._distance(pose, (waypoints[i].x, waypoints[i].y))
            if d >= self.params.L:
                return i
            
        return len(waypoints) - 1

    @property
    def name(self) -> str:
        return "pure_pursuit"
