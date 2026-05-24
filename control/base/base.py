import math
import numpy as np

from core import Waypoint
from abc import ABC, abstractmethod


class Control(ABC):
    @abstractmethod
    def update(
        self,
        state: np.ndarray, 
        waypoints: list[Waypoint], 
        index: int,
        dt: float = 0.5
    ) -> tuple[float, float, int]:
        """
        @param state: robot state `(x, y, theta)`
        @param waypoints: list of waypoints
        @param index: index of current target in waypoints list
        @return: `(v, w, new_idx)`
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    @staticmethod
    def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        dx, dy = p1[0] - p2[0], p1[1] - p2[1]
        return math.sqrt(dx ** 2 + dy ** 2)

    @staticmethod
    def _bearing(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))
