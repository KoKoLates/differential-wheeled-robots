import math

from typing import Optional
from dataclasses import dataclass


@dataclass
class Waypoint(object):
    x: float
    y: float
    v: Optional[float] = 0.0
    heading: Optional[float] = 0.0


def load_map(file_path: str) -> list[Waypoint]:
    waypoints: list[Waypoint] = []

    with open(file_path, 'r') as file:
        for line in file:
            x, y = map(float, line.strip().split(", "))
            waypoints.append(Waypoint(x, y))

    for i in range(len(waypoints) - 1):
        dx = waypoints[i+1].x - waypoints[i].x
        dy = waypoints[i+1].y - waypoints[i].y
        waypoints[i].heading = math.atan2(dy, dx)
    waypoints[-1].heading = waypoints[-2].heading
    
    return waypoints
