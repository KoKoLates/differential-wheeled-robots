import os
import csv
import math
import uuid

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .robot import Robot
from .loader import Waypoint
from control.base import Control
from visualize import LiveRenderer


ColorTuple = tuple[int, int, int]
ROBOT_COLOR: ColorTuple = (20, 255, 57)


def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    dx, dy = p1[0] - p2[0], p1[1] - p2[1]
    return math.sqrt(dx ** 2 + dy ** 2)


def _compute_bounds(
    waypoints: list[Waypoint], padding: float = 50
) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = [w.x for w in waypoints]
    ys = [w.y for w in waypoints]
    xlim = (min(xs) - padding, max(xs) + padding)
    ylim = (min(ys) - padding, max(ys) + padding)
    return xlim, ylim


@dataclass
class SimulationResult:
    controller: str
    map_name: str
    timestamps: list[float]         = field(default_factory=list)
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    headings: list[float]           = field(default_factory=list)
    velocities: list[tuple[float, float]] = field(default_factory=list)
    errors: list[float]             = field(default_factory=list)
    timeout: bool                   = False

    def save(self, log_dir: str = "logs") -> str:
        map_stem = os.path.splitext(os.path.basename(self.map_name))[0]
        folder = os.path.join(log_dir, map_stem)
        os.makedirs(folder, exist_ok=True)

        uid = uuid.uuid4().hex[:4]
        filename = f"{self.controller}_{uid}.txt"
        path = os.path.join(folder, filename)

        with open(path, "w", newline="") as f:
            f.write(f"# controller: {self.controller}\n")
            f.write(f"# map: {self.map_name}\n")
            f.write(f"# date: {datetime.now().isoformat()}\n")
            f.write(f"# timeout: {self.timeout}\n")

            writer = csv.writer(f)
            writer.writerow(["t", "x", "y", "theta", "v", "w", "error"])
            for i in range(len(self.timestamps)):
                x, y = self.trajectory[i]
                v, w = self.velocities[i]
                writer.writerow([
                    round(self.timestamps[i], 4),
                    round(x, 4), round(y, 4),
                    round(self.headings[i], 4),
                    round(v, 4), round(w, 4),
                    round(self.errors[i], 4),
                ])

        print(f"Log saved: {path}")
        return path


def simulate(
    robot: Robot,
    ctrl: Control,
    waypoints: list[Waypoint],
    map_name: str = "unknown",
    dt: float = 0.5,
    visualize: bool = False,
    save_log: bool = True,
    log_dir: str = "logs",
    max_steps_per_waypoint: int = 200,
) -> SimulationResult:
    if not isinstance(ctrl, Control):
        raise TypeError(f"Expected Control subclass, got {type(ctrl).__name__}")

    result = SimulationResult(controller=ctrl.name, map_name=map_name)

    vis = None
    if visualize:
        xlim, ylim = _compute_bounds(waypoints)
        vis = LiveRenderer(title=ctrl.name, xlim=xlim, ylim=ylim)

    waypoint_path = [(w.x, w.y) for w in waypoints]
    timeout_limit = len(waypoints) * max_steps_per_waypoint
    index = 0
    total_steps = 0
    current_time = 0.0
    v, w = 0.0, 0.0

    last = waypoints[-1]
    dist_to_end = math.inf

    while index < len(waypoints) and not (index >= len(waypoints) - 1 and dist_to_end < 15):
        p, _ = robot.state
        pose = (float(p[0, 0]), float(p[1, 0]))
        theta = float(p[2, 0])

        target = waypoints[index]
        d = _distance(pose, (target.x, target.y))
        dist_to_end = _distance(pose, (last.x, last.y))

        result.trajectory.append(pose)
        result.headings.append(theta)
        result.velocities.append((v, w))
        result.timestamps.append(current_time)
        result.errors.append(d)

        v, w, index = ctrl.update(p, waypoints, index, dt)

        robot.set_robot_speeds(v, w)
        robot.update(dt)
        current_time += dt
        total_steps += 1

        if total_steps > timeout_limit:
            result.timeout = True
            print(f"Warning: simulation timeout ({ctrl.name})")
            break

        if vis is not None:
            vis.clean()
            vis.plot_path(waypoint_path)
            vis.plot_path(result.trajectory, ROBOT_COLOR, dotted=True)
            vis.plot_polygon(robot.points, ROBOT_COLOR)
            if vis.show():
                break

    if vis is not None:
        vis.close()

    if save_log:
        result.save(log_dir)

    return result
