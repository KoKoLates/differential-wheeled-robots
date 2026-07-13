import argparse
import numpy as np

from typing import Callable
from core import Robot, Waypoint
from control import PID, PIDGains, PIDParams
from visualize import LiveRenderer


def make_square(
    x: int, y: int, border: int, step: int
) -> list[Waypoint]:
    return [
        *(Waypoint(x=i, y=y) for i in range(x, x + border, step)),
        *(Waypoint(x=x + border, y=j) for j in range(y, y + border, step)),
        *(Waypoint(x=i, y=y + border) for i in range(x + border, x, -step)),
        *(Waypoint(x=x, y=j) for j in range(y + border, y, -step)),
    ]


def make_circle(
    x: int, y: int, r: int, n: int = 100
) -> list[Waypoint]:
    angles = np.linspace(0, np.pi * 2, n, endpoint=False)
    return [Waypoint(x + int(r * np.sin(angle)), y - int(r * np.cos(angle))) for angle in angles]


def run(title: str, robot: Robot, waypoints: list[Waypoint]) -> None:
    viz = LiveRenderer(title=title)

    ctrl = PID(PIDParams(
        v=PIDGains(Kp=0.5, Ki=0.0, Kd=0.1),
        w=PIDGains(Kp=0.9, Ki=0.0, Kd=0.1),
        threshold=10
    ))

    index: int = 0
    history: list[tuple[float, float]] = []
    color = (20, 255, 57)

    while True:
        viz.clean()

        viz.plot_path([(w.x, w.y) for w in waypoints])
        viz.plot_path(history, color, dotted=True)
        viz.plot_polygon(robot.points, color)

        p, _ = robot.state
        history.append((float(p[0, 0]), float(p[1, 0])))

        if index < len(waypoints):
            v, w, index = ctrl.update(p, waypoints, index)
        else:
            v, w = 0.0, 0.0

        robot.set_robot_speeds(v, w)
        robot.update(0.5)

        if viz.show():
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predefined path tracking")
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="square",
        choices=["square", "circle"],
        help="type of simulation to be executed (default: square)",
    )
    args = parser.parse_args()

    func: dict[str, Callable[[], None]] = {
        "square": lambda: run("Square", Robot(125, 125), make_square(125, 125, 250, 30)),
        "circle": lambda: run("Circle", Robot(250, 130), make_circle(250, 250, 120))
    }

    func[args.type]()
