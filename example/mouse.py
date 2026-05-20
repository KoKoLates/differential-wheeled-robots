import cv2

from typing import Optional

from core import Robot, Visualizer, Waypoint
from control import PID


class MouseHandler:
    def __init__(self) -> None:
        self.path: list[tuple[int, int]] = []

    def callback(
        self,
        event: int,
        x: int, y: int,
        flags: Optional[int] = None,
        param: Optional[int] = None,
    ) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.path.append((x, y))
        if event == cv2.EVENT_RBUTTONDOWN and self.path:
            self.path.pop()

    @property
    def waypoints(self) -> list[Waypoint]:
        return [Waypoint(x=x, y=y) for x, y in self.path]


def main() -> None:
    handler = MouseHandler()
    viz = Visualizer((500, 500), "Mouse Tracking", handler.callback)
    robot = Robot(250, 250)
    ctrl = PID()

    index: int = 0
    history: list[tuple[int, int]] = []
    color = (20, 255, 57)

    while True:
        viz.clean()

        waypoints = handler.waypoints
        index = min(index, len(waypoints) - 1) if waypoints else 0
        
        viz.plot_path([(int(w.x), int(w.y)) for w in waypoints])
        viz.plot_path(history, color, dotted=True)
        viz.plot_polygon(robot.points, color)

        p, _ = robot.state
        history.append((int(p[0, 0]), int(p[1, 0])))

        if waypoints and index < len(waypoints):
            v, w, index = ctrl.update(p, waypoints, index)
        else:
            v, w = 0.0, 0.0

        robot.set_robot_speeds(v, w)
        robot.update(0.5)

        if viz.show() & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
