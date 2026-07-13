from core import Robot, Waypoint
from control import PID, PIDGains, PIDParams
from visualize import LiveRenderer


class MouseHandler:
    def __init__(self) -> None:
        self.path: list[tuple[float, float]] = []

    def callback(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            self.path.append((event.xdata, event.ydata))
        if event.button == 3 and self.path:
            self.path.pop()

    @property
    def waypoints(self) -> list[Waypoint]:
        return [Waypoint(x=x, y=y) for x, y in self.path]


def main() -> None:
    handler = MouseHandler()
    viz = LiveRenderer(title="Mouse Tracking", callback=handler.callback)
    robot = Robot(250, 250)
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

        waypoints = handler.waypoints
        index = min(index, len(waypoints) - 1) if waypoints else 0

        viz.plot_path([(w.x, w.y) for w in waypoints])
        viz.plot_path(history, color, dotted=True)
        viz.plot_polygon(robot.points, color)

        p, _ = robot.state
        history.append((float(p[0, 0]), float(p[1, 0])))

        if waypoints and index < len(waypoints):
            v, w, index = ctrl.update(p, waypoints, index)
        else:
            v, w = 0.0, 0.0

        robot.set_robot_speeds(v, w)
        robot.update(0.5)

        if viz.show():
            break


if __name__ == "__main__":
    main()
