import cv2
import numpy as np

from typing import Callable, Optional

ColorTuple = tuple[int, int, int]


class Visualizer:
    def __init__(
        self,
        window_size: tuple[int, int],
        window_name: str,
        callback: Optional[Callable] = None,
    ) -> None:
        self.w, self.h = window_size
        self.name = window_name

        cv2.namedWindow(self.name)
        if callback is not None:
            cv2.setMouseCallback(self.name, callback)

        self.background: ColorTuple = (0, 0, 0)
        self.canvas: np.ndarray = np.zeros((self.h, self.w, 3), dtype=np.uint8)

    def clean(self) -> None:
        self.canvas[:] = self.background

    def plot_polygon(
        self,
        points: np.ndarray,
        color: ColorTuple = (255, 0, 0),
        thickness: int = 2,
    ) -> None:
        for i in range(len(points) - 1):
            cv2.line(
                self.canvas,
                (points[i, 0], points[i, 1]),
                (points[i + 1, 0], points[i + 1, 1]),
                color, thickness,
            )
        cv2.line(
            self.canvas,
            (points[0, 0], points[0, 1]),
            (points[-1, 0], points[-1, 1]),
            color, thickness,
        )

    def plot_path(
        self,
        path: list[tuple[int, int]],
        color: ColorTuple = (128, 128, 128),
        dotted: bool = False,
    ) -> None:
        if not path:
            return

        if dotted:
            for i, point in enumerate(path[:-1]):
                if not (i % 3):
                    self._plot_circle(point, color, 2)
            return

        for p1, p2 in zip(path[:-1], path[1:]):
            cv2.line(self.canvas, p1, p2, color, 1)
        self._plot_circle(path[-1], color)

    def label(
        self,
        text: str,
        color: ColorTuple = (255, 255, 255),
        thickness: int = 2,
        font_size: float = 1.0,
        org: tuple[int, int] = (100, 50),
    ) -> None:
        cv2.putText(
            self.canvas,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def show(self, delay: int = 30) -> int:
        cv2.imshow(self.name, self.canvas)
        return cv2.waitKey(delay)

    def _plot_circle(
        self,
        point: tuple[int, int],
        color: ColorTuple,
        radius: int = 3,
    ) -> None:
        cv2.circle(self.canvas, point, radius, color, -1)
