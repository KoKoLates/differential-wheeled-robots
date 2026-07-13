import numpy as np
import matplotlib.pyplot as plt

from typing import Callable, Optional

from .style import ColorLike, draw_path, draw_polygon

DEFAULT_VIEW: tuple[float, float] = (0, 500)


class LiveRenderer:
    def __init__(
        self,
        title: str = "simulation",
        xlim: Optional[tuple[float, float]] = DEFAULT_VIEW,
        ylim: Optional[tuple[float, float]] = DEFAULT_VIEW,
        callback: Optional[Callable] = None,
    ) -> None:
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.fig.canvas.manager.set_window_title(title)

        self.xlim = xlim
        self.ylim = ylim

        self._closed = False
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        if callback is not None:
            self.fig.canvas.mpl_connect("button_press_event", callback)

        self.clean()

    def _on_close(self, _event) -> None:
        self._closed = True

    def clean(self) -> None:
        self.ax.cla()
        self.ax.set_aspect("equal")
        if self.xlim is not None:
            self.ax.set_xlim(*self.xlim)
        if self.ylim is not None:
            self.ax.set_ylim(*self.ylim)

    def plot_path(
        self,
        path: list[tuple[float, float]],
        color: ColorLike = (128, 128, 128),
        dotted: bool = False,
    ) -> None:
        draw_path(self.ax, path, color, dotted)

    def plot_polygon(self, points: np.ndarray, color: ColorLike = (255, 0, 0)) -> None:
        draw_polygon(self.ax, points, color)

    def show(self, delay: float = 0.001) -> bool:
        """Render the current frame. Returns True once the window has been closed."""
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(delay)
        return self._closed

    def close(self) -> None:
        plt.close(self.fig)
