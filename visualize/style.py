import numpy as np

from typing import Optional, Sequence, Union
from matplotlib.patches import Polygon

ColorLike = Union[str, Sequence[float]]


def to_mpl_color(color: ColorLike) -> ColorLike:
    """Normalize a color to matplotlib's expected form.

    Accepts matplotlib color strings/names as-is, and 0-255 RGB(A) tuples
    (as used throughout the rest of this codebase) by scaling to 0-1.
    """
    if isinstance(color, str):
        return color
    if any(c > 1 for c in color):
        return tuple(c / 255 for c in color)
    return color


def draw_path(
    ax,
    path: Sequence[tuple[float, float]],
    color: ColorLike = (128, 128, 128),
    dotted: bool = False,
    label: Optional[str] = None,
    marker: Optional[str] = "end",
    linewidth: float = 1.5,
) -> None:
    """Draw a polyline path onto `ax`, shared by live and offline rendering.

    @param marker: "end" to mark the last point, "start" to mark the first,
        None for no marker.
    """
    if not path:
        return

    xs, ys = zip(*path)
    c = to_mpl_color(color)

    if dotted:
        ax.scatter(xs[::3], ys[::3], color=c, s=8, label=label)
        return

    ax.plot(xs, ys, color=c, linewidth=linewidth, label=label)
    if marker == "end":
        ax.plot(xs[-1], ys[-1], "o", color=c, markersize=4)
    elif marker == "start":
        ax.plot(xs[0], ys[0], "o", color=c, markersize=6)


def draw_polygon(
    ax,
    points: np.ndarray,
    color: ColorLike = (255, 0, 0),
    linewidth: float = 2,
) -> None:
    poly = Polygon(
        points[:, :2], closed=True, fill=False,
        edgecolor=to_mpl_color(color), linewidth=linewidth,
    )
    ax.add_patch(poly)
