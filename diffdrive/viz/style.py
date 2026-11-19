"""Shared colours and robot drawing."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Polygon

from diffdrive.core.geometry import body_to_world

REFERENCE_COLOR = "#9a9a9a"
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]


def color_for(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def footprint(pose: np.ndarray, length: float, width: float) -> np.ndarray:
    """Body outline with a pointed nose so the heading is visible."""
    hl, hw = length / 2, width / 2
    body = np.array([[-hl, -hw], [hl * 0.6, -hw], [hl, 0.0], [hl * 0.6, hw], [-hl, hw]])
    return body_to_world(pose, body)


def robot_patch(ax: Axes, pose: np.ndarray, length: float, width: float, color: str) -> Polygon:
    patch = Polygon(
        footprint(pose, length, width),
        closed=True,
        fc=color,
        ec="black",
        lw=0.8,
        alpha=0.85,
        zorder=5,
    )
    ax.add_patch(patch)
    return patch


def setup_path_axes(ax: Axes, xy: np.ndarray, margin: float = 0.4) -> None:
    lo = xy.min(axis=0) - margin
    hi = xy.max(axis=0) + margin
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
