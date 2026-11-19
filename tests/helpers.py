"""Analytic test paths (independent of the shipped maps, so tests don't break when maps change)."""

from __future__ import annotations

import math

import numpy as np

from diffdrive.references import Path, save_map


def line(length: float = 6.0) -> Path:
    return Path.from_points(np.array([[0.0, 0.0], [length, 0.0]]))


def circle(radius: float = 1.5, laps: float = 1.0, direction: str = "ccw") -> Path:
    sign = 1.0 if direction == "ccw" else -1.0
    return Path.from_function(
        lambda u: (radius * np.sin(u), sign * radius * (1.0 - np.cos(u))), 0.0, 2 * math.pi * laps
    )


def figure_eight(width: float = 4.0, height: float = 2.0) -> Path:
    a, b = width / 2.0, height
    return Path.from_function(lambda u: (a * np.sin(u), b * np.sin(u) * np.cos(u)), 0.0, 2 * math.pi)


def s_curve(length: float = 6.0, amplitude: float = 0.75) -> Path:
    return Path.from_function(
        lambda u: (u, amplitude * np.sin(2 * math.pi * u / length)), 0.0, length
    )


SHAPES = {"line": line, "circle": circle, "figure_eight": figure_eight, "s_curve": s_curve}


def map_file(directory, name: str, path: Path) -> str:
    """Write ``path`` as a map in ``directory`` and return its .yaml path (for ``Scenario.map``)."""
    return str(save_map(path, directory / name, description=name))
