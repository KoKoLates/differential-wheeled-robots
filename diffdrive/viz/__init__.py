"""Plotting and animation. Matplotlib is imported lazily by the submodules."""

from diffdrive.viz.animate import animate
from diffdrive.viz.plots import (
    episode_report,
    plot_controls,
    plot_errors,
    plot_metric_bars,
    plot_paths,
    plot_sweep,
    plot_timing,
    timing_report,
)

__all__ = [
    "animate",
    "episode_report",
    "plot_controls",
    "plot_errors",
    "plot_metric_bars",
    "plot_paths",
    "plot_sweep",
    "plot_timing",
    "timing_report",
]
