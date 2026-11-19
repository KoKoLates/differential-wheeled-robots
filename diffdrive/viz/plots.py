"""Static figures from episode logs and benchmark results."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path as FsPath
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from diffdrive.evaluation.metrics import tracking_errors
from diffdrive.sim.episode import Episode
from diffdrive.viz.style import REFERENCE_COLOR, color_for, robot_patch, setup_path_axes


def plot_paths(episodes: Sequence[Episode], ax: Axes) -> None:
    ref = episodes[0].reference
    ax.plot(
        ref.path.xy[:, 0], ref.path.xy[:, 1], "--", color=REFERENCE_COLOR, lw=1.5, label="reference"
    )
    ax.plot(*ref.path.xy[0], "o", color=REFERENCE_COLOR)
    allxy = [ref.path.xy]
    for i, ep in enumerate(episodes):
        tr = ep["truth"]
        c = color_for(i)
        ax.plot(tr[:, 0], tr[:, 1], color=c, lw=1.6, label=ep.label)
        if len(tr):
            rb = ep.meta["scenario"]["robot"]
            robot_patch(ax, tr[-1], rb["length"], rb["width"], c)
            allxy.append(tr[:, :2])
    setup_path_axes(ax, np.vstack(allxy))
    ax.set_title(episodes[0].title)
    ax.legend(loc="best", fontsize=8)


def plot_errors(episodes: Sequence[Episode], ax_time: Axes, ax_path: Axes) -> None:
    for i, ep in enumerate(episodes):
        e = tracking_errors(ep)
        ax_time.plot(e["t"], e["pos"], color=color_for(i), lw=1.2, label=ep.label)
        ax_path.plot(e["t"], e["cte"], color=color_for(i), lw=1.2, label=ep.label)
    ax_time.set_title("time-indexed position error")
    ax_time.set_ylabel("‖p − p_r(t)‖ [m]")
    ax_path.set_title("cross-track error (path)")
    ax_path.set_ylabel("signed CTE [m]")
    ax_path.axhline(0, color="k", lw=0.5)
    for ax in (ax_time, ax_path):
        ax.set_xlabel("t [s]")
        ax.grid(True, alpha=0.3)


def plot_controls(episodes: Sequence[Episode], ax_v: Axes, ax_w: Axes) -> None:
    ep0 = episodes[0]
    _, ref_u, _ = ep0.reference.sample(ep0["t"])
    ax_v.plot(ep0["t"], ref_u[:, 0], "--", color=REFERENCE_COLOR, label="reference")
    ax_w.plot(ep0["t"], ref_u[:, 1], "--", color=REFERENCE_COLOR, label="reference")
    for i, ep in enumerate(episodes):
        ax_v.plot(ep["t"], ep["applied"][:, 0], color=color_for(i), lw=1.1, label=ep.label)
        ax_w.plot(ep["t"], ep["applied"][:, 1], color=color_for(i), lw=1.1, label=ep.label)
    cmd = ep0.meta["scenario"]["command"]
    for ax, lim, name in (
        (ax_v, (cmd["v_min"], cmd["v_max"]), "v [m/s]"),
        (ax_w, (-cmd["omega_max"], cmd["omega_max"]), "ω [rad/s]"),
    ):
        for b in lim:
            ax.axhline(b, color="k", lw=0.6, ls=":")
        ax.set_ylabel(name)
        ax.set_xlabel("t [s]")
        ax.grid(True, alpha=0.3)
    ax_v.set_title("applied linear velocity")
    ax_w.set_title("applied angular velocity")


def plot_timing(episodes: Sequence[Episode], ax: Axes) -> None:
    data = [ep["solve_time"] * 1e3 for ep in episodes]
    ax.boxplot(data, showfliers=True)
    ax.set_xticks(range(1, len(episodes) + 1), [ep.label for ep in episodes], rotation=20)
    dt_ms = episodes[0].meta["scenario"]["timing"]["dt_control"] * 1e3
    ax.axhline(dt_ms, color="r", ls="--", lw=0.8, label=f"control period {dt_ms:.0f} ms")
    ax.set_yscale("log")
    ax.set_ylabel("solve time [ms]")
    ax.set_title("computation per control step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")


def episode_report(episodes: Sequence[Episode], out: str | FsPath | None = None) -> Figure:
    """2×3 overview: path, time error, CTE, v, ω, solve time."""
    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    plot_paths(episodes, fig.add_subplot(gs[:, 0]))
    plot_errors(episodes, fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]))
    plot_controls(episodes, fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]))
    if out is not None:
        fig.savefig(out, dpi=130)
    return fig


def timing_report(episodes: Sequence[Episode], out: str | FsPath | None = None) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    plot_timing(episodes, ax)
    if out is not None:
        fig.savefig(out, dpi=130)
    return fig


# ------------------------------------------------------------------ benchmark
def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def plot_sweep(
    rows: list[dict[str, Any]],
    param: str,
    metric: str,
    ax: Axes | None = None,
    scenario_prefix: str | None = None,
) -> Axes:
    """Mean ± std of ``metric`` vs swept ``param`` (column ``param:<key>``), per controller."""
    col = param if param.startswith("param:") else f"param:{param}"
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    groups: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if col not in r or (isinstance(r[col], float) and math.isnan(r[col])):
            continue
        if scenario_prefix and not str(r["scenario"]).startswith(scenario_prefix):
            continue
        x = r[col]
        x = float(np.linalg.norm(np.atleast_1d(_parse_list(x)))) if isinstance(x, str) else float(x)
        groups[str(r["controller"])][x].append(_num(r.get(metric)))
    for i, (ctrl, by_x) in enumerate(sorted(groups.items())):
        xs = sorted(by_x)
        m = np.array([np.nanmean(by_x[x]) for x in xs])
        s = np.array([np.nanstd(by_x[x]) for x in xs])
        ax.plot(xs, m, "o-", color=color_for(i), label=ctrl)
        ax.fill_between(xs, m - s, m + s, color=color_for(i), alpha=0.15)
    ax.set_xlabel(param.removeprefix("param:"))
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs {param.removeprefix('param:')}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def _parse_list(x: str) -> list[float]:
    try:
        return [float(v) for v in x.strip("[]").split(",")]
    except ValueError:
        return [math.nan]


def plot_metric_bars(rows: list[dict[str, Any]], metric: str, ax: Axes | None = None) -> Axes:
    """Grouped bars: scenarios on the x-axis, one bar per controller (mean ± std over seeds)."""
    if ax is None:
        _, ax = plt.subplots(
            figsize=(max(6, len({r["scenario"] for r in rows}) * 1.2), 4), constrained_layout=True
        )
    scen = sorted({str(r["scenario"]) for r in rows})
    ctrls = sorted({str(r["controller"]) for r in rows})
    width = 0.8 / max(len(ctrls), 1)
    for j, c in enumerate(ctrls):
        means, stds = [], []
        for s in scen:
            v = np.array(
                [_num(r.get(metric)) for r in rows if r["scenario"] == s and r["controller"] == c]
            )
            means.append(np.nanmean(v) if len(v) else np.nan)
            stds.append(np.nanstd(v) if len(v) else 0.0)
        x = np.arange(len(scen)) + (j - (len(ctrls) - 1) / 2) * width
        ax.bar(x, means, width, yerr=stds, color=color_for(j), label=c, capsize=2)
    ax.set_xticks(np.arange(len(scen)), scen, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    return ax
