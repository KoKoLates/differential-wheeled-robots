"""Animate one or more episodes (same scenario) to GIF / MP4."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path as FsPath

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from diffdrive.sim.episode import Episode
from diffdrive.viz.style import REFERENCE_COLOR, color_for, footprint, robot_patch, setup_path_axes


def animate(
    episodes: Sequence[Episode],
    out: str | FsPath,
    fps: int = 20,
    speed: float = 1.0,
    show_prediction: bool = True,
    trail: bool = True,
) -> FsPath:
    """Render an animation at ``speed``× real time. Draws the robot, its trail,
    the reference point it should be at (time-indexed), and — if the controller
    logged ``predicted_states`` — its predicted horizon."""
    out = FsPath(out)
    ref = episodes[0].reference
    rb = episodes[0].meta["scenario"]["robot"]
    t_end = max(float(ep["t"][-1]) for ep in episodes if len(ep["t"]))
    frame_times = np.arange(0.0, t_end, speed / fps)

    fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)
    ax.plot(ref.path.xy[:, 0], ref.path.xy[:, 1], "--", color=REFERENCE_COLOR, lw=1.2)
    setup_path_axes(ax, np.vstack([ref.path.xy] + [ep["truth"][:, :2] for ep in episodes]))
    (ref_dot,) = ax.plot([], [], "o", ms=5, mfc="none", mec="k", label="reference(t)")
    title = ax.set_title("")

    artists = []
    for i, ep in enumerate(episodes):
        c = color_for(i)
        (line,) = ax.plot([], [], color=c, lw=1.4, label=ep.label)
        (pred,) = ax.plot([], [], ".-", color=c, lw=0.8, ms=2, alpha=0.6)
        patch = robot_patch(ax, ep["truth"][0], rb["length"], rb["width"], c)
        artists.append((ep, line, pred, patch))
    ax.legend(loc="upper right", fontsize=8)

    def update(frame: int):
        t = frame_times[frame]
        pr, _, _ = ref.sample(t)
        ref_dot.set_data([pr[0, 0]], [pr[0, 1]])
        for ep, line, pred, patch in artists:
            k = min(int(np.searchsorted(ep["t"], t, side="right")) - 1, len(ep["t"]) - 1)
            k = max(k, 0)
            tr = ep["truth"]
            if trail:
                line.set_data(tr[: k + 1, 0], tr[: k + 1, 1])
            patch.set_xy(footprint(tr[k], rb["length"], rb["width"]))
            if show_prediction and "pred_states" in ep.arrays and len(ep["ctrl_t"]):
                j = max(int(np.searchsorted(ep["ctrl_t"], t, side="right")) - 1, 0)
                ps = ep["pred_states"][j]
                ps = ps[~np.isnan(ps[:, 0])]
                pred.set_data(ps[:, 0], ps[:, 1])
        title.set_text(f"{episodes[0].title}   t = {t:5.2f} s")
        return [ref_dot, title] + [a for _, ln, pr_, pa in artists for a in (ln, pr_, pa)]

    anim = FuncAnimation(fig, update, frames=len(frame_times), blit=False)
    writer = FFMpegWriter(fps=fps) if out.suffix.lower() == ".mp4" else PillowWriter(fps=fps)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=writer)
    plt.close(fig)
    return out
