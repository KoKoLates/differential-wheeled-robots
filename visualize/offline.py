import matplotlib.pyplot as plt
import matplotlib.cm as cm

from matplotlib.animation import FuncAnimation, PillowWriter

from core.analyzer import LogData
from .style import draw_path


def _assign_colors(logs: list[LogData]) -> list:
    cmap = cm.get_cmap("tab10")
    return [cmap(i) for i in range(len(logs))]


def plot_trajectory(
    logs: list[LogData],
    reference: list[tuple[float, float]],
    save_path: str = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = _assign_colors(logs)

    draw_path(ax, reference, color="gray", label="reference", marker=None)

    for log, color in zip(logs, colors):
        draw_path(ax, log.trajectory, color=color, label=log.controller, marker="start")

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Trajectory Comparison")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_error(
    logs: list[LogData],
    save_path: str = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = _assign_colors(logs)

    for log, color in zip(logs, colors):
        ax.plot(log.timestamps, log.errors, color=color, linewidth=1.5, label=log.controller)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("distance error")
    ax.set_title("Error vs Time")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def make_gif(
    logs: list[LogData],
    reference: list[tuple[float, float]],
    save_path: str = "output.gif",
    fps: int = 30,
) -> None:
    colors = _assign_colors(logs)
    max_steps = max(len(log.trajectory) for log in logs)
    step_size = max(1, max_steps // 100)

    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        ax.clear()
        step = frame * step_size

        draw_path(ax, reference, color="gray", label="reference", marker=None)

        for log, color in zip(logs, colors):
            end = min(step, len(log.trajectory))
            if end > 0:
                draw_path(ax, log.trajectory[:end], color=color, label=log.controller)

        ax.set_aspect("equal")
        ax.legend(loc="upper left")
        ax.set_title(f"Step {step}")

    frames = max_steps // step_size
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // fps)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"GIF saved: {save_path}")
