"""Look at results: benchmark reports, or saved episodes side by side.

Benchmark run (a directory containing results.csv) → tables + figures/:

    python visualize.py runs/20261118-120000_latency
    python visualize.py runs/... --metric pos_rmse --metric solve_ms_p95
    python visualize.py runs/... --sweep param:sensor.latency

Episodes (same map and scenario) → one comparison figure and/or GIF:

    python visualize.py runs/single/A runs/single/B --output compare.png
    python visualize.py runs/single/A runs/single/B --gif compare.gif
    python visualize.py runs/<bench>/episodes/map02_nominal --output map02.png

Figures are always written to files (default next to the input).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def report(root: str | Path, metrics: list[str] | None = None, sweep: str | None = None) -> Path:
    """Aggregate ``results.csv`` (mean ± std over seeds) and draw bar / sweep figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from diffdrive.evaluation import aggregate, format_table, read_results
    from diffdrive.evaluation.report import DEFAULT_METRICS
    from diffdrive.viz import plot_metric_bars, plot_sweep

    root = Path(root)
    rows = read_results(root)
    metrics_t = tuple(metrics) if metrics else DEFAULT_METRICS
    print(format_table(aggregate(rows, metrics_t), metrics_t))
    figdir = root / "figures"
    figdir.mkdir(exist_ok=True)

    params = sorted({k for r in rows for k in r if k.startswith("param:")})
    sweeps = [sweep] if sweep else params
    for m in metrics_t:
        for p in sweeps:
            ax = plot_sweep(rows, p, m)
            ax.figure.savefig(figdir / f"sweep_{p.removeprefix('param:')}_{m}.png", dpi=130)
            plt.close(ax.figure)
        ax = plot_metric_bars(rows, m)
        ax.figure.savefig(figdir / f"bars_{m}.png", dpi=130)
        plt.close(ax.figure)
    print(f"\nfigures: {figdir}")
    return figdir


def compare(
    dirs: list[str], output: str | None, gif: str | None, timing: str | None, speed: float
) -> None:
    """Overlay episodes found under ``dirs`` in one figure and/or animation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from diffdrive.sim import Episode, find_episodes
    from diffdrive.viz import animate, episode_report, timing_report

    paths = [p for d in dirs for p in find_episodes(d)]
    if not paths:
        raise SystemExit("no episodes found")
    eps = [Episode.load(p) for p in paths]
    if len({(e.map_name, e.scenario_name) for e in eps}) > 1:
        print("warning: episodes come from different maps/scenarios; overlaying the first one's map")
    if not (output or gif):
        output = str(Path(dirs[0]) / "compare.png")
    if output:
        plt.close(episode_report(eps, out=output))
        print(f"figure: {output}")
    if timing:
        plt.close(timing_report(eps, out=timing))
        print(f"timing: {timing}")
    if gif:
        print(f"animation: {animate(eps, gif, speed=speed)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("paths", nargs="+", help="a benchmark directory, or episode directories")
    g = p.add_argument_group("benchmark report")
    g.add_argument("--metric", action="append", help="metric to report (repeatable)")
    g.add_argument("--sweep", help="swept parameter for the x-axis, e.g. param:sensor.latency")
    g = p.add_argument_group("episode comparison")
    g.add_argument("--output", help="figure file (default: <first dir>/compare.png)")
    g.add_argument("--gif", help="also write an animation (.gif or .mp4)")
    g.add_argument("--timing", help="also write a solve-time figure")
    g.add_argument("--speed", type=float, default=2.0, help="animation speed-up")
    return p


def main(args: argparse.Namespace) -> None:
    first = Path(args.paths[0])
    if len(args.paths) == 1 and (first / "results.csv").is_file():
        report(first, args.metric, args.sweep)
    else:
        compare(args.paths, args.output, args.gif, args.timing, args.speed)


if __name__ == "__main__":
    main(build_parser().parse_args())
