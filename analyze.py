import os
import argparse
import glob

from core.analyzer import load_log, load_map, print_summary, LogData
from visualize import plot_trajectory, plot_error, make_gif


def resolve_map(args, logs: list[LogData]) -> list[tuple[float, float]]:
    if args.map:
        return load_map(args.map)

    map_path = logs[0].map_name
    if not os.path.exists(map_path):
        raise FileNotFoundError(
            f"Map '{map_path}' not found, please provide --map explicitly"
        )
    return load_map(map_path)


def resolve_logs(args) -> list[str]:
    if args.logs:
        return args.logs
    if args.log_dir:
        paths = glob.glob(os.path.join(args.log_dir, "*.txt"))
        if not paths:
            raise FileNotFoundError(f"No log files found in: {args.log_dir}")
        return paths
    raise ValueError("Must provide either --log-dir or --logs")


def main(args):
    log_paths = resolve_logs(args)
    logs = [load_log(p) for p in log_paths]
    reference = resolve_map(args, logs)

    print_summary(logs, reference)
    plot_trajectory(logs, reference, save_path=args.save)
    plot_error(logs, save_path=args.save_error)

    if args.gif:
        make_gif(logs, reference, save_path=args.gif)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze simulation logs")
    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="Load all logs from folder (e.g. logs/example)"
    )
    parser.add_argument(
        "--logs", nargs="+", default=None,
        help="Specific log files (e.g. logs/example/pid_a3f2.txt logs/example/lqr_b1c4.txt)"
    )
    parser.add_argument(
        "--map", type=str, default=None,
        help="Map file (optional, inferred from log metadata if not provided)"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save trajectory plot (e.g. traj.png)"
    )
    parser.add_argument(
        "--save-error", type=str, default=None,
        help="Save error plot (e.g. error.png)"
    )
    parser.add_argument(
        "--gif", type=str, default=None,
        help="Save GIF (e.g. output.gif)"
    )
    args = parser.parse_args()
    main(args)
