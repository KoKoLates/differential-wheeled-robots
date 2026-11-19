"""Run one episode: one map, one scenario (conditions), one controller.

    python main.py --map map02 --controller pid                      # print metrics only
    python main.py --map map02 --controller pid --save               # + episode data, figure, GIF
    python main.py --map map05 --scenario noisy --controller pure_pursuit --save
    python main.py --scenario delayed --controller pid --set command.latency=0.1
    python main.py --map map01 --controller examples/my_controller.py:Kanayama
    python main.py --list

Without --map the scenario's default map is used; without --scenario, "nominal".
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def parse_kv(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"expected key=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def print_metrics(m: dict[str, Any]) -> None:
    keys = [
        ("termination", "termination"),
        ("success", "success"),
        ("pos_rmse", "pos RMSE [m]"),
        ("pos_max", "pos max [m]"),
        ("cte_rmse", "CTE RMSE [m]"),
        ("heading_rmse", "heading RMSE [rad]"),
        ("completion", "completion"),
        ("goal_distance", "goal dist [m]"),
        ("effort_deviation", "‖u−u_r‖² effort"),
        ("constraint_violations", "raw cmd violations"),
        ("solve_ms_mean", "solve mean [ms]"),
        ("solve_ms_p95", "solve P95 [ms]"),
        ("deadline_miss_rate", "deadline misses"),
    ]
    for k, name in keys:
        v = m.get(k)
        if isinstance(v, float):
            v = f"{v:.4g}"
        print(f"  {name:22s} {v}")


def list_all() -> None:
    from diffdrive.controllers import available_controllers
    from diffdrive.core.paths import SUITES_DIR
    from diffdrive.references import available_maps, map_info
    from diffdrive.sim import builtin_scenarios, load_scenario

    print("controllers:")
    for name, cls in available_controllers().items():
        state = "ready" if cls.implemented else f"TODO ({cls.milestone})"
        print(f"  {name:16s} {state:12s} {cls.__module__}:{cls.__qualname__}")
    print("maps:")
    for n in available_maps():
        print(f"  {n:16s} {map_info(n).get('description', '')}")
    print("scenarios:")
    for n in builtin_scenarios():
        sc = load_scenario(n)
        print(f"  {n:16s} [{sc.map}] {sc.description}")
    print("suites:")
    for p in sorted(SUITES_DIR.glob("*.yaml")):
        desc = (yaml.safe_load(p.read_text()) or {}).get("description", "")
        print(f"  {p.stem:16s} {desc}")


def main(args: argparse.Namespace) -> Path | None:
    if args.list:
        list_all()
        return None

    from diffdrive.sim import load_scenario

    overrides = parse_kv(args.set)
    if args.map:
        overrides["map"] = args.map
    sc = load_scenario(args.scenario).with_overrides(overrides)
    if args.show:
        print(sc.to_yaml())
        return None

    from diffdrive.controllers import make_controller
    from diffdrive.evaluation import compute_metrics
    from diffdrive.sim import run_episode

    ctrl = make_controller(args.controller, parse_kv(args.param))
    ep = run_episode(sc, ctrl, seed=args.seed, label=args.label)
    ep.metrics = compute_metrics(ep)
    print(f"{sc.map} · {sc.name} · {ep.label} · seed {args.seed}")
    print_metrics(ep.metrics)
    if not args.save:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from diffdrive import viz

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) / f"{stamp}_{sc.map}_{sc.name}_{ep.label}"
    ep.save(out)
    plt.close(viz.episode_report([ep], out=out / "summary.png"))
    files = ["episode.npz", "manifest.json", "summary.png"]
    if not args.no_gif:
        viz.animate([ep], out / "animation.gif", speed=args.speed)
        files.append("animation.gif")
    print(f"saved: {out}/  ({', '.join(files)})")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--map", help="map name (maps/) or path to a map .yaml/.csv")
    p.add_argument("--scenario", default="nominal", help="scenario name or YAML path")
    p.add_argument("--controller", default="pid", help="registry name or path/to/file.py:Class")
    p.add_argument(
        "--param", action="append", metavar="KEY=VAL", help="controller config, e.g. kp_v=1.2"
    )
    p.add_argument(
        "--set",
        action="append",
        metavar="KEY=VAL",
        help="scenario override, e.g. sensor.latency=0.05",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", help="name for this run (default: controller name)")
    p.add_argument("--save", action="store_true", help="save episode data, figure and GIF")
    p.add_argument("--no-gif", action="store_true", help="with --save: skip the (slow) GIF")
    p.add_argument("--speed", type=float, default=2.0, help="GIF playback speed-up")
    p.add_argument("--out", default="runs/single", help="where --save writes")
    p.add_argument("--list", action="store_true", help="list controllers, maps, scenarios, suites")
    p.add_argument("--show", action="store_true", help="print the resolved scenario as YAML")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
