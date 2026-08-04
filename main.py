import os
import argparse
import numpy as np

from core import Robot, load_map, simulate, Waypoint
from control import LQR, PurePursuit, PurePursuitParams, PID, PIDParams, PIDGains
from control.mpc import MPC


def get_ctrl():
    return PID(PIDParams(
        v=PIDGains(Kp=0.5, Kd=0.1),
        w=PIDGains(Kp=0.9, Kd=0.1),
        threshold=30
    ))
    # return PurePursuit(PurePursuitParams(
    #     L = 30, v_ref=10
    # ))
    # return LQR()
    # return MPC()

def get_robot(waypoints: list[Waypoint]):
    first = waypoints[0]
    return Robot(int(first.x), int(first.y))


def main(args):
    if not os.path.exists(args.map):
        raise FileNotFoundError(f"Map file not found: {args.map}")

    waypoints = load_map(args.map)
    robot = get_robot(waypoints)
    ctrl = get_ctrl()

    result = simulate(
        robot, ctrl, waypoints,
        map_name=args.map,
        visualize=args.visualize,
        save_log=not args.no_log,
        log_dir=args.log_dir,
    )

    print(f"Controller : {result.controller}")
    print(f"Map        : {result.map_name}")
    print(f"Steps      : {len(result.timestamps)}")
    print(f"Timeout    : {result.timeout}")
    print(f"Avg Error  : {sum(result.errors) / len(result.errors):.2f}")
    print(f"Max Error  : {max(result.errors):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Differential Drive Simulation")
    parser.add_argument(
        "--map", type=str, required=True,
        help="Path to waypoint map file"
    )
    parser.add_argument(
        "--visualize", action="store_true", default=False,
        help="Show simulation window"
    )
    parser.add_argument(
        "--no-log", action="store_true", default=False,
        help="Disable log saving"
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs",
        help="Directory to save logs (default: logs)"
    )
    args = parser.parse_args()
    main(args)
