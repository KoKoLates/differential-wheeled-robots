# core/analyzer.py
import os
import csv
import math

from dataclasses import dataclass, field


@dataclass
class LogData:
    controller: str
    map_name: str
    timeout: bool
    timestamps: list[float]               = field(default_factory=list)
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    headings: list[float]                 = field(default_factory=list)
    velocities: list[tuple[float, float]] = field(default_factory=list)
    errors: list[float]                   = field(default_factory=list)


def load_log(path: str) -> LogData:
    meta = {}
    rows = []

    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                key, _, val = line[2:].partition(":")
                meta[key.strip()] = val.strip()
            else:
                rows.append(line)

    reader = csv.DictReader(rows)
    log = LogData(
        controller=meta.get("controller", "unknown"),
        map_name=meta.get("map", "unknown"),
        timeout=meta.get("timeout", "False") == "True",
    )

    for row in reader:
        log.timestamps.append(float(row["t"]))
        log.trajectory.append((float(row["x"]), float(row["y"])))
        log.headings.append(float(row["theta"]))
        log.velocities.append((float(row["v"]), float(row["w"])))
        log.errors.append(float(row["error"]))

    return log


def load_map(path: str) -> list[tuple[float, float]]:
    waypoints = []
    with open(path, "r") as f:
        for line in f:
            x, y = map(float, line.strip().split(","))
            waypoints.append((x, y))
    return waypoints


def _rmse(errors: list[float]) -> float:
    return math.sqrt(sum(e ** 2 for e in errors) / len(errors))


def _ate(
    trajectory: list[tuple[float, float]],
    reference: list[tuple[float, float]]
) -> float:
    errors = []
    for x, y in trajectory:
        d = min(
            math.sqrt((x - rx) ** 2 + (y - ry) ** 2)
            for rx, ry in reference
        )
        errors.append(d)
    return sum(errors) / len(errors)

def _cte(
    trajectory: list[tuple[float, float]],
    reference: list[tuple[float, float]],
) -> list[float]:
    """Cross Track Error: perpendicular distance to nearest path segment"""
    errors = []
    for px, py in trajectory:
        min_dist = float("inf")
        for i in range(len(reference) - 1):
            ax, ay = reference[i]
            bx, by = reference[i + 1]

            abx, aby = bx - ax, by - ay
            apx, apy = px - ax, py - ay
            ab_len_sq = abx ** 2 + aby ** 2

            if ab_len_sq == 0:
                d = math.sqrt(apx ** 2 + apy ** 2)
            else:
                t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len_sq))
                cx, cy = ax + t * abx, ay + t * aby
                d = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)

            min_dist = min(min_dist, d)
        errors.append(min_dist)
    return errors


def print_summary(
    logs: list[LogData],
    reference: list[tuple[float, float]]
) -> None:
    header = (
        f"{'Controller':<20} {'Steps':>6} "
        f"{'Avg CTE':>9} {'RMSE CTE':>9} {'Max CTE':>9} "
        f"{'ATE':>8} {'Timeout':>8}"
    )
    print(header)
    print("-" * len(header))

    for log in logs:
        cte = _cte(log.trajectory, reference)
        avg  = sum(cte) / len(cte)
        rmse = _rmse(cte)
        mx   = max(cte)
        ate  = _ate(log.trajectory, reference)
        print(
            f"{log.controller:<20} "
            f"{len(log.timestamps):>6} "
            f"{avg:>9.2f} "
            f"{rmse:>9.2f} "
            f"{mx:>9.2f} "
            f"{ate:>8.2f} "
            f"{str(log.timeout):>8}"
        )