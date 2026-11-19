"""Generate reference maps (paths) for the simulator.

A map is three files in this folder:

    yaml/mapNN.yaml   description, the command that produced it, summary stats
    csv/mapNN.csv     the path sampled every DS metres:  s, x, y, yaw, kappa
    png/mapNN.png     preview (shown in README.md)

Usage (from anywhere):

    python maps/generate.py --shape line --length 6
    python maps/generate.py --shape circle --radius 1.5
    python maps/generate.py --shape eight --width 4 --height 2
    python maps/generate.py --shape s_curve --length 6 --amplitude 0.75
    python maps/generate.py --random --points 6 --seed 7
    python maps/generate.py --random --points 8 --seed 3 --name map09 --description "..."
    python maps/generate.py --catalogue         # redraw every png/ preview + the README table

New maps get the next free number unless --name is given. Published maps are
benchmark fixtures: do not regenerate or edit them, add a new number instead.

This script is self-contained (numpy, scipy, matplotlib, pyyaml) and does not
import the simulator; the CSV format is the only contract between the two.
"""

from __future__ import annotations

import argparse
import math
import re
import shlex
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import CubicSpline

MAPS_DIR = Path(__file__).resolve().parent
DS = 0.01  # output resolution [m]
_DENSE = 200_000  # samples used before resampling by arclength


# ------------------------------------------------------------------ geometry
def resample(x: np.ndarray, y: np.ndarray, ds: float = DS) -> np.ndarray:
    """Dense polyline → uniform-arclength samples with heading and curvature.

    Returns an (n, 5) array with columns s, x, y, yaw, kappa (yaw wrapped to [-π, π)).
    """
    pts = np.column_stack([x, y])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-12])
    pts = pts[keep]
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    # differentiate on a fine uniform grid, then subsample to `ds`
    fine = ds / 5
    s_f = np.linspace(0.0, cum[-1], max(int(round(cum[-1] / fine)) + 1, 3))
    xf = np.interp(s_f, cum, pts[:, 0])
    yf = np.interp(s_f, cum, pts[:, 1])
    th = np.unwrap(np.arctan2(np.gradient(yf, s_f, edge_order=2), np.gradient(xf, s_f, edge_order=2)))
    kf = np.gradient(th, s_f, edge_order=2)
    s = np.linspace(0.0, cum[-1], max(int(round(cum[-1] / ds)) + 1, 2))
    th_s = np.interp(s, s_f, th)
    yaw = (th_s + math.pi) % (2 * math.pi) - math.pi
    return np.column_stack(
        [s, np.interp(s, s_f, xf), np.interp(s, s_f, yf), yaw, np.interp(s, s_f, kf)]
    )


def shape(kind: str, args: argparse.Namespace) -> tuple[np.ndarray, str]:
    """Analytic test shapes. All start at the origin heading along +x."""
    u = np.linspace(0.0, 1.0, _DENSE)
    if kind == "line":
        x, y = args.length * u, 0.0 * u
        desc = f"Straight line, {args.length:g} m"
    elif kind == "circle":
        a = 2 * math.pi * args.laps * u
        x, y = args.radius * np.sin(a), args.radius * (1 - np.cos(a))
        desc = f"Circle, r = {args.radius:g} m, counter-clockwise"
    elif kind == "eight":
        a = 2 * math.pi * u
        x, y = args.width / 2 * np.sin(a), args.height * np.sin(a) * np.cos(a)
        desc = f"Figure-eight {args.width:g} m × {args.height:g} m (lemniscate of Gerono), self-crossing"
    elif kind == "s_curve":
        x = args.length * u
        y = args.amplitude * np.sin(2 * math.pi * args.periods * u)
        desc = f"S-curve, {args.length:g} m, amplitude {args.amplitude:g} m"
    else:
        raise SystemExit(f"unknown shape {kind!r}")
    return resample(x, y), desc


def random_spline(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    """Cubic spline through random control points, redrawn until it is drivable."""
    rng = np.random.default_rng(args.seed)
    w, h = args.area
    for _attempt in range(10_000):
        pts = [np.array([0.0, 0.0])]
        while len(pts) < args.points:
            p = rng.uniform([0.0, -h / 2], [w, h / 2])
            if np.linalg.norm(p - pts[-1]) >= args.min_spacing:
                pts.append(p)
        pts = np.array(pts)
        chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        cs = CubicSpline(chord, pts, bc_type="natural")
        xy = cs(np.linspace(0.0, chord[-1], _DENSE))
        out = resample(xy[:, 0], xy[:, 1])
        if np.abs(out[:, 4]).max() <= 1.0 / args.min_radius:
            desc = (
                f"Random cubic spline, {args.points} control points in "
                f"{w:g} m × {h:g} m, seed {args.seed}"
            )
            return out, desc
    raise SystemExit("could not find a drivable path; relax --min-radius or --min-spacing")


# ----------------------------------------------------------------------- I/O
def dirs() -> tuple[Path, Path, Path]:
    return MAPS_DIR / "yaml", MAPS_DIR / "csv", MAPS_DIR / "png"


def map_names() -> list[str]:
    return sorted(p.stem for p in dirs()[0].glob("map*.yaml") if re.fullmatch(r"map\d+", p.stem))


def next_name() -> str:
    nums = [int(n[3:]) for n in map_names()]
    return f"map{(max(nums) + 1 if nums else 0):02d}"


def write_map(name: str, data: np.ndarray, description: str, command: str, force: bool) -> None:
    ydir, cdir, _ = dirs()
    yml, csv = ydir / f"{name}.yaml", cdir / f"{name}.csv"
    if yml.exists() and not force:
        raise SystemExit(f"{yml.name} exists; maps are benchmark fixtures (use a new name or --force)")
    ydir.mkdir(exist_ok=True)
    cdir.mkdir(exist_ok=True)
    np.savetxt(csv, data, delimiter=",", header="s,x,y,yaw,kappa", comments="", fmt="%.6f")
    meta = {
        "description": description,
        "points": f"../csv/{csv.name}",  # relative to this yaml file
        "length": round(float(data[-1, 0]), 3),
        "max_curvature": round(float(np.abs(data[:, 4]).max()), 3),
        "generator": command,
    }
    yml.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
    print(f"wrote yaml/{yml.name} + csv/{csv.name}  ({meta['length']} m)")


def load(name: str) -> tuple[dict, np.ndarray]:
    yml = dirs()[0] / f"{name}.yaml"
    meta = yaml.safe_load(yml.read_text())
    data = np.loadtxt(yml.parent / meta["points"], delimiter=",", skiprows=1)
    return meta, data


# ----------------------------------------------------------------- catalogue
_TABLE_START, _TABLE_END = "<!-- maps:start -->", "<!-- maps:end -->"


def preview(name: str, data: np.ndarray, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.plot(data[:, 1], data[:, 2], color="0.35", lw=1.5)
    i = min(40, len(data) - 1)
    ax.plot(data[0, 1], data[0, 2], "o", color="tab:green", ms=5)
    ax.annotate(
        "", xy=(data[i, 1], data[i, 2]), xytext=(data[0, 1], data[0, 2]),
        arrowprops={"arrowstyle": "->", "color": "tab:green"},
    )  # fmt: skip
    ax.set_title(name, fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def catalogue() -> None:
    """Draw png/<map>.png for every map and rewrite the table in README.md."""
    import matplotlib

    matplotlib.use("Agg")

    names = map_names()
    pdir = dirs()[2]
    pdir.mkdir(exist_ok=True)
    lines = ["| Map | Preview | Length | Max κ |", "|---|---|---|---|"]
    for name in names:
        meta, d = load(name)
        preview(name, d, pdir / f"{name}.png")
        lines.append(
            f"| `{name}` | <img src=\"png/{name}.png\" width=\"240\"> "
            f"| {meta['length']:.2f} m | {meta['max_curvature']:.2f} 1/m |"
        )
    readme = MAPS_DIR / "README.md"
    text = readme.read_text() if readme.exists() else f"# Maps\n\n{_TABLE_START}\n{_TABLE_END}\n"
    table = "\n".join(lines)
    text = re.sub(
        rf"{_TABLE_START}.*?{_TABLE_END}", f"{_TABLE_START}\n{table}\n{_TABLE_END}", text, flags=re.S
    )
    readme.write_text(text)
    print(f"updated png/ and README.md ({len(names)} maps)")


# ---------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--shape", choices=["line", "circle", "eight", "s_curve"])
    what.add_argument("--random", action="store_true", help="random cubic spline")
    what.add_argument("--catalogue", action="store_true", help="only redraw png/ + README table")

    p.add_argument("--name", help="map name (default: next free mapNN)")
    p.add_argument("--description", help="override the generated description")
    p.add_argument("--force", action="store_true", help="overwrite an existing map")
    p.add_argument("--no-catalogue", action="store_true", help="skip redrawing png/ + README")

    g = p.add_argument_group("shapes")
    g.add_argument("--length", type=float, default=6.0, help="line / s_curve length [m]")
    g.add_argument("--radius", type=float, default=1.5, help="circle radius [m]")
    g.add_argument("--laps", type=float, default=1.0, help="circle laps")
    g.add_argument("--width", type=float, default=4.0, help="figure-eight width [m]")
    g.add_argument("--height", type=float, default=2.0, help="figure-eight height [m]")
    g.add_argument("--amplitude", type=float, default=0.75, help="s_curve amplitude [m]")
    g.add_argument("--periods", type=float, default=1.0, help="s_curve periods")

    g = p.add_argument_group("random")
    g.add_argument("--points", type=int, default=6, help="number of control points")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--area", type=float, nargs=2, default=(6.0, 4.0), metavar=("W", "H"))
    g.add_argument("--min-spacing", type=float, default=1.0, help="between control points [m]")
    g.add_argument(
        "--min-radius", type=float, default=0.4, help="reject paths with tighter turns [m]"
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.catalogue:
        data, desc = random_spline(args) if args.random else shape(args.shape, args)
        argv_ = argv if argv is not None else sys.argv[1:]
        kept = [a for a in argv_ if a not in ("--force", "--no-catalogue")]
        cmd = "python maps/generate.py " + shlex.join(kept)
        write_map(args.name or next_name(), data, args.description or desc, cmd, args.force)
    if args.catalogue or not args.no_catalogue:
        catalogue()


if __name__ == "__main__":
    main()
