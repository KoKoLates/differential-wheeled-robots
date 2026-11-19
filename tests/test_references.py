import math

import numpy as np
import pytest

from diffdrive.core import Limits
from helpers import SHAPES, circle, figure_eight, line, map_file, s_curve

from diffdrive.references import (
    Path,
    ProgressTracker,
    SpeedProfile,
    Trajectory,
    available_maps,
    check_feasibility,
    load_map,
    map_info,
)

LIM = Limits()


def test_circle_geometry():
    p = circle(1.5)
    assert p.length == pytest.approx(2 * math.pi * 1.5, rel=1e-6)
    assert np.allclose(p.kappa, 1 / 1.5, atol=1e-3)
    assert p.start_pose == pytest.approx([0, 0, 0], abs=1e-6)
    cw = circle(1.5, direction="cw")
    assert np.allclose(cw.kappa, -1 / 1.5, atol=1e-3)


def test_line_and_transform():
    p = line(3.0).transformed(1.0, 2.0, math.pi / 2)
    xy, th, _ = p.sample([0.0, 3.0])
    assert xy == pytest.approx(np.array([[1.0, 2.0], [1.0, 5.0]]), abs=1e-9)
    assert th == pytest.approx([math.pi / 2] * 2)


@pytest.mark.parametrize("family", ["line", "circle", "figure_eight", "s_curve"])
def test_trajectory_is_feasible_and_consistent(family):
    tr = Trajectory.from_path(SHAPES[family](), SpeedProfile(), LIM)
    rep = check_feasibility(tr, LIM)
    assert rep.ok, rep.violations
    assert rep.consistency_error < 1e-3
    _poses, u, _ = tr.sample([0.0, tr.duration + 1.0])
    assert u[0] == pytest.approx([0.0, 0.0])  # starts at rest
    assert u[1] == pytest.approx([0.0, 0.0])  # holds at rest after the end


def test_speed_profile_respects_curvature_cap():
    p = figure_eight()
    v = SpeedProfile(cruise=1.0, omega_fraction=0.5)
    tr = Trajectory.from_path(p, v, LIM)
    _, u, _ = tr.sample(np.linspace(0, tr.duration, 2000))
    assert np.abs(u[:, 1]).max() <= 0.5 * LIM.omega_max + 1e-3


def test_stop_and_dwell():
    p = line(4.0)
    tr = Trajectory.from_path(p, SpeedProfile(stops=(0.5,), stop_duration=1.5), LIM)
    plain = Trajectory.from_path(p, SpeedProfile(stops=(0.5,)), LIM)
    assert tr.duration == pytest.approx(plain.duration + 1.5)
    t_stop = float(tr.time_at(2.0)[0])
    _, u, s = tr.sample([t_stop + 0.5, t_stop + 1.0])
    assert u[:, 0] == pytest.approx([0.0, 0.0])
    assert s == pytest.approx([2.0, 2.0], abs=1e-6)


def test_time_window_and_path_window():
    tr = Trajectory.from_path(circle(), SpeedProfile(), LIM)
    w = tr.window_time(2.0, 10, 0.1)
    assert w.horizon == 10 and w.times[-1] == pytest.approx(3.0)
    assert np.all(np.diff(w.s) >= 0)
    pw = tr.window_path(1.0, 5, 0.05)
    assert pw.s == pytest.approx(1.0 + 0.05 * np.arange(6))
    assert pw.mode == "path" and not pw.finished
    assert tr.window_path(tr.length, 3, 0.05).finished


def test_progress_is_monotonic_through_figure_eight_crossing():
    p = figure_eight()
    tracker = ProgressTracker(p)
    s_true = np.linspace(0, p.length, 600)
    xy, _, _ = p.sample(s_true)
    est = [tracker.update(pt, 0.2) for pt in xy]
    assert np.all(np.diff(est) >= -1e-12)
    assert est == pytest.approx(s_true, abs=1e-3)


def test_shipped_maps_load_and_are_feasible():
    names = available_maps()
    assert {"map00", "map01", "map02", "map03"} <= set(names)
    for name in names:
        p = load_map(name)
        assert p.length == pytest.approx(map_info(name)["length"], abs=0.02)
        assert p.start_pose == pytest.approx([0.0, 0.0, p.start_pose[2]], abs=1e-6)
        rep = check_feasibility(Trajectory.from_path(p, SpeedProfile(), LIM), LIM)
        assert rep.ok, (name, rep.violations)


def test_map_roundtrip_preserves_geometry(tmp_path):
    p = figure_eight()
    q = load_map(map_file(tmp_path, "eight", p))
    s = np.linspace(0, p.length, 400)
    assert q.sample(s)[0] == pytest.approx(p.sample(s)[0], abs=1e-5)
    assert q.sample(s)[2] == pytest.approx(p.sample(s)[2], abs=1e-4)  # curvature
    assert np.all(np.abs(np.diff(q.theta)) < 0.1)  # heading unwrapped on load
    assert load_map(tmp_path / "eight.csv").length == pytest.approx(q.length)


def test_map_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="available"):
        load_map("no_such_map")
    bad = tmp_path / "bad.csv"
    bad.write_text("x,y\n0,0\n1,0\n")
    with pytest.raises(ValueError, match="header"):
        load_map(bad)


def test_trajectory_roundtrip_arrays():
    tr = Trajectory.from_path(s_curve(), SpeedProfile(), LIM)
    tr2 = Trajectory.from_arrays(tr.to_arrays())
    t = np.linspace(0, tr.duration, 50)
    assert tr2.sample(t)[0] == pytest.approx(tr.sample(t)[0])


def test_path_requires_two_points():
    with pytest.raises(ValueError):
        Path.from_points(np.zeros((3, 2)))
