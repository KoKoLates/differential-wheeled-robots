import math

import pytest

from diffdrive.controllers import available_controllers, get_controller_class, make_controller
from diffdrive.evaluation import compute_metrics
from diffdrive.sim import load_scenario, run_episode


def test_registry_lists_builtins():
    names = available_controllers()
    for n in (
        "feedforward",
        "pid",
        "pure_pursuit",
        "lqr",
        "ilqr",
        "linear_mpc",
        "nonlinear_mpc",
        "lie_lqr",
        "lie_mpc",
    ):
        assert n in names
    assert names["pid"].implemented and not names["lqr"].implemented


def test_config_validation():
    assert make_controller("pid", {"kp_theta": 2.0}).config.kp_theta == 2.0
    with pytest.raises(TypeError, match="unknown config key"):
        make_controller("pid", {"kp_thta": 2.0})
    with pytest.raises(KeyError):
        make_controller("nope")


def test_import_path_resolution():
    assert get_controller_class("diffdrive.controllers.pid:PID").name == "pid"


@pytest.mark.parametrize(
    ("map_name", "scenario"),
    [
        ("map01", "nominal"),
        ("map02", "nominal"),
        ("map03", "nominal"),
        ("map03", "stop_and_go"),
        ("map01", "lateral_offset"),
        ("map00", "heading_offset"),
    ],
)
@pytest.mark.parametrize(
    ("controller", "config", "metric", "tol"),
    [
        ("pid", {}, "pos_rmse", 0.12),
        ("pid", {"mode": "path"}, "cte_rmse", 0.12),
        ("pure_pursuit", {}, "cte_rmse", 0.12),
    ],
)
def test_baselines_track(map_name, scenario, controller, config, metric, tol):
    sc = load_scenario(scenario).with_overrides({"map": map_name})
    m = compute_metrics(run_episode(sc, make_controller(controller, config)))
    assert m["termination"] == "time_limit"
    assert m["completion"] > 0.98
    assert m[metric] < tol, m


def test_pid_rejects_lateral_offset_better_than_feedforward():
    sc = load_scenario("lateral_offset")
    pid = compute_metrics(run_episode(sc, make_controller("pid")))
    ff = compute_metrics(run_episode(sc, make_controller("feedforward")))
    assert pid["pos_final"] < 0.05 < ff["pos_final"]


def test_pid_heading_wraps():
    sc = load_scenario("heading_offset").with_overrides({"initial_error.dtheta": math.pi - 0.05})
    m = compute_metrics(run_episode(sc, make_controller("pid")))
    assert m["termination"] == "time_limit"
