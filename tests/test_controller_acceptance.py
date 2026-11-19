"""Acceptance tests for every *implemented* controller, including your own.

Templates (``implemented = False``) are skipped. When you finish one, set
``implemented = True`` on the class and these tests start checking it.
"""

import pytest

from diffdrive.controllers import available_controllers, make_controller
from diffdrive.evaluation import compute_metrics
from diffdrive.sim import load_scenario, run_episode

CONTROLLERS = sorted(n for n in available_controllers() if n != "feedforward")


def _get(name):
    cls = available_controllers()[name]
    if not cls.implemented:
        pytest.skip(f"{name} is a template ({cls.milestone})")
    return make_controller(name)


@pytest.mark.parametrize("name", CONTROLLERS)
def test_tracks_nominal_circle(name):
    ctrl = _get(name)
    m = compute_metrics(run_episode(load_scenario("nominal").with_overrides({"map": "map01"}), ctrl))
    assert m["success"], m
    metric = "cte_rmse" if ctrl.reference_request().mode == "path" else "pos_rmse"
    assert m[metric] < 0.05, m


@pytest.mark.parametrize("name", CONTROLLERS)
def test_recovers_from_lateral_offset(name):
    m = compute_metrics(run_episode(load_scenario("lateral_offset"), _get(name)))
    assert m["termination"] == "time_limit" and m["goal_distance"] < 0.1, m


@pytest.mark.parametrize("name", CONTROLLERS)
def test_survives_noise_and_latency(name):
    m = compute_metrics(run_episode(load_scenario("noisy_delayed"), _get(name), seed=0))
    assert m["termination"] == "time_limit", m
    assert m["cte_rmse"] < 0.15, m
