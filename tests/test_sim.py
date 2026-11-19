"""Simulator acceptance tests (doc §5A.11 "M0 acceptance tests")."""

from dataclasses import dataclass

import numpy as np
import pytest
from pydantic import ValidationError

from diffdrive.controllers import Controller, make_controller
from diffdrive.core import Control, ControlResult
from diffdrive.core.rng import RngStreams
from diffdrive.sim import Scenario, builtin_scenarios, load_scenario, run_episode


class Scripted(Controller):
    """Applies a fixed command sequence and records what it observed."""

    @dataclass(frozen=True)
    class Config:
        v: float = 0.5
        omega: float = 0.0
        switch_time: float = 1e9  # after this, command (v, -omega)

    def reset(self, ctx):
        super().reset(ctx)
        self.seen = []

    def compute(self, obs, ref):
        self.seen.append(obs)
        w = self.config.omega if obs.t < self.config.switch_time else -self.config.omega
        return ControlResult(Control(self.config.v, w))


def _arrays_equal(a, b, skip=("solve_time",)):
    assert set(a.arrays) == set(b.arrays)
    for k in a.arrays:
        if k not in skip:
            np.testing.assert_array_equal(a.arrays[k], b.arrays[k], err_msg=k)


def test_all_builtin_scenarios_validate():
    for name in builtin_scenarios():
        sc = load_scenario(name)
        assert sc.name == name
        sc.build_reference()  # its default map exists and is feasible


def test_ideal_feedforward_reproduces_reference(quick_circle):
    ep = run_episode(quick_circle, make_controller("feedforward"))
    ref, _, _ = ep.reference.sample(ep["t"])
    assert np.abs(ep["truth"][:, :2] - ref[:, :2]).max() < 1e-3
    assert ep.termination == "time_limit"


def test_determinism_same_seed(quick_circle):
    sc = quick_circle.with_overrides(
        {
            "sensor.noise_std": [0.02, 0.02, 0.01],
            "sensor.dropout": 0.2,
            "plant.process_noise_std": [0.01, 0.01, 0.01],
        }
    )
    a = run_episode(sc, make_controller("pid"), seed=7)
    b = run_episode(sc, make_controller("pid"), seed=7)
    _arrays_equal(a, b)
    c = run_episode(sc, make_controller("pid"), seed=8)
    assert not np.array_equal(a["truth"], c["truth"])


def test_noise_is_paired_across_controllers(quick_circle):
    """Exogenous noise must not depend on which controller runs."""
    sc = quick_circle.with_overrides({"sensor.noise_std": [0.02, 0.02, 0.01]})
    a = run_episode(sc, make_controller("pid"), seed=3)
    b = run_episode(sc, make_controller("pure_pursuit"), seed=3)
    noise_a = a["sensor_pose"][:5] - a["truth"][:: int(0.05 / 0.005)][:5]
    noise_b = b["sensor_pose"][:5] - b["truth"][:: int(0.05 / 0.005)][:5]
    assert noise_a == pytest.approx(noise_b, abs=1e-12)


def test_rng_streams_independent():
    a = RngStreams(1)
    x1 = a.get("sensor.noise").standard_normal(5)
    b = RngStreams(1)
    b.get("sensor.dropout").random(1000)  # drawing from another stream...
    x2 = b.get("sensor.noise").standard_normal(5)  # ...does not change this one
    assert x1 == pytest.approx(x2)


def test_dropout_does_not_change_noise_values(quick_circle):
    base = quick_circle.with_overrides({"sensor.noise_std": [0.02, 0.02, 0.01]})
    a = run_episode(base, make_controller("feedforward"), seed=5)
    b = run_episode(
        base.with_overrides({"sensor.dropout": 0.5}), make_controller("feedforward"), seed=5
    )
    np.testing.assert_allclose(
        a["sensor_pose"], b["sensor_pose"]
    )  # feedforward ignores obs → same truth
    assert b["sensor_dropped"].mean() > 0.2


def test_sensor_latency_never_exposes_future(quick_circle):
    sc = quick_circle.with_overrides({"sensor.latency": 0.03, "sensor.period": 0.01})
    ctrl = Scripted()
    run_episode(sc, ctrl)
    for obs in ctrl.seen[1:]:
        assert obs.t - obs.t_sample >= 0.03 - 1e-9
        assert obs.t - obs.t_sample < 0.03 + 0.01 + 1e-9  # newest available sample


def test_command_latency_delays_application(quick_circle):
    sc = quick_circle.with_overrides({"command.latency": 0.05})
    ep = run_episode(sc, Scripted(v=0.4))
    t, applied = ep["t"], ep["applied"]
    assert np.all(applied[t < 0.05 - 1e-9, 0] == 0.0)
    assert applied[np.searchsorted(t, 0.05), 0] == pytest.approx(0.4)


def test_saturation_and_slew_are_enforced(quick_circle):
    sc = quick_circle.with_overrides({"command.accel_max": 0.5, "command.v_max": 0.6})
    ep = run_episode(sc, Scripted(v=5.0, omega=10.0, switch_time=0.5))
    cmd = ep["command"]
    assert cmd[:, 0].max() <= 0.6 + 1e-12
    assert np.abs(cmd[:, 1]).max() <= 1.5 + 1e-12
    assert np.abs(np.diff(cmd[:, 0])).max() <= 0.5 * 0.05 + 1e-12
    assert (ep["raw"][:, 0] == 5.0).all()  # raw output is logged unmodified


def test_actuator_lag_is_first_order(quick_circle):
    sc = quick_circle.with_overrides({"plant.tau_v": 0.2})
    ep = run_episode(sc, Scripted(v=0.5))
    t, v = ep["t"], ep["applied"][:, 0]
    i = np.searchsorted(t, 0.2)
    assert v[i] == pytest.approx(0.5 * (1 - np.exp(-(t[i] + 0.005) / 0.2)), rel=1e-6)


def test_initial_error_in_body_frame(line_map):
    sc = Scenario.from_dict({"map": line_map, "initial_error": {"dy": 0.3, "dtheta": 0.2}})
    ep = run_episode(sc, make_controller("feedforward"))
    assert ep["truth"][0] == pytest.approx([0.0, 0.3, 0.2])


def test_tracking_failure_terminates_early(line_map):
    sc = Scenario.from_dict({"map": line_map, "initial_error": {"dtheta": 1.57}})
    ep = run_episode(sc, make_controller("feedforward"))
    assert ep.termination == "tracking_failure"


def test_controller_errors_are_recorded_when_requested(quick_circle):
    ep = run_episode(quick_circle, make_controller("lqr"), raise_errors=False)
    assert ep.termination == "controller_error" and "NotImplementedError" in ep.meta["error"]
    with pytest.raises(NotImplementedError):
        run_episode(quick_circle, make_controller("lqr"))


def test_incompatible_time_grid_fails_loudly():
    with pytest.raises(ValueError, match="integer multiple"):
        Scenario.from_dict({"timing": {"dt_physics": 0.005, "dt_control": 0.0123}})


def test_curvature_cap_keeps_fast_reference_feasible():
    sc = Scenario.from_dict(
        {
            "map": "map01",
            "speed": {"cruise": 2.0},
            "command": {"v_max": 2.0, "omega_max": 0.1},
        }
    )
    # curvature cap keeps it feasible: v ≤ 0.8·ω_max/κ
    run_episode(sc, make_controller("feedforward"))


def test_scenario_base_and_overrides(tmp_path):
    f = tmp_path / "s.yaml"
    f.write_text(
        "base: lateral_offset\noverrides: {sensor.noise_std: [0.1, 0.1, 0.0]}\n"
    )
    sc = load_scenario(f)
    assert sc.name == "s"  # defaults to the file name
    assert sc.map == "map01" and sc.initial_error.dy == 0.5  # inherited from the base
    assert sc.sensor.noise_std == (0.1, 0.1, 0.0)
    with pytest.raises(ValidationError):
        Scenario.from_dict({"sensr": {}})  # typos are rejected
