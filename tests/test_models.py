import math

import numpy as np
import pytest

from diffdrive.models import Unicycle, numerical_jacobians


@pytest.mark.parametrize("integrator", ["exact", "rk4", "euler"])
def test_zero_input_keeps_pose(integrator):
    q = np.array([0.3, -1.2, 2.0])
    assert Unicycle(integrator).step(q, np.zeros(2), 0.1) == pytest.approx(q)


@pytest.mark.parametrize("integrator", ["exact", "rk4", "euler"])
def test_straight_line(integrator):
    q = np.array([1.0, 2.0, math.pi / 6])
    nxt = Unicycle(integrator).step(q, np.array([0.8, 0.0]), 0.5)
    assert nxt == pytest.approx(
        [1.0 + 0.4 * math.cos(math.pi / 6), 2.0 + 0.4 * math.sin(math.pi / 6), math.pi / 6]
    )


def test_exact_matches_circular_arc():
    v, w, T = 0.5, 0.7, 3.0
    model = Unicycle("exact")
    q = model.step(np.zeros(3), np.array([v, w]), T)
    r = v / w
    expected = [r * math.sin(w * T), r * (1 - math.cos(w * T)), w * T]
    assert q == pytest.approx(expected, abs=1e-12)


def test_exact_is_continuous_at_small_omega():
    model = Unicycle("exact")
    q = np.array([0.0, 0.0, 0.4])
    a = model.step(q, np.array([1.0, 1e-5]), 0.05)  # series branch
    b = model.step(q, np.array([1.0, 1.0001e-3]), 0.05)  # closed-form branch
    c = model.step(q, np.array([1.0, 0.0]), 0.05)
    assert a == pytest.approx(c, abs=1e-6)
    assert b == pytest.approx(a, abs=1e-4)


def test_exact_matches_fine_rk4():
    model, oracle = Unicycle("exact"), Unicycle("rk4")
    q0 = np.array([0.1, 0.2, -2.9])
    u = np.array([0.9, -1.3])
    q_exact = model.step(q0, u, 0.2)
    q_rk = q0.copy()
    for _ in range(200):
        q_rk = oracle.step(q_rk, u, 0.001)
    assert q_exact == pytest.approx(q_rk, abs=1e-9)


def test_rollout_shape_and_wrap():
    X = Unicycle().rollout(np.zeros(3), np.tile([0.5, 1.5], (40, 1)), 0.1)
    assert X.shape == (41, 3)
    assert np.all(X[:, 2] >= -math.pi) and np.all(X[:, 2] < math.pi)


def test_numerical_jacobians_euler_closed_form():
    model = Unicycle("euler")
    q, u, dt = np.array([0.0, 0.0, 0.3]), np.array([0.7, 0.2]), 0.05
    A, B = numerical_jacobians(model.step, q, u, dt)
    A_ref = np.eye(3) + dt * np.array(
        [[0, 0, -0.7 * math.sin(0.3)], [0, 0, 0.7 * math.cos(0.3)], [0, 0, 0]]
    )
    B_ref = dt * np.array([[math.cos(0.3), 0], [math.sin(0.3), 0], [0, 1]])
    assert A == pytest.approx(A_ref, abs=1e-7) and B == pytest.approx(B_ref, abs=1e-7)


@pytest.mark.parametrize("integrator", ["euler", "exact", "rk4"])
def test_analytic_jacobians_match_finite_difference(integrator):
    """M2 acceptance test: activates once Unicycle.jacobians is implemented."""
    model = Unicycle(integrator)
    rng = np.random.default_rng(0)
    try:
        model.jacobians(np.zeros(3), np.array([0.5, 0.1]), 0.05)
    except NotImplementedError:
        pytest.skip("Unicycle.jacobians not implemented yet (M2)")
    for _ in range(20):
        q = rng.uniform([-2, -2, -math.pi], [2, 2, math.pi])
        u = rng.uniform([0, -1.5], [1, 1.5])
        A, B = model.jacobians(q, u, 0.05)
        A_n, B_n = numerical_jacobians(model.step, q, u, 0.05)
        assert A == pytest.approx(A_n, abs=1e-6)
        assert B == pytest.approx(B_n, abs=1e-6)
