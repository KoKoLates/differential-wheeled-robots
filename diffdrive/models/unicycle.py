"""Kinematic unicycle model of a differential-drive robot.

    q = [x, y, θ],  u = [v, ω],  q̇ = [v cos θ, v sin θ, ω]

The same class serves two roles:

* the simulator's **true plant** integrates it (usually with ``exact`` ZOH), and
* controllers receive a **nominal model** instance to predict with. The two may
  intentionally differ (integrator, or plant effects such as lag/slip that the
  nominal model does not know about) — that is how model mismatch is studied.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np

from diffdrive.core.geometry import wrap_angle
from diffdrive.core.types import Array

Integrator = Literal["exact", "rk4", "euler"]

_SMALL_PHI = 1e-4


def _sinc_terms(phi: float) -> tuple[float, float]:
    """Return ``a = sin(φ)/φ`` and ``b = (1 - cos φ)/φ``, stable as φ → 0."""
    if abs(phi) < _SMALL_PHI:
        phi2 = phi * phi
        return 1.0 - phi2 / 6.0, phi / 2.0 - phi * phi2 / 24.0
    return math.sin(phi) / phi, (1.0 - math.cos(phi)) / phi


class Unicycle:
    """Discrete-time unicycle with a selectable integrator.

    ``exact`` is the zero-order-hold solution for constant ``(v, ω)`` over ``dt``
    (doc §1.2), implemented with a numerically continuous small-ω branch.
    """

    state_dim = 3
    control_dim = 2

    def __init__(self, integrator: Integrator = "exact") -> None:
        if integrator not in ("exact", "rk4", "euler"):
            raise ValueError(f"unknown integrator {integrator!r}")
        self.integrator: Integrator = integrator

    # ------------------------------------------------------------------ dynamics
    @staticmethod
    def derivative(q: Array, u: Array) -> Array:
        return np.array([u[0] * math.cos(q[2]), u[0] * math.sin(q[2]), u[1]])

    def step(self, q: Array, u: Array, dt: float) -> Array:
        """Advance one step. The returned heading is wrapped to ``[-π, π)``."""
        q = np.asarray(q, dtype=float)
        u = np.asarray(u, dtype=float)
        if self.integrator == "exact":
            v, w = float(u[0]), float(u[1])
            th = float(q[2])
            a, b = _sinc_terms(w * dt)
            c, s = math.cos(th), math.sin(th)
            nxt = np.array(
                [q[0] + v * dt * (c * a - s * b), q[1] + v * dt * (s * a + c * b), th + w * dt]
            )
        elif self.integrator == "rk4":
            f = self.derivative
            k1 = f(q, u)
            k2 = f(q + 0.5 * dt * k1, u)
            k3 = f(q + 0.5 * dt * k2, u)
            k4 = f(q + dt * k3, u)
            nxt = q + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            nxt = q + dt * self.derivative(q, u)
        nxt[2] = wrap_angle(float(nxt[2]))
        return nxt

    def rollout(self, q0: Array, controls: Array, dt: float) -> Array:
        """Roll out ``controls`` ``(N, 2)`` from ``q0``; returns states ``(N+1, 3)``."""
        controls = np.atleast_2d(np.asarray(controls, dtype=float))
        out = np.empty((len(controls) + 1, 3))
        out[0] = q0
        for k, u in enumerate(controls):
            out[k + 1] = self.step(out[k], u, dt)
        return out

    # ------------------------------------------------------------ linearisation
    def jacobians(self, q: Array, u: Array, dt: float) -> tuple[Array, Array]:
        """Analytic Jacobians ``A = ∂f_d/∂q`` (3×3) and ``B = ∂f_d/∂u`` (3×2).

        TODO(M2 · LQR): derive and implement these for *this* model's discrete
        step ``f_d`` (start with ``euler``; then ``exact`` ZOH). Doc §2.3 gives
        the Euler form::

            A ≈ I + dt·[[0, 0, -v sin θ], [0, 0, v cos θ], [0, 0, 0]]
            B ≈ dt·[[cos θ, 0], [sin θ, 0], [0, 1]]

        ``tests/test_models.py::test_analytic_jacobians_match_finite_difference``
        checks your result against :func:`numerical_jacobians` once implemented.
        Until then, controllers can call :func:`numerical_jacobians`.
        """
        raise NotImplementedError("TODO(M2): implement Unicycle.jacobians (see docstring)")


def numerical_jacobians(
    step: Callable[[Array, Array, float], Array],
    q: Array,
    u: Array,
    dt: float,
    eps: float = 1e-6,
) -> tuple[Array, Array]:
    """Central finite-difference Jacobians of a discrete step function.

    Intended for verification and as a fallback. Heading differences are wrapped
    so perturbations near ±π are handled correctly.
    """
    q = np.asarray(q, dtype=float)
    u = np.asarray(u, dtype=float)

    def diff(a: Array, b: Array) -> Array:
        d = a - b
        d[2] = wrap_angle(float(d[2]))
        return d

    A = np.zeros((3, 3))
    B = np.zeros((3, 2))
    for i in range(3):
        dq = np.zeros(3)
        dq[i] = eps
        A[:, i] = diff(step(q + dq, u, dt), step(q - dq, u, dt)) / (2 * eps)
    for j in range(2):
        du = np.zeros(2)
        du[j] = eps
        B[:, j] = diff(step(q, u + du, dt), step(q, u - du, dt)) / (2 * eps)
    return A, B
