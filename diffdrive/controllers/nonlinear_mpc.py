"""Nonlinear MPC with multiple shooting (doc §2.6) — TEMPLATE, milestone M5.

    min_{q,u}  ℓ_f(q_N) + Σ_k ℓ(q_k, u_k) + ‖u_k - u_{k-1}‖²_S
    s.t. q_0 = q̂,  q_{k+1} = f_d(q_k, u_k),  u_k ∈ U,  Δu_k ∈ D

Implementation notes
--------------------
* ``pip install -e ".[mpc]"`` for CasADi + IPOPT. Build the NLP once in
  ``reset`` with ``casadi.Opti`` (or the lower-level ``nlpsol``); parameters
  are ``q̂``, the reference window and ``u_{-1}``.
* Use the *same* discrete dynamics as ``self.ctx.model`` (exact ZOH or RK4) so
  the comparison with linear MPC isolates linearisation error (doc §2.6 "Fair
  comparison"). Wrap heading errors (e.g. ``atan2(sin, cos)``) in the cost.
* Warm start primal (and dual) variables from the previous solution.
* Record solver status, iterations and objective in ``diagnostics``; return
  ``Status.FALLBACK`` with a safe command on failure — never hide failures.
"""

from __future__ import annotations

from dataclasses import dataclass

from diffdrive.core.types import ControlResult, Observation, ReferenceWindow, RefRequest
from diffdrive.controllers.base import Controller, register


@register("nonlinear_mpc")
class NonlinearMPC(Controller):
    implemented = False
    milestone = "M5"

    @dataclass(frozen=True)
    class Config:
        horizon: int = 20
        q: tuple[float, float, float] = (10.0, 10.0, 1.0)
        r: tuple[float, float] = (0.1, 0.1)
        s: tuple[float, float] = (1.0, 1.0)
        qf_scale: float = 2.0
        max_iter: int = 100
        tol: float = 1e-6

    def reference_request(self) -> RefRequest:
        return RefRequest(mode="time", horizon=self.config.horizon)

    def compute(self, obs: Observation, ref: ReferenceWindow) -> ControlResult:
        raise NotImplementedError("TODO(M5): implement nonlinear MPC — see module docstring")
