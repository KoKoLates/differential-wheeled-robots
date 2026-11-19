"""Linear time-varying MPC as a sparse QP (doc §2.5) — TEMPLATE, milestone M4.

Decision variables (deviations around the window's nominal ``q̄_k, ū_k``)::

    δq_0..δq_N,  δu_0..δu_{N-1}

    min  Σ ‖δq_k‖²_Q + ‖δu_k‖²_R + ‖u_k - u_{k-1}‖²_S  + ‖δq_N‖²_{Q_f}
    s.t. δq_0 = q̂ ⊖ q̄_0                                  (initial-state constraint)
         δq_{k+1} = A_k δq_k + B_k δu_k + d_k              (d_k = affine defect)
         u_min ≤ ū_k + δu_k ≤ u_max                        (self.ctx.limits)
         |u_k - u_{k-1}| ≤ rate·dt   (u_{-1} = obs.last_command)

Implementation notes
--------------------
* ``pip install -e ".[mpc]"`` for OSQP; build ``P``/``A`` with ``scipy.sparse``.
* Set up the OSQP problem once in ``reset`` and only ``update`` the vectors /
  matrix values each tick (warm start). Log ``status``, iterations, objective.
* Map OSQP failure to ``Status.FALLBACK`` with a safe command (e.g. ``ū_0``).
* Return ``predicted_states``/``predicted_controls`` for the visualiser.
"""

from __future__ import annotations

from dataclasses import dataclass

from diffdrive.core.types import ControlResult, Observation, ReferenceWindow, RefRequest
from diffdrive.controllers.base import Controller, register


@register("linear_mpc")
class LinearMPC(Controller):
    implemented = False
    milestone = "M4"

    @dataclass(frozen=True)
    class Config:
        horizon: int = 20
        q: tuple[float, float, float] = (10.0, 10.0, 1.0)
        r: tuple[float, float] = (0.1, 0.1)
        s: tuple[float, float] = (1.0, 1.0)  # input-rate weight
        qf_scale: float = 2.0
        eps_abs: float = 1e-4
        max_iter: int = 4000

    def reference_request(self) -> RefRequest:
        return RefRequest(mode="time", horizon=self.config.horizon)

    def compute(self, obs: Observation, ref: ReferenceWindow) -> ControlResult:
        raise NotImplementedError("TODO(M4): implement linear MPC — see module docstring")
