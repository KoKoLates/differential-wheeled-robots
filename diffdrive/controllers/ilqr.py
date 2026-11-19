"""Receding-horizon iLQR (doc §2.4) — TEMPLATE, milestone M3.

Optimise ``u_{0:N-1}`` for the nonlinear model with a quadratic tracking cost::

    J = ℓ_f(q_N) + Σ_k ℓ(q_k, u_k),   ℓ = ‖q_k ⊖ q̄_k‖²_Q + ‖u_k - ū_k‖²_R

where ``⊖`` wraps the heading difference and ``q̄, ū`` come from the window.

Suggested structure (keep each piece unit-testable)
---------------------------------------------------
* ``_rollout(q0, U)`` → ``self.ctx.model.rollout(q0, U, dt)``.
* ``_cost(X, U, ref)`` — total cost (used by the line search).
* ``_backward(X, U, ref, reg)`` → gains ``K_k`` (2×3), ``k_k`` (2,) via::

      Q_x  = ℓ_x  + A_kᵀ V_x          Q_u  = ℓ_u  + B_kᵀ V_x
      Q_xx = ℓ_xx + A_kᵀ V_xx A_k     Q_uu = ℓ_uu + B_kᵀ V_xx B_k + reg·I
      Q_ux = ℓ_ux + B_kᵀ V_xx A_k
      k_k = -Q_uu⁻¹ Q_u               K_k = -Q_uu⁻¹ Q_ux
      V_x  = Q_x  + K_kᵀ Q_uu k_k + K_kᵀ Q_u + Q_uxᵀ k_k
      V_xx = Q_xx + K_kᵀ Q_uu K_k + K_kᵀ Q_ux + Q_uxᵀ K_k

* ``_forward(X, U, K, k, alpha)`` — closed-loop rollout
  ``u_k = U_k + α k_k + K_k (x_k - X_k)``; backtrack α until the cost decreases.
* Loop until relative improvement < ``tol`` or ``max_iters``; adapt ``reg``.
* Warm start: shift last solution by one step (keep it in ``self._U``; clear in
  ``reset``). Log ``diagnostics={"iterations": ..., "cost": ...}`` and return
  ``predicted_states=X``, ``predicted_controls=U`` so the animation shows them.
* Return ``Status.FALLBACK`` (e.g. with ``ū_0``) if the cost became non-finite.
"""

from __future__ import annotations

from dataclasses import dataclass

from diffdrive.core.types import ControlResult, Observation, ReferenceWindow, RefRequest
from diffdrive.controllers.base import Controller, ControllerContext, register


@register("ilqr")
class ILQR(Controller):
    implemented = False
    milestone = "M3"

    @dataclass(frozen=True)
    class Config:
        horizon: int = 20
        q: tuple[float, float, float] = (10.0, 10.0, 1.0)
        r: tuple[float, float] = (0.5, 0.5)
        qf_scale: float = 2.0
        max_iters: int = 20
        tol: float = 1e-4
        reg_init: float = 1e-6
        line_search: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.05)

    def reset(self, ctx: ControllerContext) -> None:
        super().reset(ctx)
        self._U = None  # warm-start control sequence

    def reference_request(self) -> RefRequest:
        return RefRequest(mode="time", horizon=self.config.horizon)

    def compute(self, obs: Observation, ref: ReferenceWindow) -> ControlResult:
        raise NotImplementedError("TODO(M3): implement iLQR — see module docstring")
