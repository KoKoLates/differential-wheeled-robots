"""Time-varying trajectory-tracking LQR (doc §2.3) — TEMPLATE, milestone M2.

Status: not implemented. ``tests/test_controller_acceptance.py`` will start
checking it automatically once ``compute`` stops raising ``NotImplementedError``
and you set ``implemented = True``.

Plan
----
1. Implement :meth:`models.Unicycle.jacobians` (run
   ``pytest tests/test_models.py -k jacobian`` to verify against finite differences).
2. Each control tick, the window gives the nominal ``q̄_k = ref.poses[k]`` and
   ``ū_k = ref.controls[k]`` for ``k = 0..N``. Linearise:
   ``A_k, B_k = self.ctx.model.jacobians(q̄_k, ū_k, ref_dt)``.
3. Backward Riccati recursion from ``P_N = Q_f``::

       K_k = (R + B_kᵀ P_{k+1} B_k)⁻¹ B_kᵀ P_{k+1} A_k      # use np.linalg.solve
       P_k = Q + A_kᵀ P_{k+1} (A_k - B_k K_k)

4. Deviation ``δq = q - q̄_0`` with the heading component wrapped, then
   ``u = ū_0 - K_0 δq``.  Optionally return ``predicted_states`` by rolling the
   policy forward on ``self.ctx.model`` so the visualiser draws it.
5. Things to study: why linearising about a *stopped* reference loses lateral
   controllability (doc §2.3 caveat); finite vs infinite horizon; world-frame
   vs body-frame error coordinates (compare with ``lie_lqr``).
"""

from __future__ import annotations

from dataclasses import dataclass

from diffdrive.core.types import ControlResult, Observation, ReferenceWindow, RefRequest
from diffdrive.controllers.base import Controller, register


@register("lqr")
class LQR(Controller):
    implemented = False
    milestone = "M2"

    @dataclass(frozen=True)
    class Config:
        horizon: int = 20
        q: tuple[float, float, float] = (10.0, 10.0, 1.0)  # diag(Q) on [x, y, θ] deviation
        r: tuple[float, float] = (1.0, 1.0)  # diag(R) on [v, ω] deviation
        qf_scale: float = 1.0  # Q_f = qf_scale · Q

    def reference_request(self) -> RefRequest:
        return RefRequest(mode="time", horizon=self.config.horizon)

    def compute(self, obs: Observation, ref: ReferenceWindow) -> ControlResult:
        raise NotImplementedError("TODO(M2): implement time-varying LQR — see module docstring")
