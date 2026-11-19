"""Lie-group LQR and MPC on SE(2) (doc §2.7–2.8) — TEMPLATES, milestone M6.

Build on :mod:`core.se2` (implement and test it first:
``pytest tests/test_se2.py``).

Lie-group LQR
-------------
* Error ``e = Log(T_r⁻¹ T)^∨`` (state the left/right-invariant convention you
  pick in the docstring and keep it everywhere).
* Derive the error dynamics for ``Ṫ = T ξ̂`` with ``ξ = [v, 0, ω]`` and linearise
  in algebra coordinates (the adjoint ``Ad`` appears); discretise; then reuse
  exactly the Riccati recursion from :mod:`lqr`.
* Key experiment (doc §2.7): compare against body-frame LQR with matched
  costs; demonstrate — don't presume — any improvement.

Lie-group MPC
-------------
* Implement (a) the Lie-linearised QP first (reuse the linear MPC machinery
  with the error dynamics above), optionally (b) a nonlinear NLP with
  group-consistent integration ``T_{k+1} = T_k Exp(dt · ξ̂_k)``.
* Ablate error coordinates vs integrator vs solver separately (doc §4.5).
"""

from __future__ import annotations

from dataclasses import dataclass

from diffdrive.core.types import ControlResult, Observation, ReferenceWindow, RefRequest
from diffdrive.controllers.base import Controller, register


@register("lie_lqr")
class LieLQR(Controller):
    implemented = False
    milestone = "M6"

    @dataclass(frozen=True)
    class Config:
        horizon: int = 20
        q: tuple[float, float, float] = (10.0, 10.0, 1.0)
        r: tuple[float, float] = (1.0, 1.0)
        qf_scale: float = 1.0

    def reference_request(self) -> RefRequest:
        return RefRequest(mode="time", horizon=self.config.horizon)

    def compute(self, obs: Observation, ref: ReferenceWindow) -> ControlResult:
        raise NotImplementedError("TODO(M6): implement Lie-group LQR — see module docstring")


@register("lie_mpc")
class LieMPC(Controller):
    implemented = False
    milestone = "M6"

    @dataclass(frozen=True)
    class Config:
        horizon: int = 20
        q: tuple[float, float, float] = (10.0, 10.0, 1.0)
        r: tuple[float, float] = (0.1, 0.1)
        s: tuple[float, float] = (1.0, 1.0)
        qf_scale: float = 2.0
        formulation: str = "linearized"  # "linearized" (QP) | "nonlinear" (NLP)

    def reference_request(self) -> RefRequest:
        return RefRequest(mode="time", horizon=self.config.horizon)

    def compute(self, obs: Observation, ref: ReferenceWindow) -> ControlResult:
        raise NotImplementedError("TODO(M6): implement Lie-group MPC — see module docstring")
