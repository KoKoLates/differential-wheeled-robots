"""SE(2) Lie-group utilities — TEMPLATE, milestone M6.

Conventions to keep consistent everywhere (write them down when you choose):

* ``T = [[R(θ), p], [0, 1]]``; twist ``ξ = [ρ_x, ρ_y, φ]`` with
  ``ξ̂ = [[0, -φ, ρ_x], [φ, 0, ρ_y], [0, 0, 0]]``.
* ``Exp(ξ) = [[R(φ), V(φ) ρ], [0, 1]]`` with
  ``V(φ) = (1/φ)[[sin φ, -(1 - cos φ)], [1 - cos φ, sin φ]]`` (→ I as φ → 0).
* ``Log`` inverts ``Exp`` for ``φ ∈ (-π, π]``; use a series near ``φ = 0``.
* ``Ad(T) = [[R, [p_y, -p_x]ᵀ], [0, 1]]`` for the ``[ρ, φ]`` ordering above.

``tests/test_se2.py`` contains identity/round-trip tests that activate as soon
as these stop raising ``NotImplementedError``. Note that the translation part
of ``Log(T_r⁻¹ T)`` is *not* the body-frame position error (doc §1.4) — the
``V(φ)⁻¹`` factor is exactly the difference.
"""

from __future__ import annotations

from diffdrive.core.types import Array


def hat(xi: Array) -> Array:
    """``[ρ_x, ρ_y, φ]`` → 3×3 Lie-algebra matrix."""
    raise NotImplementedError("TODO(M6): se2.hat")


def vee(xi_hat: Array) -> Array:
    """Inverse of :func:`hat`."""
    raise NotImplementedError("TODO(M6): se2.vee")


def from_pose(pose: Array) -> Array:
    """``[x, y, θ]`` → homogeneous 3×3 matrix."""
    raise NotImplementedError("TODO(M6): se2.from_pose")


def to_pose(T: Array) -> Array:
    """Homogeneous 3×3 matrix → ``[x, y, θ]`` (θ wrapped)."""
    raise NotImplementedError("TODO(M6): se2.to_pose")


def inverse(T: Array) -> Array:
    """Closed-form inverse ``[[Rᵀ, -Rᵀp], [0, 1]]`` (do not call ``np.linalg.inv``)."""
    raise NotImplementedError("TODO(M6): se2.inverse")


def exp(xi: Array) -> Array:
    """Exponential map ``ℝ³ → SE(2)``."""
    raise NotImplementedError("TODO(M6): se2.exp")


def log(T: Array) -> Array:
    """Logarithm map ``SE(2) → ℝ³``."""
    raise NotImplementedError("TODO(M6): se2.log")


def adjoint(T: Array) -> Array:
    """3×3 adjoint matrix of ``T``."""
    raise NotImplementedError("TODO(M6): se2.adjoint")
