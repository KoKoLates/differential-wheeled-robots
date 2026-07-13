import math
import numpy as np

from core import Waypoint
from control.base import Control
from control.mpc.model import Model

from dataclasses import dataclass, field


@dataclass
class ILQRParams:
    horizon: int  = 10
    Q: np.ndarray  = field(default_factory=lambda: np.diag([10.0, 5.0]))
    R: np.ndarray  = field(default_factory=lambda: np.diag([0.5, 0.5]))
    Rd: np.ndarray = field(default_factory=lambda: np.diag([0.5, 1.0]))

    v_max: float = 10
    w_max: float = math.pi
    threshold: float = 15

    max_iters: int = 20
    tol: float = 1e-3   # stop iterating once cost improvement drops below this
    reg: float = 1e-6   # regularization added to Q_uu before inverting it


class ILQR(Control):
    def __init__(self, params: ILQRParams = ILQRParams()) -> None:
        super().__init__()
        self.params = params
        self.reset()

    def reset(self) -> None:
        """ Drop the warm-start control sequence, so the next `update()`
        re-seeds from scratch instead of continuing from a stale guess. """
        self._u_seq: np.ndarray = None

    def update(
        self,
        state: np.ndarray,
        waypoints: list[Waypoint],
        index: int,
        dt: float = 0.5,
    ) -> tuple[float, float, int]:
        """ One receding-horizon iLQR step: warm-start a control sequence,
        run it to local convergence via `_solve`, apply the first control.

        @param state: current robot state `(x, y, theta)`
        @param waypoints: full waypoint list
        @param index: index of the current target waypoint
        @param dt: timestep
        @return: `(v, w, new_index)` -- same contract as every other Control
        """
        target = np.array((waypoints[index].x, waypoints[index].y))
        pose = tuple(state[:2].flatten())

        ## TODO 1: build the warm-started initial control sequence.
        ##   - if `self._u_seq` exists, shift it by one step (drop u[0],
        ##     append a copy of the last entry) as the guess for this call.
        ##   - otherwise seed with something reasonable, e.g.
        ##     `np.tile([1.0, 0.0], (self.params.horizon, 1))`.
        ## TODO 2: call `self._solve(state, target, u_init, dt)` to get
        ##   the converged `(x_seq, u_seq)`.
        ## TODO 3: save `u_seq` to `self._u_seq` for the next call's warm start.
        ## TODO 4: take `ctrl_v, ctrl_w = u_seq[0]`, clip to
        ##   `[0, v_max]` / `[-w_max, w_max]`.
        ## TODO 5: advance `index` the same way `MPC.update` does -- +1 once
        ##   distance from `pose` to `target` drops below `self.params.threshold`.
        ...

    def _solve(
        self,
        state: np.ndarray,
        target: np.ndarray,
        u_seq: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """ Run backward/forward passes to local convergence.

        @param state: state to roll out from
        @param target: `(x, y)` target this horizon is tracking
        @param u_seq: initial/warm-started controls, shape (horizon, 2)
        @param dt: timestep
        @return: `(x_seq, u_seq)`, the converged nominal trajectory + controls
        """
        ## TODO 1: roll out `x_seq` from `state`/`u_seq` via `Model.rollout`,
        ##   and compute the starting cost via `self._cost`.
        ## TODO 2: loop up to `self.params.max_iters` times:
        ##   - backward pass: `K_seq, k_seq = self._backward_pass(x_seq, u_seq, target)`
        ##   - forward pass:  `new_x_seq, new_u_seq, new_cost = self._forward_pass(
        ##         x_seq, u_seq, K_seq, k_seq, target, dt)`
        ##   - if `cost - new_cost < self.params.tol`: stop (converged)
        ##   - else: `x_seq, u_seq, cost = new_x_seq, new_u_seq, new_cost` and continue
        ...

    def _backward_pass(
        self,
        x_seq: np.ndarray,
        u_seq: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """ Riccati-like backward recursion around the nominal trajectory.

        @return: `(K_seq, k_seq)` -- feedback gains (horizon, 2, 3) and
                 feedforward terms (horizon, 2)
        """
        ## TODO 1: initialize the terminal value function `V_x`, `V_xx` from
        ##   the terminal cost's gradient/Hessian at `x_seq[:, -1]` (get
        ##   these from `self._stage_derivatives`, ignoring its `l_u` parts).
        ## TODO 2: for k = horizon-1 down to 0:
        ##   - `A, B = Model.jacobian(x_seq[:, k], u_seq[k], dt)`
        ##   - `l_x, l_u, l_xx, l_uu, l_ux = self._stage_derivatives(
        ##         x_seq[:, k], u_seq[k], target)`
        ##   - `Q_x  = l_x  + A.T @ V_x`
        ##   - `Q_u  = l_u  + B.T @ V_x`
        ##   - `Q_xx = l_xx + A.T @ V_xx @ A`
        ##   - `Q_uu = l_uu + B.T @ V_xx @ B + self.params.reg * np.eye(2)`
        ##   - `Q_ux = l_ux + B.T @ V_xx @ A`
        ##   - `k_k = -np.linalg.solve(Q_uu, Q_u)`
        ##   - `K_k = -np.linalg.solve(Q_uu, Q_ux)`
        ##   - store `K_k`, `k_k` at index `k`
        ##   - update `V_x`, `V_xx` for the next (earlier) `k` using
        ##     `k_k`, `K_k`, `Q_x`, `Q_u`, `Q_xx`, `Q_uu`, `Q_ux`
        ...

    def _forward_pass(
        self,
        x_seq: np.ndarray,
        u_seq: np.ndarray,
        K_seq: np.ndarray,
        k_seq: np.ndarray,
        target: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """ Roll out a new trajectory using the backward pass's local policy,
        backtracking the step size until cost actually decreases.

        @return: `(new_x_seq, new_u_seq, new_cost)`
        """
        ## TODO 1: for alpha in a shrinking sequence (e.g. 1, 0.5, 0.25, ...):
        ##   - roll a new trajectory forward step by step:
        ##     `new_u[k] = u_seq[k] + alpha * k_seq[k] + K_seq[k] @ (new_x[k] - x_seq[k])`
        ##     then `new_x[k+1] = Model.step(new_x[k], new_u[k], dt)`
        ##     (note this uses the *new* state for feedback, not the nominal
        ##     one -- that's what makes it closed-loop instead of open-loop)
        ##   - clip controls to `v_max`/`w_max` as you go
        ##   - compute `new_cost = self._cost(new_x_seq, new_u_seq, target)`
        ##   - if `new_cost` improves on the nominal cost, return it
        ## TODO 2: if no alpha helps, return the nominal `(x_seq, u_seq)`
        ##   unchanged (with its own cost) so the outer loop can stop.
        ...

    def _stage_derivatives(
        self,
        x_k: np.ndarray,
        u_k: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Local quadratic model of the per-step cost at `(x_k, u_k)`.

        The cost itself is already quadratic in `(x - target)` and `u`
        (same `Q`/`R` as `MPC`), so these are just the gradient/Hessian of
        that quadratic -- no finite differences needed.

        Note: the `Rd` control-smoothness term in `MPC._cost` couples `u_k`
        and `u_{k-1}`, which doesn't fit this per-step `(l_x, l_u, l_xx,
        l_uu, l_ux)` shape cleanly. Leave `Rd` out of this method for a
        first working version -- come back to it (by augmenting the state
        with the previous control) once basic iLQR converges.

        @return: `(l_x, l_u, l_xx, l_uu, l_ux)` -- gradients/Hessians of the
                 stage cost w.r.t. state (3,)/(3,3) and control (2,)/(2,2),
                 plus the state-control cross term `l_ux` (2,3)
        """
        ## TODO: `l_x` and `l_xx` only act on the (x, y) rows (`self.params.Q`
        ## is 2x2, theta is untouched, matching `MPC._cost`'s `err = target -
        ## x_k[:2]`); `l_u = R @ u_k`, `l_uu = R`; `l_ux` is all zeros (this
        ## cost has no state-control cross term).
        ...

    def _cost(
        self,
        x_seq: np.ndarray,
        u_seq: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """ Total cost over the horizon -- used for the forward pass's line
        search and the outer loop's convergence check. Same terms as
        `MPC._cost` (tracking + control effort + smoothness). """
        ## TODO: port this from `MPC._cost` -- `x_seq`/`u_seq` here are
        ## already the rollout, so this is just the summation part.
        ...
