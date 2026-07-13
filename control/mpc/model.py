import numpy as np


class Model(object):
    @staticmethod
    def rollout(state: np.ndarray, controls: np.ndarray, dt: float) -> np.ndarray:
        """ Rollout a sequence of controls
        @param state: initial state `(x, y, theta)`
        @param controls: sequence of `(v, w)` 
        @return: predicted state 
        """
        horizon = controls.shape[0]
        states = np.zeros((3, horizon + 1))
        states[:, 0] = state.flatten()

        for i in range(horizon):
            states[:, i+1] = Model.step(states[:, i:i+1], controls[i], dt).flatten()

        return states

    @staticmethod
    def step(state: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        x, y, theta = state.flatten()
        v, w = u.flatten()
        
        x_next      = x + v * np.cos(theta) * dt
        y_next      = y + v * np.sin(theta) * dt
        theta_next  = theta + w * dt

        return np.array([x_next, y_next, theta_next]).reshape(3, 1)

    @staticmethod
    def jacobian(state: np.ndarray, u: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """ Linearize `Model.step` around `(state, u)`.

        Used by iLQR's backward pass to get a local linear approximation of
        the (nonlinear) dynamics at each point along the current nominal
        trajectory -- this is re-linearized every backward pass, which is
        what makes it "iterative" LQR rather than the one-shot LQR in
        `control/lqr.py` (which linearizes once, around a fixed reference
        heading).

        @param state: state `(x, y, theta)` to linearize around
        @param u: control `(v, w)` to linearize around
        @param dt: timestep
        @return: `(A, B)` such that, for a small perturbation,
                 `step(state + dx, u + du) - step(state, u) ~= A @ dx + B @ du`
                 `A` is `df/dx` (3x3), `B` is `df/du` (3x2)
        """
        ## TODO: differentiate x_next/y_next/theta_next above w.r.t.
        ## (x, y, theta) to get A, and w.r.t. (v, w) to get B.
        ## Hint: this is the same linearization `LQR._compute_gain` already
        ## does for A/B -- but evaluated at this step's actual (theta, v)
        ## instead of a fixed (heading, v_ref).
        ...
