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
