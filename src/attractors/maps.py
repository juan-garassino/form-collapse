"""Discrete map definitions for formCollapse.

Each map takes a state array whose first two entries are used and returns the
next state as a float array of shape (3,); the third entry is padding so maps
share the (n, 3) trajectory layout of the ODE systems.
"""
import numpy as np
from typing import Dict


def henon_map(state: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Henon map: x' = 1 - a x^2 + y, y' = b x. Params: a, b."""
    x, y = state[0], state[1]
    a, b = params['a'], params['b']
    return np.array([1 - a * x**2 + y, b * x, 0.0])


def logistic_map(state: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Logistic map x' = r x (1 - x); the previous x is stored in the second slot for plotting. Params: r."""
    x = state[0]
    r = params['r']
    return np.array([r * x * (1 - x), x, 0.0])  # store previous as y for plotting


def standard_map(state: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Chirikov standard map on the torus [0, 2 pi)^2. Params: K."""
    theta, p = state[0], state[1]
    K = params['K']
    p_new = p + K * np.sin(theta)
    theta_new = theta + p_new
    return np.array([theta_new % (2 * np.pi), p_new % (2 * np.pi), 0.0])


def ikeda_map(state: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Ikeda map. Params: u."""
    x, y = state[0], state[1]
    u = params['u']
    t = 0.4 - 6.0 / (1 + x**2 + y**2)
    x_new = 1 + u * (x * np.cos(t) - y * np.sin(t))
    y_new = u * (x * np.sin(t) + y * np.cos(t))
    return np.array([x_new, y_new, 0.0])


MAP_FUNCTIONS = {
    'henon_map': henon_map,
    'logistic_map': logistic_map,
    'standard_map': standard_map,
    'ikeda_map': ikeda_map,
}
