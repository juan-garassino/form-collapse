"""Right-hand sides of the 17 continuous (ODE) attractor systems.

Each ``*_system(X, t, params)`` takes a state ``X`` of shape (3,), a time ``t``
and a parameter dict, and returns the derivative as a float array of shape (3,).
"""
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def lorenz_system(X: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
    """
    Compute the derivatives for the Lorenz system.
    Takes the current state X, time t, and system parameters as input.
    Returns the rate of change for each variable [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = X
    sigma, beta, rho = params['sigma'], params['beta'], params['rho']
    logger.debug(f"Lorenz system: x={x}, y={y}, z={z}, sigma={sigma}, beta={beta}, rho={rho}")
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ])

def aizawa_system(X: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
    """
    Compute the derivatives for the Aizawa system.
    Takes the current state X, time t, and system parameters as input.
    Returns the rate of change for each variable [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = X
    a, b, c, d, e, f = [params[key] for key in 'abcdef']
    logger.debug(f"Aizawa system: x={x}, y={y}, z={z}, a={a}, b={b}, c={c}, d={d}, e={e}, f={f}")
    return np.array([
        (z - b) * x - d * y,
        d * x + (z - b) * y,
        c + a * z - z**3 / 3 - (x**2 + y**2) * (1 + e * z) + f * z * x**3
    ])

def rabinovich_fabrikant_system(X: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
    """
    Compute the derivatives for the Rabinovich-Fabrikant system.
    Takes the current state X, time t, and system parameters as input.
    Returns the rate of change for each variable [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = X
    alpha, gamma = params['alpha'], params['gamma']
    logger.debug(f"Rabinovich-Fabrikant system: x={x}, y={y}, z={z}, alpha={alpha}, gamma={gamma}")
    return np.array([
        y * (z - 1 + x**2) + gamma * x,
        x * (3 * z + 1 - x**2) + gamma * y,
        -2 * z * (alpha + x * y)
    ])

def three_scroll_system(X: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
    """
    Compute the derivatives for the Three-Scroll system.
    Takes the current state X, time t, and system parameters as input.
    Returns the rate of change for each variable [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = X
    a, b, c = params['a'], params['b'], params['c']
    logger.debug(f"Three-Scroll system: x={x}, y={y}, z={z}, a={a}, b={b}, c={c}")
    return np.array([
        a * (y - x),
        (b - a) * x - x * z + b * y,
        x * y - c * z
    ])

def rossler_system(X: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
    """
    Compute the derivatives for the Rossler system.
    Takes the current state X, time t, and system parameters as input.
    Returns the rate of change for each variable [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = X
    a, b, c = params['a'], params['b'], params['c']
    logger.debug(f"Rossler system: x={x}, y={y}, z={z}, a={a}, b={b}, c={c}")
    return np.array([
        -y - z,
        x + a * y,
        b + z * (x - c)
    ])

def chen_system(X: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
    """
    Compute the derivatives for the Chen system.
    Takes the current state X, time t, and system parameters as input.
    Returns the rate of change for each variable [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = X
    a, b, c = params['a'], params['b'], params['c']
    logger.debug(f"Chen system: x={x}, y={y}, z={z}, a={a}, b={b}, c={c}")
    return np.array([
        a * (y - x),
        (c - a) * x - x * z + c * y,
        x * y - b * z
    ])

def halvorsen_system(X: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
    """
    Compute the derivatives for the Halvorsen system.
    Takes the current state X, time t, and system parameters as input.
    Returns the rate of change for each variable [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = X
    a = params['a']
    logger.debug(f"Halvorsen system: x={x}, y={y}, z={z}, a={a}")
    return np.array([
        -a * x - 4 * y - 4 * z - y**2,
        -a * y - 4 * z - 4 * x - z**2,
        -a * z - 4 * x - 4 * y - x**2
    ])

def anishchenko_system(X: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
    """
    Compute the derivatives for the Anishchenko system.
    Takes the current state X, time t, and system parameters as input.
    Returns the rate of change for each variable [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = X
    a, b, c = params['a'], params['b'], params['c']
    logger.debug(f"Anishchenko system: x={x}, y={y}, z={z}, a={a}, b={b}, c={c}")
    return np.array([
        y,
        -a * y + b * x - z,
        c * y
    ])

def arnold_system(X: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
    """
    Compute the derivatives for the Arnold system.
    Takes the current state X, time t, and system parameters as input.
    Returns the rate of change for each variable [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = X
    omega = params['omega']
    logger.debug(f"Arnold system: x={x}, y={y}, z={z}, omega={omega}")
    return np.array([
        x + omega,
        y + omega,
        z + omega
    ])

def burke_shaw_system(X: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
    """
    Compute the derivatives for the Burke-Shaw system.
    Takes the current state X, time t, and system parameters as input.
    Returns the rate of change for each variable [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = X
    a, b, c = params['a'], params['b'], params['c']
    logger.debug(f"Burke-Shaw system: x={x}, y={y}, z={z}, a={a}, b={b}, c={c}")
    return np.array([
        a * x - y * z,
        b * y + x * z,
        c * z + x * y / 3
    ])

def chen_celikovsky_system(X: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
    """
    Compute the derivatives for the Chen-Celikovsky system.
    Takes the current state X, time t, and system parameters as input.
    Returns the rate of change for each variable [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = X
    a, c, d = params['a'], params['c'], params['d']
    logger.debug(f"Chen-Celikovsky system: x={x}, y={y}, z={z}, a={a}, c={c}, d={d}")
    return np.array([
        a * (y - x),
        (d - a) * x - x * z + d * y,
        x * y - c * z
    ])

def finance_system(X: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
    """
    Compute the derivatives for the Finance system.
    Takes the current state X, time t, and system parameters as input.
    Returns the rate of change for each variable [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = X
    a, b, c = params['a'], params['b'], params['c']
    logger.debug(f"Finance system: x={x}, y={y}, z={z}, a={a}, b={b}, c={c}")
    return np.array([
        y * (z - 1 + x**2) + b * x,
        x * (3 * z + 1 - x**2) + c * y,
        -a * z * (1 + x * y)
    ])

def newton_leipnik_system(X: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
    """
    Compute the derivatives for the Newton-Leipnik system.
    Takes the current state X, time t, and system parameters as input.
    Returns the rate of change for each variable [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = X
    a, b = params['a'], params['b']
    logger.debug(f"Newton-Leipnik system: x={x}, y={y}, z={z}, a={a}, b={b}")
    return np.array([
        -a * x + y + 10 * y * z,
        -x - 0.4 * y + 5 * x * z,
        b * z - 5 * x * y
    ])

def qi_chen_system(X: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
    """
    Compute the derivatives for the Qi-Chen system.
    Takes the current state X, time t, and system parameters as input.
    Returns the rate of change for each variable [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = X
    a, b, c = params['a'], params['b'], params['c']
    logger.debug(f"Qi-Chen system: x={x}, y={y}, z={z}, a={a}, b={b}, c={c}")
    return np.array([
        a * (y - x),
        (c - a) * x - x * z + c * y,
        x * y - b * z
    ])

def rayleigh_benard_system(X: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
    """
    Compute the derivatives for the Rayleigh-Benard system.
    Takes the current state X, time t, and system parameters as input.
    Returns the rate of change for each variable [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = X
    a, b, c = params['a'], params['b'], params['c']
    logger.debug(f"Rayleigh-Benard system: x={x}, y={y}, z={z}, a={a}, b={b}, c={c}")
    return np.array([
        -a * x + y * z,
        b * y + x * z,
        c * z + x * y
    ])

def tsucs1_system(X: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
    """
    Compute the derivatives for the TSUCS1 system.
    Takes the current state X, time t, and system parameters as input.
    Returns the rate of change for each variable [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = X
    a, b, c = params['a'], params['b'], params['c']
    logger.debug(f"TSUCS1 system: x={x}, y={y}, z={z}, a={a}, b={b}, c={c}")
    return np.array([
        -a * x + y * z,
        b * y + x * z,
        c * z + x * y
    ])

def liu_chen_system(X: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
    """
    Compute the derivatives for the Liu-Chen system.
    Takes the current state X, time t, and system parameters as input.
    Returns the rate of change for each variable [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = X
    a, b, c, d = params['a'], params['b'], params['c'], params['d']
    logger.debug(f"Liu-Chen system: x={x}, y={y}, z={z}, a={a}, b={b}, c={c}, d={d}")
    return np.array([
        a * (y - x),
        (d - a) * x - x * z + d * y,
        x * y - b * z
    ])