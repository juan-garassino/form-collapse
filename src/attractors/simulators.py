import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from typing import Dict, Any, Callable, Tuple, Optional
import logging
import time
import warnings
from .attractors import (
    lorenz_system,
    aizawa_system,
    rabinovich_fabrikant_system,
    three_scroll_system,
    rossler_system,
    chen_system,
    halvorsen_system,
    anishchenko_system,
    arnold_system,
    burke_shaw_system,
    chen_celikovsky_system,
    finance_system,
    newton_leipnik_system,
    qi_chen_system,
    rayleigh_benard_system,
    tsucs1_system,
    liu_chen_system,
)
from .maps import MAP_FUNCTIONS

logger = logging.getLogger(__name__)

SYSTEM_FUNCTIONS: Dict[str, Callable] = {
    'lorenz_system': lorenz_system,
    'aizawa_system': aizawa_system,
    'rabinovich_fabrikant_system': rabinovich_fabrikant_system,
    'chen_system': chen_system,
    'halvorsen_system': halvorsen_system,
    'newton_leipnik_system': newton_leipnik_system,
    'three_scroll_system': three_scroll_system,
    'rossler_system': rossler_system,
    'anishchenko_system': anishchenko_system,
    'arnold_system': arnold_system,  # Linear divergence — control/test case
    'burke_shaw_system': burke_shaw_system,
    'chen_celikovsky_system': chen_celikovsky_system,
    'finance_system': finance_system,
    'qi_chen_system': qi_chen_system,
    'rayleigh_benard_system': rayleigh_benard_system,
    'tsucs1_system': tsucs1_system,
    'liu_chen_system': liu_chen_system,
}

SYSTEM_SCALES: Dict[str, Dict[str, Any]] = {
    'lorenz_system': {'scale': 20.0, 'params': {'sigma': (9, 11), 'beta': (2, 3), 'rho': (20, 30)}},
    'aizawa_system': {'scale': 1.0, 'params': {'a': (0.7, 0.95), 'b': (0.7, 0.85), 'c': (0.4, 0.6), 'd': (3.0, 3.5), 'e': (0.25, 0.3), 'f': (0.07, 0.12)}},
    'rabinovich_fabrikant_system': {'scale': 1.0, 'params': {'alpha': (0.14, 0.17), 'gamma': (0.98, 1.02)}},
    'three_scroll_system': {'scale': 20.0, 'params': {'a': (38, 42), 'b': (52, 58), 'c': (1.7, 2.0)}},
    'rossler_system': {'scale': 10.0, 'params': {'a': (0.1, 0.2), 'b': (0.1, 0.2), 'c': (5.0, 5.7)}},
    'chen_system': {'scale': 20.0, 'params': {'a': (35, 36), 'b': (3, 3.5), 'c': (20, 28)}},
    'halvorsen_system': {'scale': 5.0, 'params': {'a': (1.27, 1.3)}},
    'anishchenko_system': {'scale': 10.0, 'params': {'a': (1.0, 1.2), 'b': (0.5, 0.7), 'c': (0.7, 0.9)}},
    'burke_shaw_system': {'scale': 10.0, 'params': {'a': (10, 11), 'b': (4, 4.5), 'c': (2.7, 3.0)}},
    'chen_celikovsky_system': {'scale': 20.0, 'params': {'a': (35, 36), 'c': (27, 28), 'd': (1.8, 2.0)}},
    'finance_system': {'scale': 1.0, 'params': {'a': (0.95, 1.0), 'b': (0.2, 0.3), 'c': (1.0, 1.1)}},
    'newton_leipnik_system': {'scale': 0.5, 'params': {'a': (0.3, 0.5), 'b': (0.1, 0.2)}},
    'qi_chen_system': {'scale': 15.0, 'params': {'a': (35, 36), 'b': (3, 3.5), 'c': (20, 28)}},
    'rayleigh_benard_system': {'scale': 10.0, 'params': {'a': (9, 10), 'b': (5, 6), 'c': (12, 13)}},
    'tsucs1_system': {'scale': 1.0, 'params': {'a': (40, 41), 'b': (0.16, 0.17), 'c': (0.95, 1.05)}},
    'liu_chen_system': {'scale': 20.0, 'params': {'a': (5, 5.5), 'b': (-10, -9.5), 'c': (-3.8, -3.6), 'd': (1, 1.1)}}
}

def iterate_map(map_func, initial_state, params, n_iterations, transient=1000):
    """Iterate a discrete map and return trajectory."""
    state = initial_state.copy()
    trajectory = []
    for i in range(n_iterations + transient):
        state = map_func(state, params)
        if i >= transient:
            trajectory.append(state.copy())
    return np.array(trajectory)


def generate_initial_condition(scale: float, attempt: int) -> np.ndarray:
    """Generate an initial condition based on the system scale and attempt number."""
    if attempt == 0:
        # First attempt: use a small perturbation from origin
        return np.random.randn(3) * 0.1 * scale
    else:
        # Subsequent attempts: use full scale with increasing randomness
        return np.random.randn(3) * scale * (1 + attempt * 0.2)

def adaptive_simulation(
    system_name: str,
    system_func: str,
    params: Dict[str, float],
    sim_time: float,
    sim_steps: int,
    max_attempts: int = 10,
    max_time: float = 30.0,
    scale: float = 1.0
) -> Tuple[bool, Optional[np.ndarray], str]:
    start_time = time.time()

    # Check if it's a discrete map
    if system_func in MAP_FUNCTIONS:
        map_function = MAP_FUNCTIONS[system_func]
        n_iterations = int(sim_steps)
        initial_state = np.random.randn(3) * 0.1
        try:
            data = iterate_map(map_function, initial_state, params, n_iterations)
            valid, message = check_simulation_validity(data, n_iterations)
            if valid:
                return True, data, "Successful map iteration"
        except Exception as e:
            return False, None, str(e)
        return False, None, "Map iteration failed validation"

    system_function = SYSTEM_FUNCTIONS.get(system_func)
    if system_function is None:
        raise ValueError(f"Unknown system function: {system_func}")

    # Define base tolerances and step size
    base_rtol, base_atol = 1e-6, 1e-9
    base_max_step = sim_time / 100

    for attempt in range(max_attempts):
        # Adjust parameters based on the attempt number
        rtol = base_rtol * (10 ** (attempt // 3))
        atol = base_atol * (10 ** (attempt // 3))
        max_step = base_max_step / (2 ** (attempt // 2))

        initial_condition = generate_initial_condition(scale, attempt)
        
        # Try RK45 first; if it uses too many evaluations, fall back to stiff solvers
        methods_to_try = ['RK45', 'BDF', 'Radau']
        for method in methods_to_try:
            try:
                solution = solve_ivp(
                    lambda t, y: system_function(y, t, params),
                    (0, sim_time),
                    initial_condition,
                    method=method,
                    t_eval=np.linspace(0, sim_time, sim_steps),
                    rtol=rtol,
                    atol=atol,
                    max_step=max_step
                )

                if solution.success:
                    # If RK45 needed too many evaluations, skip to stiff solvers
                    if method == 'RK45' and hasattr(solution, 'nfev') and solution.nfev > sim_steps * 10:
                        logger.info(f"RK45 used {solution.nfev} evaluations (>{sim_steps*10}), trying stiff solvers")
                        continue

                    data = solution.y.T
                    valid, message = check_simulation_validity(data, sim_steps)
                    if valid:
                        elapsed_time = time.time() - start_time
                        if elapsed_time > max_time:
                            return True, data[:int(len(data) * max_time / elapsed_time)], f"Successful simulation (truncated due to time limit) using {method}"
                        return True, data, f"Successful simulation using {method}"

                logger.info(f"Attempt {attempt + 1} with {method} failed: {solution.message}")

            except Exception as solver_error:
                logger.warning(f"Solver {method} failed: {str(solver_error)}")
                continue

        if time.time() - start_time > max_time:
            return False, None, f"Simulation exceeded time limit of {max_time} seconds"
    
    return False, None, f"Failed to produce valid simulation after {max_attempts} attempts"

def check_simulation_validity(data: np.ndarray, sim_steps: int) -> Tuple[bool, str]:
    """Check if the simulation results are valid."""
    if data is None or len(data) < sim_steps // 2:
        return False, "Simulation terminated too early"
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        return False, "Simulation produced NaN or Inf values"
    if np.all(np.std(data, axis=0) < 1e-3):
        return False, "Simulation produced simple output"
    
    # You can add more checks here if needed
    
    return True, "Simulation valid"

def compute_lyapunov_exponent(
    data: np.ndarray,
    dt: float = 1.0,
    theiler_window: int = 50,
    max_divergence_steps: int = 200,
    max_reference_points: int = 1000,
) -> float:
    """Estimate the maximal Lyapunov exponent using Rosenstein's nearest-neighbor method.

    For a subset of reference points, find the nearest neighbor (excluding temporal
    neighbors within the Theiler window), then track how pairs diverge over time.
    MLE = slope of <ln(divergence)> vs time.
    """
    n = len(data)
    if n < theiler_window + max_divergence_steps + 1:
        max_divergence_steps = max(1, n - theiler_window - 1)

    usable = n - max_divergence_steps
    if usable <= 0:
        return 0.0

    tree = cKDTree(data)

    # Subsample reference points for speed
    if usable > max_reference_points:
        ref_indices = np.linspace(0, usable - 1, max_reference_points, dtype=int)
    else:
        ref_indices = np.arange(usable)

    # Batch query: get k nearest neighbors for all reference points
    k_query = min(theiler_window + 5, n)
    dists_all, idxs_all = tree.query(data[ref_indices], k=k_query)

    nn_indices = np.full(len(ref_indices), -1, dtype=int)
    for qi, i in enumerate(ref_indices):
        for j_idx in range(1, k_query):
            j = idxs_all[qi, j_idx]
            if abs(i - j) > theiler_window and j + max_divergence_steps < n:
                nn_indices[qi] = j
                break

    valid_mask = nn_indices >= 0
    valid_qi = np.where(valid_mask)[0]

    if len(valid_qi) < 10:
        warnings.warn("Too few valid neighbor pairs for Lyapunov estimation")
        return 0.0

    # Track divergence using vectorized operations
    ln_divergence = np.zeros(max_divergence_steps)
    counts = np.zeros(max_divergence_steps)

    vi = ref_indices[valid_qi]
    vj = nn_indices[valid_qi]

    for k in range(max_divergence_steps):
        diffs = data[vi + k] - data[vj + k]
        dists = np.linalg.norm(diffs, axis=1)
        pos = dists > 0
        if pos.any():
            ln_divergence[k] = np.mean(np.log(dists[pos]))
            counts[k] = pos.sum()

    valid_steps = counts > 0
    if valid_steps.sum() < 2:
        return 0.0

    time_axis = np.arange(max_divergence_steps)[valid_steps] * dt
    coeffs = np.polyfit(time_axis, ln_divergence[valid_steps], 1)
    return float(coeffs[0])


def estimate_lyapunov_exponent(data: np.ndarray, dt: float = 1.0) -> float:
    """Deprecated: use compute_lyapunov_exponent instead."""
    warnings.warn(
        "estimate_lyapunov_exponent is deprecated, use compute_lyapunov_exponent",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_lyapunov_exponent(data, dt=dt)


def classify_attractor(
    data: np.ndarray, lyapunov_exp: float
) -> Tuple[str, Dict[str, Any]]:
    """Classify the attractor type based on trajectory data and Lyapunov exponent.

    Returns (classification_str, metrics_dict).
    """
    metrics: Dict[str, Any] = {
        "lyapunov_exponent": lyapunov_exp,
        "std": np.std(data, axis=0).tolist(),
        "range": (np.max(data, axis=0) - np.min(data, axis=0)).tolist(),
        "mean": np.mean(data, axis=0).tolist(),
    }

    # Check divergent
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        return "divergent", metrics
    data_range = np.max(data) - np.min(data)
    if data_range > 1e6:
        return "divergent", metrics

    # Check fixed point
    if np.all(np.std(data, axis=0) < 1e-3):
        return "fixed_point", metrics

    # Check strange attractor
    if lyapunov_exp > 0.01:
        return "strange_attractor", metrics

    return "limit_cycle", metrics


def _numerical_jacobian(
    system_func: Callable,
    state: np.ndarray,
    t: float,
    params: Dict[str, float],
    h: float = 1e-7,
) -> np.ndarray:
    """Compute numerical Jacobian of system_func at the given state."""
    n = len(state)
    f0 = system_func(state, t, params)
    jac = np.zeros((n, n))
    for i in range(n):
        perturbed = state.copy()
        perturbed[i] += h
        fi = system_func(perturbed, t, params)
        jac[:, i] = (fi - f0) / h
    return jac


def analyze_simulation_results(data: np.ndarray, dt: float = 1.0) -> Dict[str, Any]:
    """Analyze the simulation results and return various metrics."""
    mle = compute_lyapunov_exponent(data, dt=dt)
    classification, metrics = classify_attractor(data, mle)
    metrics["classification"] = classification
    return metrics