"""Configuration loading, parameter generation (random and Latin Hypercube) and presets."""
import yaml
import logging
from typing import Dict, Any, List
import os
import numpy as np

logger = logging.getLogger(__name__)

KNOWN_GOOD_PRESETS: Dict[str, Dict[str, float]] = {
    'lorenz_system': {'sigma': 10.0, 'rho': 28.0, 'beta': 8.0 / 3.0},
    'aizawa_system': {'a': 0.95, 'b': 0.7, 'c': 0.6, 'd': 3.5, 'e': 0.25, 'f': 0.1},
    'rabinovich_fabrikant_system': {'alpha': 0.14, 'gamma': 0.10},
    'chen_system': {'a': 35.0, 'b': 3.0, 'c': 28.0},
    'halvorsen_system': {'a': 1.27},
    'newton_leipnik_system': {'a': 0.4, 'b': 0.175},
    'three_scroll_system': {'a': 40.0, 'b': 55.0, 'c': 1.833},
    'rossler_system': {'a': 0.2, 'b': 0.2, 'c': 5.7},
    'anishchenko_system': {'a': 1.2, 'b': 0.5, 'c': 0.6},
    'arnold_system': {'omega': 1.0},
    'burke_shaw_system': {'a': 10.0, 'b': 4.272, 'c': 2.73},
    'chen_celikovsky_system': {'a': 36.0, 'c': 20.0, 'd': 1.833},
    'finance_system': {'a': 0.001, 'b': 0.2, 'c': 1.1},
    'qi_chen_system': {'a': 38.0, 'b': 2.666, 'c': 80.0},
    'rayleigh_benard_system': {'a': 9.0, 'b': 5.0, 'c': 12.0},
    'tsucs1_system': {'a': 40.0, 'b': 0.16, 'c': 0.65},
    'liu_chen_system': {'a': 5.0, 'b': -10.0, 'c': -3.78, 'd': 1.0},
    'henon_map': {'a': 1.4, 'b': 0.3},
    'ikeda_map': {'u': 0.918},
}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")

    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {str(e)}")
            raise


def generate_system_params(config: Dict[str, Any], system_name: str) -> Dict[str, Any]:
    """Draw one random parameter set for a system from its ranges in ``config``.

    Returns the sampled parameters plus ``sim_time``, ``sim_steps`` and ``scale``.
    """
    system_config = config['systems'][system_name]
    params = {}
    for param_name, param_config in system_config['params'].items():
        if isinstance(param_config, dict) and 'range' in param_config:
            params[param_name] = np.random.uniform(*param_config['range'])
        else:
            params[param_name] = param_config

    # Add simulation time and steps, using global defaults if not specified
    params['sim_time'] = system_config.get('sim_time', config['global']['default_sim_time'])
    params['sim_steps'] = system_config.get('sim_steps', config['global']['default_sim_steps'])
    params['scale'] = system_config['scale']

    return params


def generate_params_lhs(
    config: Dict[str, Any], system_name: str, n_samples: int
) -> List[Dict[str, Any]]:
    """Generate parameter sets using Latin Hypercube Sampling.

    Returns a list of n_samples parameter dicts for the given system.
    """
    from scipy.stats.qmc import LatinHypercube

    system_config = config['systems'][system_name]
    param_names = []
    param_ranges = []

    for param_name, param_config in system_config['params'].items():
        if isinstance(param_config, dict) and 'range' in param_config:
            param_names.append(param_name)
            param_ranges.append(param_config['range'])

    if not param_names:
        # No variable parameters — return n_samples copies of fixed params
        base = generate_system_params(config, system_name)
        return [base.copy() for _ in range(n_samples)]

    sampler = LatinHypercube(d=len(param_names))
    samples = sampler.random(n=n_samples)  # shape (n_samples, d) in [0, 1]

    results = []
    for row in samples:
        params = {}
        for i, name in enumerate(param_names):
            lo, hi = param_ranges[i]
            params[name] = lo + row[i] * (hi - lo)
        # Fill in fixed params
        for param_name, param_config in system_config['params'].items():
            if param_name not in params:
                params[param_name] = param_config

        params['sim_time'] = system_config.get('sim_time', config['global']['default_sim_time'])
        params['sim_steps'] = system_config.get('sim_steps', config['global']['default_sim_steps'])
        params['scale'] = system_config['scale']
        results.append(params)

    return results


def get_config(config_path: str) -> Dict[str, Any]:
    """Get the configuration from the specified YAML file."""
    if not config_path:
        raise ValueError("Configuration file path must be provided.")
    return load_config(config_path)
