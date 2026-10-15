"""Shared fixtures for formCollapse tests."""
import pytest
import tempfile

import numpy as np


@pytest.fixture
def lorenz_data():
    """Pre-computed Lorenz trajectory for testing."""
    from scipy.integrate import solve_ivp
    from src.attractors.attractors import lorenz_system

    params = {'sigma': 10.0, 'rho': 28.0, 'beta': 8.0 / 3.0}
    sol = solve_ivp(
        lambda t, y: lorenz_system(y, t, params),
        (0, 33), [1.0, 1.0, 1.0],
        method='RK45', t_eval=np.linspace(0, 33, 9999),
        rtol=1e-6, atol=1e-9,
    )
    return sol.y.T


@pytest.fixture
def constant_data():
    """Constant trajectory (fixed point)."""
    return np.ones((1000, 3)) * 5.0


@pytest.fixture
def tmp_output_dir():
    """Temporary output directory for test artifacts."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def config():
    """Load the project config."""
    from src.utils.config import get_config
    return get_config('config.yaml')
