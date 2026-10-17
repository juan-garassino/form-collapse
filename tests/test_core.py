"""Core tests for formCollapse — Phase 1D."""
import json
import os

import numpy as np
import torch


# ── Lyapunov known-answer ──────────────────────────────────────────────────

def test_lyapunov_lorenz_known_answer(lorenz_data):
    """Lorenz (sigma=10, rho=28, beta=8/3) should give MLE roughly 0.3–2.0."""
    from src.attractors.simulators import compute_lyapunov_exponent

    dt = 33.0 / 9999
    mle = compute_lyapunov_exponent(lorenz_data, dt=dt)
    assert 0.3 < mle < 2.0, f"Lorenz MLE {mle} outside expected range [0.3, 2.0]"


# ── Classification ─────────────────────────────────────────────────────────

def test_classify_strange_attractor(lorenz_data):
    """Lorenz data should be classified as strange_attractor."""
    from src.attractors.simulators import compute_lyapunov_exponent, classify_attractor

    dt = 33.0 / 9999
    mle = compute_lyapunov_exponent(lorenz_data, dt=dt)
    classification, _ = classify_attractor(lorenz_data, mle)
    assert classification == "strange_attractor"


def test_classify_fixed_point(constant_data):
    """Constant data should be classified as fixed_point."""
    from src.attractors.simulators import classify_attractor

    classification, _ = classify_attractor(constant_data, lyapunov_exp=-0.5)
    assert classification == "fixed_point"


# ── Solver fallback ────────────────────────────────────────────────────────

def test_solver_fallback_produces_valid_output():
    """adaptive_simulation should produce valid output even for stiff systems."""
    from src.attractors.simulators import adaptive_simulation

    params = {'sigma': 10.0, 'rho': 28.0, 'beta': 8.0 / 3.0,
              'sim_time': 10, 'sim_steps': 2000, 'scale': 20.0}
    success, data, msg = adaptive_simulation(
        'Lorenz', 'lorenz_system', params,
        sim_time=10, sim_steps=2000,
        max_attempts=5, max_time=15.0, scale=20.0,
    )
    assert success
    assert data is not None
    assert data.shape[1] == 3
    assert not np.any(np.isnan(data))
    assert not np.any(np.isinf(data))


# ── Session schema ─────────────────────────────────────────────────────────

def test_session_schema(tmp_output_dir):
    """session.json should have session_timestamp, results[0].system_name."""
    from src.session import Session

    session = Session(base_dir=tmp_output_dir)
    session.add_result(
        system_name="Lorenz",
        classification="strange_attractor",
        lyapunov=0.905,
        params={"sigma": 10.0, "rho": 28.0},
        files=["test.png"],
    )
    path = session.save()

    with open(path) as f:
        data = json.load(f)

    assert "session_timestamp" in data
    assert "results" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["system_name"] == "Lorenz"
    assert data["results"][0]["classification"] == "strange_attractor"
    assert isinstance(data["results"][0]["lyapunov_exponent"], float)


# ── Config loading ─────────────────────────────────────────────────────────

def test_config_loading(config):
    """config.yaml should load successfully and have expected structure."""
    assert 'global' in config
    assert 'default_sim_time' in config['global']
    assert 'systems' in config
    assert len(config['systems']) >= 6
    assert 'Lorenz' in config['systems']
    assert 'func' in config['systems']['Lorenz']


# ── GAN shapes ─────────────────────────────────────────────────────────────

def test_gan_generator_shape():
    """Generator should produce (batch, seq_len, 3) tensors."""
    from src.gan.models import create_models

    latent_dim = 100
    seq_len = 64
    generator, _ = create_models(latent_dim, seq_len=seq_len)
    generator.eval()

    z = torch.randn(8, latent_dim)
    with torch.no_grad():
        output = generator(z)

    assert output.shape == (8, seq_len, 3), f"Generator output shape {output.shape} != expected (8, {seq_len}, 3)"


def test_gan_discriminator_shape():
    """Discriminator should accept (batch, seq_len, 3) and return (batch, 1)."""
    from src.gan.models import create_models

    latent_dim = 100
    seq_len = 64
    _, discriminator = create_models(latent_dim, seq_len=seq_len)
    discriminator.eval()

    x = torch.randn(8, seq_len, 3)
    with torch.no_grad():
        output = discriminator(x)

    assert output.shape == (8, 1), f"Discriminator output shape {output.shape} != expected (8, 1)"


# ── All systems registered ─────────────────────────────────────────────────

def test_all_systems_registered():
    """All 17 ODE systems should be in SYSTEM_FUNCTIONS."""
    from src.attractors.simulators import SYSTEM_FUNCTIONS

    expected = [
        'lorenz_system', 'aizawa_system', 'rabinovich_fabrikant_system',
        'chen_system', 'halvorsen_system', 'newton_leipnik_system',
        'three_scroll_system', 'rossler_system', 'anishchenko_system',
        'arnold_system', 'burke_shaw_system', 'chen_celikovsky_system',
        'finance_system', 'qi_chen_system', 'rayleigh_benard_system',
        'tsucs1_system', 'liu_chen_system',
    ]
    for name in expected:
        assert name in SYSTEM_FUNCTIONS, f"{name} not in SYSTEM_FUNCTIONS"


# ── Discrete maps ──────────────────────────────────────────────────────────

def test_henon_map_iteration():
    """Henon map should produce a valid trajectory."""
    from src.attractors.simulators import iterate_map
    from src.attractors.maps import henon_map

    params = {'a': 1.4, 'b': 0.3}
    data = iterate_map(henon_map, np.array([0.1, 0.1, 0.0]), params, 1000, transient=100)
    assert data.shape == (1000, 3)
    assert not np.any(np.isnan(data))


# ── Projection ─────────────────────────────────────────────────────────────

def test_projection_3d_to_2d(lorenz_data):
    """project_3d_to_2d should return (N, 2) array."""
    from src.utils.projection import project_3d_to_2d

    proj = project_3d_to_2d(lorenz_data, elevation=30, azimuth=-60)
    assert proj.shape == (len(lorenz_data), 2)


def test_optimal_projection(lorenz_data):
    """find_optimal_projection should return angles and projected data."""
    from src.utils.projection import find_optimal_projection

    el, az, proj = find_optimal_projection(lorenz_data[:1000], n_trials=10)
    assert -90 <= el <= 90
    assert proj.shape == (1000, 2)


# ── SVG/G-code (RDP, pen-lifts) ───────────────────────────────────────────

def test_rdp_simplification():
    """RDP should reduce point count on a noisy line."""
    from src.utils.svg_gcode import rdp_simplify

    # Straight line with noise
    t = np.linspace(0, 1, 1000)
    points = np.column_stack([t, t * 0.5 + np.random.randn(1000) * 0.001])
    simplified = rdp_simplify(points, epsilon=0.01)
    assert len(simplified) < len(points)


def test_gcode_has_pen_lifts(tmp_output_dir):
    """G-code should contain pen-lift commands for discontinuous trajectories."""
    from src.utils.svg_gcode import generate_gcode

    # Create data with a big jump in the middle
    segment1 = np.column_stack([np.linspace(0, 1, 50), np.linspace(0, 1, 50), np.zeros(50)])
    segment2 = np.column_stack([np.linspace(5, 6, 50), np.linspace(5, 6, 50), np.zeros(50)])
    data = np.vstack([segment1, segment2])

    generate_gcode(data, "test_penlift", tmp_output_dir)

    gcode_path = os.path.join(tmp_output_dir, "gcode", "test_penlift.gcode")
    assert os.path.exists(gcode_path)

    with open(gcode_path) as f:
        content = f.read()

    # Should have multiple pen-up commands (more than just the initial and final)
    z_up_count = content.count("Z5")
    assert z_up_count >= 3, f"Expected >= 3 pen-up commands, got {z_up_count}"
