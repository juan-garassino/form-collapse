"""Attractor discovery engine — find interesting parameter regimes."""
import logging
import numpy as np
from typing import Dict, Any, List

from .utils.config import generate_params_lhs
from .attractors.simulators import adaptive_simulation, compute_lyapunov_exponent, classify_attractor

logger = logging.getLogger(__name__)


def discover_attractors(
    system_name: str,
    config: Dict[str, Any],
    n_candidates: int = 500,
    output_dir: str = "results/discovery",
    top_n: int = 25,
) -> List[Dict[str, Any]]:
    """Run n_candidates parameter sets, keep strange attractors, rank by coverage."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    system_config = config['systems'][system_name]
    func_name = system_config['func']

    param_sets = generate_params_lhs(config, system_name, n_candidates)

    results = []
    for i, params in enumerate(param_sets):
        if (i + 1) % 50 == 0:
            logger.info(f"Discovery: {i+1}/{n_candidates} candidates tested")

        try:
            success, data, message = adaptive_simulation(
                system_name, func_name, params,
                params.get('sim_time', 10),  # shorter for speed
                params.get('sim_steps', 5000),
                max_attempts=3, max_time=10.0,
                scale=system_config['scale'],
            )

            if not success or data is None:
                continue

            dt = params.get('sim_time', 10) / params.get('sim_steps', 5000)
            mle = compute_lyapunov_exponent(data, dt=dt)
            classification, metrics = classify_attractor(data, mle)

            if classification == "strange_attractor":
                # Coverage metric: count occupied voxels in a grid
                coverage = _compute_coverage(data, grid_size=50)
                results.append({
                    'params': params,
                    'classification': classification,
                    'lyapunov': mle,
                    'coverage': coverage,
                    'data': data,
                })
        except Exception as e:
            logger.debug(f"Candidate {i} failed: {e}")
            continue

    logger.info(f"Discovery: {len(results)}/{n_candidates} strange attractors found")

    # Rank by coverage
    results.sort(key=lambda r: r['coverage'], reverse=True)
    return results[:top_n]


def _compute_coverage(data: np.ndarray, grid_size: int = 50) -> float:
    """Count fraction of occupied voxels in bounding grid."""
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0

    normalized = ((data - mins) / ranges * (grid_size - 1)).astype(int)
    normalized = np.clip(normalized, 0, grid_size - 1)

    voxels = set(map(tuple, normalized))
    return len(voxels) / (grid_size ** 3)


def create_gallery(results: List[Dict[str, Any]], output_dir: str, system_name: str = "") -> str:
    """Create a grid of thumbnails with param annotations."""
    import matplotlib.pyplot as plt
    import os

    n = len(results)
    if n == 0:
        return ""

    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows),
                              subplot_kw={'projection': '3d'})
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for i, result in enumerate(results):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        data = result['data']
        # Normalize
        d_min, d_max = data.min(axis=0), data.max(axis=0)
        d_range = d_max - d_min
        d_range[d_range == 0] = 1
        nd = (data - d_min) / d_range

        ax.plot(nd[:, 0], nd[:, 1], nd[:, 2], lw=0.3, alpha=0.8)
        ax.set_title(f"MLE={result['lyapunov']:.3f}\ncov={result['coverage']:.3f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    # Hide unused axes
    for i in range(n, rows * cols):
        r, c = i // cols, i % cols
        axes[r, c].set_visible(False)

    plt.suptitle(f"{system_name} Discovery Gallery (top {n})", fontsize=14)
    plt.tight_layout()

    gallery_path = os.path.join(output_dir, f"{system_name}_discovery_gallery.png")
    fig.savefig(gallery_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Gallery saved: {gallery_path}")
    return gallery_path
