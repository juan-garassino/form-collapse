"""CSV export of raw trajectories."""
import os
from typing import Dict

import numpy as np


def save_data(results: Dict[str, np.ndarray], output_dir: str) -> None:
    """Write each trajectory (n, 3) to ``<output_dir>/data/<name>.csv`` (comma-separated)."""
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    for name, data in results.items():
        np.savetxt(os.path.join(data_dir, f"{name}.csv"), data, delimiter=",")
    print(f"Saved raw data in {data_dir}")