"""Session manager for organizing results into timestamped directories."""

import json
import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class Session:
    def __init__(self, base_dir: str = "results"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sessions started in the same second get a numeric suffix instead of
        # sharing (and overwriting) one directory.
        name, n = self.timestamp, 1
        while True:
            self.session_dir = os.path.join(base_dir, name)
            try:
                os.makedirs(self.session_dir)
                break
            except FileExistsError:
                n += 1
                name = f"{self.timestamp}_{n}"
        self.results: List[Dict[str, Any]] = []
        logger.info(f"Session created: {self.session_dir}")

    def get_output_dir(self) -> str:
        """Return the session output directory.

        Existing viz functions already create png/, svg/, gcode/ subdirs
        inside the output_dir, so this automatically organizes into session folders.
        """
        return self.session_dir

    def add_result(
        self,
        system_name: str,
        classification: str,
        lyapunov: Optional[float],
        params: Dict[str, Any],
        files: List[str],
    ) -> None:
        # Filter out non-serializable keys from params
        serializable_params = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                serializable_params[k] = v
            elif isinstance(v, (np.integer, np.floating)):
                serializable_params[k] = v.item()
            elif isinstance(v, np.ndarray):
                serializable_params[k] = v.tolist()

        self.results.append({
            "system_name": system_name,
            "classification": classification,
            "lyapunov_exponent": float(lyapunov) if lyapunov is not None else None,
            "params": serializable_params,
            "files": files,
            "timestamp": datetime.now().isoformat(),
        })

    def save(self) -> str:
        """Write session.json to the session directory. Returns the path."""
        path = os.path.join(self.session_dir, "session.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "session_timestamp": self.timestamp,
                    "num_results": len(self.results),
                    "results": self.results,
                },
                f,
                indent=2,
                cls=_NumpyEncoder,
            )
        logger.info(f"Session saved to {path}")
        return path
