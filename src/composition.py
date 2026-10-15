"""Multi-attractor composition engine."""
import os
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CompositionLayer:
    """One attractor layer in a composition.

    Attributes
    ----------
    name : str
        Layer label.
    data : np.ndarray
        Trajectory of shape (n, 2) or (n, 3), float.
    color : str
        SVG stroke colour.
    pen_number : int
        Tool number used for G-code tool changes.
    scale : float
        Scale factor applied to the normalised layer.
    translate : tuple of float
        (dx, dy) offset.
    rotate : float
        Rotation in degrees.
    """
    name: str
    data: np.ndarray
    color: str = "black"
    pen_number: int = 1
    scale: float = 1.0
    translate: Tuple[float, float] = (0.0, 0.0)
    rotate: float = 0.0  # degrees


def _apply_transform(data_2d: np.ndarray, layer: CompositionLayer) -> np.ndarray:
    """Apply scale, rotate, translate to 2D data."""
    # Normalize to [0, 1]
    d_min = data_2d.min(axis=0)
    d_max = data_2d.max(axis=0)
    d_range = d_max - d_min
    d_range[d_range == 0] = 1
    normalized = (data_2d - d_min) / d_range

    # Scale
    normalized *= layer.scale

    # Rotate
    if layer.rotate != 0:
        theta = np.radians(layer.rotate)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        center = normalized.mean(axis=0)
        normalized = (normalized - center) @ R.T + center

    # Translate
    normalized[:, 0] += layer.translate[0]
    normalized[:, 1] += layer.translate[1]

    return normalized


def compose_layers(layers: List[CompositionLayer], output_dir: str,
                   canvas_size: Tuple[int, int] = (1000, 1000)) -> Dict[str, str]:
    """Compose multiple attractor layers into combined SVG and G-code."""
    os.makedirs(output_dir, exist_ok=True)
    files = {}

    # --- SVG ---
    svg_dir = os.path.join(output_dir, 'svg')
    os.makedirs(svg_dir, exist_ok=True)

    import svgwrite
    svg_path = os.path.join(svg_dir, "composition.svg")
    dwg = svgwrite.Drawing(svg_path, size=canvas_size,
                           viewBox=f"0 0 {canvas_size[0]} {canvas_size[1]}")

    for layer in layers:
        data_2d = layer.data[:, :2] if layer.data.shape[1] >= 2 else layer.data
        transformed = _apply_transform(data_2d, layer)
        scaled = transformed * (canvas_size[0] - 1)

        g = dwg.g(id=layer.name)
        points = scaled.tolist()
        polyline = dwg.polyline(points=points, fill='none',
                                stroke=layer.color, stroke_width=1)
        g.add(polyline)
        dwg.add(g)

    dwg.save()
    files['svg'] = svg_path
    logger.info(f"Composition SVG saved: {svg_path}")

    # --- G-code ---
    gcode_dir = os.path.join(output_dir, 'gcode')
    os.makedirs(gcode_dir, exist_ok=True)

    gcode_path = os.path.join(gcode_dir, "composition.gcode")
    gcode_canvas = (100, 100)

    with open(gcode_path, 'w') as f:
        f.write("G21 ; Set units to millimeters\n")
        f.write("G90 ; Use absolute coordinates\n")
        f.write("G0 Z5 ; Raise pen\n")

        for layer in layers:
            f.write(f"\n; --- Layer: {layer.name} (pen {layer.pen_number}) ---\n")
            f.write(f"M6 T{layer.pen_number} ; Tool change\n")

            data_2d = layer.data[:, :2] if layer.data.shape[1] >= 2 else layer.data
            transformed = _apply_transform(data_2d, layer)
            scaled = transformed * (gcode_canvas[0] - 1)

            f.write("G0 Z5 ; Raise pen\n")
            f.write(f"G0 X{scaled[0, 0]:.2f} Y{scaled[0, 1]:.2f} ; Move to start\n")
            f.write("G0 Z0 ; Lower pen\n")

            for x, y in scaled[1:]:
                f.write(f"G1 X{x:.2f} Y{y:.2f}\n")

            f.write("G0 Z5 ; Raise pen\n")

        f.write("\nG0 X0 Y0 ; Return to origin\n")

    files['gcode'] = gcode_path
    logger.info(f"Composition G-code saved: {gcode_path}")

    return files
