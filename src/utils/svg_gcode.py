"""SVG and G-code export with RDP simplification, pen-lift detection and stroke ordering."""
import svgwrite
import os
import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 4A. Ramer-Douglas-Peucker simplification
# ---------------------------------------------------------------------------

def rdp_simplify(points: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    """Ramer-Douglas-Peucker line simplification.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, D) with D-dimensional points.
    epsilon : float
        Maximum perpendicular distance threshold.  Points whose distance
        from the line between the start and end is less than *epsilon* are
        discarded.  Applied after normalisation, so 0.01 means 1 % of the
        data range.

    Returns
    -------
    np.ndarray
        Simplified array of shape (M, D) with M <= N.
    """
    if len(points) <= 2:
        return points

    # Find the point with the maximum distance from the line between
    # the first and last point.
    start, end = points[0], points[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len == 0:
        # All points collapse to a single location – keep first & last.
        dists = np.linalg.norm(points - start, axis=1)
    else:
        # Perpendicular distance of every interior point to the line.
        unit = line_vec / line_len
        diff = points - start
        proj = np.outer(diff @ unit, unit)
        perp = diff - proj
        dists = np.linalg.norm(perp, axis=1)

    idx = np.argmax(dists)
    d_max = dists[idx]

    if d_max > epsilon:
        left = rdp_simplify(points[: idx + 1], epsilon)
        right = rdp_simplify(points[idx:], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return np.array([start, end])


# ---------------------------------------------------------------------------
# 4B. Pen-lift / segment detection
# ---------------------------------------------------------------------------

def detect_segments(data: np.ndarray, threshold: float = 0.05) -> List[np.ndarray]:
    """Split trajectory into continuous segments based on inter-point distance.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (N, D) – typically normalised to [0, 1].
    threshold : float
        Fraction of canvas diagonal.  Consecutive points farther apart than
        ``threshold * sqrt(D)`` (the diagonal of a unit hypercube) trigger a
        pen-lift.

    Returns
    -------
    List[np.ndarray]
        List of arrays, each representing one continuous stroke.
    """
    if len(data) < 2:
        return [data]

    ndim = data.shape[1]
    canvas_diag = np.sqrt(ndim)  # diagonal of unit hypercube
    abs_threshold = threshold * canvas_diag

    dists = np.linalg.norm(np.diff(data, axis=0), axis=1)
    split_indices = np.where(dists > abs_threshold)[0] + 1  # +1 because diff shifts

    segments: List[np.ndarray] = []
    prev = 0
    for idx in split_indices:
        seg = data[prev:idx]
        if len(seg) > 0:
            segments.append(seg)
        prev = idx
    # Remaining tail
    seg = data[prev:]
    if len(seg) > 0:
        segments.append(seg)

    return segments


# ---------------------------------------------------------------------------
# 4E. TSP greedy nearest-neighbour stroke reorder
# ---------------------------------------------------------------------------

def reorder_segments(segments: List[np.ndarray]) -> List[np.ndarray]:
    """Greedy nearest-neighbor reorder to minimize pen travel.

    Starting from the first segment, always jump to whichever remaining
    segment has its start *or* end closest to the current pen position.
    If the closest point is the *end* of a segment, the segment is reversed
    so the pen enters at that end.

    Parameters
    ----------
    segments : List[np.ndarray]
        List of continuous stroke arrays, each of shape (M_i, D).

    Returns
    -------
    List[np.ndarray]
        Reordered (and possibly reversed) list of segments.
    """
    if len(segments) <= 1:
        return segments

    remaining = list(range(len(segments)))
    ordered: List[np.ndarray] = []

    # Start with the first segment
    current_idx = remaining.pop(0)
    ordered.append(segments[current_idx])
    pen = ordered[-1][-1]  # pen is at end of first segment

    while remaining:
        best_dist = np.inf
        best_i = 0
        best_reverse = False

        for i, seg_idx in enumerate(remaining):
            seg = segments[seg_idx]
            d_start = np.linalg.norm(pen - seg[0])
            d_end = np.linalg.norm(pen - seg[-1])
            if d_start < best_dist:
                best_dist = d_start
                best_i = i
                best_reverse = False
            if d_end < best_dist:
                best_dist = d_end
                best_i = i
                best_reverse = True

        chosen = remaining.pop(best_i)
        seg = segments[chosen]
        if best_reverse:
            seg = seg[::-1]
        ordered.append(seg)
        pen = ordered[-1][-1]

    return ordered


# ---------------------------------------------------------------------------
# Shared normalisation helper
# ---------------------------------------------------------------------------

def _normalise_and_scale(data: np.ndarray, canvas_size, canvas_margin: float = 0.0,
                         preserve_aspect: bool = True):
    """Normalise *data* into [0, 1], then scale to *canvas_size* with margin.

    Returns
    -------
    scaled_data : np.ndarray
        Scaled points (only first 2 dims used for x/y).
    """
    # Work on a copy, use only first 2 dims for spatial scaling
    pts = data[:, :2].astype(float).copy()
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0  # avoid division by zero

    # Normalise to [0, 1]
    normalised = (pts - mins) / ranges

    if preserve_aspect:
        # Use the same scale factor for both axes so aspect ratio is kept
        aspect = ranges / ranges.max()
        normalised = normalised * aspect

    # Compute drawable area inside margins
    cw, ch = float(canvas_size[0]), float(canvas_size[1])
    margin_x = cw * canvas_margin
    margin_y = ch * canvas_margin
    draw_w = cw - 2 * margin_x
    draw_h = ch - 2 * margin_y

    scaled = normalised.copy()
    scaled[:, 0] = normalised[:, 0] * draw_w + margin_x
    scaled[:, 1] = normalised[:, 1] * draw_h + margin_y

    return scaled


# ---------------------------------------------------------------------------
# 4D. SVG output
# ---------------------------------------------------------------------------

def save_svg(data: np.ndarray, filename: str, output_dir: str,
             canvas_size=(1000, 1000), stroke_width: float = 1.0,
             epsilon: float = 0.01) -> None:
    """Save trajectory data as an SVG file.

    Improvements over the original implementation:
    - RDP simplification applied before output.
    - Pen-lift detection splits the trajectory into separate ``<path>``
      elements, each with its own ``M`` / ``L`` commands.
    - Segments are reordered via greedy nearest-neighbour to minimise
      pen travel.
    - Proper ``viewBox`` attribute for resolution-independent rendering.
    - Configurable ``stroke_width``.

    Parameters
    ----------
    data : np.ndarray
        Shape (N, D) with D >= 2.  Only the first two columns are used for
        x/y coordinates.
    filename : str
        Base name (without extension) for the output file.
    output_dir : str
        Root output directory.  SVGs are saved under ``output_dir/svg/``.
    canvas_size : tuple of int
        Width and height of the SVG canvas.
    stroke_width : float
        Stroke width for the path elements.
    epsilon : float
        RDP simplification tolerance (post-normalisation units).
    """
    svg_dir = os.path.join(output_dir, 'svg')
    os.makedirs(svg_dir, exist_ok=True)
    full_path = os.path.join(svg_dir, f"{filename}.svg")

    try:
        # Normalise to canvas
        scaled = _normalise_and_scale(data, canvas_size)

        # Simplify
        scaled = rdp_simplify(scaled, epsilon * max(canvas_size))

        # Split into segments, reorder
        segments = detect_segments(
            scaled / np.array([canvas_size[0], canvas_size[1]], dtype=float),
            threshold=0.05,
        )
        # Scale segments back
        segments = [seg * np.array([canvas_size[0], canvas_size[1]], dtype=float)
                    for seg in segments]
        segments = reorder_segments(segments)

        # Build SVG
        vb = f"0 0 {canvas_size[0]} {canvas_size[1]}"
        dwg = svgwrite.Drawing(
            full_path,
            size=(f"{canvas_size[0]}px", f"{canvas_size[1]}px"),
            viewBox=vb,
        )

        for seg in segments:
            if len(seg) < 2:
                continue
            d_parts = [f"M {seg[0][0]:.2f} {seg[0][1]:.2f}"]
            for pt in seg[1:]:
                d_parts.append(f"L {pt[0]:.2f} {pt[1]:.2f}")
            path_d = " ".join(d_parts)
            dwg.add(dwg.path(
                d=path_d,
                fill='none',
                stroke='black',
                stroke_width=stroke_width,
            ))

        dwg.save()
        logger.info(f"Saved SVG as {full_path}")
    except Exception as e:
        logger.error(f"Failed to save SVG: {str(e)}")
        raise


# ---------------------------------------------------------------------------
# 4B + 4C. G-code output
# ---------------------------------------------------------------------------

def generate_gcode(data: np.ndarray, filename: str, output_dir: str,
                   canvas_size=(100, 100), z_up: float = 5, z_down: float = 0,
                   feed_rate: float = None, canvas_margin: float = 0.05,
                   preserve_aspect: bool = True, epsilon: float = 0.01) -> None:
    """Generate G-code from trajectory data.

    Improvements over the original implementation:
    - RDP simplification reduces point count while preserving shape.
    - Pen-lift detection inserts ``G0 Z{z_up}`` / ``G0 Z{z_down}``
      transitions between discontinuous segments instead of drawing
      through jumps.
    - Segments are reordered (greedy nearest-neighbour) to minimise
      total pen-up travel.
    - ``feed_rate`` appends ``F{rate}`` to ``G1`` draw commands.
    - ``canvas_margin`` keeps artwork away from the physical edges.
    - ``preserve_aspect`` (default ``True``) locks the aspect ratio.

    Parameters
    ----------
    data : np.ndarray
        Shape (N, D) with D >= 2.  First two columns = x/y.
    filename : str
        Base name (without extension) for the output file.
    output_dir : str
        Root output directory.  G-code saved under ``output_dir/gcode/``.
    canvas_size : tuple
        Physical canvas width and height (mm).
    z_up : float
        Z height when pen is raised (mm).
    z_down : float
        Z height when pen is lowered (mm).
    feed_rate : float or None
        If given, ``F{feed_rate}`` is appended to every ``G1`` command.
    canvas_margin : float
        Fraction of canvas reserved as margin on each side (0.05 = 5 %).
    preserve_aspect : bool
        If True, scale uniformly so the aspect ratio is preserved.
    epsilon : float
        RDP simplification tolerance (post-normalisation units).
    """
    gcode_dir = os.path.join(output_dir, 'gcode')
    os.makedirs(gcode_dir, exist_ok=True)
    full_path = os.path.join(gcode_dir, f"{filename}.gcode")

    try:
        # Normalise / scale
        scaled = _normalise_and_scale(data, canvas_size,
                                      canvas_margin=canvas_margin,
                                      preserve_aspect=preserve_aspect)

        # Simplify
        scaled = rdp_simplify(scaled, epsilon * max(canvas_size))

        # Detect segments on normalised coords, then map back
        norm_for_detect = scaled / np.array(
            [float(canvas_size[0]), float(canvas_size[1])])
        segments = detect_segments(norm_for_detect, threshold=0.05)
        segments = [seg * np.array([float(canvas_size[0]), float(canvas_size[1])])
                    for seg in segments]

        # Reorder to minimise travel
        segments = reorder_segments(segments)

        # Build feed-rate suffix
        f_suffix = f" F{feed_rate:.1f}" if feed_rate is not None else ""

        with open(full_path, 'w') as f:
            # Preamble
            f.write("G21 ; Set units to millimeters\n")
            f.write("G90 ; Use absolute coordinates\n")
            f.write(f"G0 Z{z_up} ; Raise pen\n")

            for seg in segments:
                if len(seg) == 0:
                    continue
                # Rapid move to segment start (pen up)
                x0, y0 = seg[0]
                f.write(f"G0 X{x0:.2f} Y{y0:.2f} ; Rapid to segment start\n")
                f.write(f"G0 Z{z_down} ; Lower pen\n")

                # Draw segment
                for pt in seg[1:]:
                    f.write(f"G1 X{pt[0]:.2f} Y{pt[1]:.2f}{f_suffix} ; Draw\n")

                # Lift pen at end of segment
                f.write(f"G0 Z{z_up} ; Raise pen\n")

            # Return home
            f.write("G0 X0 Y0 ; Return to origin\n")

        logger.info(f"Saved G-code as {full_path}")
    except Exception as e:
        logger.error(f"Failed to generate G-code: {str(e)}")
        raise
