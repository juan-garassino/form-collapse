"""Regression tests added by the guardian agent (see GUARDIAN.md)."""
import json
import os

import numpy as np


def test_gallery_escapes_session_fields(tmp_output_dir):
    """G-001: values from session.json must be HTML-escaped in the gallery."""
    from src.gallery import generate_gallery

    session_dir = os.path.join(tmp_output_dir, "20260101_000000")
    os.makedirs(session_dir)
    payload = '<script>alert(1)</script>'
    with open(os.path.join(session_dir, "session.json"), "w") as f:
        json.dump({"results": [{
            "system_name": payload,
            "classification": '"><img src=x onerror=alert(1)>',
            "lyapunov_exponent": 0.5,
            "params": {payload: payload},
            "files": [],
        }]}, f)

    out = generate_gallery(base_dir=tmp_output_dir)
    with open(out) as f:
        html = f.read()
    assert "<script>alert(1)</script>" not in html
    assert "onerror=alert(1)>" not in html
    assert "&lt;script&gt;" in html


def test_zero_elevation_and_azimuth_are_respected():
    """G-002: elevation=0 / azimuth=0 are valid angles, not 'unset'."""
    from src.router import RenderRequest, _plotter_projection
    from src.utils.projection import project_3d_to_2d

    rng = np.random.default_rng(0)
    data = rng.normal(size=(50, 3))
    req = RenderRequest(name="t", data=data, output_dir=".", modes=[],
                        elevation=0.0, azimuth=0.0)
    np.testing.assert_allclose(_plotter_projection(req), project_3d_to_2d(data, 0.0, 0.0))


def test_preserve_aspect_on_non_square_canvas():
    """G-003: preserve_aspect must keep a square bounding box square on a 200x100 canvas."""
    from src.utils.svg_gcode import _normalise_and_scale

    square = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    scaled = _normalise_and_scale(square, (200, 100), preserve_aspect=True)
    extent = scaled.max(axis=0) - scaled.min(axis=0)
    assert np.isclose(extent[0], extent[1])
    assert scaled[:, 0].max() <= 200 and scaled[:, 1].max() <= 100


def test_preserve_aspect_false_fills_canvas():
    """Without preserve_aspect the data stretches to the full drawable area."""
    from src.utils.svg_gcode import _normalise_and_scale

    square = np.array([[0.0, 0.0], [1.0, 1.0]])
    scaled = _normalise_and_scale(square, (200, 100), preserve_aspect=False)
    np.testing.assert_allclose(scaled.max(axis=0), [200, 100])
