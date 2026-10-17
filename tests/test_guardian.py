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


def test_rdp_simplify_long_spiral_no_recursion_error():
    """G-008: an outward spiral makes recursive RDP recurse ~N deep and crash."""
    from src.utils.svg_gcode import rdp_simplify

    t = np.arange(5000)
    pts = np.c_[t * np.cos(t * 0.5), t * np.sin(t * 0.5)]
    pts = (pts - pts.min(0)) / (pts.max(0) - pts.min(0)) * 800
    out = rdp_simplify(pts, 4.0)
    assert 2 <= len(out) <= len(pts)
    np.testing.assert_array_equal(out[0], pts[0])
    np.testing.assert_array_equal(out[-1], pts[-1])


def test_rdp_simplify_matches_reference():
    """G-008: iterative RDP keeps exactly the vertices of the textbook algorithm."""
    from src.utils.svg_gcode import rdp_simplify

    pts = np.array([[0, 0], [1, 0.1], [2, -0.1], [3, 5], [4, 6], [5, 7], [6, 8.1], [7, 9]],
                   dtype=float)
    out = rdp_simplify(pts, 1.0)
    np.testing.assert_array_equal(out, pts[[0, 2, 3, 7]])
    # Degenerate inputs
    assert len(rdp_simplify(pts[:2], 1.0)) == 2
    loop = np.array([[0, 0], [1, 1], [0, 0]], dtype=float)
    np.testing.assert_array_equal(rdp_simplify(loop, 0.5), loop)


def test_sessions_created_same_second_do_not_collide(tmp_output_dir, monkeypatch):
    """G-009: two sessions in the same second must not share (and overwrite) a directory."""
    import src.session as session_mod

    class _FrozenDT(session_mod.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 1, 1, 0, 0, 0)

    monkeypatch.setattr(session_mod, "datetime", _FrozenDT)
    a = session_mod.Session(base_dir=tmp_output_dir)
    b = session_mod.Session(base_dir=tmp_output_dir)
    assert a.session_dir != b.session_dir
    assert os.path.isdir(a.session_dir) and os.path.isdir(b.session_dir)


def test_compute_curvature_planar_circle():
    """G-010: curvature must use the full |r' x r''| (a circle in the xy-plane is 1/R, not 0)."""
    from src.utils.animations import compute_curvature

    R = 5.0
    t = np.linspace(0, 2 * np.pi, 2000)
    k = compute_curvature(R * np.cos(t), R * np.sin(t), np.zeros_like(t))
    np.testing.assert_allclose(k[5:-5], 1 / R, rtol=1e-3)
    # Rotating the plane must not change curvature.
    k2 = compute_curvature(np.zeros_like(t), R * np.cos(t), R * np.sin(t))
    np.testing.assert_allclose(k2[5:-5], 1 / R, rtol=1e-3)


def test_router_projection_failure_does_not_abort_render(tmp_output_dir, monkeypatch):
    """G-011: plotter projection ran unguarded even with no SVG/G-code export,
    so a projection error crashed render() and lost every generated file."""
    import src.utils.projection as projection
    from src.router import Mode, ModeRouter, RenderRequest

    def boom(*a, **k):
        raise ValueError("degenerate data")

    monkeypatch.setattr(projection, "find_optimal_projection", boom)
    router = ModeRouter()
    monkeypatch.setitem(router._handlers, Mode.trajectory, lambda req: "x.png")
    req = RenderRequest(name="t", data=np.random.rand(50, 3), output_dir=tmp_output_dir,
                        modes=[Mode.trajectory], auto_project=True)
    assert router.render(req) == ["x.png"]
    req.export_svg = True
    assert router.render(req) == ["x.png"]  # export failure is logged, not raised


def test_compose_layers_fits_non_square_canvas(tmp_output_dir):
    """G-012: compose_layers scaled y by the canvas WIDTH, so on a wide canvas the
    drawing overflowed the SVG viewBox vertically."""
    import xml.etree.ElementTree as ET
    from src.composition import CompositionLayer, compose_layers

    t = np.linspace(0, 2 * np.pi, 200)
    layer = CompositionLayer(name="c", data=np.c_[np.cos(t), np.sin(t), t])
    files = compose_layers([layer], tmp_output_dir, canvas_size=(1000, 500))
    poly = [e for e in ET.parse(files["svg"]).iter() if e.tag.endswith("polyline")][0]
    pts = np.array([p.split(",") for p in poly.get("points").split()], dtype=float)
    assert pts[:, 0].max() <= 1000 and pts[:, 1].max() <= 500
    assert pts[:, 1].max() > 400  # still uses the full height
