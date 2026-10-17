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


def test_map_initial_state_is_in_basin():
    """G-013: maps started from randn*0.1, so a negative x0 sent the logistic map
    to -inf (NaN) and the run failed validation (~20% of seeds)."""
    import warnings

    from src.attractors.simulators import adaptive_simulation

    cases = [('logistic_map', {'r': 3.9}), ('henon_map', {'a': 1.4, 'b': 0.3}),
             ('ikeda_map', {'u': 0.9}), ('standard_map', {'K': 1.2})]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for func, params in cases:
            for seed in range(20):
                np.random.seed(seed)
                ok, _, msg = adaptive_simulation('t', func, params, 10, 2000)
                assert ok, (func, seed, msg)


def test_gan_preprocess_constant_column_is_finite():
    """G-014: discrete maps have a constant z column; std=0 made every GAN input NaN."""
    import torch

    from src.gan.training import preprocess_data

    data = np.c_[np.random.rand(256, 2), np.zeros(256)]
    out = preprocess_data({"henon": data}, seq_len=64)
    assert out.shape == (7, 64, 3)
    assert torch.isfinite(out).all()


def test_save_svg_paths_stay_on_canvas(tmp_output_dir, lorenz_data):
    """Export path: SVG is written, parseable, and every vertex lies on the canvas."""
    import re
    import xml.etree.ElementTree as ET

    from src.utils.svg_gcode import save_svg

    save_svg(lorenz_data[:, :2], "lz", tmp_output_dir, canvas_size=(800, 400))
    path = os.path.join(tmp_output_dir, "svg", "lz.svg")
    paths = [e for e in ET.parse(path).iter() if e.tag.endswith("path")]
    assert paths
    nums = np.array([float(v) for p in paths for v in re.findall(r"-?\d+\.?\d*", p.get("d"))])
    xy = nums.reshape(-1, 2)
    assert (xy >= 0).all() and (xy[:, 0] <= 800).all() and (xy[:, 1] <= 400).all()


def test_load_config_errors(tmp_output_dir):
    """Config loading: missing file, malformed YAML and empty path fail loudly."""
    import pytest
    import yaml

    from src.utils.config import get_config, load_config

    with pytest.raises(FileNotFoundError):
        load_config(os.path.join(tmp_output_dir, "nope.yaml"))
    bad = os.path.join(tmp_output_dir, "bad.yaml")
    with open(bad, "w") as f:
        f.write("systems: [unclosed\n")
    with pytest.raises(yaml.YAMLError):
        load_config(bad)
    with pytest.raises(ValueError):
        get_config("")


def test_draw_attractor_smoke(config, tmp_output_dir, monkeypatch):
    """Idea 2: simulate -> classify -> render (mocked) -> session, end to end."""
    import src.router as router
    from src.session import Session

    seen = {}

    def fake_render(req):
        seen["req"] = req
        return ["fake.png"]

    monkeypatch.setattr(router._router, "render", fake_render)
    session = Session(base_dir=tmp_output_dir)
    params = {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0,
              "sim_time": 20.0, "sim_steps": 2000, "scale": 20.0}
    np.random.seed(0)
    out = router.draw_attractor("Lorenz", config, [router.Mode.trajectory], tmp_output_dir,
                                session=session, override_params=params, elevation=0.0)
    assert out["files"] == ["fake.png"]
    assert out["data"].shape == (2000, 3)
    assert out["classification"] != "failed"
    assert seen["req"].elevation == 0.0 and seen["req"].modes == [router.Mode.trajectory]
    assert session.results[0]["system_name"] == "Lorenz"
    with open(session.save()) as f:
        assert json.load(f)["num_results"] == 1


def test_draw_attractor_simulation_failure(config, tmp_output_dir, monkeypatch):
    """A failed simulation returns a 'failed' result instead of rendering."""
    import src.router as router

    monkeypatch.setattr(router, "adaptive_simulation", lambda *a, **k: (False, None, "boom"))
    monkeypatch.setattr(router._router, "render", lambda req: _render_must_not_run())
    out = router.draw_attractor("Lorenz", config, [router.Mode.all], tmp_output_dir, use_preset=True)
    assert out["classification"] == "failed" and out["files"] == [] and out["message"] == "boom"


def _render_must_not_run():
    raise AssertionError("render must not be called after a failed simulation")


def test_plotter_cli_options_reach_exporters(tmp_output_dir, monkeypatch):
    """G-016: --rdp-epsilon / --feed-rate / --canvas-margin were parsed but never
    passed to the SVG / G-code writers, so they were silently ignored."""
    import src.router as router

    calls = {}
    monkeypatch.setattr(router, "save_svg", lambda *a, **k: calls.setdefault("svg", k))
    monkeypatch.setattr(router, "generate_gcode", lambda *a, **k: calls.setdefault("gcode", k))
    req = router.RenderRequest(name="t", data=np.random.rand(50, 3), output_dir=tmp_output_dir,
                               modes=[], export_svg=True, export_gcode=True,
                               plotter_opts={"epsilon": 0.2, "feed_rate": 900.0,
                                             "canvas_margin": 0.1})
    router.ModeRouter().render(req)
    assert calls["svg"] == {"epsilon": 0.2}
    assert calls["gcode"] == {"epsilon": 0.2, "feed_rate": 900.0, "canvas_margin": 0.1}
