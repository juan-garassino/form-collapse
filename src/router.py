"""Mode router for dispatching attractor visualizations."""

import logging
import time
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable

import numpy as np

from .attractors.simulators import (
    adaptive_simulation,
    SYSTEM_FUNCTIONS,
    compute_lyapunov_exponent,
    classify_attractor,
)
from .utils.config import generate_system_params, KNOWN_GOOD_PRESETS
from .utils.visualization import (
    plot_attractor,
    plot_phase_space,
    plot_time_series,
    plot_poincare_section,
    plot_power_spectrum,
    plot_bifurcation,
    plot_lyapunov_exponent,
)
from .utils.svg_gcode import save_svg, generate_gcode
from .utils.animations import animate_3d

logger = logging.getLogger(__name__)


class Mode(Enum):
    trajectory = "trajectory"
    phase = "phase"
    poincare = "poincare"
    spectrum = "spectrum"
    bifurcation = "bifurcation"
    timeseries = "timeseries"
    lyapunov = "lyapunov"
    animation = "animation"
    all = "all"


ALL_RENDER_MODES = [m for m in Mode if m != Mode.all]


@dataclass
class RenderRequest:
    name: str
    data: np.ndarray
    output_dir: str
    modes: List[Mode]
    smooth: bool = True
    export_svg: bool = False
    export_gcode: bool = False
    system_config: Optional[Dict[str, Any]] = None
    system_func: Optional[Callable] = None
    lyapunov_exp: Optional[float] = None
    params: Optional[Dict[str, Any]] = None
    elevation: Optional[float] = None
    azimuth: Optional[float] = None
    auto_project: bool = False


class ModeRouter:
    def __init__(self):
        self._handlers: Dict[Mode, Callable] = {
            Mode.trajectory: self._render_trajectory,
            Mode.phase: self._render_phase,
            Mode.poincare: self._render_poincare,
            Mode.spectrum: self._render_spectrum,
            Mode.bifurcation: self._render_bifurcation,
            Mode.timeseries: self._render_timeseries,
            Mode.lyapunov: self._render_lyapunov,
            Mode.animation: self._render_animation,
        }

    def render(self, request: RenderRequest) -> List[str]:
        """Dispatch rendering for all requested modes. Returns list of generated file paths."""
        modes = ALL_RENDER_MODES if Mode.all in request.modes else request.modes
        files = []

        for mode in modes:
            handler = self._handlers.get(mode)
            if handler is None:
                logger.warning(f"Unknown mode: {mode}")
                continue
            try:
                result = handler(request)
                if result:
                    files.extend(result if isinstance(result, list) else [result])
            except Exception as e:
                logger.error(f"Failed to render mode {mode.value} for {request.name}: {e}")

        # Cross-cutting: SVG / G-code export (projection only when needed, and
        # guarded so a projection error cannot discard the files rendered above)
        if not (request.export_svg or request.export_gcode):
            return files
        try:
            data_2d = _plotter_projection(request)
        except Exception as e:
            logger.error(f"Plotter projection failed for {request.name}: {e}")
            return files

        if request.export_svg:
            try:
                save_svg(data_2d, f"{request.name}_svg", request.output_dir)
                files.append(f"{request.name}_svg.svg")
            except Exception as e:
                logger.error(f"SVG export failed for {request.name}: {e}")

        if request.export_gcode:
            try:
                generate_gcode(data_2d, f"{request.name}_gcode", request.output_dir)
                files.append(f"{request.name}_gcode.gcode")
            except Exception as e:
                logger.error(f"G-code export failed for {request.name}: {e}")

        return files

    def _render_trajectory(self, req: RenderRequest) -> Optional[str]:
        plot_attractor(req.name, req.data, req.output_dir, smooth=req.smooth)
        return f"{req.name}_attractor_3d.png"

    def _render_phase(self, req: RenderRequest) -> Optional[str]:
        plot_phase_space(req.name, req.data, req.output_dir, smooth=req.smooth)
        return f"{req.name}_phase_space.png"

    def _render_timeseries(self, req: RenderRequest) -> Optional[str]:
        plot_time_series(req.name, req.data, req.output_dir, smooth=req.smooth)
        return f"{req.name}_time_series.png"

    def _render_poincare(self, req: RenderRequest) -> List[str]:
        files = []
        for plane in ['xy', 'yz', 'xz']:
            plot_poincare_section(req.name, req.data, req.output_dir, plane=plane)
            files.append(f"{req.name}_poincare_section_{plane}.png")
        return files

    def _render_spectrum(self, req: RenderRequest) -> Optional[str]:
        plot_power_spectrum(req.name, req.data, req.output_dir)
        return f"{req.name}_power_spectrum.png"

    def _render_bifurcation(self, req: RenderRequest) -> Optional[str]:
        if req.system_config and 'bifurcation_param' in req.system_config and req.system_func:
            bp = req.system_config['bifurcation_param']
            param_range = np.linspace(bp['start'], bp['stop'], bp['num'])
            plot_bifurcation(req.name, req.system_func, param_range, bp['name'], req.output_dir, base_params=req.params)
            return f"{req.name}_bifurcation_diagram.png"
        logger.info(f"Skipping bifurcation for {req.name}: no bifurcation_param in config")
        return None

    def _render_lyapunov(self, req: RenderRequest) -> Optional[str]:
        if req.lyapunov_exp is not None:
            plot_lyapunov_exponent(req.name, np.array([req.lyapunov_exp]), req.output_dir)
            return f"{req.name}_lyapunov_spectrum.png"
        return None

    def _render_animation(self, req: RenderRequest) -> Optional[str]:
        try:
            animate_3d(req.name, req.data, req.output_dir)
            return f"{req.name}_3d_animation.mp4"
        except Exception as e:
            logger.warning(f"Animation failed for {req.name} (ffmpeg may not be installed): {e}")
            return None


def _plotter_projection(request: RenderRequest) -> np.ndarray:
    """Compute the 2D projection used for SVG / G-code plotter output."""
    if request.auto_project:
        from .utils.projection import find_optimal_projection
        _, _, data_2d = find_optimal_projection(request.data)
        return data_2d
    if request.elevation is not None or request.azimuth is not None:
        from .utils.projection import project_3d_to_2d
        # 0.0 is a valid angle: only fall back to defaults when truly unset.
        return project_3d_to_2d(
            request.data,
            30.0 if request.elevation is None else request.elevation,
            -60.0 if request.azimuth is None else request.azimuth,
        )
    return request.data[:, :2]


# Global router instance
_router = ModeRouter()


def draw_attractor(
    system_name: str,
    config: Dict[str, Any],
    modes: List[Mode],
    output_dir: str,
    smooth: bool = True,
    export_svg: bool = False,
    export_gcode: bool = False,
    use_preset: bool = False,
    session=None,
    progress_callback: Optional[Callable] = None,
    elevation: Optional[float] = None,
    azimuth: Optional[float] = None,
    auto_project: bool = False,
    override_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Unified entry point: simulate an attractor, classify it, and render requested modes.

    Returns dict with keys: data, classification, lyapunov, files, params.
    """
    system_config = config['systems'][system_name]
    func_name = system_config['func']

    # Generate or use preset/override params
    if override_params is not None:
        params = dict(override_params)
    elif use_preset and func_name in KNOWN_GOOD_PRESETS:
        params = dict(KNOWN_GOOD_PRESETS[func_name])
        params['sim_time'] = system_config.get('sim_time', config['global']['default_sim_time'])
        params['sim_steps'] = system_config.get('sim_steps', config['global']['default_sim_steps'])
        params['scale'] = system_config['scale']
    else:
        params = generate_system_params(config, system_name)

    if progress_callback:
        progress_callback("simulating")

    max_time = 60.0 if system_name.lower() == 'three_scroll_system' else 30.0
    t_start = time.time()

    success, data, message = adaptive_simulation(
        system_name,
        func_name,
        params,
        params['sim_time'],
        params['sim_steps'],
        max_attempts=10,
        max_time=max_time,
        scale=system_config['scale'],
    )

    sim_time_elapsed = time.time() - t_start

    if not success or data is None:
        logger.error(f"Simulation failed for {system_name}: {message}")
        return {
            "data": None,
            "classification": "failed",
            "lyapunov": None,
            "files": [],
            "params": params,
            "message": message,
            "sim_time": sim_time_elapsed,
        }

    if progress_callback:
        progress_callback("analyzing")

    # Compute Lyapunov exponent and classify
    dt = params['sim_time'] / params['sim_steps']
    mle = compute_lyapunov_exponent(data, dt=dt)
    classification, metrics = classify_attractor(data, mle)

    if progress_callback:
        progress_callback("rendering")

    # Build render request
    from datetime import datetime
    key = f"{system_name}_{datetime.now().strftime('%m%d_%H%M%S')}"

    system_func_ref = SYSTEM_FUNCTIONS.get(func_name)
    request = RenderRequest(
        name=key,
        data=data,
        output_dir=output_dir,
        modes=modes,
        smooth=smooth,
        export_svg=export_svg,
        export_gcode=export_gcode,
        system_config=system_config,
        system_func=system_func_ref,
        lyapunov_exp=mle,
        params=params,
        elevation=elevation,
        azimuth=azimuth,
        auto_project=auto_project,
    )

    files = _router.render(request)

    # Record in session if provided
    if session is not None:
        session.add_result(system_name, classification, mle, params, files)

    return {
        "data": data,
        "classification": classification,
        "lyapunov": mle,
        "metrics": metrics,
        "files": files,
        "params": params,
        "message": message,
        "sim_time": sim_time_elapsed,
        "key": key,
    }
