import os
import sys
import argparse
import logging
from typing import Dict, Any, List

import torch

from src.utils.config import get_config, generate_params_lhs
from src.router import Mode, draw_attractor
from src.session import Session
from src.gan.training import train_gan_on_results
from src.utils.visualization import create_summary_plot
from src.utils.data_handling import save_data


def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate strange attractors and create visualizations."
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml).')
    parser.add_argument('--num_simulations', type=int, default=1,
                        help='Number of simulations per attractor.')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Base directory to save results.')
    parser.add_argument('--attractor', type=str, default=None,
                        help='Run a single attractor by name (e.g. Lorenz).')
    parser.add_argument('--mode', type=str, default='all',
                        help='Visualization mode(s), comma-separated. '
                             'Choices: trajectory, phase, poincare, spectrum, '
                             'bifurcation, timeseries, lyapunov, animation, all')
    parser.add_argument('--export', type=str, default=None,
                        help='Export format: svg, gcode, or both.')
    parser.add_argument('--preset', action='store_true',
                        help='Use known-good preset parameters.')
    parser.add_argument('--smooth', action='store_true', default=True,
                        help='Smooth trajectories (default: True).')
    parser.add_argument('--no-smooth', action='store_false', dest='smooth',
                        help='Disable trajectory smoothing.')
    parser.add_argument('--train_gan', action='store_true',
                        help='Train GAN on simulation results.')

    # Sweep mode
    parser.add_argument('--sweep', type=int, default=None,
                        help='Run N LHS-sampled parameter sweeps per attractor.')

    # Discovery mode
    parser.add_argument('--discover', type=int, default=None,
                        help='Run attractor discovery with N candidates.')

    # Live preview
    parser.add_argument('--live', action='store_true',
                        help='Launch Textual TUI for live attractor preview.')

    # Composition
    parser.add_argument('--compose', type=str, default=None,
                        help='Compose multiple attractors (comma-separated names).')

    # Gallery
    parser.add_argument('--gallery', action='store_true',
                        help='Generate HTML gallery from all sessions.')

    # 3D projection
    parser.add_argument('--elevation', type=float, default=None,
                        help='Elevation angle (degrees) for 3D to 2D projection.')
    parser.add_argument('--azimuth', type=float, default=None,
                        help='Azimuth angle (degrees) for 3D to 2D projection.')
    parser.add_argument('--auto-project', action='store_true',
                        help='Automatically find optimal 3D projection angles.')

    # Plotter quality
    parser.add_argument('--rdp-epsilon', type=float, default=0.01,
                        help='RDP simplification epsilon (default: 0.01).')
    parser.add_argument('--feed-rate', type=float, default=None,
                        help='G-code feed rate (F value on G1 commands).')
    parser.add_argument('--canvas-margin', type=float, default=0.05,
                        help='Canvas margin fraction (default: 0.05).')

    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--log_file', type=str, help='File to save logs to.')
    return parser.parse_args()


def _parse_modes(mode_str: str) -> list:
    modes = []
    for m in mode_str.split(','):
        m = m.strip()
        try:
            modes.append(Mode(m))
        except ValueError:
            logging.warning(f"Unknown mode '{m}', skipping")
    return modes if modes else [Mode.all]


def _run_sweep(
    system_name: str,
    config: Dict[str, Any],
    n_samples: int,
    modes: List[Mode],
    output_dir: str,
    session: Session,
    smooth: bool,
    export_svg: bool,
    export_gcode: bool,
    logger: logging.Logger,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Run N LHS-sampled parameter sweeps for a system."""
    param_sets = generate_params_lhs(config, system_name, n_samples)
    sweep_results = []

    for i, params in enumerate(param_sets):
        logger.info(f"Sweep {i+1}/{n_samples} for {system_name}")

        result = draw_attractor(
            system_name=system_name,
            config=config,
            modes=modes,
            output_dir=output_dir,
            smooth=smooth,
            export_svg=export_svg,
            export_gcode=export_gcode,
            use_preset=False,
            session=session,
            override_params=params,
            **kwargs,
        )

        sweep_results.append(result)

        if result['data'] is not None:
            logger.info(
                f"  {result['classification']} "
                f"(MLE={result['lyapunov']:.4f})"
            )
        else:
            logger.warning(f"  Failed: {result['message']}")

    return sweep_results


def main() -> None:
    # No args at all -> interactive mode
    if len(sys.argv) == 1:
        config_path = 'config.yaml'
        if not os.path.exists(config_path):
            print(f"No config file found at {config_path}. Please provide --config.")
            return
        config = get_config(config_path)
        setup_logging()

        from src.cli import interactive_mode
        interactive_mode(config)
        return

    args = parse_arguments()
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger("main")

    logger.info("Loading configuration")
    try:
        config = get_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        return

    # --- Gallery mode (no simulation needed) ---
    if args.gallery:
        from src.gallery import generate_gallery
        path = generate_gallery(base_dir=args.output_dir)
        logger.info(f"Gallery generated: {path}")
        return

    # --- Live preview mode ---
    if args.live:
        if not args.attractor:
            logger.error("--live requires --attractor")
            return
        from src.tui import live_preview
        live_preview(config, args.attractor, use_preset=args.preset)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Parse modes and export flags
    modes = _parse_modes(args.mode)
    export_svg = args.export in ('svg', 'both') if args.export else False
    export_gcode = args.export in ('gcode', 'both') if args.export else False

    # Determine which systems to run
    if args.attractor:
        if args.attractor not in config['systems']:
            logger.error(f"Unknown attractor: {args.attractor}. "
                         f"Available: {list(config['systems'].keys())}")
            return
        system_names = [args.attractor]
    else:
        system_names = list(config['systems'].keys())

    # Common kwargs for projection/plotter options
    extra_kwargs = {}
    if args.elevation is not None:
        extra_kwargs['elevation'] = args.elevation
    if args.azimuth is not None:
        extra_kwargs['azimuth'] = args.azimuth
    if args.auto_project:
        extra_kwargs['auto_project'] = True
    extra_kwargs['plotter_opts'] = {
        'epsilon': args.rdp_epsilon,
        'feed_rate': args.feed_rate,
        'canvas_margin': args.canvas_margin,
    }

    # --- Composition mode ---
    if args.compose:
        compose_names = [n.strip() for n in args.compose.split(',')]
        from src.composition import CompositionLayer, compose_layers
        session = Session(base_dir=args.output_dir)
        output_dir = session.get_output_dir()
        layers = []

        colors = ['black', 'red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta']
        for i, name in enumerate(compose_names):
            if name not in config['systems']:
                logger.error(f"Unknown attractor for composition: {name}")
                continue
            result = draw_attractor(
                system_name=name, config=config, modes=[Mode.trajectory],
                output_dir=output_dir, smooth=args.smooth,
                export_svg=False, export_gcode=False,
                use_preset=args.preset, session=session, **extra_kwargs,
            )
            if result['data'] is not None:
                layers.append(CompositionLayer(
                    name=name, data=result['data'],
                    color=colors[i % len(colors)], pen_number=i + 1,
                ))

        if layers:
            files = compose_layers(layers, output_dir)
            logger.info(f"Composition: {files}")

        session.save()
        return

    # --- Discovery mode ---
    if args.discover:
        if not args.attractor:
            logger.error("--discover requires --attractor")
            return
        from src.discovery import discover_attractors, create_gallery
        session = Session(base_dir=args.output_dir)
        output_dir = session.get_output_dir()
        results = discover_attractors(
            args.attractor, config, n_candidates=args.discover, output_dir=output_dir,
        )
        if results:
            create_gallery(results, output_dir, system_name=args.attractor)
        logger.info(f"Discovery complete: {len(results)} attractors found")
        session.save()
        return

    # Create session
    session = Session(base_dir=args.output_dir)
    output_dir = session.get_output_dir()

    logger.info(f"Session output: {output_dir}")
    logger.info(f"Attractors: {system_names}")
    logger.info(f"Modes: {[m.value for m in modes]}")

    results = {}
    all_sweep_results = {}

    for system_name in system_names:
        system_config = config['systems'][system_name]
        if 'func' not in system_config:
            logger.error(f"Missing 'func' in configuration for {system_name}. Skipping.")
            continue

        # --- Sweep mode ---
        if args.sweep:
            sweep_results = _run_sweep(
                system_name=system_name, config=config,
                n_samples=args.sweep, modes=modes,
                output_dir=output_dir, session=session,
                smooth=args.smooth, export_svg=export_svg,
                export_gcode=export_gcode, logger=logger, **extra_kwargs,
            )
            all_sweep_results[system_name] = sweep_results

            # Collect successful results
            for sr in sweep_results:
                if sr['data'] is not None:
                    results[sr.get('key', system_name)] = sr['data']
            continue

        # --- Normal mode ---
        for i in range(args.num_simulations):
            logger.info(f"Simulation {i+1}/{args.num_simulations} for {system_name}")

            result = draw_attractor(
                system_name=system_name,
                config=config,
                modes=modes,
                output_dir=output_dir,
                smooth=args.smooth,
                export_svg=export_svg,
                export_gcode=export_gcode,
                use_preset=args.preset,
                session=session,
                **extra_kwargs,
            )

            if result['data'] is not None:
                key = result.get('key', f"{system_name}_{i}")
                results[key] = result['data']
                logger.info(
                    f"{system_name}: {result['classification']} "
                    f"(MLE={result['lyapunov']:.4f}, {result['sim_time']:.1f}s)"
                )
            else:
                logger.warning(f"Simulation failed for {system_name}: {result['message']}")

    logger.info(f"Total successful simulations: {len(results)}")

    # Generate sweep heatmaps
    if args.sweep and all_sweep_results:
        from src.utils.visualization import plot_parameter_heatmap
        for system_name, sweep_results in all_sweep_results.items():
            try:
                plot_parameter_heatmap(sweep_results, system_name, output_dir)
            except Exception as e:
                logger.warning(f"Heatmap failed for {system_name}: {e}")

    if results:
        create_summary_plot(results, output_dir, smooth=args.smooth)
        save_data(results, output_dir)

        if args.train_gan:
            logger.info("Starting GAN training")
            train_gan_on_results(results, device, config, output_dir)

    # Save session
    session_path = session.save()
    logger.info(f"Session saved: {session_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
