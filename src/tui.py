"""Textual TUI for live attractor preview."""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def live_preview(config: Dict[str, Any], system_name: str, use_preset: bool = False) -> None:
    """Launch a terminal UI for live attractor exploration."""
    try:
        from textual.app import App, ComposeResult
        from textual.widgets import Header, Footer, Static, Log
        from textual.containers import Horizontal, Vertical
    except ImportError:
        print("Textual is not installed. Install with: pip install textual plotext")
        return

    try:
        import plotext as plt
    except ImportError:
        print("plotext is not installed. Install with: pip install plotext")
        return

    from .attractors.simulators import adaptive_simulation, compute_lyapunov_exponent, classify_attractor
    from .utils.config import generate_system_params, KNOWN_GOOD_PRESETS

    class AttractorApp(App):
        CSS = """
        #plot-area { width: 3fr; height: 100%; }
        #sidebar { width: 1fr; height: 100%; background: $surface; padding: 1; }
        #status-bar { height: 3; background: $accent; padding: 0 1; }
        """
        BINDINGS = [
            ("q", "quit", "Quit"),
            ("r", "regenerate", "Regenerate"),
            ("p", "toggle_preset", "Toggle Preset"),
        ]

        def __init__(self, config, system_name, use_preset):
            super().__init__()
            self.config = config
            self.system_name = system_name
            self.use_preset = use_preset
            self.result_data = None

        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal():
                yield Static(id="plot-area")
                with Vertical(id="sidebar"):
                    yield Static("[bold]Parameters[/bold]", id="params-title")
                    yield Static("", id="params-display")
                    yield Static("", id="metrics-display")
            yield Static("", id="status-bar")
            yield Footer()

        def on_mount(self) -> None:
            self.title = f"formCollapse — {self.system_name}"
            self.action_regenerate()

        def action_regenerate(self) -> None:
            system_config = self.config['systems'][self.system_name]
            func_name = system_config['func']

            if self.use_preset and func_name in KNOWN_GOOD_PRESETS:
                params = dict(KNOWN_GOOD_PRESETS[func_name])
                params['sim_time'] = system_config.get('sim_time', self.config['global']['default_sim_time'])
                params['sim_steps'] = system_config.get('sim_steps', self.config['global']['default_sim_steps'])
                params['scale'] = system_config['scale']
            else:
                params = generate_system_params(self.config, self.system_name)

            success, data, message = adaptive_simulation(
                self.system_name, func_name, params,
                params['sim_time'], params['sim_steps'],
                max_attempts=5, max_time=15.0,
                scale=system_config['scale'],
            )

            if success and data is not None:
                self.result_data = data
                dt = params['sim_time'] / params['sim_steps']
                mle = compute_lyapunov_exponent(data, dt=dt)
                classification, metrics = classify_attractor(data, mle)

                # Update params display
                param_text = "\n".join(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}"
                                       for k, v in params.items() if k not in ('sim_time', 'sim_steps', 'scale'))
                self.query_one("#params-display").update(param_text)

                # Update metrics
                metrics_text = (f"[bold]MLE:[/bold] {mle:.6f}\n"
                              f"[bold]Class:[/bold] {classification}\n"
                              f"[bold]Points:[/bold] {len(data)}")
                self.query_one("#metrics-display").update(metrics_text)

                # Update status bar
                color = {"strange_attractor": "green", "limit_cycle": "blue",
                         "fixed_point": "gray", "divergent": "red"}.get(classification, "white")
                self.query_one("#status-bar").update(
                    f"[{color}]{classification}[/{color}] | MLE: {mle:.4f} | Press [bold]r[/bold] to regenerate"
                )

                # ASCII plot using plotext
                try:
                    plt.clear_figure()
                    plt.scatter(data[::10, 0].tolist(), data[::10, 1].tolist(), marker='dot')
                    plt.title(f"{self.system_name} (XY)")
                    plt.theme("dark")
                    plot_str = plt.build()
                    self.query_one("#plot-area").update(plot_str)
                except Exception:
                    self.query_one("#plot-area").update("[dim]Plot rendering failed[/dim]")
            else:
                self.query_one("#status-bar").update(f"[red]Simulation failed: {message}[/red]")

        def action_toggle_preset(self) -> None:
            self.use_preset = not self.use_preset
            self.action_regenerate()

    app = AttractorApp(config, system_name, use_preset)
    app.run()
