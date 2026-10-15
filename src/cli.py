"""Rich interactive CLI for formCollapse."""

import logging
from typing import Dict, Any, List

from .router import Mode, draw_attractor
from .session import Session

logger = logging.getLogger(__name__)


def _get_available_systems(config: Dict[str, Any]) -> List[str]:
    return list(config['systems'].keys())


def interactive_mode(config: Dict[str, Any]) -> None:
    """Run the interactive Rich CLI when no args are provided."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.prompt import Prompt, Confirm
        from rich.table import Table
        from rich.progress import Progress, SpinnerColumn, TextColumn
    except ImportError:
        print("Rich is not installed. Install it with: pip install rich")
        print("Falling back to non-interactive mode.")
        return

    console = Console()

    # Figlet banner
    try:
        from pyfiglet import Figlet
        fig = Figlet(font="slant")
        banner = fig.renderText("formCollapse")
        console.print(f"[bold cyan]{banner}[/bold cyan]", highlight=False)
    except ImportError:
        pass
    console.print(Panel.fit(
        "[bold cyan]formCollapse[/bold cyan] — Strange Attractor Studio",
        border_style="cyan",
    ))

    systems = _get_available_systems(config)

    # Display available attractors
    table = Table(title="Available Attractors")
    table.add_column("#", style="dim")
    table.add_column("Name", style="bold")
    table.add_column("Function")
    for i, name in enumerate(systems, 1):
        func = config['systems'][name].get('func', '?')
        table.add_row(str(i), name, func)
    console.print(table)

    # Pick attractor
    choice = Prompt.ask(
        "Select attractor (number or name, or 'all')",
        default="all",
    )
    if choice.lower() == 'all':
        selected_systems = systems
    elif choice.isdigit() and 1 <= int(choice) <= len(systems):
        selected_systems = [systems[int(choice) - 1]]
    elif choice in systems:
        selected_systems = [choice]
    else:
        console.print(f"[red]Invalid choice: {choice}[/red]")
        return

    # Pick mode(s)
    mode_names = [m.value for m in Mode]
    console.print(f"\n[bold]Available modes:[/bold] {', '.join(mode_names)}")
    mode_input = Prompt.ask("Select mode(s) (comma-separated)", default="all")
    modes = []
    for m in mode_input.split(","):
        m = m.strip()
        try:
            modes.append(Mode(m))
        except ValueError:
            console.print(f"[yellow]Unknown mode '{m}', skipping[/yellow]")
    if not modes:
        modes = [Mode.all]

    # Export options
    export_svg = Confirm.ask("Export SVG?", default=False)
    export_gcode = Confirm.ask("Export G-code?", default=False)
    use_preset = Confirm.ask("Use known-good preset params?", default=False)
    smooth = Confirm.ask("Smooth trajectories?", default=True)

    # Parameter sweep option
    sweep_input = Prompt.ask("Parameter sweep sample count (0 = no sweep)", default="0")
    sweep_count = int(sweep_input) if sweep_input.isdigit() else 0

    # Run
    session = Session()
    output_dir = session.get_output_dir()

    all_sweep_results = {}

    for system_name in selected_systems:
        console.print(f"\n[bold green]>>> {system_name}[/bold green]")

        if sweep_count > 0:
            # Sweep mode
            from .utils.config import generate_params_lhs
            param_sets = generate_params_lhs(config, system_name, sweep_count)
            sweep_results = []

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Sweeping {system_name}...", total=sweep_count)

                for i, params in enumerate(param_sets):
                    progress.update(task, completed=i, description=f"{system_name}: sweep {i+1}/{sweep_count}")

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
                    )
                    sweep_results.append(result)

                progress.update(task, completed=sweep_count)

            all_sweep_results[system_name] = sweep_results

            # Show sweep summary
            classifications = {}
            for r in sweep_results:
                cls = r['classification']
                classifications[cls] = classifications.get(cls, 0) + 1

            summary = Table.grid(padding=1)
            summary.add_row("[bold]Total runs:[/bold]", str(sweep_count))
            for cls, count in sorted(classifications.items()):
                summary.add_row(f"  {cls}:", str(count))
            console.print(Panel(summary, title=f"{system_name} Sweep Results", border_style="cyan"))

        else:
            # Normal single-run mode
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Processing {system_name}...", total=None)

                def update_progress(stage):
                    progress.update(task, description=f"{system_name}: {stage}")

                result = draw_attractor(
                    system_name=system_name,
                    config=config,
                    modes=modes,
                    output_dir=output_dir,
                    smooth=smooth,
                    export_svg=export_svg,
                    export_gcode=export_gcode,
                    use_preset=use_preset,
                    session=session,
                    progress_callback=update_progress,
                )

            # Summary panel
            if result['data'] is not None:
                summary = Table.grid(padding=1)
                summary.add_row("[bold]Classification:[/bold]", result['classification'])
                summary.add_row("[bold]Lyapunov (MLE):[/bold]", f"{result['lyapunov']:.6f}")
                summary.add_row("[bold]Sim time:[/bold]", f"{result['sim_time']:.2f}s")
                summary.add_row("[bold]Files:[/bold]", str(len(result['files'])))
                console.print(Panel(summary, title=f"{system_name} Results", border_style="green"))
            else:
                console.print(f"[red]Simulation failed: {result['message']}[/red]")

    # Generate heatmaps for sweep results
    if all_sweep_results:
        from .utils.visualization import plot_parameter_heatmap
        for system_name, sweep_results in all_sweep_results.items():
            try:
                plot_parameter_heatmap(sweep_results, system_name, output_dir)
                console.print(f"[dim]Heatmap saved for {system_name}[/dim]")
            except Exception as e:
                console.print(f"[yellow]Heatmap failed for {system_name}: {e}[/yellow]")

    session_path = session.save()
    console.print(f"\n[bold]Session saved:[/bold] {session_path}")
    console.print(f"[bold]Output dir:[/bold]  {output_dir}")
