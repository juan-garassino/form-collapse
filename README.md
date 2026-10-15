# formCollapse

Strange attractor simulation studio. Generates artistic outputs from chaotic dynamical systems — 3D plots, phase portraits, SVG files, G-code for pen plotters, GAN-generated trajectories, and more.

```
     ____                     ______      ____
    / __/___  _________ ___  / ____/___  / / /___ _____  ________
   / /_/ __ \/ ___/ __ `__ \/ /   / __ \/ / / __ `/ __ \/ ___/ _ \
  / __/ /_/ / /  / / / / / / /___/ /_/ / / / /_/ / /_/ (__  )  __/
 /_/  \____/_/  /_/ /_/ /_/\____/\____/_/_/\__,_/ .___/____/\___/
                                                /_/
```

## Quick Start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

```bash
uv sync

# Smallest run (CPU only, about 15 s): one Lorenz trajectory with preset parameters
uv run python main.py --attractor Lorenz --mode trajectory --preset

# Interactive mode (Rich CLI with menus)
uv run python main.py

# Command-line mode
uv run python main.py --attractor Lorenz --mode all --export both
uv run python main.py --mode phase,timeseries --export svg
```

The examples below use `python main.py`; prefix with `uv run` if you have not activated the `.venv` that `uv sync` creates.

### Example output

The smallest run above writes a session directory under `results/` (git-ignored) containing `session.json`, `png/` (3D plot, 2D projections, summary plot) and `data/` (trajectory CSV). Its log ends with a classification line like:

```
Lorenz: strange_attractor (MLE=1.0304, 0.2s)
```

The MLE value varies between runs because the initial condition is random: two runs on 2026-10-15 with the pinned dependencies gave 1.0304 and 0.7594. These numbers are not stored in the repo. The repo's own check is `tests/test_core.py`, which asserts that the Lorenz MLE estimate lies in the range 0.3 to 2.0 (the Lorenz system's MLE is positive, as the method in [docs/lyapunov.md](docs/lyapunov.md) expects for chaos).

## Features

- **19 attractor systems**: 17 ODE systems (Lorenz, Aizawa, Rabinovich-Fabrikant, Chen, Halvorsen, Newton-Leipnik, Three-Scroll, Rossler, Anishchenko, Arnold, Burke-Shaw, Chen-Celikovsky, Finance, Qi-Chen, Rayleigh-Benard, TSUCS1, Liu-Chen) + 2 discrete maps (Henon, Ikeda)
- **9 visualization modes**: trajectory, phase, poincare, spectrum, bifurcation, timeseries, lyapunov, animation, all
- **Lyapunov exponent estimation**: Rosenstein nearest-neighbor method (proper MLE)
- **Attractor classification**: automatic detection of strange attractors, limit cycles, fixed points, divergent trajectories
- **Parameter sweep**: LHS-sampled sweeps with heatmap visualization (`--sweep N`)
- **Attractor discovery**: automated search for interesting parameter regimes (`--discover N`)
- **3D projection controls**: elevation/azimuth angles or auto-optimized projection for plotter output
- **Plotter-quality export**: RDP simplification, pen-lift detection, feed rate control, canvas margins, TSP stroke reorder
- **Multi-attractor composition**: layer multiple attractors with transforms and tool changes (`--compose`)
- **Live preview**: Textual TUI with ASCII plots and real-time parameter exploration (`--live`)
- **Web gallery**: HTML gallery from all sessions with classification filtering (`--gallery`)
- **Session management**: timestamped output directories with `session.json` metadata
- **GAN training**: trajectory-segment generative model on attractor data
- **Interactive CLI**: Rich-powered menus with pyfiglet banner, progress bars, result panels

## CLI Reference

### Interactive Mode

```bash
python main.py
```

No arguments launches the Rich interactive CLI. Pick attractors, modes, export options, and sweep count from menus.

### Command-Line Mode

```bash
python main.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `config.yaml` | Path to configuration file |
| `--attractor` | all systems | Run a single attractor by name |
| `--mode` | `all` | Visualization mode(s), comma-separated |
| `--export` | none | Export format: `svg`, `gcode`, or `both` |
| `--preset` | off | Use known-good preset parameters |
| `--smooth` / `--no-smooth` | on | Trajectory smoothing |
| `--num_simulations` | 1 | Simulations per attractor |
| `--output_dir` | `results` | Base results directory |
| `--train_gan` | off | Train GAN on results |
| `--sweep N` | off | Run N LHS-sampled parameter sweeps |
| `--discover N` | off | Run attractor discovery with N candidates |
| `--live` | off | Launch Textual TUI live preview |
| `--compose A,B` | off | Compose multiple attractors into one output |
| `--gallery` | off | Generate HTML gallery from all sessions |
| `--elevation` | none | Elevation angle for 3D projection (degrees) |
| `--azimuth` | none | Azimuth angle for 3D projection (degrees) |
| `--auto-project` | off | Auto-find optimal projection angles |
| `--rdp-epsilon` | 0.01 | RDP simplification epsilon |
| `--feed-rate` | none | G-code feed rate (F value) |
| `--canvas-margin` | 0.05 | Canvas margin fraction |
| `--log_level` | `INFO` | Logging verbosity |
| `--log_file` | none | Log to file |

### Modes

| Mode | Output |
|------|--------|
| `trajectory` | 3D attractor plot + 2D projections |
| `phase` | Phase space density (KDE) |
| `timeseries` | X/Y/Z vs time |
| `poincare` | Poincare sections (xy, yz, xz planes) |
| `spectrum` | Power spectral density |
| `bifurcation` | Bifurcation diagram (if param configured) |
| `lyapunov` | Lyapunov exponent spectrum plot |
| `animation` | 3D animation (requires ffmpeg) |
| `all` | All of the above |

### Examples

```bash
# Single attractor with preset parameters
python main.py --attractor Lorenz --mode trajectory --preset

# Multiple modes with SVG export
python main.py --attractor Chen --mode trajectory,phase,timeseries --export svg

# Parameter sweep with 20 LHS samples
python main.py --attractor Lorenz --sweep 20 --mode trajectory

# Discover interesting parameter regimes (top 25 from 500 candidates)
python main.py --attractor Lorenz --discover 500

# Export G-code with optimal projection and feed rate
python main.py --attractor Lorenz --preset --export gcode --auto-project --feed-rate 1000

# Compose two attractors into one SVG/G-code
python main.py --compose Lorenz,Rossler --preset --export both

# Live terminal preview
python main.py --live --attractor Lorenz --preset

# Generate HTML gallery from all past sessions
python main.py --gallery

# Discrete map
python main.py --attractor Henon --mode trajectory --preset
```

## Output Structure

Each run creates a timestamped session directory:

```
results/
  20260328_142417/
    session.json          # Metadata: params, classification, Lyapunov, file list
    png/                  # Plot images (trajectory, phase, heatmap, etc.)
    svg/                  # SVG vector files
    gcode/                # G-code for pen plotters
    data/                 # Raw trajectory CSV
    animations/           # MP4 animations
  gallery/
    index.html            # HTML gallery (generated with --gallery)
```

## Configuration

`config.yaml` defines attractor systems and their parameter ranges. See [docs/configuration.md](docs/configuration.md) for full details.

## Architecture

```
main.py                    # Entry point: CLI args, sweep, discover, compose, gallery
src/
  cli.py                   # Rich interactive CLI with sweep support
  router.py                # Mode enum, ModeRouter, draw_attractor() entry point
  session.py               # Session manager (timestamped dirs, session.json)
  discovery.py             # Attractor discovery engine with coverage ranking
  composition.py           # Multi-attractor layer composition
  gallery.py               # HTML gallery generator
  tui.py                   # Textual TUI live preview
  attractors/
    attractors.py           # ODE system definitions (17 systems)
    maps.py                 # Discrete map definitions (Henon, Logistic, Standard, Ikeda)
    simulators.py           # Adaptive solver, map iterator, Lyapunov, classification
  utils/
    config.py               # YAML config, param generation, LHS, presets (19 systems)
    visualization.py        # Matplotlib plots + parameter heatmaps
    svg_gcode.py            # SVG/G-code with RDP, pen-lifts, TSP reorder
    projection.py           # 3D-to-2D projection (elevation/azimuth/auto)
    animations.py           # 3D animation with curvature-based line thickness
    data_handling.py        # CSV data saving
    gan_interpolations.py   # Latent space interpolation
  gan/
    models.py               # Generator/Discriminator (Conv1d trajectory segments)
    training.py             # GAN training loop with segment windowing
tests/
  test_core.py              # 14 tests: Lyapunov, classification, GAN, maps, projection, RDP
```

## Requirements

Python 3.12+. Runtime dependencies (see `pyproject.toml`): numpy, scipy, matplotlib, seaborn, torch, pyyaml, svgwrite, rich, pyfiglet, textual, plotext (pytest is also listed there).

Optional: `ffmpeg` (for animation export)

## Tests

```bash
uv run pytest tests/ -v
```

14 tests in `tests/test_core.py`, all passing at the time of this README update.

## License

MIT
