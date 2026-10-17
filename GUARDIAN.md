# GUARDIAN

Automated security & bug-hunting ledger maintained by the guardian agent on branch `claude/guardian`.

## Checks

Environment: Python 3.12 venv (scratch, not committed), deps from `uv.lock` (torch 2.14.0 from PyPI; the CPU wheel index was unreachable through the proxy). The PyPI torch wheel needs its `nvidia-*`/`cuda-*` wheels installed even on CPU-only machines (import fails on `libcudart.so.13` otherwise).

| Area | Command |
|---|---|
| Dependency CVEs | `uv export --frozen --no-hashes --format requirements-txt > req.txt && pip-audit -r req.txt --no-deps --disable-pip` |
| npm audit | n/a (no `package.json`) |
| Python SAST | `bandit -q -r src main.py` |
| Risky patterns | `grep -rnE "eval\(\|exec\(\|pickle\|torch\.load\|yaml\.load\|shell=True\|verify=False\|os\.system\|0\.0\.0\.0" --include=*.py .` (only `nn.Module.eval()` and `yaml.safe_load` hits) |
| Secrets (history) | `gitleaks detect --source . --redact -v` (18 commits on run #2) + regex sweep of `git log -p --all` |
| Secrets (tree) | `gitleaks detect --no-git --source . --redact` |
| Tracked env/log files | `git ls-files \| grep -iE 'env\|log'` |
| GitHub Actions | n/a (no `.github/workflows`) |
| Dockerfiles | n/a (none) |
| Tests | `MPLBACKEND=Agg python -m pytest -q --cov=src` |
| Lint | `ruff check --isolated --select E9,F .` (no ruff config in repo, so pyflakes + syntax errors) |
| PR / CI | open review threads on the guardian PR; check runs on its head (none configured) |

## Findings ledger

| id | severity | type | file | status | date |
|---|---|---|---|---|---|
| G-001 | medium | HTML/script injection (stored XSS) - `session.json` fields written unescaped into gallery HTML | src/gallery.py | fixed | 2026-10-15 |
| G-002 | low | bug - `--elevation 0` / `--azimuth 0` silently replaced by defaults (30 / -60) via `or` | src/router.py | fixed | 2026-10-15 |
| G-003 | low | bug - `preserve_aspect=True` still distorts on non-square canvases (per-axis scaling) | src/utils/svg_gcode.py | fixed | 2026-10-15 |
| G-004 | low | bandit B112 try/except/continue (intentional: skip degenerate hulls / bad points) | src/utils/projection.py, src/utils/visualization.py | wontfix | 2026-10-15 |
| G-005 | low | info disclosure - tracked `repo_status_puller.log` reveals a local directory layout (no secrets) | repo_status_puller.log | open | 2026-10-15 |
| G-006 | low | quality - 28 unused imports (F401) / 1 redefinition (F811) | 13 files | fixed | 2026-10-15 |
| G-007 | low | quality - `from .attractors import *` hides names (F403/F405) [fixed run #2]; `pytest` listed as runtime dependency [still open] | src/attractors/simulators.py, pyproject.toml | open (partly fixed 2026-10-17) | 2026-10-15 |
| G-008 | medium | bug (crash) - recursive `rdp_simplify` hits `RecursionError` on long outward-spiral trajectories, aborting SVG/G-code export | src/utils/svg_gcode.py | fixed | 2026-10-17 |
| G-009 | low | bug - two `Session`s started in the same second share one directory and overwrite `session.json` | src/session.py | fixed | 2026-10-17 |
| G-010 | low | bug - `compute_curvature` used only the x-component of r' x r'' (planar xy curves = 0 curvature) and a fixed 1e-6 denominator bias; constant curvature made animation widths NaN | src/utils/animations.py | fixed | 2026-10-17 |
| G-011 | low | bug - plotter projection ran unguarded (even without export) so a projection error crashed `ModeRouter.render` | src/router.py | fixed | 2026-10-17 |
| G-012 | low | bug - `compose_layers` scaled y by canvas width, overflowing non-square canvases | src/composition.py | fixed | 2026-10-17 |
| G-013 | low | bug - discrete maps started from `randn*0.1`; negative x0 diverges the logistic map (~20% of seeds fail) | src/attractors/simulators.py | fixed | 2026-10-17 |
| G-014 | low | bug - GAN preprocessing divided by std=0 for constant columns (2D maps), producing all-NaN training data | src/gan/training.py | fixed | 2026-10-17 |
| G-015 | low | quality - unused `textual.widgets.Log` import (F401) | src/tui.py | fixed | 2026-10-17 |

## Metrics

| metric | run #1 before | run #1 after | run #2 before | run #2 after |
|---|---|---|---|---|
| open vulns crit/high/med/low | 0/0/1/2 | 0/0/0/2 | 0/0/0/2 | 0/0/0/2 |
| open bugs found this run (crash/logic) | - | - | 7 (1 med, 6 low) | 0 |
| dependency CVEs (pip-audit) | 0 | 0 | 0 | 0 |
| secrets (gitleaks history + tree) | 0 | 0 | 0 | 0 |
| failing tests | 0 | 0 | 0 | 0 |
| test count | 14 | 18 | 18 | 26 |
| coverage (`src`) | 22% | 30% | 30% | 40% |
| lint errors (`ruff --isolated --select E9,F`) | 48 | 20 | 20 | 0 |

## Backlog

- G-005: delete `repo_status_puller.log` from the tree and add `*.log` to `.gitignore` (owner decision; it is harmless but noise).
- G-007 (rest): move `pytest` to a `[dependency-groups] dev` group. Tried on run #2 but the local `uv lock` (newer uv) rewrote platform markers across the whole lock; needs the owner's uv version to re-lock cleanly.
- Map configs in `config.yaml` set `n_iterations` / `transient`, but `adaptive_simulation` ignores them (uses `sim_steps` and a hard-coded transient of 1000).
- `discovery._compute_coverage` divides by `grid_size**3` regardless of dimension, so 2D-map coverage scores are not comparable with flows.
- `adaptive_simulation` truncation on time-out (`data[:len*max_time/elapsed]`) drops the tail of an already-computed valid run; probably should keep it.
- Coverage gaps: `cli.py`, `tui.py`, `visualization.py`, `gan/training.py` (train loop) are mostly untested.
- Non-isolated `ruff check .` (a machine-level config, not the repo's) reports ~230 style findings (UP006/I001/BLE001...); not tracked as a metric.

## Learnings

- The project requires Python >= 3.12; `uv venv -p 3.12` works offline-ish via uv-managed interpreters.
- `download.pytorch.org` is blocked by the proxy; plain PyPI torch installs fine (large).
- Tests need `MPLBACKEND=Agg` on headless machines.
- PyPI torch 2.14 on Linux x86_64 is the CUDA build: install `nvidia-*`, `cuda-*` and `triton` from the lock too, or `import torch` fails.
- gitleaks is not preinstalled; the v8 release tarball from GitHub downloads fine through the proxy.
- Running `uv lock` with a different uv version churns markers in `uv.lock`; avoid re-locking unless necessary.

## Ideas

1. Add a minimal GitHub Actions workflow (pinned SHAs, `permissions: contents: read`) running pytest + ruff.
2. Add a smoke test for `draw_attractor` with a tiny `sim_steps` and a mocked render to cover `router.py`.
3. Validate `config.yaml` against a schema at load time (missing `scale`/`func` currently raise bare `KeyError`).
4. Property-based tests (hypothesis) for `rdp_simplify` / `detect_segments` / `reorder_segments` invariants (endpoints kept, output subset of input, no point lost across segments).
5. A tiny end-to-end `draw_attractor` test with `sim_steps=500` and `Mode.trajectory` to cover simulate -> classify -> render -> session.
6. Make `Session` take an optional clock/name so tests do not have to monkeypatch `datetime`.

## Log

- 2026-10-15 run #1: first guardian run. No dependency CVEs, no secrets in tree or history, no CI/Docker. Fixed gallery HTML injection (G-001), zero-angle projection bug (G-002) and non-square aspect-ratio bug (G-003), each with a regression test in `tests/test_guardian.py`; removed unused imports. Tests 14 -> 18 passing, coverage 22% -> 30%, lint 48 -> 20.
- 2026-10-17 run #2: no open review threads or CI on the guardian PR; merged main (no-op). No dependency CVEs, no secrets (gitleaks history + tree, regex sweep), bandit only the known B112 wontfix. Fixed 7 bugs, each with a failing-first regression test in `tests/test_guardian.py`: RecursionError in RDP on long spirals (G-008, rewritten iteratively and checked equivalent to the recursive version on 300 random curves), same-second session collision (G-009), wrong 3D curvature formula + NaN widths (G-010), unguarded plotter projection in the router (G-011), non-square composition overflow (G-012), logistic-map start outside its basin (G-013), NaN GAN inputs for constant columns (G-014). Replaced the star import (G-007 part) and an unused import; isolated lint 20 -> 0. Tests 18 -> 26, coverage 30% -> 40%. Moving pytest to a dev group was reverted (uv lock churn).
