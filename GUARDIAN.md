# GUARDIAN

Automated security & bug-hunting ledger maintained by the guardian agent on branch `claude/guardian`.

## Checks

Environment: Python 3.12 venv (scratch, not committed), deps from `uv.lock` (torch 2.14.0 from PyPI; the CPU wheel index was unreachable through the proxy).

| Area | Command |
|---|---|
| Dependency CVEs | `uv export --frozen --no-hashes --format requirements-txt > req.txt && pip-audit -r req.txt --no-deps --disable-pip` |
| npm audit | n/a (no `package.json`) |
| Python SAST | `bandit -q -r src main.py` |
| Risky patterns | `grep -rnE "eval\(\|exec\(\|pickle\|torch\.load\|yaml\.load\|shell=True\|verify=False\|os\.system\|0\.0\.0\.0" --include=*.py .` (only `nn.Module.eval()` and `yaml.safe_load` hits) |
| Secrets (history) | `gitleaks detect --source . --redact -v` (15 commits) |
| Secrets (tree) | `gitleaks detect --no-git --source . --redact` |
| Tracked env/log files | `git ls-files \| grep -iE 'env\|log'` |
| GitHub Actions | n/a (no `.github/workflows`) |
| Dockerfiles | n/a (none) |
| Tests | `MPLBACKEND=Agg python -m pytest -q --cov=src` |
| Lint | `ruff check --isolated --select E9,F .` (no ruff config in repo, so pyflakes + syntax errors) |

## Findings ledger

| id | severity | type | file | status | date |
|---|---|---|---|---|---|
| G-001 | medium | HTML/script injection (stored XSS) - `session.json` fields written unescaped into gallery HTML | src/gallery.py | fixed | 2026-10-15 |
| G-002 | low | bug - `--elevation 0` / `--azimuth 0` silently replaced by defaults (30 / -60) via `or` | src/router.py | fixed | 2026-10-15 |
| G-003 | low | bug - `preserve_aspect=True` still distorts on non-square canvases (per-axis scaling) | src/utils/svg_gcode.py | fixed | 2026-10-15 |
| G-004 | low | bandit B112 try/except/continue (intentional: skip degenerate hulls / bad points) | src/utils/projection.py, src/utils/visualization.py | wontfix | 2026-10-15 |
| G-005 | low | info disclosure - tracked `repo_status_puller.log` reveals a local directory layout (no secrets) | repo_status_puller.log | open | 2026-10-15 |
| G-006 | low | quality - 28 unused imports (F401) / 1 redefinition (F811) | 13 files | fixed | 2026-10-15 |
| G-007 | low | quality - `from .attractors import *` hides names (F403/F405), `pytest` listed as runtime dependency | src/attractors/simulators.py, pyproject.toml | open | 2026-10-15 |

## Metrics

| metric | before | after |
|---|---|---|
| open vulns crit/high/med/low | 0/0/1/2 (+1 info-disclosure) | 0/0/0/2 |
| dependency CVEs (pip-audit) | 0 | 0 |
| secrets (gitleaks history + tree) | 0 | 0 |
| failing tests | 0 | 0 |
| test count | 14 | 18 |
| coverage (`src`) | 22% | 30% |
| lint errors (`ruff --isolated --select E9,F`) | 48 | 20 (all F403/F405 star import + 1 F401 + 1 F841) |

## Backlog

- G-005: delete `repo_status_puller.log` from the tree and add `*.log` to `.gitignore` (owner decision; it is harmless but noise).
- G-007: replace the star import in `simulators.py` with explicit imports; move `pytest` to a dev dependency group.
- `Session` directories are named by second-resolution timestamp; two sessions in the same second share a directory and overwrite `session.json`.
- `rdp_simplify` is recursive; extremely long, pathological curves could hit the recursion limit. An iterative version would be safer.
- Coverage gaps: `router.py`, `cli.py`, `visualization.py`, `gan/training.py` are mostly untested.

## Learnings

- The project requires Python >= 3.12; `uv venv -p 3.12` works offline-ish via uv-managed interpreters.
- `download.pytorch.org` is blocked by the proxy; plain PyPI torch installs fine (large).
- Tests need `MPLBACKEND=Agg` on headless machines.

## Ideas

1. Add a minimal GitHub Actions workflow (pinned SHAs, `permissions: contents: read`) running pytest + ruff.
2. Add a smoke test for `draw_attractor` with a tiny `sim_steps` and a mocked render to cover `router.py`.
3. Validate `config.yaml` against a schema at load time (missing `scale`/`func` currently raise bare `KeyError`).

## Log

- 2026-10-15 run #1: first guardian run. No dependency CVEs, no secrets in tree or history, no CI/Docker. Fixed gallery HTML injection (G-001), zero-angle projection bug (G-002) and non-square aspect-ratio bug (G-003), each with a regression test in `tests/test_guardian.py`; removed unused imports. Tests 14 -> 18 passing, coverage 22% -> 30%, lint 48 -> 20.
