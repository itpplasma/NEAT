# Repository Guidelines

## Project Structure & Module Organization

- `src/neat/`: primary Python package (import as `neat`).
- `src/neatpp/`: native extension sources (C++/Fortran via CMake + pybind11).
- `external/gyronimo/`: vendored dependency (git submodule) used for orbit tracing.
- `external/simple/`: optional benchmark dependency (enabled via CMake `INSTALL_SIMPLE`).
- `tests/`: `unittest` suite; test data lives in `tests/inputs/`.
- `examples/`: runnable scripts that demonstrate common workflows.
- `docs/`: Sphinx documentation (`docs/source/` is the source tree).

## Build, Test, and Development Commands

- Install Python deps: `pip install -r requirements.txt`
- Editable install (builds the native extension): `pip install -e .`
- Run unit tests: `./run_tests` (wrapper that runs `python3 -m unittest -v`)
- Format and lint Python: `./run_autopep` (autopep8, isort, black, flake8, pylint)
- Build docs locally: `make -C docs html`

If CMake cannot find netCDF, provide paths via `NETCDF_INC_PATH` and `NETCDF_LIB_PATH`
when configuring/building.

## Coding Style & Naming Conventions

- Python: `black` formatting, `isort` import ordering, `flake8` checks.
  Keep modules small and prefer `snake_case.py` filenames under `src/neat/`.
- Native code: keep headers as `.hh` and implementations as `.cc` under `src/neatpp/`.
  Follow existing naming and keep Python bindings minimal and explicit.
- Avoid committing generated artifacts (build outputs, `docs/build/`, large result files).

## Testing Guidelines

- Tests must be deterministic and fast; add coverage alongside bug fixes.
- Test files follow `tests/test_*.py`. Put new test inputs under `tests/inputs/`.

## Commit & Pull Request Guidelines

- Commit messages in this repo are short, imperative sentences; keep them specific
  (example: `Fix objective normalization in QS driver`).
- PRs should include: a concise summary, how to reproduce/validate, and any relevant
  plots or numerical output (referencing an `examples/` script when possible).
  Before opening/updating a PR, run `./run_autopep` and `./run_tests`.
