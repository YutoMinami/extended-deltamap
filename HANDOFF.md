# HANDOFF

## Current branch
- `feature/fix-deltamap-main-entrypoint`

## What changed
- Fixed the `extended_deltamap.deltamap` module entrypoint by adding the missing `sys` import so `uv run python -m extended_deltamap.deltamap` no longer fails with `NameError`.
- Added `pyproject.toml` for setuptools-based packaging with `uv` support.
- Set `requires-python = ">=3.12"` and added `.python-version` with `3.12`.
- Created a real package directory at `extended_deltamap/`.
- Copied the core modules into the package:
  `extended_deltamap/deltamap.py`,
  `extended_deltamap/dmatrix.py`,
  `extended_deltamap/covariance.py`,
  `extended_deltamap/templates.py`.
- Added `extended_deltamap/__init__.py` exporting `Covariance`, `DeltaMap`, `DMatrix`, and `Templates`.
- Bundled the two theory spectrum `.dat` files into `extended_deltamap/files/` and registered them in `pyproject.toml`.
- Updated `README.md` to describe `uv` usage and Python 3.12.
- Updated `examples/TestDeltamap.py` to import from `extended_deltamap` instead of relying on `sys.path.append('../')`.
- Updated `examples/TestDeltamap.py` and `examples/run_pysm3.py` so config-relative paths work from the repo root, not only from inside `examples/`.
- Replaced hard-coded research-cluster FITS paths in `examples/TestDeltamap.py` with environment variables:
  `DELTAMAP_DUST_BETA_MAP`
  `DELTAMAP_DUST_TEMP_MAP`
- Added `.gitignore` for `.venv` and Python cache files.
- Added a SciPy compatibility shim in `extended_deltamap/covariance.py` so Python 3.12 + current SciPy resolves spherical harmonics correctly.
- Removed the legacy top-level modules and duplicate root-level bundled spectra so `extended_deltamap/` is now the only source of package code and theory data.
- Generated `uv.lock`.

## Environment status
- A local uv virtual environment exists at `.venv/`.
- It was created with:
  `uv venv .venv --python 3.12 --prompt delta-map`
- `uv sync` completed successfully in this repository.
- `uv run python` package import check passed after the SciPy compatibility fix.

## Known limitations
- The package layout is improved, but the repository has not been fully normalized around `src/` layout or CLI entry points.
- `examples/TestDeltamap.py` is still large and research-script shaped.
- End-to-end scientific validation was not completed in this session because the external FITS inputs are not bundled in the repository.
- `uv.lock` exists but still needs a normal review and commit with the rest of the branch changes.
- A code-vs-paper review found several likely bugs that still need implementation work:
  `examples/TestDeltamap.py` appears to regenerate `anoise_freq` even when a cached file was loaded, which breaks reproducibility.
  `extended_deltamap/dmatrix.py` accumulates columns across repeated matrix-preparation calls because `D_matrix_template` is not reset.
  The handling of the shared `anoise` term may be inconsistent between the simulated maps and the covariance model, so it should be checked against the intended likelihood assumptions in the paper.

## Suggested next steps
1. Fix the remaining review findings in `examples/TestDeltamap.py`, `extended_deltamap/deltamap.py`, and `extended_deltamap/dmatrix.py`.
2. Add regression tests for the module entrypoint, D-matrix rebuild behavior, and prior accumulation.
3. Run the example workflows with real external FITS inputs and confirm the path changes behave as intended.
4. Review and commit `uv.lock`.

## Useful commands
- Create or refresh the environment:
  `uv venv .venv --python 3.12 --prompt delta-map`
- Install default dependencies:
  `uv sync`
- Install optional PySM support:
  `uv sync --extra pysm`
- Activate the environment:
  `source .venv/bin/activate`
- Run a package import smoke check:
  `uv run python - <<'PY'
import extended_deltamap
from extended_deltamap import DeltaMap, DMatrix, Covariance, Templates
print(extended_deltamap.__all__)
PY`
- Run the PySM example:
  `uv run python examples/run_pysm3.py examples/LTD_config.ini`
- Run the fit example:
  `export DELTAMAP_DUST_BETA_MAP=/path/to/dust_beta.fits`
  `export DELTAMAP_DUST_TEMP_MAP=/path/to/dust_temp.fits`
  `uv run python examples/TestDeltamap.py examples/LTD_config.ini examples/Synch_var_3freq_r1e-2.ini 1`

## Files most relevant to continue from here
- [pyproject.toml](/home/yminami/workdir/DeltaMap/extended-deltamap/pyproject.toml:1)
- [README.md](/home/yminami/workdir/DeltaMap/extended-deltamap/README.md:1)
- [examples/TestDeltamap.py](/home/yminami/workdir/DeltaMap/extended-deltamap/examples/TestDeltamap.py:1)
- [examples/run_pysm3.py](/home/yminami/workdir/DeltaMap/extended-deltamap/examples/run_pysm3.py:1)
- [extended_deltamap/covariance.py](/home/yminami/workdir/DeltaMap/extended-deltamap/extended_deltamap/covariance.py:1)
