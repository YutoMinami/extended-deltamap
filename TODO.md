# TODO

## Current refactor plan
- Silence current warnings, starting with `healpy` deprecation warnings around the `verbose` argument.
- Fix path handling so generated inputs, outputs, and config-relative resources resolve consistently from the repo root and from arbitrary working directories.
- Add type hints to the actively maintained code paths, prioritizing `examples/TestDeltamap.py`, `examples/run_pysm3.py`, and the public package modules.
- Normalize formatting and style, including indentation cleanup and low-risk readability improvements.
- Add Google-style docstrings to public classes, functions, and operational entry points.

## Packaging
- Add an explicit license decision to `pyproject.toml` instead of the current placeholder `Proprietary`.
- Consider adding console scripts in `pyproject.toml` for the example workflows if they are meant to be supported entry points.

## uv / Python 3.12
- Review and commit the generated `uv.lock` once the team is happy with the dependency set.
- Decide whether `pysm3` should stay as an optional extra or become part of the default environment for contributors.
- Add a short developer setup section documenting `uv sync`, `uv sync --extra pysm`, and `source .venv/bin/activate`.

## Code cleanup
- Refactor `examples/TestDeltamap.py` into smaller functions or a package CLI module; it is currently the operational entry point and still mixes I/O, simulation, fitting, and environment-specific assumptions.
- Replace broad `except:` blocks in the example scripts with targeted exceptions.
- Standardize naming and style across the codebase. The current code mixes research-script naming, inconsistent indentation, and package-style imports.
- Audit old commented-out blocks in `extended_deltamap/deltamap.py` and `extended_deltamap/covariance.py`; remove dead code where it no longer serves as research context.

## Configuration and paths
- Move the external dust template FITS paths out of environment variables into config keys if reproducible batch execution is a priority.
- Validate that all paths in example `.ini` files resolve correctly when launched from arbitrary working directories.
- Decide whether example outputs such as `inputs/`, `output_pysm3_*`, and `test_results/` should be gitignored more explicitly.

## Validation
- Run end-to-end checks with:
  `uv run python examples/run_pysm3.py examples/LTD_config.ini`
- Run end-to-end checks with:
  `uv run python examples/TestDeltamap.py examples/LTD_config.ini examples/Synch_var_3freq_r1e-2.ini 1`
- Confirm that the new SciPy compatibility shim for `sph_harm` behaves correctly on the exact SciPy versions the team plans to support.
- Add at least a smoke-test suite that verifies package import, bundled data access, and one minimal covariance calculation under Python 3.12.

## Documentation
- Document the role of `extended_deltamap` modules and the intended public API.
- Explain the required external data inputs for the example fit flow, especially `DELTAMAP_DUST_BETA_MAP` and `DELTAMAP_DUST_TEMP_MAP`.
- Add a short migration note in release or PR text explaining that old top-level module imports were removed in favor of `extended_deltamap`.
