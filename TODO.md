# TODO

## Active
- Add a regression test to cover the already-fixed `extended_deltamap.deltamap` module entrypoint import bug.
- Run an end-to-end check with:
  `uv run python examples/TestDeltamap.py examples/LTD_config.ini examples/Synch_var_3freq_r1e-2.ini 1`
- Run an end-to-end check with:
  `uv run python examples/run_pysm3.py examples/LTD_config.ini`
- Add at least a smoke-test suite that verifies package import, bundled data access, and one minimal covariance calculation under Python 3.12.
- Confirm that the new SciPy compatibility shim for `sph_harm` behaves correctly on the exact SciPy versions the team plans to support.

## Backlog
- Silence current warnings, starting with `healpy` deprecation warnings around the `verbose` argument.
- Add type hints to the actively maintained code paths, prioritizing `examples/TestDeltamap.py`, `examples/run_pysm3.py`, and the public package modules.
- Normalize formatting and style, including indentation cleanup and low-risk readability improvements.
- Add Google-style docstrings to public classes, functions, and operational entry points.
- Refactor `examples/TestDeltamap.py` into smaller functions or a package CLI module; it is currently the operational entry point and still mixes I/O, simulation, fitting, and environment-specific assumptions.
- Replace broad `except:` blocks in the example scripts with targeted exceptions.
- Continue exception cleanup by replacing ad-hoc `print()` diagnostics with structured errors or logging where failures need operator context.
- Factor repeated config reads and prior-loading branches in `examples/TestDeltamap.py` into helpers so exception handling can stay local and explicit.
- Standardize naming and style across the codebase. The current code mixes research-script naming, inconsistent indentation, and package-style imports.
- Audit old commented-out blocks in `extended_deltamap/deltamap.py` and `extended_deltamap/covariance.py`; remove dead code where it no longer serves as research context.
- Move the external dust template FITS paths out of environment variables into config keys if reproducible batch execution is a priority.
- Validate that all paths in example `.ini` files resolve correctly when launched from arbitrary working directories.
- Decide whether example outputs such as `inputs/`, `output_pysm3_*`, and `test_results/` should be gitignored more explicitly.
- Document the role of `extended_deltamap` modules and the intended public API.
- Explain the required external data inputs for the example fit flow, especially `DELTAMAP_DUST_BETA_MAP` and `DELTAMAP_DUST_TEMP_MAP`.
- Add a short migration note in release or PR text explaining that old top-level module imports were removed in favor of `extended_deltamap`.
- Add an explicit license decision to `pyproject.toml` instead of the current placeholder `Proprietary`.
- Consider adding console scripts in `pyproject.toml` for the example workflows if they are meant to be supported entry points.
- Review and commit the generated `uv.lock` once the team is happy with the dependency set.
- Decide whether `pysm3` should stay as an optional extra or become part of the default environment for contributors.
- Add a short developer setup section documenting `uv sync`, `uv sync --extra pysm`, and `source .venv/bin/activate`.

## Done
- Fixed config-relative path handling so generated inputs, outputs, and resource paths resolve from the repo root and arbitrary working directories.
- Fixed the `extended_deltamap.deltamap` module entrypoint so `uv run python -m extended_deltamap.deltamap` no longer fails due to a missing `sys` import.
- Fixed `examples/TestDeltamap.py` additional-noise generation so cached `anoise_freq` inputs are reused instead of being unconditionally regenerated after load.
- Updated `extended_deltamap/dmatrix.py` so repeated `PrepareDMatrix()`-style calls rebuild from a fresh template while keeping room for future higher-order derivative terms.
- Clarified the intended artificial-noise split in `examples/TestDeltamap.py`: common artificial noise for CMB and per-frequency artificial noise for instrument.

## Future methodology update: second-order Delta-map expansion
- Design the second-order foreground expansion before changing `extended_deltamap/dmatrix.py`; this is a future methodology update, not part of the current bug-fix pass.
- When extending beyond first order, generate the basis terms fresh for each build while still allowing the current build step to append 0th-, 1st-, and future 2nd-order terms in a controlled order.
- Include both pure second derivatives and mixed derivatives for multi-parameter models, e.g. for dust with `beta_d` and `T_d1` or `x^R` include:
  `∂²f/∂beta_d²`, `∂²f/∂T_d1²`, and `∂²f/(∂beta_d ∂T_d1)`.
- Use `sympy.diff(...)` to generate second derivatives, but avoid duplicating symmetric mixed terms such as `∂²/∂a∂b` and `∂²/∂b∂a`.
- Decide and document the column ordering convention for 0th-, 1st-, and 2nd-order terms before implementation so `D_matrix`, parameter bookkeeping, and downstream fitting code stay aligned.
- Specify the corresponding second-order sky-side unknowns explicitly, including the meaning of terms like `(δp_i)^2 s_b` and `(δp_i δp_j) s_b`.
- Confirm the correct Taylor-expansion coefficients, especially whether diagonal second-order terms should carry a `1/2` factor and how that factor should be absorbed between the frequency-space basis and the sky-side amplitudes.
- Update the likelihood and fitting code only after the second-order basis definition is fixed on paper; adding second-derivative columns alone is not sufficient.
