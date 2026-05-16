# TODO

## Active
- Finish the in-progress `DMatrix` API cleanup so `PrepareDMatrix(order=...)`
  becomes the public entry point for 0th-, 1st-, and future 2nd-order builds,
  while `dim_params` always matches the actual column count.
- Pause deeper second-order Delta-map fitting work and shift the next
  methodology work toward region-wise foreground parameters.
- Keep the second-order findings documented before pushing the
  current `order=2` path further; 9-band dust tests show that
  `order=1` can recover `r` while `order=2` still becomes unstable or
  collapses to `r ≈ 0`, even after restoring the explicit Taylor `1/2`
  coefficients on diagonal second-derivative terms.
- Check whether `nside=4` is simply too coarse to constrain the added
  second-order sky-side terms; low spatial information may be part of why the
  `order=2` likelihood becomes unstable even when the first-order model works.
- Pre-Step-4 cleanup A: replace `_count_component_terms` in `dmatrix.py` with a
  call to `_build_component_terms(sympy.Integer(1), ...)` and `len()` so the two
  implementations cannot silently diverge. Update the corresponding smoke test to
  confirm it still passes.
- Pre-Step-4 cleanup B: move `expand_to_qu` and `validate_region_masks` from
  `scripts/make_synch_brightness_regions.py` into `extended_deltamap/regions.py`;
  update the script to import from there; export both from `__init__.py`; add a
  smoke test confirming the import and that `validate_region_masks` raises on
  overlapping inputs. Do not move the script-level I/O helpers.
- Add a regression test for CalcH_matrix confirming that the no-mask path
  (all column_masks=None) produces the same DTNID, DTNIDc, and DTNIM values
  as before Step 4. Can be a unit test with a small synthetic NI_list and D
  matrix rather than a full end-to-end run.
- Implement region-wise foreground-parameter prototype following the 5-step
  plan (see HANDOFF.md for detail). Start with synchrotron-only, 2 fixed
  regions, beta_s only, first-order Delta-map.
  Step 1: region mask creation — boolean arrays of shape (size,), expand pixel
    mask to Q/U, validate disjoint and full coverage.
  Step 2: add symbol_name kwarg to Templates methods so per-region symbols
    beta_s_sreg0, beta_s_sreg1, etc. can be generated.
  Step 3: add column_masks list to DMatrix; AddD() accepts optional region_mask;
    each derivative column inherits the same mask as its parent component.
  Step 4: update CalcH_matrix() to apply mask operators row/column-wise when
    building each (i,j) pixel-space block of DTNID, DTNIDc, and DTNIM.
  Step 5: add parameter expansion helper in TestDeltamap.py to convert
    beta_s=[v0,v1] into beta_s_sreg0, beta_s_sreg1 inits entries.
  Validate with column_masks=None first (existing tests must still pass), then
  enable masks and check beta_s recovery for the 2-region synchrotron case.
- Keep dust and synchrotron region sets independent in the design; their
  clustering maps, region masks, parameter names, and final region counts may
  differ.
- When finalizing supported SciPy versions, confirm that the `sph_harm` compatibility shim still matches the intended API and numerical behavior.

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
- Added a `unittest` smoke/regression suite covering package import, the fixed `extended_deltamap.deltamap` module entrypoint, one minimal covariance build, and repeated `DMatrix` rebuilds.
- Confirmed that `uv run python examples/run_pysm3.py examples/LTD_config.ini` completes successfully in the local environment.
- Confirmed that `uv run python examples/TestDeltamap.py examples/LTD_config.ini examples/Synch_var_3freq_r1e-2.ini 1` completes successfully when `DELTAMAP_DUST_BETA_MAP` and `DELTAMAP_DUST_TEMP_MAP` are set to local FITS inputs.

## Future methodology update: second-order Delta-map expansion
- Design the second-order foreground expansion before changing `extended_deltamap/dmatrix.py`; this is a future methodology update, not part of the current bug-fix pass.
- Agreed design direction so far:
  mixed derivatives should appear only once per unordered parameter pair.
- Agreed design direction so far:
  derivative multi-indices should follow parameter-name order.
- Agreed design direction so far:
  component ordering should continue to follow the current convention
  (`dust` block first, then `synchrotron` block).
- Agreed design direction so far:
  `PrepareOneDiffDMatrix()` should remain a first-derivative inspection helper.
- Agreed design direction so far:
  `PrepareUniformDMatrix()` can remain as a compatibility wrapper around
  `PrepareDMatrix(order=0)`.
- When extending beyond first order, generate the basis terms fresh for each build while still allowing the current build step to append 0th-, 1st-, and future 2nd-order terms in a controlled order.
- Include both pure second derivatives and mixed derivatives for multi-parameter models, e.g. for dust with `beta_d` and `T_d1` or `x^R` include:
  `∂²f/∂beta_d²`, `∂²f/∂T_d1²`, and `∂²f/(∂beta_d ∂T_d1)`.
- Use `sympy.diff(...)` to generate second derivatives, but avoid duplicating symmetric mixed terms such as `∂²/∂a∂b` and `∂²/∂b∂a`.
- Decide and document the column ordering convention for 0th-, 1st-, and 2nd-order terms before implementation so `D_matrix`, parameter bookkeeping, and downstream fitting code stay aligned.
- Specify the corresponding second-order sky-side unknowns explicitly, including the meaning of terms like `(δp_i)^2 s_b` and `(δp_i δp_j) s_b`.
- Confirm the correct Taylor-expansion coefficients, especially whether diagonal second-order terms should carry a `1/2` factor and how that factor should be absorbed between the frequency-space basis and the sky-side amplitudes.
- Update:
  diagonal second-derivative columns now carry the explicit Taylor `1/2`
  prefactor in `DMatrix`, generalized through a multiplicity-based factorial
  prefactor helper that also extends naturally to higher orders.
- Update the likelihood and fitting code only after the second-order basis definition is fixed on paper; adding second-derivative columns alone is not sufficient.

## Future methodology update: region-wise foreground parameters
- Prototype is synchrotron-only, 2 fixed regions, beta_s only (see Active).
- After prototype: extend to dust with separate region sets; dust regions
  clustered from 353 GHz dust-dominated maps, synchrotron regions from
  low-frequency maps. Do not reuse the same masks across components.
- Longer-term target: k-means clustering with around 10 regions per component.
  Grow region count in stages (2 → 4 → 8 → 10) and check condition numbers
  and parameter stability at each stage.
- Region count does not increase required frequency bands. Spatial exclusivity
  means each pixel sees only its own region's foreground parameters. Main costs
  are global parameter count, effective pixels per region, block matrix size,
  and region-level statistical noise.
- Physical motivation for separate dust/synchrotron region sets: beta_s spatial
  variation is driven by cosmic-ray and magnetic-field structure; T_d and beta_d
  variation is driven by ISM density and radiation-field structure. These
  spatial patterns are independent and should not share a common region mask.
