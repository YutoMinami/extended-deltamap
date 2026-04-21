# HANDOFF

## Current branch
- `feature/fix-deltamap-main-entrypoint`

## Current state
- The package layout is in place under `extended_deltamap/` and the main
  example workflows run from the repo root with `uv`.
- The README now documents the reference paper and the map-to-likelihood
  pipeline at a higher level.
- Basic smoke/regression checks exist under `tests/test_smoke.py`.
- Local end-to-end checks were completed with the bundled local dust FITS files
  under `data/pysm_2/`.
- There is active in-progress work in `extended_deltamap/dmatrix.py` to turn
  `PrepareDMatrix(order=...)` into the public order-aware API for future
  second-order expansion support.

## What has been completed on this branch
- Fixed the `extended_deltamap.deltamap` module entrypoint by adding the missing
  `sys` import.
- Fixed `examples/TestDeltamap.py` so cached `anoise_freq` arrays are actually
  reused instead of being regenerated after load.
- Clarified the paper's artificial-noise split in code comments:
  common artificial noise for the CMB-side shared component and
  per-frequency artificial noise for the instrument-side channel component.
- Made `extended_deltamap/dmatrix.py` rebuild its template columns from a fresh
  list so repeated matrix preparation no longer accumulates duplicate columns.
- Added smoke/regression tests covering:
  package import,
  the fixed module entrypoint,
  one minimal covariance build,
  and repeated `DMatrix` rebuild behavior.
- Confirmed these commands complete successfully in the local environment:
  `uv run python -m unittest discover -s tests -v`
  `uv run python examples/run_pysm3.py examples/LTD_config.ini`
  `uv run python examples/TestDeltamap.py examples/LTD_config.ini examples/Synch_var_3freq_r1e-2.ini 1`
  with
  `DELTAMAP_DUST_BETA_MAP=data/pysm_2/dust_beta.fits`
  `DELTAMAP_DUST_TEMP_MAP=data/pysm_2/dust_temp.fits`

## Current in-progress code changes
- `extended_deltamap/dmatrix.py`
  Work is underway to:
  expose `PrepareDMatrix(order=...)`,
  make `PrepareUniformDMatrix()` a compatibility wrapper,
  and redefine `dim_params` so it matches the actual number of generated
  columns at the chosen expansion order.
- This work is not committed yet in the current worktree.

## Second-order design decisions already agreed
- Taylor-series coefficients such as `1/2` should be absorbed on the sky-side
  unknowns rather than into the frequency-space derivative columns.
- Mixed derivatives should appear only once per unordered parameter pair.
- Derivative multi-indices should follow parameter-name order.
- Component block order should stay aligned with the current implementation:
  `dust` block first, then `synchrotron` block.
- `PrepareOneDiffDMatrix()` should remain a first-derivative inspection helper.
- `PrepareUniformDMatrix()` can remain for compatibility, but should conceptually
  map to `PrepareDMatrix(order=0)`.

## Local design notes
- A gitignored Japanese design memo for the second-order work lives at:
  [SECOND_ORDER_DELTAMAP_JA.md](/home/yminami/workdir/DeltaMap/extended-deltamap/data/notes/SECOND_ORDER_DELTAMAP_JA.md:1)
- Because it is under `data/`, it is not tracked by git.

## Known limitations
- `examples/TestDeltamap.py` is still large and research-script shaped.
- There is not yet a committed public API for second-order Delta-map expansion;
  only the groundwork and design notes exist so far.
- The SciPy `sph_harm` compatibility shim works in the current environment, but
  supported-version policy still needs to be finalized.
- `uv.lock` exists locally and is used, but still needs a normal review and
  commit decision.

## Suggested next steps
1. Finish the in-progress `DMatrix` API cleanup and add tests for
   `PrepareDMatrix(order=0|1|2)` and `dim_params`.
2. Compare generated second-order dust/synch columns against the local design
   memo before touching `DeltaMap` likelihood code.
3. Decide whether to commit `uv.lock`.
4. Continue doc cleanup once the second-order API direction settles.

## Useful commands
- Run smoke/regression tests:
  `uv run python -m unittest discover -s tests -v`
- Run the PySM map-generation example:
  `uv run python examples/run_pysm3.py examples/LTD_config.ini`
- Run the fit example with local dust FITS files:
  `DELTAMAP_DUST_BETA_MAP=data/pysm_2/dust_beta.fits DELTAMAP_DUST_TEMP_MAP=data/pysm_2/dust_temp.fits uv run python examples/TestDeltamap.py examples/LTD_config.ini examples/Synch_var_3freq_r1e-2.ini 1`

## Files most relevant to continue from here
- [README.md](/home/yminami/workdir/DeltaMap/extended-deltamap/README.md:1)
- [TODO.md](/home/yminami/workdir/DeltaMap/extended-deltamap/TODO.md:1)
- [examples/TestDeltamap.py](/home/yminami/workdir/DeltaMap/extended-deltamap/examples/TestDeltamap.py:1)
- [extended_deltamap/dmatrix.py](/home/yminami/workdir/DeltaMap/extended-deltamap/extended_deltamap/dmatrix.py:1)
- [tests/test_smoke.py](/home/yminami/workdir/DeltaMap/extended-deltamap/tests/test_smoke.py:1)
