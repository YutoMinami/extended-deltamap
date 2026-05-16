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
- Order-aware `DMatrix` preparation and config-driven `order=...` selection are
  implemented, and current work is focused on understanding why the
  second-order dust likelihood remains unstable.
- The methodology direction is shifting away from pushing `order=2` fitting
  harder and toward region-wise foreground parameters over broad sky patches.

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
- Added explicit Taylor prefactors to repeated higher-order derivative columns
  in `extended_deltamap/dmatrix.py` using a multiplicity-based factorial helper
  so diagonal second derivatives now carry `1/2`, and the same logic extends
  naturally to future third-order terms.
- Added smoke/regression tests covering:
  package import,
  the fixed module entrypoint,
  one minimal covariance build,
  and repeated `DMatrix` rebuild behavior.
- Extended the smoke tests to cover:
  second-order helper ordering,
  order-aware `DMatrix` column counts,
  config validation for second-order dust runs,
  and the generalized Taylor prefactor logic.
- Confirmed these commands complete successfully in the local environment:
  `uv run python -m unittest discover -s tests -v`
  `uv run python examples/run_pysm3.py examples/LTD_config.ini`
  `uv run python examples/TestDeltamap.py examples/LTD_config.ini examples/Synch_var_3freq_r1e-2.ini 1`
  with
  `DELTAMAP_DUST_BETA_MAP=data/pysm_2/dust_beta.fits`
  `DELTAMAP_DUST_TEMP_MAP=data/pysm_2/dust_temp.fits`

## Current in-progress code changes
- `extended_deltamap/deltamap.py`
  Numerical-stability helpers now use jittered Cholesky factorization for some
  nearly-positive-definite blocks, but second-order dust runs can still fail in
  `CalcB()` when the optimizer reaches genuinely non-positive-definite regions.
- `examples/`
  Several local second-order dust configs were added for experiments with
  `order=2`, `anoise`, and `fgnoise_fac`.
- These current changes are not committed yet in the current worktree.

## Second-order design decisions already agreed
- Diagonal higher-order derivative columns should carry the explicit Taylor
  prefactors in frequency space, using a multiplicity-based factorial rule
  rather than ad-hoc per-order special cases.
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
- A gitignored Japanese design memo for region-wise foreground parameters lives
  at:
  [REGIONWISE_PARAMETER_DELTAMAP_JA.md](/home/yminami/workdir/DeltaMap/extended-deltamap/data/notes/REGIONWISE_PARAMETER_DELTAMAP_JA.md:1)
- Because it is under `data/`, it is not tracked by git.

## Known limitations
- `examples/TestDeltamap.py` is still large and research-script shaped.
- The order-aware public API exists, but the statistical treatment of the
  second-order sky-side terms is still unresolved.
- In local 9-band dust-only tests with `anoise=1.0`:
  `order=1` recovers `r ≈ 9.8e-3`,
  while `order=2` either collapses toward `r ≈ 0` or fails when the optimizer
  reaches a non-positive-definite `B` block.
- The emerging alternative is to keep the Delta-map expansion first-order and
  allow foreground parameters to vary by broad sky region. Dust and synchrotron
  should be allowed to use separate region sets.
- The SciPy `sph_harm` compatibility shim works in the current environment, but
  supported-version policy still needs to be finalized.
- `uv.lock` exists locally and is used, but still needs a normal review and
  commit decision.

## Region-wise parameter implementation plan

Design settled in conversation on 2026-05-16. Implement in the order below.
Start with synchrotron-only, 2 fixed regions, beta_s only, first-order fit.

### Step 1 — Region mask creation
- Create boolean pixel masks of shape `(n_pix,)` for each region.
- Expand to full Q/U size: `numpy.concatenate([mask_pix, mask_pix])` → shape `(size,)`.
- Validate: masks are disjoint (`mask0 & mask1 == 0`) and together cover the
  observed pixels.
- For the first prototype, use a simple fixed split (e.g. north/south by
  HEALPix theta). Clustering comes later.
- Store as `synch_region_masks = [mask_sreg0, mask_sreg1]`.
- Files: new utility function in `TestDeltamap.py` or a small helper module.

### Step 2 — Templates: per-region symbol names
- Add `symbol_name="beta_s"` kwarg to `ReturnPowerLawSynch()` and equivalent
  dust methods so callers can pass `"beta_s_sreg0"`, `"beta_s_sreg1"`, etc.
- Call `AddD()` in a loop over regions, passing the region-specific symbol.
- Files: `extended_deltamap/templates.py` (1-line change per method), setup
  code in `TestDeltamap.py`.

### Step 3 — DMatrix: column-to-region metadata
- Add `self.column_masks = []` to `DMatrix.__init__`.
- Add optional `region_mask=None` argument to `AddD()`; store one entry per
  component.
- In `_set_matrix_from_template_terms()`, expand `column_masks` so every
  derivative column inherits the same mask as its parent component term.
- With `region_mask=None` the behaviour is identical to today (no-mask path).
- Files: `extended_deltamap/dmatrix.py`.

### Step 4 — DeltaMap: mask-aware CalcH_matrix
- `CalcH_matrix()` reads `self.dmtrx.column_masks` for each column pair `(i, j)`.
- Replace `self.NI_list[k]` with a masked version:
  ```
  NI_k = self.NI_list[k]
  if mask_i is not None: NI_k = NI_k * mask_i[:, None]
  if mask_j is not None: NI_k = NI_k * mask_j[None, :]
  element += D[k, i] * NI_k * D[k, j]
  ```
- Apply same masking in the `DTNIDc` and `DTNIM` loops (row-mask only, `mask_i`).
- Matrix shapes are unchanged: DTNID stays `(n_cols*size) × (n_cols*size)`;
  A, Delta, B stay `size × size`.
- Validate: with all masks None the likelihood and parameter recovery must match
  the pre-change baseline.
- Files: `extended_deltamap/deltamap.py` (`CalcH_matrix` only).

### Step 5 — Config/inits parameter expansion
- Add a helper that converts `{"beta_s": ([v0, v1], (lo, hi))}` into
  `{"beta_s_sreg0": (v0, (lo, hi)), "beta_s_sreg1": (v1, (lo, hi))}`.
- Validate that the expanded parameter count matches `len(synch_region_masks)`.
- Files: `examples/TestDeltamap.py`.

### Key design constraints to preserve
- `A = S0_CMB_I + Σ NI_k` does not change; it has no D dependence.
- Cross-region DTNID blocks are zero when noise is pixel-diagonal. Full CMB
  covariance couples regions only through A (via `S0_CMB_I`), not through DTNID.
- Dust and synchrotron must use separate region mask lists. Their spatial
  variation patterns are physically independent and should not share masks.
- Internal parameter names use `_sreg{i}` for synchrotron and `_dreg{i}` for
  dust. User-facing config uses plain arrays.

## Suggested next steps
1. Implement the 5-step plan above, validating each step before the next.
2. After the 2-region synchrotron prototype runs cleanly, extend to dust with
   separate region masks.
3. Decide whether to commit `uv.lock`.

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
