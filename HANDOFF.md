# HANDOFF

## Current branch
- `feature/regionwise-foreground-parameters`

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

## Region-wise implementation status

Implemented on branch `feature/regionwise-foreground-parameters` through the
first end-to-end 2-region synchrotron fit. The working tree was clean after
commit `a90ced4 Add two-region synchrotron fit example`.

Completed commits for this phase:
- `c17e641 Add synchrotron brightness region mask script`
- `4ea17eb Allow region-specific template symbols`
- `f55d4cd Track DMatrix masks per column`
- `e8d72f4 Document pre-step-4 cleanup tasks`
- `b737e64 Derive DMatrix mask counts from component terms`
- `704c76d Move region mask helpers into package`
- `c8c2315 Document dust template factory caution`
- `e48584a Apply DMatrix column masks in CalcH matrix`
- `7c39722 Document CalcH no-mask regression test`
- `36635dd Cover CalcH matrix no-mask regression`
- `84174bb Add synchrotron region parameter setup`
- `535d040 Document Step 5 review follow-ups`
- `f2634a1 Tighten region mask setup validation`
- `a90ced4 Add two-region synchrotron fit example`

What is now implemented:
- `scripts/make_synch_brightness_regions.py` creates two synchrotron regions
  from low-frequency polarization brightness using the median of
  `sqrt(Q^2 + U^2)` inside the analysis mask.
- Running the script on the local 40 GHz synchrotron template produced:
  - valid pixels: `107`
  - faint region: `53` pixels
  - bright region: `54` pixels
  - output files under gitignored `data/regions/`
- `extended_deltamap.regions` now provides reusable:
  - `expand_to_qu`
  - `validate_region_masks`
- `Templates` methods accept region-specific symbol names:
  - synchrotron: `ReturnPowerLawSynch(symbol_name=...)`
  - power-law dust: `ReturnPowerLawDust(symbol_name=...)`
  - MBB dust: `beta_symbol_name`, `temperature_symbol_name`, or
    `xref_symbol_name` as appropriate.
- `DMatrix.AddD(..., region_mask=...)` records component masks, and prepared
  matrix columns inherit the parent component mask through `column_masks`.
- `_count_component_terms()` now delegates to `_build_component_terms()` with
  `sympy.Integer(1)` to avoid duplicate term-counting logic.
- `DeltaMap.CalcH_matrix()` applies `column_masks` to:
  - `DTNID`: row and column masks
  - `DTNIDc`: row mask
  - `DTNIM`: row mask
- `examples/TestDeltamap.py` now has Step 5 helper support:
  - `region_parameter_name`
  - `expand_region_parameter_inits`
  - `add_synch_components_to_dmatrix`
  - `load_synch_region_masks`
  - `restrict_region_mask_to_observed_qu`
  - synch and dust template factory helpers to keep keyword differences local.
- Optional config keys now supported:
  - `[regions] synch_region_masks`
  - `[regions] beta_s_region_inits`
  - legacy `[par]` keys with the same names also work.
- `examples/Synch_var_3freq_regionwise_r1e-2.ini` enables the local two-region
  synchrotron prototype using the median-split masks under `data/regions/`.
- `DeltaMap` now identifies the tensor-to-scalar ratio by exact parameter name
  (`param.name == "r"`), so region names such as `beta_s_sreg0` are not
  accidentally treated as `r` parameters.

Important implementation detail:
- Saved region masks may be pixel masks, full-sky Q/U masks, or already in the
  observed Q/U layout. `load_synch_region_masks()` converts them to the
  DeltaMap internal observed vector layout:
  `[Q observed pixels..., U observed pixels...]`.
- For the local nside 4 example, full Q/U masks have length `384`, while
  DeltaMap's observed Q/U vectors have length `214`. The conversion is required
  before passing masks into `DMatrix`.

Validation completed:
- `uv run python -m unittest tests.test_smoke -v`
  - `24` tests passed.
- `uv run python -m py_compile examples/TestDeltamap.py`
- `uv run python -m py_compile extended_deltamap/deltamap.py`
- `git diff --check`
- Existing unmasked synchrotron example still completes:
  `DELTAMAP_DUST_BETA_MAP=data/pysm_2/dust_beta.fits DELTAMAP_DUST_TEMP_MAP=data/pysm_2/dust_temp.fits uv run python examples/TestDeltamap.py examples/LTD_config.ini examples/Synch_var_3freq_r1e-2.ini 1`
- Region-wise setup with the local 40 GHz median masks reaches
  `CalcH_matrix()` successfully. Observed check:
  - params: `['r', 'beta_s_sreg0', 'beta_s_sreg1']`
  - initial values:
    `{'r': 0.0, 'beta_s_sreg0': -3.1, 'beta_s_sreg1': -3.6}`
  - `DTNIDc.shape == (856, 214)`
  - `DTNIM.shape == (856, 1)`
- Actual 2-region synchrotron fit now completes with:
  `DELTAMAP_DUST_BETA_MAP=data/pysm_2/dust_beta.fits DELTAMAP_DUST_TEMP_MAP=data/pysm_2/dust_temp.fits uv run python examples/TestDeltamap.py examples/LTD_config.ini examples/Synch_var_3freq_regionwise_r1e-2.ini 1`
- Local seed-1 output:
  - `r = 0.0118065453 +/- 0.0035853502`
  - `beta_s_sreg0 = -2.9322997718 +/- 0.0287338953`
  - `beta_s_sreg1 = -2.9819950386 +/- 0.0459051613`

Remaining immediate work:
- Run multiple seeds for `examples/Synch_var_3freq_regionwise_r1e-2.ini` and
  compare recovered `r`, `beta_s_sreg0`, and `beta_s_sreg1` against the
  unregioned synchrotron baseline.
- Inspect condition numbers and mask-size sensitivity if any seed becomes
  unstable.
- After the 2-region baseline looks stable, try 4 regions before moving toward
  the longer-term 8-10 region clustering target.

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

### Pre-Step-4 cleanup (complete before starting Step 4)

Two issues found in review of Steps 1–3. Fix both before implementing Step 4.

**Issue A — `_count_component_terms` can silently diverge from `_build_component_terms`**

`dmatrix.py` has two parallel implementations that must stay in sync:
- `_build_component_terms` — builds the actual symbolic terms
- `_count_component_terms` — counts how many terms `_build_component_terms` would produce

If one is updated without the other (e.g. a new derivative order rule is added), `column_masks`
will be the wrong length and the `_set_matrix_from_template_terms` length check will catch it
at runtime rather than at test time.

Fix: replace `_count_component_terms` with a call to `_build_component_terms` and `len()`:
```python
def _count_component_terms(self, params, max_order=1, diff_param=None):
    return len(self._build_component_terms(
        sympy.Integer(1), params, max_order=max_order, diff_param=diff_param
    ))
```
Using `sympy.Integer(1)` as a dummy function avoids symbolic computation overhead while
keeping the branch logic identical. Update the existing smoke test
`test_count_component_terms_matches_second_order_dust_columns` to confirm it still passes.

**Issue B — `expand_to_qu` and `validate_region_masks` are stranded in a script**

Both functions are defined only in `scripts/make_synch_brightness_regions.py`.
Step 4 (`CalcH_matrix`) and Step 5 (param expansion) will need to call them from fitting code,
but importing from `scripts/` is not a clean dependency.

Fix:
1. Move `expand_to_qu` and `validate_region_masks` into a new module
   `extended_deltamap/regions.py`.
2. In `scripts/make_synch_brightness_regions.py`, replace the local definitions with imports:
   `from extended_deltamap.regions import expand_to_qu, validate_region_masks`
3. Export both from `extended_deltamap/__init__.py` if they will be used by callers outside
   the package (e.g. `TestDeltamap.py`).
4. Add a minimal smoke test confirming the import works and `validate_region_masks` raises on
   overlapping inputs.

Do not move `split_by_median`, `read_synch_pol_brightness`, or `read_analysis_mask` — those
are script-level I/O helpers that belong in `scripts/`.

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

**Watch out — dust template methods have inconsistent keyword argument names:**
When building the DMatrix loop for dust regions, the kwarg names differ by method:
- `ReturnMBB1` / `ReturnMBB1_Norm`: `beta_symbol_name`, `temperature_symbol_name`
- `ReturnMBB1_xRef` / `ReturnMBB1_xRef_Norm`: `beta_symbol_name`, `xref_symbol_name`
- `ReturnPowerLawDust`: `symbol_name`

Do not call these methods with positional arguments or hardcoded kwarg names scattered
across the setup loop. Instead, build a small factory or registry in the Step 5 helper
that maps component type to (method, kwarg dict), e.g.:
```python
def make_dust_template(templates, beta_name, temp_name):
    return templates.ReturnMBB1(
        beta_symbol_name=beta_name,
        temperature_symbol_name=temp_name,
    )
```
This keeps the kwarg names in one place and makes the region loop readable.

### Key design constraints to preserve
- `A = S0_CMB_I + Σ NI_k` does not change; it has no D dependence.
- Cross-region DTNID blocks are zero when noise is pixel-diagonal. Full CMB
  covariance couples regions only through A (via `S0_CMB_I`), not through DTNID.
- Dust and synchrotron must use separate region mask lists. Their spatial
  variation patterns are physically independent and should not share masks.
- Internal parameter names use `_sreg{i}` for synchrotron and `_dreg{i}` for
  dust. User-facing config uses plain arrays.

## Suggested next steps
1. Create a region-wise synchrotron fit config using the generated
   `data/regions/synch40_median_ns4_*_qu.npy` masks.
2. Run the 2-region synchrotron fit end-to-end and inspect recovery of `r`,
   `beta_s_sreg0`, and `beta_s_sreg1`.
3. After the 2-region synchrotron prototype runs cleanly, extend to dust with
   separate region masks.
4. Decide whether to commit `uv.lock`.

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
