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
- Seeds 1-10 were run for both the unregioned and 2-region synchrotron
  configs. All runs completed and wrote CSV outputs under
  `examples/test_results/`.
- Seed 1-10 unregioned summary:
  - `r`: mean `0.011031288`, sample std `0.0033153087`,
    min `0.0067588135`, max `0.01589814`
  - `beta_s`: mean `-2.9924456`, sample std `0.12544139`,
    min `-3.2529467`, max `-2.8314063`
- Seed 1-10 2-region summary:
  - `r`: mean `0.010986448`, sample std `0.0036426419`,
    min `0.0064164764`, max `0.015859678`
  - `beta_s_sreg0`: mean `-2.9780265`, sample std `0.058016306`,
    min `-3.0541201`, max `-2.8910757`
  - `beta_s_sreg1`: mean `-3.014152`, sample std `0.055480556`,
    min `-3.0802615`, max `-2.9309276`
- `scripts/make_synch_brightness_regions.py` now supports
  `--region-count`, while preserving the existing 2-region median filenames.
- `examples/Synch_var_3freq_regionwise4_r1e-2.ini` enables a 4-region
  synchrotron quantile split. Local mask generation command:
  `uv run python scripts/make_synch_brightness_regions.py examples/output_pysm3_ns4/test01_nu0040p00GHz_s1_nside0004.fits examples/files/mask_p06_Nside4.v2.fits --output-dir data/regions --prefix synch40_quantile4_ns4 --nside 4 --region-count 4`
- The local 4-region masks split 107 valid pixels into `27/26/27/27` pixels.
  In observed Q/U layout, the prepared D-matrix has shape `(3, 8)`, all 8
  columns are masked, `DTNIDc.shape == (1712, 214)`, and
  `DTNIM.shape == (1712, 1)`.
- A direct full iterative 4-region seed-1 fit was stopped after about 4 minutes
  with no CSV output yet. This looks like a runtime/optimization-cost issue
  rather than a setup failure; the same setup reaches `CalcH_matrix()`.
- Feasibility note for reducing matrix size:
  use a spatial coefficient / region expansion representation instead of
  duplicating D columns per region. For synchrotron first-order, keep the
  sky-side columns as `[f_s, df_s/dbeta_s]` and evaluate their coefficients as
  pixel vectors:
  `c_0(k,p) = sum_r m_r(p) f_s(nu_k; beta_r)` and
  `c_1(k,p) = sum_r m_r(p) df_s/dbeta_s(nu_k; beta_r)`.
  Then `CalcH_matrix()` uses
  `c_i[:, None] * N_k^{-1} * c_j[None, :]` instead of scalar `D[k,i]`
  times column masks. This preserves region-dependent scalar parameters while
  avoiding the current D-column and H-block growth with region count.
- The local design note
  `data/notes/REGIONWISE_PARAMETER_DELTAMAP_JA.md` now records this formula
  and the implementation plan. The note is under gitignored `data/`, so it is
  local documentation rather than a tracked artifact.
- Spatial coefficient path has now been implemented for synchrotron-only,
  first-order region fits:
  - `DMatrix.SetSpatialCoefficients(...)` stores symbolic template terms plus a
    coefficient builder.
  - `DeltaMap.CalcH_matrix()` uses the spatial path when present; no-region
    scalar broadcast is covered by a regression test.
  - `examples/TestDeltamap.py` selects the spatial synchrotron path when
    `synch_region_masks` are present, `isdust=False`, `uni=False`, and
    `order == 1`.
  - 2-region setup now reports `D.shape == (3, 2)`,
    `DTNIDc.shape == (428, 214)`, and `DTNIM.shape == (428, 1)`.
  - 4-region seed-1 fit now completes locally; output was
    `r = 0.0115087091 +/- 0.0035587728`, with all four
    `beta_s_sreg*` values near `-3`.
- Seeds 1-10 were run for the 4-region spatial-coefficient config. All runs
  completed and wrote CSV outputs under
  `examples/test_results/Synch_var_3freq_regionwise4_r1e-2/`.
- Seed 1-10 4-region summary:
  - `r`: mean `0.010672412`, sample std `0.0035385558`,
    min `0.0056552733`, max `0.015234329`
  - `beta_s_sreg0`: mean `-3.0380741`, sample std `0.11750109`,
    min `-3.1729155`, max `-2.8003778`
  - `beta_s_sreg1`: mean `-2.9645623`, sample std `0.11000173`,
    min `-3.2106326`, max `-2.8286982`
  - `beta_s_sreg2`: mean `-3.032954`, sample std `0.098366799`,
    min `-3.2021492`, max `-2.9157604`
  - `beta_s_sreg3`: mean `-3.0463102`, sample std `0.074396639`,
    min `-3.2098041`, max `-2.9676918`

Remaining immediate work:
- Retire the old column-mask region path from normal use. The spatial
  coefficient path is now the supported region-wise implementation. The
  column-mask code in `CalcH_matrix()` and `DMatrix` can be kept temporarily
  as dead code for reference, but should not be the active path for any
  region-wise fit. End-to-end regression against the column-mask path is not
  required; the spatial path is validated by direct coefficient checks and
  multi-seed fit results.
- Planned region-wise progression (no weak priors needed; scatter at nside=4
  is expected and the method is designed to scale with Nside):
  Step A: COMPLETE. Spatial coefficient path extended to dust (2 regions,
    nside=4). Dust-only seed-1 fit completes. Combined dust+synch at nside=4
    does not converge reliably due to MBB (beta_d, T_d1) within-component
    degeneracy and near-identical SED parameters across regions; this is
    expected at nside=4 and will be resolved by moving to nside=8.
  Step B: Move to nside=8 with synchrotron and dust each at 2 regions.
    ~428 valid pixels at nside=8 (4x nside=4); validates Nside scaling.
    First target: dust+synch combined fit converges reliably at nside=8.
  Step C: k-means ~5 regions each for synchrotron and dust at nside=8.
    Synchrotron regions from low-frequency maps; dust regions from 353 GHz.
    Final region counts and shapes kept independent between components.

Step B nside=8 setup status:
- Added `examples/LTD_config_ns8.ini` and nside=8 dust+synch 2-region configs:
  `examples/DustSynch_var_9freq_regionwise2_ns8_r1e-2.ini`,
  `examples/DustSynch_var_9freq_regionwise2_ns8_profile.ini`, and
  `examples/DustSynch_var_9freq_regionwise2_ns8_trace_ncall20.ini`.
- Generated local nside=8 PySM maps with:
  `uv run python examples/run_pysm3.py examples/LTD_config_ns8.ini`.
  The generated `examples/output_pysm3_ns8/` directory is ignored by git.
- Generated 2-region median masks:
  `synch40_median_ns8_*` from the 40 GHz synchrotron map and
  `dust337_median_ns8_*` from the 337 GHz dust proxy map. Both use the
  nside=4 analysis mask ud-graded to nside=8 and split 428 valid pixels into
  214/214 pixels.
- The nside=8 combined profile completed with
  `D_matrix.shape == (9, 5)` and active spatial coefficients, confirming that
  the region count still does not grow the D basis. However, one likelihood
  evaluation initially took ~4.56 s on seed 2. Timing showed
  `CalcH_matrix()` ~2.66 s, `CalcDelta()` ~0.63 s, and `ReturnlnDNID()` ~0.97 s.
  `ReturnlnDNID()` was optimized to use the Cholesky diagonal identity
  `log det(L L^T) = 2 sum(log(diag(L)))`, reducing that term to ~0.0001 s and
  total evaluation time to ~3.49 s on seed 3. A full Minuit run is still likely
  long; next engineering target is `CalcH_matrix()` / `CalcDelta()` caching or
  block optimization, or defining a cheaper staged convergence test before
  multi-seed full fits.

Step A implementation status:
- Dust spatial coefficients are implemented for `beta_d_dreg{i}` and
  `T_d1_dreg{i}`. Dust-only region-wise setup keeps the D basis at 3 columns
  regardless of the 2 dust regions; dust+synch keeps the D basis at 5 columns.
- Local dust masks were generated from the available 337 GHz dust-dominated
  proxy map with:
  `uv run python scripts/make_synch_brightness_regions.py examples/output_pysm3_ns4/test01_nu0337p00GHz_d1_nside0004.fits examples/files/mask_p06_Nside4.v2.fits --output-dir data/regions --prefix dust337_median_ns4 --nside 4 --region-count 2`
  The split produced 107 valid pixels, with 53 faint and 54 bright pixels.
- Added configs:
  `examples/Dust_var_7freq_regionwise2_r1e-2.ini`,
  `examples/DustSynch_var_9freq_regionwise2_r1e-2.ini`,
  plus `*_profile.ini` and `*_trace_ncall20.ini` diagnostics.
- Dust-only seed-1 completed:
  `examples/test_results/Dust_var_7freq_regionwise2_r1e-2/num0001.csv`.
  The CSV contains region parameters `beta_d_dreg0/1` and `T_d1_dreg0/1`;
  seed-1 gave `r = 0.02642685318736789`.
- The combined dust+synch full fit was stopped after running for a long time.
  It was CPU-bound, not blocked on mask I/O. Diagnostics show the matrix size
  is controlled (`D_matrix.shape == (9, 5)`), but the 7-parameter Minuit search
  is expensive. Profile timings:
  dust-only `(7, 3)` mean likelihood evaluation time ~0.077 s;
  dust+synch `(9, 5)` mean likelihood evaluation time ~0.186 s.
  With `migrad_ncall = 20`, dust-only used 36+22 evaluations per outer
  iteration, while dust+synch used 57+27 evaluations per outer iteration and
  remained invalid at that low cap. This points to optimizer/search cost and
  parameter degeneracy, not region mask construction, as the immediate issue.

Review follow-ups completed after the spatial coefficient implementation:
- The old `add_synch_components_to_dmatrix(..., synch_region_masks=...)`
  column-mask helper now raises if region masks are passed. Normal region-wise
  fits must use the spatial coefficient path; lower-level `DMatrix` column-mask
  support remains temporarily as reference code and for focused tests.
- `add_spatial_synch_components_to_dmatrix()` now uses explicit summed template
  terms for symbol collection instead of averaging region-specific templates.
- `test_fg_with_noise_cov()` and `_xref` now raise early when
  `synch_region_masks` are used with unsupported settings such as
  `isdust=True`, `uni=True`, or `order != 1`, avoiding silent fallback to the
  slow column-mask path.
- The spatial coefficient builder now uses `sympy.lambdify` for synchrotron SED
  and derivative evaluations in the Minuit hot path.
- Regression coverage was added for the unsupported-mode guard and for numeric
  spatial coefficients. A 4-region seed-11 fit completed after these changes.

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
