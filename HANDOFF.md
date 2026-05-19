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
  Step C: combined-feature k-means at nside=8 (design settled 2026-05-18).
    Single clustering on a 6-dimensional feature vector per pixel:
      [log(P_40), log(P_60/P_40), log(P_78/P_60),
       log(P_337), log(P_337/P_235), log(P_402/P_280)]
    Synchrotron features use ≤78 GHz only; 100 GHz and above are CMB
    channels (LTD full band: 40–402 GHz, CMB channels ~100–195 GHz).
    where P_nu = sqrt(Q_nu^2 + U_nu^2). Features must be standardised
    (zero mean, unit variance) before k-means so amplitude and SED shape
    contribute equally; without standardisation amplitude dominates and
    the clustering degenerates to a brightness split.
    Target k=8 (~54 px/region); bracket with k=4 (~107 px/region) and
    k=16 (~27 px/region) to assess stability.
    Dust and synchrotron share the same region masks from this clustering.
    Physical motivation: galactic structure drives both components; amplitude
    correlation is real (galactic centre bright in both, high latitude faint).
    Full design in data/notes/REGIONWISE_PARAMETER_DELTAMAP_JA.md.
    The pure feature-space k-means produced fragmented masks
    (connected component counts 5/5/10/15 for k=4 regions), but spatial
    contiguity is not required and not the goal. The purpose of region-wise
    parameterisation is to reduce model mismatch: as Nside increases, the
    actual foreground SED varies spatially, and unmodelled within-region
    SED variation leaks into the CMB estimate and biases r. The clustering
    criterion is therefore within-region SED parameter homogeneity, not
    spatial contiguity. Pixels that share similar SED features but are
    spatially distant should be in the same region. Feature-space k-means
    is correct for this purpose; spatially-constrained clustering would
    prevent grouping SED-similar but spatially-separated pixels and would
    increase within-region SED variation, working against the goal.

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
  long; next engineering target is `CalcH_matrix()` through the einsum
  refactor described below.

### CalcH_matrix einsum optimization (next Codex task)

Target: `extended_deltamap/deltamap.py`, method
`_calc_h_matrix_from_spatial_coefficients`. The current triple Python loop
(n_columns × n_columns × n_freqs = 225 iterations for nside=8 with 5 columns
and 9 frequencies) is the dominant cost at ~2.57 s per evaluation. NI is
block-diagonal across frequencies (no cross-frequency coupling), but each
per-frequency NI_k is a dense `(size × size)` matrix — so we cannot treat
pixels as independent scalars.

**Step 1 — Precompute `_NI_stack` once before the Minuit loop.**

In `DeltaMap.__init__`, add:
```python
self._NI_stack = None
```
At the END of `CalcNInvArray()`, after the loop that fills `self.NI_list`:
```python
self._NI_stack = numpy.stack(self.NI_list)
```
Shape: `(n_freqs, size, size)`, where `size` is the internal Q/U-expanded
vector length, not the number of valid sky pixels. For nside=8 there are
428 valid sky pixels, so `size = 2 * 428 = 856`; `_NI_stack` has shape
`(9, 856, 856)` and uses ≈ 53 MB in float64. This stack is computed once per
fit; `NI_list` does not change during Minuit.

**Step 2 — Replace DTNID triple loop with an einsum loop over i.**

Replace the current:
```python
matrix_list = []
for i in range(n_columns):
    i_list = []
    for j in range(n_columns):
        element = numpy.zeros_like(self.NI_list[0])
        for k, ni_k in enumerate(self.NI_list):
            element += (
                coefficients[k, i][:, None]
                * ni_k
                * coefficients[k, j][None, :]
            )
        i_list.append(element)
    matrix_list.append(i_list)
DTNID = numpy.block(matrix_list)
```
with:
```python
NI_stack = self._NI_stack  # (n_freqs, size, size)
n_columns = coefficients.shape[1]
size = coefficients.shape[2]

DTNID_blocks = numpy.empty((n_columns, n_columns, size, size))
for i in range(n_columns):
    # T_i[k,p,q] = c[k,i,p] * NI_k[p,q]: shape (n_freqs, size, size)
    T_i = coefficients[:, i, :, None] * NI_stack
    # sum over k: DTNID_blocks[i,j,p,q] = sum_k T_i[k,p,q] * c[k,j,q]
    DTNID_blocks[i] = numpy.einsum('kpq,kjq->jpq', T_i, coefficients)

# Block-matrix reshape: (i,j,p,q) -> (i*size, j*size)
# transpose(0,2,1,3) gives (i,p,j,q); reshape gives row i*size+p, col j*size+q
DTNID = DTNID_blocks.transpose(0, 2, 1, 3).reshape(n_columns * size, n_columns * size)
```
This replaces 225 Python iterations with 5 iterations, each calling one
vectorized einsum. Peak intermediate memory per iteration is ~53 MB at
nside=8, the same shape as `_NI_stack`. `DTNID_blocks` itself has shape
`(5, 5, 856, 856)` for the 9-band, 5-column nside=8 profile, about 146 MB;
the final `DTNID` has shape `(4280, 4280)`, also about 146 MB. This is still
acceptable at nside=8 but should be revisited before increasing nside.

**Step 3 — Replace DTNIDc loop with a single einsum.**

Replace the current:
```python
matrix_list = []
for i in range(n_columns):
    i_list = []
    element = numpy.zeros_like(self.NI_list[0])
    for k, ni_k in enumerate(self.NI_list):
        element += coefficients[k, i][:, None] * ni_k
    i_list.append(element)
    matrix_list.append(i_list)
self.DTNIDc = numpy.block(matrix_list)
```
with:
```python
# DTNIDc_blocks[i,p,q] = sum_k c[k,i,p] * NI_k[p,q]
# p appears in BOTH tensors and in the output: it is NOT summed, only k is.
DTNIDc_blocks = numpy.einsum('kip,kpq->ipq', coefficients, NI_stack)
self.DTNIDc = DTNIDc_blocks.reshape(n_columns * size, size)
```
`reshape(n_columns * size, size)` stacks the n_columns blocks of `(size, size)`
vertically, which is identical to the `numpy.block([[e_0], ..., [e_{n-1}]])`.

**Step 4 — Replace DTNIM loop with two einsums.**

Replace the current:
```python
matrix_list = []
for i in range(n_columns):
    i_list = []
    element = numpy.zeros_like(self.meanvec[0])
    for k, ni_k in enumerate(self.NI_list):
        element += coefficients[k, i][:, None] * (ni_k @ self.meanvec[k])
    i_list.append(element)
    matrix_list.append(i_list)
self.DTNIM = numpy.block(matrix_list)
```
with:
```python
# meanvec[k] has shape (size, 1); ravel to (size,) for einsum
meanvec_stack = numpy.stack([m.ravel() for m in self.meanvec])  # (n_freqs, size)
# NIM[k,p] = sum_q NI_k[p,q] * meanvec[k,q]
NIM_stack = numpy.einsum('kpq,kq->kp', NI_stack, meanvec_stack)
# DTNIM[i,p] = sum_k c[k,i,p] * NIM[k,p]
DTNIM_flat = numpy.einsum('kip,kp->ip', coefficients, NIM_stack)
self.DTNIM = DTNIM_flat.reshape(n_columns * size, 1)
```

**Correctness notes:**

- The `transpose(0,2,1,3).reshape` for DTNID produces the same block layout as
  `numpy.block(matrix_list)`: element `[i*size+p, j*size+q]` equals
  `DTNID_blocks[i,j,p,q]`. Verify by checking `numpy.allclose(DTNID, DTNID.T)`.
- `DTNIDc.reshape(n_columns * size, size)` is a vertical stack of n_columns
  blocks of shape `(size, size)`. This matches `numpy.block([[e_0], ..., [e_{n-1}]])`.
- `DTNIM.reshape(n_columns * size, 1)` matches the current `numpy.block` output.
- `_NI_stack` must already be populated before
  `_calc_h_matrix_from_spatial_coefficients` is called. Confirm that
  `CalcNInvArray()` is always called before `CalcH_matrix()` in the fit flow.

**Test plan after the change:**

1. `uv run python -m unittest tests.test_smoke -v` — all 31 tests pass after
   the einsum rewrite.
2. Re-run the nside=8 profile config to confirm `CalcH_matrix()` timing. The
   einsum rewrite reduced block-construction overhead (`build_DTNID_blocks`
   is ~0.17 s), but total `CalcH_matrix()` remains ~2.45 s because the
   dominant pieces are dense linear algebra:
   `cholesky_DTNID` ~0.73 s, `build_DT_SpNI_D` ~0.71 s, and
   `cholesky_DT_SpNI_D` ~0.67 s. Disabling finite checks on the largest
   Cholesky/solve calls brought the full nside=8 profile to ~3.37 s/eval.
   This means the Python-loop overhead was not the main bottleneck; dense
   Cholesky and `DTNIDc A^{-1} DTNIDc^T` now dominate.
3. Spot-check the seed-1 fit result for `DustSynch_var_9freq_regionwise2_ns8_r1e-2`
   stays numerically close to prior runs (r ≈ 0.026 for dust-only at nside=4
   was the reference; nside=8 results are expected to differ, but the optimizer
   should converge rather than diverge).

**CalcDelta note (secondary target):**

`CalcDelta()` at ~0.63 s is:
```python
self.Delta = self.DTNIDc.T @ scipy.linalg.cho_solve((self.DNIDL, True), self.DTNIDc)
```
`DTNIDc` is `(n_cols*size, size)` = `(4280, 856)` and DNIDL is `(4280, 4280)`
for nside=8 dust+synch with 5 D columns.
The `cho_solve` is already dispatched to optimized LAPACK. There is no easy
structural optimization here without rethinking the algorithm. Do not touch
`CalcDelta` in this pass; revisit only if CalcH_matrix improvement is confirmed
and total time is still too slow for multi-seed fits.

### Next: remove dead `DT_SpNI_D` computation (~1.38 s savings)

Code inspection shows that `DT_SpNI_DL` is set in both CalcH_matrix paths but
is only referenced in `ReturnlnDT_SpNI_D()`, which is called **only from the
commented-out block** at lines 1422–1431. The active likelihood path
(line 1433: `denomi = lnS0 + lnB + lnDNID`) does not use it.

**Remove from `_calc_h_matrix_from_spatial_coefficients` (around lines 657–667):**
```python
# DELETE THESE LINES
DT_SpNI_D = DTNID - self.DTNIDc @ scipy.linalg.cho_solve(
    (self.AL, True), self.DTNIDc.T, check_finite=False
)
self.DT_SpNI_DL = self.stable_cholesky(DT_SpNI_D, lower=True)
# (and the record_timing calls around them)
```

**Also remove the same pattern from the non-spatial `CalcH_matrix()` path
(around lines 558–565):**
```python
# DELETE THESE LINES
DT_SpNI_D = DTNID - self.DTNIDc @ scipy.linalg.cho_solve(
    (self.AL, True), self.DTNIDc.T
)
self.DT_SpNI_DL = self.stable_cholesky(DT_SpNI_D, lower=True)
```

Leave `self.DT_SpNI_DL = None` in `__init__` so `ReturnlnDT_SpNI_D()` does
not NameError if the commented-out path is ever re-enabled.

Expected result after removal:
```
build_DT_SpNI_D:    0.71 s → gone
cholesky_DT_SpNI_D: 0.67 s → gone
CalcH_matrix total: ~2.45 s → ~1.16 s
per-eval total:     ~3.37 s → ~2.12 s
```

Remaining hard costs after this fix:
- `cholesky_DTNID`: ~0.73 s  (factorize 4280×4280)
- `CalcDelta`: ~0.61 s  (cho_solve for 856 RHS)
Both are LAPACK-dispatched and need algorithmic change to go further.

Validation:
- `uv run python -m py_compile extended_deltamap/deltamap.py examples/TestDeltamap.py`
- `uv run python -m unittest tests.test_smoke -v`
- `git diff --check`
- nside=8 dust+synch profile seed 7:
  `CalcH_matrix()` ~1.16 s, total likelihood evaluation ~2.12 s.

### Next direction: staged nside=8 fits

After removing the dead `DT_SpNI_D` code the per-eval should be ~2.0 s.
The remaining dominant cost is `cholesky_DTNID` + `CalcDelta` (~1.34 s),
which are hard to reduce without algorithmic changes. Further per-evaluation
optimization is unlikely to make full combined multi-seed fits cheap enough
by itself.

Recommended next step is to reduce optimizer evaluation count:
1. Add nside=8 dust-only 2-region configs, analogous to the nside=4
   `Dust_var_7freq_regionwise2_*` configs.
2. Add nside=8 synch-only 2-region configs, analogous to
   `Synch_var_3freq_regionwise_r1e-2.ini`.
3. Run short `migrad_ncall` trace/profile jobs for dust-only and synch-only to
   check whether each component converges at nside=8.
4. Use the recovered `beta_d_dreg*`, `T_d1_dreg*`, and `beta_s_sreg*` values as
   stronger initial values for a single nside=8 combined dust+synch seed.
5. Only after that single combined fit behaves reasonably, consider multi-seed
   nside=8 runs.

Initial estimation traces:
- Added nside=8 component-only configs:
  `examples/Dust_var_7freq_regionwise2_ns8_r1e-2.ini`,
  `examples/Dust_var_7freq_regionwise2_ns8_trace_ncall20.ini`,
  `examples/Synch_var_3freq_regionwise2_ns8_r1e-2.ini`, and
  `examples/Synch_var_3freq_regionwise2_ns8_trace_ncall20.ini`.
- Seed-8 `migrad_ncall=20` traces complete but are not Minuit-valid at that
  low cap. This is expected for a diagnostic cap, but useful for identifying
  which parameters move.
- Synch-only seed 8 looks reasonable even at low cap:
  `r = 0.0135263 +/- 0.0041186`,
  `beta_s_sreg0 = -2.9348 +/- 0.0460`,
  `beta_s_sreg1 = -3.1363 +/- 0.0327`.
- Dust-only seed 8 is finite but dust temperature remains at the initial value:
  `r = 0.0172187 +/- 0.017829`,
  `beta_d_dreg0 = 1.6091`, `beta_d_dreg1 = 1.5`,
  `T_d1_dreg0 = 20.0`, `T_d1_dreg1 = 20.0`.
- Combined seed 8 remains mostly stuck on dust initial values under the same
  cap, while one synch region moves:
  `r = 0.0284168`, `beta_s_sreg1 = -3.1816`, dust params mostly at
  `beta_d=1.5`, `T_d1=20.0`.
- Interpretation: no obvious numerical/algebraic failure (finite likelihoods,
  finite errors, no Cholesky failure), but dust MBB parameters need better
  initialization or longer/staged optimization. The synch-only result is a good
  candidate source for combined-fit `beta_s_region_inits`.

Working hypothesis about the dust split:
- The current dust regions are a median split by high-frequency brightness
  (337 GHz polarization amplitude proxy). This is useful for implementation
  validation but may be a poor split for estimating region-wise dust SED
  parameters. The faint region is faint precisely because the dust signal is
  weak, so `beta_d_dreg*` / `T_d1_dreg*` in that region may be weakly
  constrained and can remain near initial values under short Minuit caps.
- This is distinct from the generic MBB `beta_d`/`T_d1` degeneracy. A parameter
  can have a large error and still converge; the concern here is that the faint
  brightness region gives the optimizer almost no local likelihood curvature for
  its region-specific dust parameters.
- Before moving to k-means, a useful diagnostic is to tabulate each current dust
  region's polarization amplitude, fitted parameter movement, and errors. This
  would test whether the faint region is indeed information-poor.
- For k-means, prefer SED-like color features rather than brightness-only
  features. Example dust feature vectors:
  `log(P_337/P_235)`, `log(P_402/P_280)` where
  `P_nu(p) = sqrt(Q_nu(p)^2 + U_nu(p)^2)`. Avoid letting `log(P_337)` dominate,
  or clustering will reproduce bright/faint regions. A weak amplitude feature
  may still be useful only after standardization/clipping.
- Synchrotron can use analogous low-frequency color features, e.g.
  `log(P_60/P_40)` and `log(P_140/P_60)`.

### IterateMinimize convergence improvements

Background: r and the foreground SED parameters (beta_d, T_d1, beta_s) enter
the likelihood through structurally different mechanisms:
- **r** adjusts the CMB covariance matrix (B = Delta − A, where A ∝ r via S0_CMB).
- **fg params** adjust the foreground template D and hence DTNID, DTNIDc, DTNIM.

Jointly optimizing them with a single Minuit call is known to work poorly because
their gradient structures are incompatible. `IterateMinimize` separates them via
coordinate descent (fg-only → r-only → repeat), but the current convergence
conditions and initialization have several fixable problems.

**Note on r cold start (do not change)**

`ReturnMinimize` always restarts the r-only step from `self.inits[param.name][0]`
rather than the current best. This was intentional: restarting from the initial
value avoids getting trapped in a local minimum found in a previous outer
iteration. Since r optimisation is 1D the restart cost is low. Do not change
this to a warm start.

**Priority 1 — fg sensitivity check for outer-loop termination**

The current criterion `abs(pre_r - tmp_r) > 1e-5` uses a hardcoded absolute
threshold that has no physical scale. The correct question is: "does the new r
change the likelihood enough to warrant re-optimising fg parameters?"

After the r-only step returns `r_new`, evaluate the likelihood at
`(fg_old, r_new)` with one CalcInOneLoop call. If the change from
`(fg_old, r_old)` is small, skip the next fg-optimisation step:

```python
lh_new_r_old_fg = self._eval_lh_at_current_params()   # one CalcInOneLoop call
delta_lh_from_r = abs(lh_new_r_old_fg - self.lh)
if delta_lh_from_r < fg_retrigger_threshold:
    break   # fg does not need re-optimising; outer loop is converged
```

`fg_retrigger_threshold` should scale with pixel count; `1.0 / self.size` is a
reasonable default (change of less than ~1 unit of likelihood per pixel). This
also handles the r=0 boundary naturally: when r is stuck at 0, `delta_lh ≈ 0`
and the loop terminates without re-running fg.

**Priority 2 — likelihood change criterion normalised by pixel count**

The existing `(pre_lh - self.lh) > 1e-2` threshold does not scale with nside.
At nside=8 the likelihood value is ~4× larger than at nside=4. Replace with:

```python
lh_tol = max(1e-2 / self.size, 1e-6)
```

**Priority 3 — replace Minuit r-only step with bounded scalar minimiser**

The r-only optimisation is a 1D problem over `r ∈ [0, r_max]`.
`scipy.optimize.minimize_scalar(method='bounded')` handles the boundary
natively, needs no gradient, and converges in ~20 evaluations for a unimodal
function. Replace the `self.m = Minuit(self.MinimizeOnlyR, ...)` block with:

```python
from scipy.optimize import minimize_scalar
result = minimize_scalar(
    self.MinimizeOnlyR_scalar,   # wrapper that takes a scalar r, not a list
    bounds=(r_lo, r_hi),
    method='bounded',
    options={'xatol': lh_tol},
)
self.param_values['r'] = result.x
```

This removes the HESSE-at-boundary underestimation problem for r's error
estimate (compute r's error separately via likelihood profiling if needed).

**Priority 4 — ln(r) reparameterisation (larger change, future work)**

Theoretical predictions for r span orders of magnitude (Starobinsky-like
r~0.004 vs. large-field r~0.1), so the natural scale for r is logarithmic.
Reparameterise as `q = ln(r)`, q unconstrained, so that Minuit's step sizing
is in log space. This removes the need for a lower-bound clamp and makes the
gradient structure more uniform across r's physically plausible range.

Requires: choosing r_min > 0 for the lower bound (e.g. r_min = 1e-4 or the
lensing B-mode equivalent), updating all `inits['r']` handling, and verifying
that recovered `r` values and errors are consistent with the sqrt(r) or
MINOS-based reference at nside=4.

Do not implement this until Priorities 1–3 are confirmed and timed.

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
### Latest optimizer status
- The apparent r-only likelihood worsening was a diagnostic mistake: the
  foreground-only Minuit step reports `ReturnChiSquare()`, while the r-only
  step reports `ReturnLikelihood()`. Those fvals must not be subtracted.
- `IterateMinimize()` now evaluates `ReturnLikelihood()` once immediately
  after the foreground step and uses that as `L_fg` for the r-step improvement
  test, `Delta L_r = L_fg - L_r`.
- The old hard `abs(delta r) > 1e-5` outer-loop criterion has been replaced by
  the r-step full-likelihood improvement threshold. This avoids forcing an
  extra foreground step solely because r was cold-started.
- r-only cold start is still intentional and should be preserved.
- r-only optimizer default is Minuit again, so `r_error` remains available.
  `scipy.optimize.minimize_scalar(method="bounded")` is available only as a
  diagnostic path with `diagnostics.r_minimizer = scalar`; it gives nearly the
  same likelihood on capped nside=8 traces but no Minuit error.
- Capped nside=8 seed-16 comparison:
  dust-only Minuit `r=0.021421`, `r_error=0.012243`;
  dust-only scalar `r=0.024956`, `r_error=nan`.
  dust+synch Minuit `r=0.019299`, `r_error=0.016423`;
  dust+synch scalar `r=0.019043`, `r_error=nan`.

### Fix: r_step_lh_tol does not scale at nside≤8

Current code (`deltamap.py`, `IterateMinimize`):
```python
r_step_lh_tol = max(1.0, 1.0e-3 * self.size)
```
The `max(1.0, ...)` floor dominates for all current nside values
(nside=4: `1e-3×214=0.214`, nside=8: `1e-3×856=0.856`, both < 1.0),
so the threshold is effectively constant at 1.0 rather than scaling with
pixel count. Since `delta_lh_r_step` is a likelihood improvement that
itself scales with `size`, the threshold should also scale.

**Fix:** remove the floor:
```python
r_step_lh_tol = 1.0e-3 * self.size
```
This gives nside=4 → 0.214, nside=8 → 0.856, nside=16 → 3.424,
and scales naturally to larger nside.

Implemented in `extended_deltamap/deltamap.py`.

Validation: re-run nside=8 seed-16 traces and confirm outer-loop
iteration count is unchanged or more sensible than with the 1.0 floor.

### Step C clustering prototype status
- Added `scripts/make_combined_feature_regions.py`.
- The script builds the 6-D combined-feature vector from nside=8 PySM maps:
  `log(P40_s)`, `log(P60_s/P40_s)`, `log(P78_s/P60_s)`,
  `log(P337_d)`, `log(P337_d/P235_d)`, `log(P402_d/P280_d)`.
- Features are standardized before k-means. This is important to avoid
  degenerating back to a brightness split.
- Generated masks with:
  `uv run python scripts/make_combined_feature_regions.py --k 4 8 16 --seed 0`
- Outputs are under gitignored `data/regions/`:
  `combined_feature_kmeans_ns8_k{04,08,16}_regXX_{pix,qu}.npy` and
  `combined_feature_kmeans_ns8_summary.csv`.
- Counts/diagnostics:
  k=4 counts `79/134/83/132`, inertia `1127.43`, mean SED std `0.617`;
  k=8 counts `33/57/52/48/56/86/26/70`, inertia `748.55`,
  mean SED std `0.472`;
  k=16 counts `14/15/16/25/26/25/17/29/31/43/20/20/29/47/21/50`,
  inertia `509.39`, mean SED std `0.389`.
- Because k=8 already has a 26-pixel region, start fitting with k=4.
- Added trace configs:
  `examples/Synch_var_3freq_regionwise_combinedk4_ns8_trace_ncall20.ini`
  and `examples/DustSynch_var_9freq_regionwise_combinedk4_ns8_trace_ncall20.ini`.
- Synch-only k=4 seed-18 trace completed under `migrad_ncall=20`.
  It is not Minuit-valid at the cap, but it validates mask loading and
  4-region parameter expansion. CSV:
  `examples/test_results/Synch_var_3freq_regionwise_combinedk4_ns8_trace_ncall20/num0018.csv`.
  Result: `r=0.009104 +/- 0.003554`; only `beta_s_sreg3` moved strongly
  (`-2.8511 +/- 0.0373`), while other regions stayed near `-3.0` with large
  errors.
- Next immediate run: k=4 combined dust+synch short trace, then inspect whether
  the larger parameter count is still tractable and which parameters move.
- k=4 combined dust+synch seed-18 short trace completed but is slow:
  foreground Minuit used `nfcn=214` and `218` in the two outer iterations.
  Result:
  `r=0.0145065 +/- 0.0168837`;
  `beta_s_sreg1=-3.1042`, `beta_s_sreg3=-2.8685`;
  all `beta_d` stayed at `1.5` and all `T_d1` stayed at `20.0`.
- Added optional final foreground Minuit matrix output:
  set `[diagnostics] save_minuit_matrices = True`.
  The final foreground-only covariance and correlation are written next to the
  result CSV as `*_fg_cov.csv` and `*_fg_corr.csv`.
  `DeltaMap.last_fg_minuit` and `last_fg_parameter_names` hold the source.
- k=4 combined seed-19 was run with matrix saving enabled. It produced
  `num0019_fg_cov.csv` and `num0019_fg_corr.csv`. The first version used
  Minuit's generic `x0...` names; this was fixed so subsequent runs use
  physical names such as `beta_s_sreg0`, `beta_d_dreg0`, `T_d1_dreg0`.
  The seed-19 correlation matrix shows very strong foreground degeneracies
  (many |corr| values around 0.8-0.99), consistent with dust parameters
  staying at initial values under the short cap.
- Added `scripts/plot_fg_correlation_heatmap.py` to visualize saved foreground
  correlation matrices. For the legacy seed-19 file, run:
  `uv run python scripts/plot_fg_correlation_heatmap.py examples/test_results/DustSynch_var_9freq_regionwise_combinedk4_ns8_trace_ncall20/num0019_fg_corr.csv --legacy-k4-combined-names`.
  Outputs:
  `data/plots/num0019_fg_corr_heatmap.png` and
  `data/plots/num0019_fg_corr_heatmap_strong_pairs.csv`.
  The strongest correlations include cross-component pairs, e.g.
  `corr(beta_s_sreg2, beta_d_dreg1)=0.9967`,
  `corr(beta_s_sreg2, T_d1_dreg0)=0.9901`, and
  `corr(beta_d_dreg1, T_d1_dreg0)=0.9900`.
  This suggests the combined k=4 short-cap difficulty is not only the usual
  dust `beta_d`/`T_d` degeneracy, but also strong synch-vs-dust foreground
  mixing.
- Important interpretation: the strongest synch-vs-dust correlations are not
  limited to matching region indices. If the dominant pattern had been
  `beta_s_sreg{i}` strongly correlated only with `beta_d_dreg{i}` or
  `T_d1_dreg{i}`, that would point directly to the shared dust/synch region
  mask as the problem and motivate separate component-specific region sets.
  Instead, seed-19 already shows very strong cross-component correlations
  across different region indices, e.g. synch region 2 with dust region 1 or
  dust-temperature region 0. So do not assume that simply separating dust and
  synch sky partitions will solve the degeneracy. First quantify same-region
  vs different-region cross-component correlations; if same-region correlations
  dominate, split the dust/synch region sets, otherwise focus on global
  synch/dust/r degeneracy controls such as staged fits, frequency leverage, or
  weak priors.

### Next Codex task: full-fit correlation matrix for k=4 combined

The ncall=20 short-cap correlation matrix showed strong cross-region
correlations (e.g. corr(beta_s_sreg2, beta_d_dreg1)=0.997), which is
physically unexpected because synch and dust share the same region masks
(confirmed: sreg_i and dreg_i are identical pixel sets). The likely cause
is that Minuit's HESSE at a non-converged point is numerically unreliable.

**Task:** run one full (uncapped) seed of the k=4 combined dust+synch fit
and save the foreground correlation matrix.

1. Add a new config
   `examples/DustSynch_var_9freq_regionwise_combinedk4_ns8_r1e-2.ini`
   based on the existing `*_trace_ncall20.ini` but with `migrad_ncall`
   removed (or commented out) and `save_minuit_matrices = True` kept.
2. Run one seed (e.g. seed 1):
   ```
   DELTAMAP_DUST_BETA_MAP=data/pysm_2/dust_beta.fits \
   DELTAMAP_DUST_TEMP_MAP=data/pysm_2/dust_temp.fits \
   uv run python examples/TestDeltamap.py \
     examples/LTD_config_ns8.ini \
     examples/DustSynch_var_9freq_regionwise_combinedk4_ns8_r1e-2.ini 1
   ```
3. Visualise the saved correlation matrix:
   ```
   uv run python scripts/plot_fg_correlation_heatmap.py \
     examples/test_results/DustSynch_var_9freq_regionwise_combinedk4_ns8_r1e-2/num0001_fg_corr.csv
   ```
4. Record in HANDOFF: whether within-region correlations
   (e.g. corr(beta_s_sreg_i, beta_d_dreg_i)) dominate over cross-region
   ones; and what r, beta_s, beta_d, T_d1 values were recovered.

Expected outcome if ncall=20 was the issue: within-region same-index
pairs should show the strongest synch-dust correlation; cross-region
pairs should be weaker.

Attempted 2026-05-18/19:
- Added `examples/DustSynch_var_9freq_regionwise_combinedk4_ns8_r1e-2.ini`
  with `save_minuit_matrices = True` and no `migrad_ncall`.
- Started seed 1 with the command above.
- The uncapped foreground Minuit did not return any step output after more than
  five hours wall time. Host `ps` showed the Python process at ~91% CPU with
  more than 3.5 hours CPU time. It was killed manually; no result CSV or
  covariance/correlation files were produced.
- Conclusion: full uncapped k=4 combined dust+synch is not practical in the
  current interactive workflow. To get a more reliable correlation matrix than
  `ncall=20`, use a large but finite cap (for example `migrad_ncall=200` or
  staged component initialisation) rather than truly uncapped Minuit.
- Suggested next attempt: create a `trace_ncall200` config with
  `save_minuit_matrices = True`, run one seed, then compare same-region vs
  different-region cross-component correlations. If `ncall=200` is still too
  slow, reduce k or freeze/priors on dust `T_d1` before revisiting full combined
  fits.
- Follow-up discussion (2026-05-19): keep the original `r` initial value
  (`inits = 0.5 ...`). Even if the current `r` estimate is not yet near the
  simulated truth, the outer loop is now judged by
  `Delta L_r = L_fg - L_r`, so short capped coordinate-descent steps can still
  be useful diagnostics. Do not change `r_init` for this comparison; instead
  run the existing k=4 combined `migrad_ncall=20` setup with
  `save_minuit_matrices=True` and inspect the final foreground correlation
  matrix.
- Ran the original-r-init k=4 combined capped trace, seed 20:
  `examples/DustSynch_var_9freq_regionwise_combinedk4_ns8_trace_ncall20.ini`.
  A briefly started `r_init=0.01` variant was killed before producing fit
  output after discussion clarified that `r_init` should remain unchanged.
- Seed 20 trace:
  - outer 0: foreground `nfcn=209`, r-only `nfcn=22`,
    `r=0.0122794376`, `Delta L_r=57.1305`, continued.
  - outer 1: foreground `nfcn=211`, r-only `nfcn=21`,
    `r=0.0112598438`, `Delta L_r=0.00733 < tol=0.856`, stopped.
  - result CSV:
    `examples/test_results/DustSynch_var_9freq_regionwise_combinedk4_ns8_trace_ncall20/num0020.csv`.
  - final foreground correlation/covariance now use physical parameter names:
    `num0020_fg_corr.csv`, `num0020_fg_cov.csv`.
  - heatmap:
    `data/plots/num0020_fg_corr_heatmap.png`.
  - strong-pair CSV:
    `data/plots/num0020_fg_corr_heatmap_strong_pairs.csv`.
- Seed 20 final parameters remain a capped diagnostic, not a valid converged
  fit: `r=0.01126 +/- 0.01123`; `beta_s_sreg0=-2.759`,
  `beta_s_sreg3=-3.151`, while `beta_s_sreg1/2`, all `beta_d`, and all
  `T_d1` remain at initial values.
- Strong-correlation summary for `|corr| >= 0.8`: 57 pairs total, 42
  cross-component pairs. Of those cross-component pairs, 12 are same-region and
  30 are different-region. This again argues that the current k=4 combined
  degeneracy is not simply a same-sky-region dust/synch mixing problem; the
  cross-component correlations are largely cross-region/global.
- Visualized the current k=4 combined-feature region masks with
  `scripts/plot_region_mollview.py`.
  Outputs:
  `data/plots/combined_feature_kmeans_ns8_k04_mollview.png` and
  `data/plots/combined_feature_kmeans_ns8_k04_mollview_regions.png`.
  HEALPix-neighbor connected component counts are:
  region 00: 5 components (largest 51/25 pixels);
  region 01: 5 components (largest 67/51/14);
  region 02: 10 components (largest 41/17);
  region 03: 15 components (largest 73/14/12/10).
- Interpretation: current `combined_feature_kmeans` is a spectral-tracer-value
  clustering, closer in spirit to MC-NILC-style partitions where disconnected
  sky areas may share a region. For patch-wise parametric foreground
  parameters, this is likely not the final target. The more relevant direction
  is a 2109.11562-like spherical graph clustering: combine angular proximity
  on the HEALPix sphere with SED/tracer similarity, then use a graph Laplacian
  embedding or connectedness-aware agglomerative clustering.
- Keep the three decision axes separate:
  1. Strong foreground-parameter correlations are acceptable if `r` bias/error
     improve; if `r` does not improve, regionwise parameters are unnecessary.
  2. Dust `T_d`/`beta_d` degeneracy can be a compute problem rather than an
     `r` problem, provided it does not bias or inflate `r`.
  3. The current clustering may be inappropriate for patch-wise parametric
     fitting because it ignores sky adjacency.
- Added a lightweight spatial-proximity clustering prototype:
  `scripts/make_spatial_feature_regions.py`.
  It builds a HEALPix-neighbor graph over the analysis mask, weights each edge
  by angular distance plus standardized SED-ratio feature distance, computes a
  minimum spanning tree, and cuts high-weight edges while enforcing
  `--min-pixels`.
- The analysis mask has two HEALPix-neighbor connected components at nside=8:
  224 and 204 pixels. Therefore any strictly connected partition has at least
  two components. Running
  `uv run python scripts/make_spatial_feature_regions.py --k 4 --min-pixels 50`
  produced `spatial_feature_mst_ns8` masks with counts
  `[56, 62, 142, 168]`, all one connected component.
  Outputs:
  `data/regions/spatial_feature_mst_ns8_k04_reg{00..03}_{pix,qu}.npy`,
  `data/regions/spatial_feature_mst_ns8_summary.csv`,
  `data/plots/spatial_feature_mst_ns8_k04_mollview.png`, and
  `data/plots/spatial_feature_mst_ns8_k04_mollview_regions.png`.
- Caveat: the k=4 spatial MST masks are spatially connected, but the SED-ratio
  feature means are very similar across regions. The split is therefore
  mostly a spatial/amplitude patching, not yet a strong SED-shape partition.
- Added configs for these masks:
  `examples/Synch_var_3freq_regionwise_spatialk4_ns8_trace_ncall20.ini` and
  `examples/DustSynch_var_9freq_regionwise_spatialk4_ns8_trace_ncall20.ini`.
- Ran synch-only spatial-k4 seed 21 as a smoke/diagnostic run. It completed:
  outer 0 `r=0.00326991`, `Delta L_r=311.395`, continued; outer 1
  `Delta L_r=0`, stopped. Final CSV:
  `examples/test_results/Synch_var_3freq_regionwise_spatialk4_ns8_trace_ncall20/num0021.csv`.
  Result: `r=0.00327 +/- 0.00222`; only `beta_s_sreg3` moved
  (`-2.870`), other `beta_s` stayed at `-3.0`. Treat this as a capped
  mask-loading/parameter-expansion check, not a fit-quality result.
- Ran same-seed capped combined diagnostics for the key `r` question:
  spatial-MST k=4 seed 22 and combined-feature k=4 seed 22.
  These are not Minuit-valid converged fits, but they are useful for comparing
  the current coordinate-descent behavior.
  - Spatial-MST k=4:
    `examples/DustSynch_var_9freq_regionwise_spatialk4_ns8_trace_ncall20.ini`
    seed 22. Outer 0: `nfcn_fg=310`, `r=0.0452063`,
    `Delta L_r=35.1298`; outer 1: `nfcn_fg=211`, `r=0.0519172`,
    `Delta L_r=0.0518 < tol`, stopped. Final:
    `r=0.05192 +/- 0.02969`.
  - Combined-feature k=4:
    `examples/DustSynch_var_9freq_regionwise_combinedk4_ns8_trace_ncall20.ini`
    seed 22. Outer 0: `nfcn_fg=209`, `r=0.0536543`,
    `Delta L_r=30.7707`; outer 1: `nfcn_fg=215`, `r=0.0561449`,
    `Delta L_r=0.00465 < tol`, stopped. Final:
    `r=0.05614 +/- 0.03453`.
  - Both are high relative to true `r=0.01`. Spatial-MST is slightly lower in
    `r` and `sigma_r`, but this single capped seed does not demonstrate an
    `r`-precision improvement.
  - Correlation structure differs strongly: combined-feature seed 22 has 50
    strong pairs with `|corr| >= 0.8` (37 cross-component; 10 same-region
    cross-component; 27 different-region cross-component), while spatial-MST
    seed 22 has zero pairs above 0.8.
  - Heatmaps were regenerated with non-colliding names:
    `data/plots/combinedk4_num0022_fg_corr_heatmap.png`,
    `data/plots/combinedk4_num0022_fg_corr_heatmap_strong_pairs.csv`,
    `data/plots/spatialk4_num0022_fg_corr_heatmap.png`,
    `data/plots/spatialk4_num0022_fg_corr_heatmap_strong_pairs.csv`.
- Important next evaluation: foreground correlation is secondary. Compare
  `r_hat`, `sigma_r`, and `(r_hat-r_true)/sigma_r` over multiple seeds for
  baseline/global, combined-feature k=4, and spatial-MST k=4. If regionwise
  does not improve `r`, the extra foreground freedom is not justified.
- Added and ran the no-region/global baseline with the same seed 22 and same
  capped diagnostic settings:
  `examples/DustSynch_var_9freq_ns8_trace_ncall20.ini`.
  This uses the same 9 frequencies, `r true=0.01`, `r init=0.5`,
  `migrad_ncall=20`, and `save_minuit_matrices=True`, but no `[regions]`.
  Result:
  `examples/test_results/DustSynch_var_9freq_ns8_trace_ncall20/num0022.csv`.
  Final `r=0.05292 +/- 0.03341`; pull `(r-0.01)/sigma_r=1.285`;
  likelihood `-13203518595.573286`.
- Same-seed seed-22 comparison:
  - global: `r=0.05292 +/- 0.03341`, pull `1.285`, strong pairs `0`.
  - combined-feature k=4: `r=0.05614 +/- 0.03453`, pull `1.336`,
    strong pairs `50`.
  - spatial-MST k=4: `r=0.05192 +/- 0.02969`, pull `1.412`,
    strong pairs `0`.
  The regionwise variants do not show a clear `r` improvement for this seed.
  Combined-feature creates strong foreground correlations; spatial-MST removes
  those strong correlations, but this does not translate into a better
  normalized `r` pull in seed 22.
  Global heatmap outputs:
  `data/plots/global_num0022_fg_corr_heatmap.png` and
  `data/plots/global_num0022_fg_corr_heatmap_strong_pairs.csv`.

### Key open question: at which Nside does r bias appear?

Motivation: the region-wise approach is meant to reduce model mismatch from
unmodelled spatial SED variation, which leaks into r and causes bias as Nside
increases. However, this has not been empirically verified; the claim that
higher Nside causes bias comes from others, not from this project's own fits.

Reference point: nside=4 global (no-region) full fits, seeds 1–10,
give mean r ≈ 0.011 (true r=0.01). This is the baseline that region-wise
must beat to be justified at any Nside.

What is needed:
1. nside=8 global (no-region) full multi-seed fit to establish the nside=8
   baseline. Compare mean r, scatter, and bias to nside=4. If bias is small,
   region-wise is not yet justified at nside=8.
2. nside=16 global full multi-seed fit (requires new PySM map generation
   with LTD_config_ns16.ini first). This is where model mismatch is expected
   to be more visible.
3. Only after observing bias at a specific Nside does region-wise fitting
   at that Nside have an empirical justification.

Compute bottleneck: nside=8 uncapped full fit was killed after >5 hours
(single seed). The per-eval cost is ~2.12 s. Need either:
  (a) a faster per-eval path (currently maxed out at nside=8), or
  (b) acceptance that nside=8 multi-seed runs require long wall time
      (e.g. overnight batch), or
  (c) a staged/frozen-parameter approach to reduce Minuit evaluations.

The seed-22 capped (ncall=20) comparison showed global/combined-k4/
spatial-MST-k4 all give r ≈ 0.05 at that cap, which is not informative
about converged bias; it reflects non-convergence, not model mismatch.

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

---

## Codex task: close-out commit + README update

This is a wrap-up task. The goal is to commit all uncommitted work on
`feature/regionwise-foreground-parameters` in logical groups, update README.md
to reflect current state, then open a PR to `main`.

Do not change any behaviour. Do not run fits. Only commit, document, and open
the PR.

### Step 1 — Commit in five logical groups

Run `uv run python -m unittest discover -s tests -v` after step 1d to confirm
the tree is still valid before opening the PR.

**1a. r-step loop fix**
Stage and commit only `extended_deltamap/deltamap.py`.
Commit message:
```
Fix r-step likelihood threshold scaling and outer-loop comparison
```

**1b. nside=8 example configs**
Stage and commit the following new files:
- `examples/DustSynch_var_9freq_ns8_trace_ncall20.ini`
- `examples/DustSynch_var_9freq_regionwise2_ns8_trace_ncall20_scalar.ini`
- `examples/DustSynch_var_9freq_regionwise_combinedk4_ns8_r1e-2.ini`
- `examples/DustSynch_var_9freq_regionwise_combinedk4_ns8_trace_ncall20.ini`
- `examples/DustSynch_var_9freq_regionwise_spatialk4_ns8_trace_ncall20.ini`
- `examples/Dust_var_7freq_regionwise2_ns8_r1e-2.ini`
- `examples/Dust_var_7freq_regionwise2_ns8_trace_ncall20.ini`
- `examples/Dust_var_7freq_regionwise2_ns8_trace_ncall20_scalar.ini`
- `examples/Synch_var_3freq_regionwise2_ns8_r1e-2.ini`
- `examples/Synch_var_3freq_regionwise2_ns8_trace_ncall20.ini`
- `examples/Synch_var_3freq_regionwise_combinedk4_ns8_trace_ncall20.ini`
- `examples/Synch_var_3freq_regionwise_spatialk4_ns8_trace_ncall20.ini`
Commit message:
```
Add nside=8 region-wise and baseline example configs
```

**1c. clustering and visualisation scripts**
Stage and commit the following new files:
- `scripts/make_combined_feature_regions.py`
- `scripts/make_spatial_feature_regions.py`
- `scripts/plot_dust_sed_by_region.py`
- `scripts/plot_fg_correlation_heatmap.py`
- `scripts/plot_region_mollview.py`
Commit message:
```
Add clustering and region visualisation scripts
```

**1d. TestDeltamap, pyproject, uv.lock**
Stage and commit:
- `examples/TestDeltamap.py`
- `pyproject.toml`
- `uv.lock`
Commit message:
```
Update TestDeltamap, dependencies, and dev tooling
```

**1e. HANDOFF and TODO**
Stage and commit:
- `HANDOFF.md`
- `TODO.md`
Commit message:
```
Update handoff notes and TODO for Step C and nside scaling
```

### Step 2 — Update README.md

Make the following targeted edits. Do not rewrite sections that are not listed
here.

**2a. `## Current status` — append three bullet points** after the existing
last bullet in that section:

```
- implemented region-wise foreground parameters with a spatial coefficient path
  supporting synchrotron and dust independently, validated at nside=4 and nside=8
- added combined-feature k-means and spatial-MST clustering scripts for patch
  region generation under scripts/
- fixed r-step likelihood threshold scaling and outer-loop comparison criterion
  in IterateMinimize
```

**2b. `## Caveats` — replace the region-wise bullet**

Remove this line (it is now outdated):
```
- Region-wise foreground parameters are still at the design-note stage. A first
  prototype should start with a small number of broad dust regions before moving
  toward clustered dust and synchrotron region sets.
```

Replace it with:
```
- Region-wise foreground parameters are implemented and validated for
  synchrotron and dust independently at nside=4 and nside=8 using the spatial
  coefficient path. Cluster-based region masks (combined-feature k-means,
  spatial MST) are generated by scripts under scripts/.
- Multi-seed validation at nside=8 and the nside-scaling study are not yet
  complete; see TODO.md for the planned next steps.
```

**2c. `## Example configs` — add a region-wise block** after the existing list
of example configs (before the paragraph that begins "In practice, the ini
format matters less"):

```
Region-wise configs (nside=8):

- `examples/DustSynch_var_9freq_ns8_trace_ncall20.ini`
  No-region baseline for nside=8 dust+synch 9-band fits.
- `examples/DustSynch_var_9freq_regionwise_combinedk4_ns8_trace_ncall20.ini`
  Combined-feature k-means k=4 region-wise dust+synch fit.
- `examples/DustSynch_var_9freq_regionwise_spatialk4_ns8_trace_ncall20.ini`
  Spatial-MST k=4 region-wise dust+synch fit.
```

**2d. Commit README.md alone**
Commit message:
```
Update README to reflect region-wise implementation status
```

### Step 3 — Open a PR to main

After all commits pass the smoke tests, open a pull request from
`feature/regionwise-foreground-parameters` to `main` using the gh CLI.

PR title:
```
Region-wise foreground parameters: nside=8 validation and clustering
```

PR body (use a HEREDOC):
```
## Summary

- Implements region-wise foreground parameters for synchrotron and dust
  independently via the spatial coefficient path, validated at nside=4 and
  nside=8.
- Adds combined-feature k-means and spatial-MST clustering scripts under
  `scripts/` for generating patch region masks.
- Fixes `r_step_lh_tol` scaling (removes incorrect constant floor) and
  corrects the outer-loop likelihood comparison in `IterateMinimize`.
- Adds nside=8 example configs for no-region baseline, combined-feature k=4,
  and spatial-MST k=4 fits.
- Updates README and documentation to reflect current implementation status.

## Test plan

- [ ] `uv run python -m unittest discover -s tests -v` passes
- [ ] No unintended behaviour changes in `extended_deltamap/deltamap.py`
- [ ] README Caveats section no longer describes region-wise as design-note only
```
