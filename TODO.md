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
- Region-wise foreground-parameter prototype is implemented and validated for
  synchrotron-only fits up to 4 regions. Spatial coefficient path is the
  supported implementation. Planned progression:
  Step A — Extend spatial coefficient path to dust (2 regions, nside=4).
    Dust template has 2 SED parameters (beta_d, T_d1), so template_terms will
    have 3 columns (0th order + 2 derivatives). Region masks derived from
    353 GHz dust-dominated maps, independent of synchrotron masks.
    Dust-only seed-1 fit completes locally with `D_matrix.shape == (7, 3)`.
    Combined dust+synch at nside=4 does not converge reliably: the (beta_d,
    T_d1) pair introduces a within-component MBB degeneracy that is independent
    of region count, and the near-identical SED parameters across the two regions
    at nside=4 create an additional between-region swap degeneracy. This is
    expected at nside=4 and not worth debugging further; nside=8 will provide
    enough per-region pixels to relax both degeneracies.
    Step A is considered validated: the spatial coefficient path works for dust.
  Step B — Move to nside=8 with 2 regions each for synchrotron and dust.
    nside=8 gives ~4x more pixels (~428 valid), providing enough per-region
    statistics to support more regions. Confirms the method scales with Nside.
    Initial setup is complete: PySM nside=8 maps were generated locally, and
    synch/dust median masks split 428 valid pixels into 214/214 regions.
    Combined dust+synch profile keeps `D_matrix.shape == (9, 5)`, but one
    likelihood evaluation initially took ~4.56 s. Replacing the `DNIDL`
    logdet with the Cholesky diagonal formula reduced `ReturnlnDNID` from
    ~0.97 s to ~0.0001 s and total evaluation time to ~3.49 s. An einsum
    rewrite of the spatial `CalcH_matrix()` reduced Python-loop overhead, but
    the remaining time is dominated by dense Cholesky and `DTNIDc A^{-1}
    DTNIDc^T`; with finite checks disabled on the largest solves, the profile
    was ~3.37 s/eval. Removing the unused `DT_SpNI_D` / `DT_SpNI_DL` build from
    current likelihood evaluations reduced the nside=8 profile to ~2.12 s/eval.
    Do not launch multi-seed full fits until either this dense linear algebra
    cost is acceptable or a cheaper staged test is chosen.
    Next direction: reduce optimizer evaluation count rather than chasing
    small per-evaluation gains. Add nside=8 dust-only and synch-only 2-region
    configs, run short `migrad_ncall` traces, then use their recovered
    foreground parameters as stronger initial values for a single combined
    dust+synch seed before any multi-seed run.
    Initial seed-8 `migrad_ncall=20` traces are finite but not valid. Synch-only
    gives plausible values (`r≈0.0135`, `beta_s≈-2.93,-3.14`), while dust-only
    and combined fits leave several dust parameters at their initial values.
    This points to dust MBB exploration/initialization rather than an algebraic
    failure of the spatial coefficient path.
    Optimizer-loop cleanup: the outer-loop continuation criterion now compares
    full likelihood values before/after the r-only step rather than comparing
    `r` shifts. A transient negative `Delta L_r` diagnosis was traced to
    comparing FG `ReturnChiSquare()` fvals against r-only `ReturnLikelihood()`
    fvals. With the corrected full-likelihood comparison, Minuit r-only and
    scalar bounded r-only give nearly identical likelihoods on capped nside=8
    traces. Keep Minuit as the default r-only optimizer to preserve `r_error`;
    scalar remains a diagnostic option via `diagnostics.r_minimizer = scalar`.
    The r-step likelihood threshold now uses `1e-3 * self.size` without a
    constant floor, so the stopping criterion scales with nside.
    New hypothesis: the current dust median brightness split is a poor
    parameter-estimation split because the faint region has too little dust
    signal to constrain its own `beta_d`/`T_d1`. For later clustering, prefer
    grouping pixels by SED-like color features, e.g. log polarization-amplitude
    ratios across frequencies, rather than by brightness alone.
  Step C — k-means clustering with ~5 regions each for synchrotron and dust
    at nside=8. Synchrotron regions from low-frequency polarization maps;
    dust regions from 353 GHz dust-dominated maps. Final region counts and
    shapes are independent between components.
    Started combined-feature k-means prototype using six standardized features:
    log synch amplitude, two synch SED ratios, log dust amplitude, and two dust
    SED ratios. Generated k=4/8/16 masks with
    `scripts/make_combined_feature_regions.py`; k=4 has healthy counts
    (79/134/83/132), while k=8 already has a 26-pixel cluster and should wait.
    Added k=4 combined-feature synch-only and dust+synch trace configs. The
    synch-only k=4 seed-18 trace loads the masks and expands four `beta_s`
    parameters successfully, though the short cap is not Minuit-valid.
    The k=4 combined dust+synch trace is much slower (`nfcn≈214` per FG step)
    and leaves dust parameters at initial values under `migrad_ncall=20`.
    Added `save_minuit_matrices` diagnostic output for the final foreground
    Minuit covariance/correlation; seed-19 shows very strong foreground
    correlations, so inspect these matrices before increasing k or ncall.
    Added `scripts/plot_fg_correlation_heatmap.py`; seed-19 heatmap/strong-pair
    outputs show strong cross-component synch-vs-dust correlations as well as
    dust beta/T correlations. These cross-component correlations are not only
    same-region pairs; strong different-region pairs appear too. Therefore,
    do not jump immediately to separate dust/synch region sets as the fix.
    First summarize same-region vs different-region cross-component correlations.
    Tried full uncapped k=4 combined seed 1 for a better correlation matrix;
    the foreground Minuit did not return after >5h wall time and was killed
    without outputs. Next use a large finite cap (e.g. `migrad_ncall=200`) or
    staged/frozen dust parameters rather than true uncapped Minuit.
    Keep the original `r_init=0.5`; the next comparison should not change the
    initial condition. Because the outer loop now uses `Delta L_r`, capped
    coordinate-descent steps may still be preferable. Ran the existing k=4
    combined trace with `migrad_ncall=20`, seed 20. It stopped after outer 1
    with `r=0.01126` and produced physically named `num0020_fg_corr.csv`.
    Strong correlations remain mostly cross-component and different-region
    (57 strong pairs, 42 cross-component, 30 different-region cross-component),
    so this does not yet support a simple same-region dust/synch split as the
    main fix.
    Visualized the k=4 masks with `scripts/plot_region_mollview.py`. The
    current feature-space k-means masks are highly non-connected
    (component counts 5/5/10/15 for regions 0..3). This is acceptable for an
    MC-NILC-like "same spectral tracer value" clustering, but it is probably
    not the right final definition for patch-wise parametric foreground
    parameters. Next clustering prototype should add spherical proximity:
    build a HEALPix neighbor graph with angular-distance and SED-tracer terms,
    then apply spectral/agglomerative clustering with connectedness in mind.
    Added first lightweight spatial-proximity prototype:
    `scripts/make_spatial_feature_regions.py`. It uses a HEALPix-neighbor graph
    with angular+SED-ratio edge weights, cuts a minimum spanning tree with a
    `--min-pixels` guard, and generated connected k=4 masks
    `spatial_feature_mst_ns8` with counts 56/62/142/168. The SED-ratio means
    are still very similar across regions, so this is a spatially connected
    patch prototype rather than a strong SED-shape clustering. Added spatial-k4
    synch-only and dust+synch trace configs; synch-only seed 21 completed as a
    mask-loading diagnostic.
    Ran same-seed dust+synch capped diagnostics for seed 22. Spatial-MST k=4
    gives `r=0.05192 +/- 0.02969`; combined-feature k=4 gives
    `r=0.05614 +/- 0.03453`; both are high versus true `r=0.01`, so this does
    not yet show an `r`-precision benefit. Spatial-MST does remove the very
    strong foreground-parameter correlations (`|corr|>=0.8`: 0 pairs versus 50
    for combined-feature), but correlation reduction is secondary. Next compare
    `r_hat`, `sigma_r`, and normalized pull over multiple seeds and against a
    no-region/global baseline.
    Added the no-region baseline
    `examples/DustSynch_var_9freq_ns8_trace_ncall20.ini` and ran seed 22.
    Same-seed comparison: global `r=0.05292 +/- 0.03341` (pull 1.285),
    combined-feature k=4 `r=0.05614 +/- 0.03453` (pull 1.336), spatial-MST k=4
    `r=0.05192 +/- 0.02969` (pull 1.412). Regionwise does not clearly improve
    `r` for this seed.
  The region-wise approach is motivated by Nside scaling: more pixels allow
  more regions, and the spatial coefficient path keeps H matrix size fixed
  regardless of region count.
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
- Completed the synchrotron-only 2-region foreground-parameter prototype:
  median low-frequency brightness masks, region-specific `beta_s` symbols,
  `DMatrix` column masks, masked `CalcH_matrix()` blocks, config-driven
  region parameter expansion, and one end-to-end fit.
- Confirmed that
  `uv run python examples/TestDeltamap.py examples/LTD_config.ini examples/Synch_var_3freq_regionwise_r1e-2.ini 1`
  completes locally with the two median-split region masks.

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
