# extended-deltamap

`extended-deltamap` is a Python package for foreground modeling and parameter
estimation in cosmic microwave background (CMB) analyses.

The current repository contains:

- the reusable package code under `extended_deltamap/`
- example scripts under `examples/`
- bundled theory spectra used by the covariance code
- a `uv`-managed Python 3.12 development workflow

## What this repo does

This project focuses on:

- modeling thermal dust and synchrotron foregrounds
- building pixel-space covariance matrices from CMB power spectra
- constructing DeltaMap fitting matrices
- fitting CMB and foreground parameters from simulated or prepared inputs

## Reference paper

This repository implements and experiments with the extended Delta-map method
described in:

- Y. Minami and K. Ichiki, "Extended delta-map: A map-based foreground removal
  method for CMB polarization observations," *PTEP* **2023** (2023) no.3,
  033E01.
  DOI: [10.1093/ptep/ptad016](https://doi.org/10.1093/ptep/ptad016)
  arXiv: [2212.01773](https://arxiv.org/abs/2212.01773)

The package and example scripts follow the paper's map-based workflow: build
multi-frequency polarization maps, assemble the Delta-map foreground basis,
construct pixel-space covariance matrices, and evaluate the likelihood for CMB
and foreground parameters.

## Project layout

- `extended_deltamap.deltamap`
  Main fitting class. Computes chi-square and likelihood terms and runs the
  iterative minimization flow.
- `extended_deltamap.dmatrix`
  Builds symbolic foreground mixing matrices and their parameter derivatives.
- `extended_deltamap.covariance`
  Converts theory power spectra into pixel-space covariance matrices.
- `extended_deltamap.templates`
  Provides symbolic foreground templates used by the fitting workflow.
- `examples/run_pysm3.py`
  Generates PySM foreground maps from a config file.
- `examples/TestDeltamap.py`
  End-to-end example script for running DeltaMap fits from config files.

## Environment

- Python `3.12`
- `uv`

## Setup

Install the default environment:

```bash
uv sync
```

Activate the virtual environment if you want an interactive shell:

```bash
source .venv/bin/activate
```

The project is configured as a package, so package imports work through `uv run`
and from the activated environment.

## Data and external inputs

Some example workflows expect external FITS files for dust template parameters.
At runtime, set:

```bash
export DELTAMAP_DUST_BETA_MAP=/path/to/dust_beta.fits
export DELTAMAP_DUST_TEMP_MAP=/path/to/dust_temp.fits
```

For example, PySM-compatible dust parameter maps can be used for these values.
If you keep local copies under this repository, a setup such as the following is
fine:

```bash
export DELTAMAP_DUST_BETA_MAP=$PWD/data/pysm_2/dust_beta.fits
export DELTAMAP_DUST_TEMP_MAP=$PWD/data/pysm_2/dust_temp.fits
```

If you keep local large files inside this repository, `data/` is already
gitignored.

## Pipeline overview

The main example flow is split between map generation and likelihood
construction:

1. Generate or prepare foreground maps.
   `examples/run_pysm3.py` creates frequency maps from a config file, while
   external FITS inputs provide dust parameter maps such as `beta_d` and `T_d`.
2. Build simulated multi-frequency inputs.
   `examples/TestDeltamap.py` uses `return_map_with_noise_cov()` to combine CMB,
   synchrotron, dust, detector noise, and the paper's artificial-noise terms
   into masked Q/U data vectors `mvec`.
3. Build the foreground mixing model.
   `extended_deltamap.templates` defines symbolic synchrotron and dust scaling
   laws, and `extended_deltamap.dmatrix` turns them into the Delta-map
   frequency matrix and first-derivative columns.
4. Build the signal and noise covariance terms.
   `extended_deltamap.covariance` converts theory spectra into pixel-space CMB
   covariance matrices, and `examples/TestDeltamap.py` assembles per-frequency
   noise covariances before calling `DeltaMap.SetS0()` and `DeltaMap.SetNoiseList()`.
5. Evaluate the likelihood and fit parameters.
   `extended_deltamap.deltamap` builds the internal matrices used by the
   extended Delta-map likelihood, evaluates the chi-square and log-determinant
   terms, and runs the iterative minimization over `r` and the foreground
   parameters.

At a high level, `examples/run_pysm3.py` is the map-production entry point and
`examples/TestDeltamap.py` is the end-to-end fitting entry point.

## Quick start

Generate foreground maps from the example configuration:

```bash
uv run python examples/run_pysm3.py examples/LTD_config.ini
```

Run one example fit:

```bash
export DELTAMAP_DUST_BETA_MAP=/path/to/dust_beta.fits
export DELTAMAP_DUST_TEMP_MAP=/path/to/dust_temp.fits
uv run python examples/TestDeltamap.py \
  examples/LTD_config.ini \
  examples/Synch_var_3freq_r1e-2.ini \
  1
```

## Example configs

The repository currently keeps the runtime configuration in `ini` files. This
is still a reasonable fit for the project because the example runs mainly need
flat values such as booleans, numeric parameters, paths, and frequency lists.

The most useful example configs are:

- `examples/LTD_config.ini`
  Base simulation setup with the full LiteBIRD-like 15-band frequency list,
  beam widths, noise levels, and input naming patterns.
- `examples/Synch_var_3freq_r1e-2.ini`
  Minimal synchrotron-only first-order fit example.
- `examples/Dust_var_4freq_r1e-2.ini`
  Minimal dust-only first-order fit example.
- `examples/Dust_var_7freq_r1e-2.ini`
  Dust-only 7-band setup used as a second-order minimum-frequency candidate.
- `examples/Dust_var_9freq_r1e-2.ini`
  Dust-only 9-band setup using all channels from `100 GHz` through `402 GHz`,
  closer to the band selection discussed in the paper.
- `examples/Dust_var_9freq_r1e-2_order2.ini`
  Dust-only 9-band setup with `order = 2`, intended for second-order D-matrix
  tests with the minimum band count for two dust parameters.
- `examples/LTD_config_sections.ini`
  New-section version of the shared simulation config, provided as a migration
  example for `[instrument]`, `[foreground]`, and `[simulation]`.
- `examples/Dust_var_9freq_r1e-2_sections.ini`
  New-section version of the 9-band dust-only fit config, provided as a
  migration example for `[fit]`, `[templates]`, and `[io]`.

Region-wise configs (nside=8):

- `examples/DustSynch_var_9freq_ns8_trace_ncall20.ini`
  No-region baseline for nside=8 dust+synch 9-band fits.
- `examples/DustSynch_var_9freq_regionwise_combinedk4_ns8_trace_ncall20.ini`
  Combined-feature k-means k=4 region-wise dust+synch fit.
- `examples/DustSynch_var_9freq_regionwise_spatialk4_ns8_trace_ncall20.ini`
  Spatial-MST k=4 region-wise dust+synch fit.

In practice, the `ini` format matters less than keeping the examples readable
and safe to run. To help with that, `examples/TestDeltamap.py` now performs
basic config validation before the expensive map and likelihood steps:

- the map config must provide matching `nu`, `noise`, and `fwhm` lengths
- the fit config must provide matching `params` and `inits` lengths
- fit frequencies must be a subset of the simulation frequencies
- at least one foreground component must be enabled
- the fit model must have enough frequency channels for its current parameter
  count and configured D-matrix order
- fit outputs are expected to use `.csv`

When adding new configs, the preferred section layout is now:

- `[instrument]`
  Shared observing setup such as `nu`, `noise`, `fwhm`, and `nside`
- `[foreground]`
  PySM component names and generated foreground output patterns such as
  `components`, `fg_dir`, and `fg_name`
- `[simulation]`
  Input cache and auxiliary map patterns such as `input_dir`, `noise_name`,
  `anoise_name`, `anoise_freq_name`, `cmb_name`, and `maskname`
- `[fit]`
  Fit-time settings such as selected channels, enabled components, parameter
  lists, initial values, D-matrix `order`, `anoise`, `fgnoise_fac`, `fixTd`, `migrad`, and
  `simul`
- `[templates]`
  Foreground template map patterns such as `dust_temp` and `synch_temp`
- `[io]`
  Output directory and filename pattern such as `odir` and `oname`
- `[priors]`
  Optional prior toggles and widths such as `Tdprior`, `Tdsigma`, `xRprior`,
  and `xRsigma`

The example scripts currently accept both the legacy sections (`[par]`,
`[fgpar]`, `[simpar]`) and the newer section names above, so configs can be
migrated gradually.

## Typical workflow

1. Sync the environment with `uv sync`.
2. Prepare or generate example foreground maps.
3. Point `DELTAMAP_DUST_BETA_MAP` and `DELTAMAP_DUST_TEMP_MAP` to the required
   FITS files.
4. Run `examples/TestDeltamap.py` with a simulation config, fit config, and
   seed.

Generated inputs and fit outputs are written under `examples/` according to the
paths defined in the config files.

## Development notes

- The repository currently uses `uv.lock` for dependency reproducibility.
- Smoke and regression checks currently live under `tests/` and run with:
  `uv run python -m unittest discover -s tests -v`
- The active refactor notes live in `TODO.md`.
- Repo-local working agreements for automated contributors live in `AGENTS.md`.
- Current handoff notes for future work live in `HANDOFF.md`.
- Local Japanese design notes for second-order Delta-map work and region-wise
  foreground-parameter work can be kept under `data/notes/`; this directory is
  gitignored on purpose.

## Current status

This repository has recently been updated to:

- use `uv` as the primary workflow tool
- target Python `3.12`
- package the code under `extended_deltamap/`
- improve path handling and example-script readability
- add docstrings across the actively used modules
- add exploratory support for order-aware `DMatrix` construction while the next
  methodology direction moves toward region-wise foreground parameters instead
  of deeper second-order fitting
- implemented region-wise foreground parameters with a spatial coefficient path
  supporting synchrotron and dust independently, validated at nside=4 and nside=8
- added combined-feature k-means and spatial-MST clustering scripts for patch
  region generation under scripts/
- fixed r-step likelihood threshold scaling and outer-loop comparison criterion
  in IterateMinimize
- stabilized the spin-2 Legendre basis construction in
  `extended_deltamap.covariance` by replacing raw `lpmv` basis precomputation
  with normalized associated-Legendre values and log-space normalization; the
  new path was checked against the previous implementation at `nside=4`,
  `lmax=8`

## Caveats

- Some scripts still reflect their research-code origins and remain larger than
  ideal.
- Example fitting currently depends on external dust parameter FITS files
  provided through environment variables.
- The example scripts are still the main operational entry points.
- Region-wise foreground parameters are implemented and validated for
  synchrotron and dust independently at nside=4 and nside=8 using the spatial
  coefficient path. Cluster-based region masks (combined-feature k-means,
  spatial MST) are generated by scripts under scripts/.
- Multi-seed validation at nside=8 and the nside-scaling study are not yet
  complete; see TODO.md for the planned next steps.
