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
- The active refactor notes live in `TODO.md`.
- Repo-local working agreements for automated contributors live in `AGENTS.md`.
- Current handoff notes for future work live in `HANDOFF.md`.

## Current status

This repository has recently been updated to:

- use `uv` as the primary workflow tool
- target Python `3.12`
- package the code under `extended_deltamap/`
- improve path handling and example-script readability
- add docstrings across the actively used modules

## Caveats

- Some scripts still reflect their research-code origins and remain larger than
  ideal.
- Example fitting currently depends on external dust parameter FITS files
  provided through environment variables.
- The example scripts are still the main operational entry points.
