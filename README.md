# extended-deltamap

This is a package to remove themal dust emission and synchrotron radiation foreground in the cosmic microwave background (CMB) map
and to estimate CMB parameter.


## enviroment
- Python 3.12
- uv

## required packages
- Managed via `uv sync`

## optional packages
- `uv sync --extra pysm`

## Explanation
- `extended_deltamap.deltamap`: a class whcih can return $\chi^2$ and likelihood, and can minimise them
- `extended_deltamap.dmatrix`: a class to deal with foreground $\tilde{D}$ matrix
- `extended_deltamap.covariance`: a class to calculated pixel space covariance matrix from power spectra
- `extended_deltamap.templates`: a class to deal with foreground models with sympy symbol

## Usage
Please see `examples/TestDeltamap.py` to see how to fit cmb and fg parameters
## Example
### Preparation
```bash
uv sync --extra pysm
uv run python examples/run_pysm3.py examples/LTD_config.ini
```
### Test
```bash
export DELTAMAP_DUST_BETA_MAP=/path/to/dust_beta.fits
export DELTAMAP_DUST_TEMP_MAP=/path/to/dust_temp.fits
uv run python examples/TestDeltamap.py examples/LTD_config.ini examples/Synch_var_3freq_r1e-2.ini 1
```
