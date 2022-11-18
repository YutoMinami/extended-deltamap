# extended-deltamap

## enviroment
- python>=3.7

## required packages
- numpy
- scipy 
- sympy
- healpy
- astropy
- iminuit

## optional packages
- pysm3

## Explanation
- deltamap.py: a class whcih can return $\chi^2$ and likelihood, and can minimise them
- dmatrix.py: a class to deal with foreground $\tilde{D}$ matrix
- covariance.py : a class to calculated pixel space covariance matrix from power spectra
- templates.py: a class to deal with foreground models with sympy symbol

## Usage
Please see example/TestDeltamap.py to see how to fit cmb and fg parameters
## Example
### Preparation
cd example
python run_pysm3.py LTD_config.ini
### Test
python TestDeltamap.py LTD_config.ini Synch_var_3freq_r1e-2.ini 1


