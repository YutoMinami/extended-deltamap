#!/usr/bin/env python
import sys
from pathlib import Path
import configparser
import healpy as hp
import numpy as np
import pysm3 as pysm
import pysm3.units as u
from pysm3 import models
import warnings
warnings.filterwarnings("ignore")
def main():
  if len(sys.argv) == 1:
    filename = './config.ini'
  else:
    filename = sys.argv[1]
  config_path = Path(filename).resolve()
  config_dir = config_path.parent

  ##Get Parameters##
  parser = configparser.ConfigParser()
  parser.read(config_path)
  nu = np.array([float(i) for i in parser.get('par','nu').split()])
  fwhm =np.array([float(i) for i in parser.get('par','fwhm').split()]) 
  noise =np.array([float(i) for i in parser.get('par','noise').split()]) 
  Nside = parser.getint('par','nside')

  ofdir = (config_dir / parser.get('fgpar', 'fg_dir').format(Nside)).resolve()
  ofdir.mkdir(parents=True, exist_ok=True)

  nu_template = 'nu{0:07.2f}GHz'

  ofname_template = parser.get('fgpar', 'fg_name')

  comp_names = parser.get('fgpar','components' ).split()
  
  for comp in comp_names:
    sky = pysm.Sky(nside = Nside, preset_strings = [ comp ], output_unit = "uK_CMB")
    for nu_i in nu:
      nu_name = nu_template.format(nu_i).replace('.','p')
      ofname = ofdir / ofname_template.format(nu_name, comp, Nside)
      if ofname.exists():
        continue
      hp.write_map(str(ofname), sky.get_emission( nu_i *u.GHz  ), nest =False )
  return 0
if __name__ == '__main__':
  sys.exit(main())
