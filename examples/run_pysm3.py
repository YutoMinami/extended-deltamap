#!/usr/bin/env python
import  sys  
import os
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

  ##Get Parameters##
  parser = configparser.ConfigParser()
  parser.read(filename)
  nu = np.array([float(i) for i in parser.get('par','nu').split()])
  fwhm =np.array([float(i) for i in parser.get('par','fwhm').split()]) 
  noise =np.array([float(i) for i in parser.get('par','noise').split()]) 
  Nside = parser.getint('par','nside')

  ofdir = parser.get('fgpar', 'fg_dir').format(Nside)
  if not  os.path.exists( ofdir ):
    os.mkdir(ofdir)

  nu_template = 'nu{0:07.2f}GHz'

  ofname_template = ofdir + parser.get('fgpar', 'fg_name')

  comp_names = parser.get('fgpar','components' ).split()
  
  for comp in comp_names:
    sky = pysm.Sky(nside = Nside, preset_strings = [ comp ], output_unit = "uK_CMB")
    for nu_i in nu:
      nu_name = nu_template.format(nu_i).replace('.','p')
      ofname = ofname_template.format(nu_name, comp, Nside)
      if os.path.exists(ofname):
        continue
      hp.write_map(ofname, sky.get_emission( nu_i *u.GHz  ), nest =False )
  return 0
if __name__ == '__main__':
  sys.exit(main())

