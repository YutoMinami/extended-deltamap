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
  # Nside = parser.getint('par','nside')
  Nside = 16


  ofdir ='./output_pysm3_ns{0:d}_02/'.format(Nside)
  nu_template = 'nu{0:07.2f}GHz'
  ofname_template = ofdir + 'test001_{0}_{1}_nside{2:04d}.fits'
  comp_names = ['dust','synchrotron']
  #comp_names = ['d4','synchrotron']
  #Get Dust
  sky =  pysm.Sky(nside = Nside, preset_strings = ["d1"], output_unit = "uK_CMB")
  #sky =  pysm.Sky(nside = Nside, preset_strings = ["d4"], output_unit = "uK_CMB")
  for nu_i in nu:
    nu_name = nu_template.format(nu_i).replace('.','p')
    ofname = ofname_template.format( nu_name, comp_names[0] ,Nside)
    if os.path.exists(ofname):
      continue
    hp.write_map(ofname, sky.get_emission( nu_i *u.GHz  ), nest =False )


  sky =  pysm.Sky(nside = Nside, preset_strings = ["s1"], output_unit = "uK_CMB")
  for nu_i in nu:
    nu_name = nu_template.format(nu_i).replace('.','p')
    ofname = ofname_template.format( nu_name, comp_names[1] ,Nside)
    if os.path.exists(ofname):
      continue
    hp.write_map(ofname, sky.get_emission( nu_i *u.GHz  ), nest =False )
  return 0
if __name__ == '__main__':
  sys.exit(main())

