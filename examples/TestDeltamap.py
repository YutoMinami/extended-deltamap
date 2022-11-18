import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy
import healpy
import scipy
import sympy
import sys
sys.path.append('../')
from DeltaMap import templates, dmatrix,deltamap,covariance
import scipy.constants.constants as constants
import configparser
import argparse
from iminuit import Minuit

from copy import deepcopy


def ReadCell(cl_s_name, nside , isScalar = True):
    cl_s = numpy.loadtxt(cl_s_name)
    if len(cl_s[0]) in (4, 6) and isScalar:
        cl_s = numpy.c_[cl_s[:,:3], numpy.zeros(len(cl_s))[:,numpy.newaxis],cl_s[:,3]]
    cls, ls = cl_s.T[1:5], cl_s.T[0]
    cl = cls * 2.*numpy.pi/(ls*(ls+1.))
    cl = numpy.c_[numpy.zeros([cl.shape[0],2]),cl]

    return cl
def ReturnBell(ell, fwhm):
    s=2.
    sigma_b =( fwhm * numpy.pi/10800.)/numpy.sqrt(8.*numpy.log(2))
    return numpy.exp(-(ell*(ell+1)-s**2)*pow(sigma_b,2)/2)

def ReturnNoiseSigma(noise ,nside,):
    npix = healpy.nside2npix(nside)
    pix_ster = 4.*numpy.pi/npix
    pix_amin = numpy.rad2deg(numpy.sqrt(pix_ster) ) *60.
    sigma = noise/pix_amin 
    return sigma

def ReturnCMBMap(r, nside, fwhm):
    Cl_s = ReadCell('/home/cmb/yminami/Git/DeltaMap/files/test_lensedcls_49T7H5WT3X.dat', nside, True)
    Cl_t = ReadCell('/home/cmb/yminami/Git/DeltaMap/files/test_tenscls_49T7H5WT3X.dat', nside, False)
    minlen= min( len(Cl_s[1]) , len(Cl_t[1]))
    cmbmap = healpy.synfast(
        Cl_s[:,:minlen] + Cl_t[:,:minlen] * r, lmax=nside*2, nside=nside, new=True, fwhm= fwhm * numpy.pi / 10800., pixwin=True ,
         verbose=False
    )
    return cmbmap


def ReturnANoiseMap(anoise, nside, nonzero_len):
    asigma = ReturnNoiseSigma(anoise, nside)
    random_anoise = numpy.random.randn(nonzero_len)*asigma
    return random_anoise

def ReturnNoiseCov(noi, nside, beam, cov, pixwin = True):
    ell = numpy.arange(0, nside*2+1,1)
    bl_nominal  = ReturnBell(ell, beam)
    pw = healpy.pixwin( lmax= nside*2, nside = nside, pol=True)[1]
    Cell = numpy.zeros((6, nside*2+1) )
    if pixwin:
        bl_nominal[pw!=0] /= pw[pw!=0]
    Cell[1] = numpy.full( (1, nside*2+1), pow( noi * numpy.pi / 10800., 2) ) * pow( 1. / bl_nominal ,2) 
    Cell[2] = numpy.full( (1, nside*2+1), pow( noi * numpy.pi / 10800., 2) ) * pow( 1. / bl_nominal ,2)
    cov.CalcCov02(Cell[:,2:])
    noise_cov = numpy.block([ [cov.QQ, cov.QU],[cov.QU.T, cov.UU]])
    return noise_cov




def ReturnNoiseMap(noise, nside, beam, fwhm):
    npix = healpy.nside2npix(nside)
    noise_map = numpy.zeros( shape=(3, npix  ) )
    sigma = ReturnNoiseSigma(noise, nside)
    
    random_ar = numpy.random.randn(2 , npix )    
    noise_map[1] = random_ar[0] * sigma
    noise_map[2] = random_ar[1] * sigma
    
    tmp_noise = numpy.copy(noise_map)
    
    alm = healpy.map2alm(tmp_noise, lmax= nside*2, pol=True)
    
    noise_map = healpy.alm2map(alm,  nside = nside, lmax=nside*2,  pixwin=True, verbose=False,
                   fwhm = fwhm*numpy.sqrt(1- pow(beam/fwhm,2) )*numpy.pi/ 10800. )[1:]
    return noise_map



def ReturnMapWithNoiseCov(maskname , anoise, fwhm, nside, r,fgfac , nfac,
             freqs, noises,fwhm_list, dust_model, synch_model, param_defs,uni = False, isdust=True, 
        issynch =True,
          re_noise = False, re_cmb = False, dust_template = None, synch_template= None, 
          fixTd = False, fgnoise_fac = None, seed = 1
             ):
    print('Deltamap03')
    mvec= []
    mask= healpy.read_map( maskname, field=(0), verbose=False, dtype = numpy.float64)
    mask = healpy.ud_grade(mask,nside_out=nside)
   
    params = dust_model.free_symbols
    for param in params:
        if param.name in param_defs.keys():
            dust_model= dust_model.subs(param, param_defs[param.name])
    params = synch_model.free_symbols
    for param in params:
        if param.name in param_defs.keys():
            synch_model = synch_model.subs(param, param_defs[param.name])
     
    if issynch and uni:
      piv_synch = float(synch_model.evalf(subs={'nu':40}) )
    if isdust and uni:
      piv_mbb1 = float(dust_model.evalf(subs={'nu':402}) )
  
    noise_template = './inputs/noise_nu{0}GHz_ns{1:04d}_{2}amin_fwhm{3:d}_beam{4}_{5:04d}.fits'
    anoise_freq_template = './inputs/anoise_nu{0}GHz_ns{1:04d}_{2:04d}.npy'
    cmb_template = './inputs/cmb_ns{0:04d}_fwhm{1:d}_r{2}_{3:04d}.fits'
    anoise_template = './inputs/anoise_ns{0:04d}_{1}amin_{2:04d}.npy'

    if 4 == nside:                
        if synch_template is None:
            synchmapname = './PySM/output02_ns4/test002_nu{0}GHz_synchrotron_nside0004.fits'
        else:
            synchmapname = synch_template
        if dust_template is None:
            if fixTd: 
                dustmapname = './PySM/output02_ns4/TdFix_nu{0}GHz_dust_nside0004.fits'
            else:
                dustmapname = './PySM/output02_ns4/test002_nu{0}GHz_dust_nside0004.fits'
        else:
            dustmapname = dust_template
    elif 16==nside:
        synchmapname = './PySM/output_ns16_r0/test002_nu{0}GHz_synchrotron_nside0016.fits'
        if fixTd: 
            dustmapname = './PySM/output_ns16_r0/TdFix_nu{0}GHz_dust_nside0016.fits'
        else:
            dustmapname = './PySM/output_ns16_r0/test002_nu{0}GHz_dust_nside0016.fits'
    else:
        synchmapname = synch_template
        if fixTd: 
          return 1
        dustmapname = dust_template

    nonzero_len =len(  mask[mask != 0.0] )*2
   
    anoise_name = anoise_template.format(nside,'{0:.1e}'.format(anoise).replace('.', 'p'), seed) 
    if re_noise or not os.path.exists(anoise_name):
      random_anoise = ReturnANoiseMap(anoise, nside, nonzero_len)
      if not os.path.exists(anoise_name):
        numpy.save(anoise_name, random_anoise)
    else:
      print('read old anoise map')
      random_anoise = numpy.load(anoise_name)
    
    cmb_writename = cmb_template.format( nside, int(fwhm), '{0:.1e}'.format(r).replace('.', 'p'), seed )
    if re_cmb or not os.path.exists(cmb_writename):
        cmbmap = ReturnCMBMap(r, nside, fwhm)
        if not os.path.exists(cmb_writename):
          healpy.write_map(cmb_writename, cmbmap, nest =False)    
    else:
        print('read old cmb map')
        cmbmap =  healpy.read_map(cmb_writename, field=[0,1,2], verbose = False, dtype = numpy.float64)

   
    nu_str = '{0:07.2f}'.format(402.0).replace('.','p')
    dustmap = healpy.read_map( dustmapname.format(nu_str) , field=(0,1,2), verbose=False, dtype = numpy.float64)
    alm = healpy.map2alm( dustmap, lmax=nside*2, pol=True)

    dustmap = healpy.alm2map(alm, nside=nside, lmax=nside*2, pixwin=True, verbose=False, fwhm=fwhm*numpy.pi/10800.)
    nu_str = '{0:07.2f}'.format(40.0).replace('.','p')
    synchmap = healpy.read_map( synchmapname.format(nu_str), field=(0,1,2), verbose=False, dtype = numpy.float64)
    alm = healpy.map2alm( synchmap, lmax=nside*2, pol=True)
    synchmap = healpy.alm2map(alm, nside=nside, lmax=nside*2, pixwin=True, verbose=False, fwhm=fwhm*numpy.pi/10800.)
    for nu,noise,beam in zip( freqs, noises, fwhm_list ):
        #print(nu,noise)
        fgmap = numpy.zeros_like(cmbmap)    
        nu_str = '{0:07.2f}'.format(nu).replace('.','p')
        if uni:
            fac_mbb1 =  float( dust_model.evalf(subs={'nu':nu}) )
            fac_synch = float(synch_model.evalf(subs={'nu':nu}) )
            fgmap = numpy.zeros_like(cmbmap)
            if isdust:
                fgmap += dustmap * fac_mbb1/piv_mbb1
            if issynch:
                fgmap += synchmap * fac_synch/piv_synch
        else:
            dustmap = healpy.read_map( dustmapname.format(nu_str), field=(0,1,2), verbose=False, dtype = numpy.float64)
            alm = healpy.map2alm( dustmap, lmax=nside*2, pol=True)
            dustmap = healpy.alm2map(alm, nside=nside, lmax=nside*2, pixwin=True, verbose=False, fwhm=fwhm*numpy.pi/10800.)

            synchmap = healpy.read_map( synchmapname.format(nu_str), field=(0,1,2), verbose=False, dtype = numpy.float64)
            alm = healpy.map2alm( synchmap, lmax=nside*2, pol=True)
            synchmap = healpy.alm2map(alm, nside=nside, lmax=nside*2, pixwin=True, verbose=False, fwhm=fwhm*numpy.pi/10800.)
            if isdust:
                fgmap += dustmap
            if issynch:
                fgmap +=  synchmap
                
        inmap = cmbmap[1:]+  fgfac * fgmap[1:]
        noisename =  noise_template.format( nu_str, nside, '{0:.3f}'.format(noise).replace('.','p'), 
        int(fwhm), '{0:.1f}'.format(beam).replace('.0','p0'), seed )
        if re_noise or not os.path.exists(noisename):
          noise_map = ReturnNoiseMap(noise, nside, beam, fwhm)
          if not os.path.exists(noisename) :
            healpy.write_map( noisename, noise_map, nest = False) 
        else:
          print('read old noise map')
          noise_map = healpy.read_map(noisename, field=[0,1], verbose = False, dtype = numpy.float64)

        inmap += noise_map * nfac
        mvec_each = numpy.concatenate([inmap[0][mask != 0.0],inmap[1][mask != 0.0]])
        mvec_each += random_anoise
        anoise_freq_name = anoise_freq_template.format(nu_str, nside, seed)
        if re_noise or not os.path.exists(anoise_freq_name):
          if fgnoise_fac is None:
            random_freq_anoise = ReturnANoiseMap(anoise, nside, nonzero_len) 
          else:
            random_freq_anoise = ReturnANoiseMap( noise / fgnoise_fac, nside, nonzero_len) 
          if  not os.path.exists(anoise_freq_name):
            numpy.save(anoise_freq_name, random_freq_anoise)
        else:
          random_freq_anoise = numpy.load( anoise_freq_name )
          print('read old anoise freq  map')


          random_anoise = ReturnANoiseMap(anoise, nside, nonzero_len)
          if not os.path.exists(anoise_name):
            numpy.save(anoise_name, random_anoise)
 
        if fgnoise_fac is None:
            random_freq_anoise = ReturnANoiseMap(anoise, nside, nonzero_len) 
            #random_freq_anoise = numpy.random.randn(nonzero_len) * asigma
        else:
            random_freq_anoise = ReturnANoiseMap( noise / fgnoise_fac, nside, nonzero_len) 
            #noise_sigma_freq = ReturnNoiseSigma( noise / fgnoise_fac, nside)
            #random_freq_anoise = numpy.random.randn(nonzero_len) * noise_sigma_freq
        mvec_each += random_freq_anoise
        
        mvec.append(mvec_each)
    return mvec



def TestFGWithNoiseCov(freq_list, n_list, fwhm_list,nside=4, fwhm=2200., 
                           isdust = True, issynch = True, r = 1.0e-3, anoise = 2.0e-2,
                       param_defs ={'beta_s':-3.0, 'beta_d':1.5, 'T_d1':20.9}, 
                        dust_template =None, synch_template= None,
                       uni =False, fixTd = False, fixbetad = False, fgnoise_fac = None,
                           fgfac = 1.0, dmp = None, T_d1_mean = 20, beta_d_mean = 1.5, seed = 1, re_noise= False, re_cmb = False, isdust_map = None, issynch_map = None,
    ):
    maskname= './mask_p06_Nside4.v2.fits'
    anoise = anoise
    fgfac = fgfac
    nfac =1 
    nuRef = 353.0

    tmpl = templates.Templates()

    mbb1 = tmpl.ReturnMBB1()
    synch = tmpl.ReturnPowerLawSynch()
    if fixTd:
        mbb1 = mbb1.subs('T_d1', T_d1_mean)
    if fixbetad:
        mbb1 = mbb1.subs('beta_d', beta_d_mean )
    if isdust_map is None:
      isdust_map = isdust
    if issynch_map is None:
      issynch_map = issynch
    mvec = ReturnMapWithNoiseCov( maskname, anoise, fwhm, nside , r,
                     fgfac, nfac, freq_list, n_list, fwhm_list, mbb1, synch, param_defs,uni = uni, 
                     issynch = issynch_map, isdust = isdust_map,
                     re_noise = re_noise, re_cmb = re_cmb, dust_template = dust_template, 
                    synch_template=synch_template, fgnoise_fac = fgnoise_fac, seed = seed
                    )
    if dmp is not None:
      dmp.SetMvec(mvec)
      dmp.initialise()
      return dmp

    dmt = dmatrix.DMatrix()
    if isdust:
        dmt.AddD(mbb1)
    if issynch:
        dmt.AddD(synch)

    dmp = deltamap.DeltaMap(verbose= False)
    dmt.SetFreqs( freq_list, [None]*len(freq_list) )
    if uni:
        dmt.PrepareUniformDMatrix()
    else:
        dmt.PrepareDMatrix()

    cov = covariance.Covariance( nside= nside, maskname= maskname,verbose=False, pixwin=True , lmax=nside*2 , fwhm=fwhm )
    cov.Initialise()

    S0_SM = cov.ReturnCovMatrix(True)
    S0_BSM = cov.ReturnCovMatrix(False)

    asigma = ReturnNoiseSigma(anoise, nside)
    #aNoise_Cov = numpy.eye( S0_SM.shape[0] )*pow(asigma,2)
    aNoise_Cov = numpy.eye( S0_SM.shape[0] )*pow(asigma,2)
    dmp.SetS0(S0_SM + aNoise_Cov, S0_BSM)
    dmp.SetFgDmatrix(dmt)

    #noise = numpy.array(n_list)
    #sigma = ReturnNoiseSigma(noise, nside)

    Noise_list = []
    for nu,noi, beam in zip(freq_list, n_list, fwhm_list):
        noise_cov = ReturnNoiseCov(noi, nside, beam, cov, pixwin = True)
        if fgnoise_fac is None:
            Noise_list.append( noise_cov + numpy.eye(S0_SM.shape[0]) * pow( asigma, 2) )
        else:
            noise_sigma_freq = ReturnNoiseSigma( noi / fgnoise_fac, nside)
            Noise_list.append( noise_cov + numpy.eye(S0_SM.shape[0]) * pow(noise_sigma_freq,2) )
    #dmp.SetNoiseArray( sigma**2 )
    dmp.SetNoiseList(Noise_list)
    dmp.SetMvec( mvec )
    dmp.initialise()
    if isdust and issynch:
        if not 'T_d1' in param_defs.keys():
            dmp.SetParameterInitial(
            {'r':[0.0,  (0.0, 2.0) ], 'beta_d':[1.5, ( 0.1, 10)],'beta_s':[-3.47, (-10.0, -0.01)]}    
             )
        else:
            dmp.SetParameterInitial(
            {'r':[0.0,  (0.0, 2.0) ], 'T_d1':[ 20.0,(5.0, 40.0)], 'beta_d':[1.5, ( 0.1, 10)],'beta_s':[-3.47, (-10.0, -0.01)]}    
             )
    elif isdust:
        if fixTd:
            dmp.SetParameterInitial(
                {'r':[0.0,  (0.0, 2.0) ], 'beta_d':[1.5, ( 0.1, 10.0)]}    
                 )
        elif fixbetad:
            dmp.SetParameterInitial(
                {'r':[0.0,  (0.0, 2.0) ],  'T_d1':[ 20.0,(5.0, 40.0)]}    
                 )
        else:
            dmp.SetParameterInitial(
                {'r':[0.0,  (0.0, 2.0) ], 'T_d1':[ 20.0,(5.0, 40.0)], 'beta_d':[1.5, ( 0.1, 10.0)]}    
                 )
            
    elif issynch:
        dmp.SetParameterInitial( {'r':[0.0,  (0.0, 2.0) ],'beta_s':[-3.47, (-10.0, -0.01)],} )
    return dmp


def TestFGWithNoiseCovXRef(freq_list, n_list, fwhm_list,nside=4, fwhm=2200., 
                           isdust = True, issynch = True, r = 1.0e-3, anoise = 2.0e-2,
                       param_defs ={'beta_s':-3.0, 'beta_d':1.5, 'x^R':0.81}, 
                        dust_template =None, synch_template= None,
                       uni =False, fixTd = False, fixbetad = False, fgnoise_fac = None,
                           fgfac = 1.0, dmp = None, T_d1_mean = 20, beta_d_mean = 1.5, seed = 1, re_noise = False, re_cmb = False, isdust_map = None, issynch_map = None,
    ):
    maskname= './mask_p06_Nside4.v2.fits'
    anoise = anoise
    fgfac = fgfac
    nfac =1 
    nuRef = 353.0

    tmpl = templates.Templates()
    mbb1 = tmpl.ReturnMBB1_xRef()
    synch = tmpl.ReturnPowerLawSynch()
    if fixTd:
        mbb1 = mbb1.subs('x^R',(constants.h * nuRef * 1.0e9)/( T_d1_mean * constants.k ) )
    if fixbetad:
        mbb1 = mbb1.subs('beta_d', beta_d_mean )
    if isdust_map is None:
      isdust_map = isdust
    if issynch_map is None:
      issynch_map = issynch
    mvec = ReturnMapWithNoiseCov( maskname, anoise, fwhm, nside , r,
                     fgfac, nfac, freq_list, n_list, fwhm_list, mbb1, synch, param_defs,uni = uni, 
                     issynch = issynch_map, isdust = isdust_map,
                     re_noise = re_noise, re_cmb = re_cmb, dust_template = dust_template, 
                    synch_template = synch_template, fgnoise_fac = fgnoise_fac, seed = seed
                    )

    if dmp is not None:
      dmp.SetMvec(mvec)
      dmp.initialise()
      return dmp

    dmt = dmatrix.DMatrix()
    if isdust:
        dmt.AddD(mbb1)
    if issynch:
        dmt.AddD(synch)

    dmp = deltamap.DeltaMap(verbose= False)
    dmt.SetFreqs( freq_list, [None]*len(freq_list) )
    if uni:
        dmt.PrepareUniformDMatrix()
    else:
        dmt.PrepareDMatrix()

    cov = covariance.Covariance( nside= nside, maskname= maskname,verbose=False, pixwin=True , lmax=nside*2 , fwhm=fwhm )
    cov.Initialise()

    S0_SM = cov.ReturnCovMatrix(True)
    S0_BSM = cov.ReturnCovMatrix(False)
    asigma = ReturnNoiseSigma(anoise, nside)
    #aNoise_Cov = numpy.eye( S0_SM.shape[0] )*pow(asigma,2)
    aNoise_Cov = numpy.eye( S0_SM.shape[0] )*pow(asigma,2)
    dmp.SetS0(S0_SM + aNoise_Cov, S0_BSM)
    dmp.SetFgDmatrix(dmt)

    #noise = numpy.array(n_list)
    #sigma = ReturnNoiseSigma(noise, nside)

    Noise_list = []
    for nu,noi, beam in zip(freq_list, n_list, fwhm_list):
        noise_cov = ReturnNoiseCov(noi, nside, beam, cov, pixwin = True)
        if fgnoise_fac is None:
            Noise_list.append( noise_cov + numpy.eye(S0_SM.shape[0]) * pow( asigma, 2) )
        else:
            noise_sigma_freq = ReturnNoiseSigma( noi / fgnoise_fac, nside)
            Noise_list.append( noise_cov + numpy.eye(S0_SM.shape[0]) * pow(noise_sigma_freq,2) )
    #dmp.SetNoiseArray( sigma**2 )
    dmp.SetNoiseList(Noise_list)
    dmp.SetMvec( mvec )
    dmp.initialise()
    if isdust and issynch:
        if not 'x^R' in param_defs.keys():
            dmp.SetParameterInitial(
            {'r':[0.0,  (0.0, 2.0) ], 'beta_d':[1.5, ( 0.1, 10)],'beta_s':[-3.47, (-10.0, -0.01)]}    
             )
        else:
            dmp.SetParameterInitial(
            {'r':[0.0,  (0.0, 2.0) ], 'x^R':[ 0.81,(0.1, 10.0)], 'beta_d':[1.5, ( 0.1, 10)],'beta_s':[-3.47, (-10.0, -0.01)]}    
             )
    elif isdust:
        if fixTd:
            dmp.SetParameterInitial(
                {'r':[0.0,  (0.0, 2.0) ], 'beta_d':[1.5, ( 0.1, 10.0)]}    
                 )
        elif fixbetad:
            dmp.SetParameterInitial(
                {'r':[0.0,  (0.0, 2.0) ],  'x^R':[ 0.81,(0.1, 10.0)]}    
                 )
        else:
            dmp.SetParameterInitial(
                {'r':[0.0,  (0.0, 2.0) ], 'x^R':[ 0.81,(0.1, 10.0)], 'beta_d':[1.5, ( 0.1, 10.0)]}    
                 )
            
    elif issynch:
        dmp.SetParameterInitial( {'r':[0.0,  (0.0, 2.0) ],'beta_s':[-3.47, (-10.0, -0.01)],} )
    return dmp


def main():
  parser = argparse.ArgumentParser(description='hoge')
  parser.add_argument('config', help='all config file', default = './LTD_config+M0.ini' )
  parser.add_argument('fitconfig', help='fit config file', default = './configs/Dust_var_M5_4freq.ini' )
  parser.add_argument('seed', help='fit config file', type = int)

  args = parser.parse_args()

  map_parser = configparser.ConfigParser()
  map_parser.read( args.config )
  fit_parser = configparser.ConfigParser()
  fit_parser.read( args.fitconfig )
  numpy.random.seed( args.seed )

  nu_list = numpy.array([float(i) for i in map_parser.get('par','nu').split()])
  fwhm_list = numpy.array([float(i) for i in map_parser.get('par','fwhm').split()])
  alpha_list = numpy.array([float(i) for i in map_parser.get('par','alpha').split()])
  noise_list = numpy.array([float(i) for i in map_parser.get('par','noise').split()])
  nside = 4
  fwhm_norm = 2200
  try:
    nside = fit_parser.getint('par', 'nside') 
  except:
    pass
  if nside !=4:
    fwhm_norm = 2200 * pow(4./nside,2)
  try:
    dust_template = fit_parser.get('par','dust_temp')
    synch_template = fit_parser.get('par','sync_temp')
  except:
    dust_template = './PySM/output_pysm3_ns4_02/test001_nu{0}GHz_dust_nside0004.fits'
    synch_template = './PySM/output_pysm3_ns4_02/test001_nu{0}GHz_synchrotron_nside0004.fits'
  print('nside ', nside)
  print('dust_template ', dust_template)

  dust_beta_d = healpy.read_map('/gpfs/group/cmb/litebird/usr/yminami/workdir/ForeGround/PySM/template/dust_beta.fits', field = (0), verbose=False, dtype = numpy.float64)
  dust_Td1 = healpy.read_map('/gpfs/group/cmb/litebird/usr/yminami/workdir/ForeGround/PySM/template/dust_temp.fits', field = (0), verbose=False, dtype = numpy.float64)

  nu_list_fit = numpy.array([float(i) for i in fit_parser.get('par','nu').split()])

  dust_beta_d = healpy.ud_grade( map_in=dust_beta_d, nside_out= nside, order_in= 'RING', order_out = 'RING')
  dust_Td1 = healpy.ud_grade( map_in=dust_Td1, nside_out = nside, order_in= 'RING', order_out = 'RING')

  temp_freqs = nu_list_fit
  temp_noise = noise_list
  anoise = fit_parser.getfloat('par','anoise')
  if fit_parser.getfloat('par','fgnoise_fac') < 0:
    fgnoise_fac = None
  else: 
    fgnoise_fac = fit_parser.getfloat('par','fgnoise_fac') 

  fit_params = fit_parser.get('par','params').split()
  fit_inits = numpy.array([float(i) for i in fit_parser.get('par', 'inits').split()])
  params = {}
  param_defs ={}
  for par,ini in zip(fit_params, fit_inits):
    params[par] = ini
    if not par == 'r':
      param_defs[par] = ini
    noise_comb = temp_noise[numpy.isin( nu_list, temp_freqs )]
  fwhm_comb = fwhm_list[numpy.isin( nu_list, temp_freqs )]


  odir = './results4paper/{0}/'
  oname = 'num{0:04d}.npy'
  oname_r = 'num{0:04d}_r.npy'
  odir = odir.format( (args.fitconfig).replace('.ini', '').rsplit('/',1)[-1] )
  #chekc directory
  if not os.path.isdir(odir):
    os.makedirs( odir )
  oname = odir + oname.format(args.seed)
  if os.path.exists(oname):
    return 0

  dmp = None

  results = []
  if not 'x^R' in fit_params:
    dmp = TestFGWithNoiseCov(temp_freqs, noise_comb, fwhm_comb, nside= nside, fwhm = fwhm_norm, 
      isdust = fit_parser.getboolean('par','isdust'), issynch = fit_parser.getboolean('par', 'issynch'),
      r = fit_parser.getfloat('par','r'),
      uni = fit_parser.getboolean('par', 'uni'),
      param_defs = param_defs, 
      dust_template =  dust_template,
      synch_template = synch_template,
      anoise= anoise,
      fgnoise_fac= fgnoise_fac,
      fixTd = fit_parser.getboolean('par','fixTd'),
      dmp = dmp,
      T_d1_mean = dust_Td1.mean(),  beta_d_mean = dust_beta_d.mean(),
      seed =  args.seed 
      )
  else:
    dmp = TestFGWithNoiseCovXRef(temp_freqs, noise_comb, fwhm_comb, nside= nside, fwhm = fwhm_norm, 
      isdust = fit_parser.getboolean('par','isdust'), issynch = fit_parser.getboolean('par', 'issynch'),
      r = fit_parser.getfloat('par','r'),
      uni = fit_parser.getboolean('par', 'uni'),
      param_defs = param_defs, 
      dust_template =  dust_template,
      synch_template = synch_template,
      anoise= anoise,
      fgnoise_fac= fgnoise_fac,
      fixTd = fit_parser.getboolean('par','fixTd'),
      dmp = dmp,
      T_d1_mean = dust_Td1.mean(),  beta_d_mean = dust_beta_d.mean(),
      seed =  args.seed 
      )

  dmp.SetParameters(values= params)

  withTdPrior = False
  Tdsigma = None
  try:
    withTdPrior = fit_parser.getboolean('par', 'Tdprior')
  except:
    pass
  try:
    Tdsigma = fit_parser.getfloat('par', 'Tdsigma')
  except:
    Tdsigma = 3.0
  withxRPrior = False
  xRsigma = None
  try:
    withxRPrior = fit_parser.getboolean('par', 'xRprior')
  except:
    pass

  try:
    xRsigma = fit_parser.getfloat('par', 'xRsigma')
  except:
    xRsigma = 3.0

  maskname= './mask_p06_Nside4.v2.fits'
  if withTdPrior:
    mask_map = healpy.read_map(maskname, field=(0), verbose = False, nest = False, dtype = numpy.float64)
    mask_map = healpy.ud_grade(mask_map, nside_out = nside)
    dmp.withTdPrior = withTdPrior
    dmp.SetTdPrior( dust_Td1[mask_map ==1].mean(), dust_Td1[mask_map ==1].std() * Tdsigma )
  if withxRPrior:
    mask_map = healpy.read_map(maskname, field=(0), verbose = False, nest = False, dtype = numpy.float64)
    mask_map = healpy.ud_grade(mask_map, nside_out = nside)
    dmp.withxRPrior = withxRPrior
    nuRef = 353.0
    xRmean = constants.h * nuRef * 1.0e9/( constants.k *  dust_Td1[mask_map ==1].mean() )
    xRstd = constants.h * nuRef * 1.0e9/( constants.k * pow(dust_Td1[mask_map ==1].mean(), 2 )  ) * dust_Td1[mask_map ==1].std()
    print('xR \pm = {0:.2f} \pm {1:.2f}'.format(xRmean, xRstd))
    dmp.SetxRPrior( xRmean, xRstd * xRsigma )

  withmigrad = True
  try:
    withmigrad = fit_parser.getboolean('par', 'migrad')
  except:
    pass

  dmp.migrad = withmigrad
  dmp.r_verbose = False

  simul = False
  try : 
    simul = fit_parser.getboolean('par', 'simul')
  except:
    simul = False
  if not simul:
    dmp.IterateMinimize()
  else:
    r_valid = False
    n_iter = 0
    while(not r_valid and n_iter <= 10):
      parameter_initial = []
      limits = []
      err_params = []
      for param in dmp.params:
        print(param.name)
        parameter_initial.append(dmp.param_values[param.name])
        limits.append(dmp.inits[param.name][1])
        err_params.append(0.5)
      dmp.m = Minuit(
                dmp.MinimizeWithR , 
              parameter_initial    
              )
      dmp.m.limits = limits
      dmp.m.errordef = 1
      dmp.m.print_level = 1
      dmp.m.strategy = 2
      print(dmp.m.params)
      if withmigrad:
        dmp.m.migrad() 
      else:
        dmp.m.scipy()
      dmp.m.hesse()

      r_valid = dmp.m.valid
      if r_valid:
        dmp.m.minos()
      dmp.param_errors = {}
      for idx,param in enumerate(dmp.params):
        dmp.param_values[param.name] = dmp.m.params[idx].value
        dmp.param_errors[param.name] = dmp.m.params[idx].error
      #dmp.r_minos = [ dmp.m.params[0].merror]
      dmp.lh =  dmp.m.fmin.fval
      print(dmp.param_values)
      n_iter += 1

  tmp_res = []
  tmp_res.append(args.seed)
  tmp_res.append(dmp.lh)
  for key in dmp.param_values.keys():
    tmp_res.append( dmp.param_values[key] )
    tmp_res.append( dmp.param_errors[key] )
  results.append( tmp_res )
  numpy.save(oname, numpy.asarray(results )  )

  return 0

if  '__main__' == __name__:
  sys.exit( main() )
