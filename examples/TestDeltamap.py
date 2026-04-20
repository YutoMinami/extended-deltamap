from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import configparser
import importlib.resources
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import healpy
import numpy
import numpy.typing as npt
import scipy.constants.constants as constants
import sympy
from iminuit import Minuit

from extended_deltamap import Covariance, DeltaMap, DMatrix, Templates

FloatArray = npt.NDArray[numpy.float64]


class SafeDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def ResolvePath(base_dir: Path, path_text: str) -> str:
    return str((base_dir / path_text).resolve())


def ReadCell(cl_s_name: str | Path, nside: int, isScalar: bool = True) -> FloatArray:
    cl_s = numpy.loadtxt(cl_s_name)
    if len(cl_s[0]) in (4, 6) and isScalar:
        cl_s = numpy.c_[cl_s[:, :3], numpy.zeros(len(cl_s))[:, numpy.newaxis], cl_s[:, 3]]
    cls, ls = cl_s.T[1:5], cl_s.T[0]
    cl = cls * 2.0 * numpy.pi / (ls * (ls + 1.0))
    cl = numpy.c_[numpy.zeros([cl.shape[0], 2]), cl]

    return cl


def ReturnBell(ell: FloatArray, fwhm: float) -> FloatArray:
    s = 2.0
    sigma_b = (fwhm * numpy.pi / 10800.0) / numpy.sqrt(8.0 * numpy.log(2))
    return numpy.exp(-(ell * (ell + 1) - s**2) * pow(sigma_b, 2) / 2)


def ReturnNoiseSigma(noise: float, nside: int) -> float:
    npix = healpy.nside2npix(nside)
    pix_ster = 4.0 * numpy.pi / npix
    pix_amin = numpy.rad2deg(numpy.sqrt(pix_ster)) * 60.0
    sigma = noise / pix_amin
    return sigma


def ReturnCMBMap(r: float, nside: int, fwhm: float) -> FloatArray:
    data_dir = importlib.resources.files("extended_deltamap").joinpath("files")
    Cl_s = ReadCell(data_dir / "test_lensedcls_49T7H5WT3X.dat", nside, True)
    Cl_t = ReadCell(data_dir / "test_tenscls_49T7H5WT3X.dat", nside, False)
    minlen = min(len(Cl_s[1]), len(Cl_t[1]))
    cmbmap = healpy.synfast(
        Cl_s[:, :minlen] + Cl_t[:, :minlen] * r,
        lmax=nside * 2,
        nside=nside,
        new=True,
        fwhm=fwhm * numpy.pi / 10800.0,
        pixwin=True,
    )
    return cmbmap


def ReturnANoiseMap(anoise: float, nside: int, nonzero_len: int) -> FloatArray:
    asigma = ReturnNoiseSigma(anoise, nside)
    random_anoise = numpy.random.randn(nonzero_len) * asigma
    return random_anoise


def ReturnNoiseCov(
    noi: float,
    nside: int,
    beam: float,
    cov: Covariance,
    pixwin: bool = True,
) -> FloatArray:
    ell = numpy.arange(0, nside * 2 + 1, 1)
    bl_nominal = ReturnBell(ell, beam)
    pw = healpy.pixwin(lmax=nside * 2, nside=nside, pol=True)[1]
    Cell = numpy.zeros((6, nside * 2 + 1))
    noise_level = pow(noi * numpy.pi / 10800.0, 2)
    if pixwin:
        bl_nominal[pw != 0] /= pw[pw != 0]
    Cell[1] = numpy.full((1, nside * 2 + 1), noise_level) * pow(1.0 / bl_nominal, 2)
    Cell[2] = numpy.full((1, nside * 2 + 1), noise_level) * pow(1.0 / bl_nominal, 2)
    cov.CalcCov02(Cell[:, 2:])
    noise_cov = numpy.block([[cov.QQ, cov.QU], [cov.QU.T, cov.UU]])
    return noise_cov


def ReturnNoiseMap(noise: float, nside: int, beam: float, fwhm: float) -> FloatArray:
    npix = healpy.nside2npix(nside)
    noise_map = numpy.zeros(shape=(3, npix))
    sigma = ReturnNoiseSigma(noise, nside)

    random_ar = numpy.random.randn(2, npix)
    noise_map[1] = random_ar[0] * sigma
    noise_map[2] = random_ar[1] * sigma

    tmp_noise = numpy.copy(noise_map)

    alm = healpy.map2alm(tmp_noise, lmax=nside * 2, pol=True)

    noise_map = healpy.alm2map(
        alm,
        nside=nside,
        lmax=nside * 2,
        pixwin=True,
        fwhm=fwhm * numpy.sqrt(1 - pow(beam / fwhm, 2)) * numpy.pi / 10800.0,
    )[1:]
    return noise_map


def ReturnMapWithNoiseCov(
    map_parser: configparser.ConfigParser,
    maskname: str,
    anoise: float,
    fwhm: float,
    nside: int,
    r: float,
    fgfac: float,
    nfac: float,
    freqs: Sequence[float],
    noises: Sequence[float],
    fwhm_list: Sequence[float],
    dust_model: sympy.Expr,
    synch_model: sympy.Expr,
    param_defs: Mapping[str, float],
    uni: bool = False,
    isdust: bool = True,
    issynch: bool = True,
    re_noise: bool = False,
    re_cmb: bool = False,
    dust_template: str | None = None,
    synch_template: str | None = None,
    fixTd: bool = False,
    fgnoise_fac: float | None = None,
    seed: int = 1,
    input_dir: str | None = None,
) -> list[FloatArray]:
    mvec = []
    mask = healpy.read_map(maskname, field=(0), dtype=numpy.float64)
    mask = healpy.ud_grade(mask, nside_out=nside)

    params = dust_model.free_symbols
    for param in params:
        if param.name in param_defs.keys():
            dust_model = dust_model.subs(param, param_defs[param.name])
    params = synch_model.free_symbols
    for param in params:
        if param.name in param_defs.keys():
            synch_model = synch_model.subs(param, param_defs[param.name])

    if issynch and uni:
        piv_synch = float(synch_model.evalf(subs={"nu": 40}))
    if isdust and uni:
        piv_mbb1 = float(dust_model.evalf(subs={"nu": 402}))

    input_dir_value = input_dir if input_dir is not None else map_parser.get("simpar", "input_dir")
    input_dir_path = Path(input_dir_value)
    input_dir_path.mkdir(parents=True, exist_ok=True)
    noise_template = str(input_dir_path / map_parser.get("simpar", "noise_name"))
    anoise_freq_template = str(input_dir_path / map_parser.get("simpar", "anoise_freq_name"))
    anoise_template = str(input_dir_path / map_parser.get("simpar", "anoise_name"))
    cmb_template = str(input_dir_path / map_parser.get("simpar", "cmb_name"))

    synchmapname = synch_template
    dustmapname = dust_template

    nonzero_len = len(mask[mask != 0.0]) * 2

    anoise_scale = "{0:.1e}".format(anoise).replace(".", "p")
    anoise_name = anoise_template.format(nside, anoise_scale, seed)
    if re_noise or not os.path.exists(anoise_name):
        random_anoise = ReturnANoiseMap(anoise, nside, nonzero_len)
        if not os.path.exists(anoise_name):
            numpy.save(anoise_name, random_anoise)
    else:
        print("read old anoise map")
        random_anoise = numpy.load(anoise_name)

    r_scale = "{0:.1e}".format(r).replace(".", "p")
    cmb_writename = cmb_template.format(nside, int(fwhm), r_scale, seed)
    if re_cmb or not os.path.exists(cmb_writename):
        cmbmap = ReturnCMBMap(r, nside, fwhm)
        if not os.path.exists(cmb_writename):
            healpy.write_map(cmb_writename, cmbmap, nest=False)
    else:
        print("read old cmb map")
        cmbmap = healpy.read_map(cmb_writename, field=[0, 1, 2], dtype=numpy.float64)

    smoothing_fwhm = fwhm * numpy.pi / 10800.0
    dust_ref_nu = "{0:07.2f}".format(402.0).replace(".", "p")
    synch_ref_nu = "{0:07.2f}".format(40.0).replace(".", "p")

    nu_str = dust_ref_nu
    dustmap = healpy.read_map(dustmapname.format(nu=nu_str), field=(0, 1, 2), dtype=numpy.float64)
    alm = healpy.map2alm(dustmap, lmax=nside * 2, pol=True)

    dustmap = healpy.alm2map(alm, nside=nside, lmax=nside * 2, pixwin=True, fwhm=smoothing_fwhm)
    nu_str = synch_ref_nu
    synchmap = healpy.read_map(synchmapname.format(nu=nu_str), field=(0, 1, 2), dtype=numpy.float64)
    alm = healpy.map2alm(synchmap, lmax=nside * 2, pol=True)
    synchmap = healpy.alm2map(alm, nside=nside, lmax=nside * 2, pixwin=True, fwhm=smoothing_fwhm)
    for nu, noise, beam in zip(freqs, noises, fwhm_list):
        # print(nu,noise)
        fgmap = numpy.zeros_like(cmbmap)
        nu_str = "{0:07.2f}".format(nu).replace(".", "p")
        noise_scale = "{0:.3f}".format(noise).replace(".", "p")
        beam_scale = "{0:.1f}".format(beam).replace(".0", "p0")
        if uni:
            fac_mbb1 = float(dust_model.evalf(subs={"nu": nu}))
            fac_synch = float(synch_model.evalf(subs={"nu": nu}))
            fgmap = numpy.zeros_like(cmbmap)
            if isdust:
                fgmap += dustmap * fac_mbb1 / piv_mbb1
            if issynch:
                fgmap += synchmap * fac_synch / piv_synch
        else:
            dustmap = healpy.read_map(dustmapname.format(nu=nu_str), field=(0, 1, 2), dtype=numpy.float64)
            alm = healpy.map2alm(dustmap, lmax=nside * 2, pol=True)
            dustmap = healpy.alm2map(alm, nside=nside, lmax=nside * 2, pixwin=True, fwhm=smoothing_fwhm)

            synchmap = healpy.read_map(synchmapname.format(nu=nu_str), field=(0, 1, 2), dtype=numpy.float64)
            alm = healpy.map2alm(synchmap, lmax=nside * 2, pol=True)
            synchmap = healpy.alm2map(alm, nside=nside, lmax=nside * 2, pixwin=True, fwhm=smoothing_fwhm)
            if isdust:
                fgmap += dustmap
            if issynch:
                fgmap += synchmap

        inmap = cmbmap[1:] + fgfac * fgmap[1:]
        noisename = noise_template.format(
            nu_str,
            nside,
            noise_scale,
            int(fwhm),
            beam_scale,
            seed,
        )
        if re_noise or not os.path.exists(noisename):
            noise_map = ReturnNoiseMap(noise, nside, beam, fwhm)
            if not os.path.exists(noisename):
                healpy.write_map(noisename, noise_map, nest=False)
        else:
            print("read old noise map")
            noise_map = healpy.read_map(noisename, field=[0, 1], dtype=numpy.float64)

        inmap += noise_map * nfac
        mvec_each = numpy.concatenate([inmap[0][mask != 0.0], inmap[1][mask != 0.0]])
        mvec_each += random_anoise
        anoise_freq_name = anoise_freq_template.format(nu_str, nside, seed)
        if re_noise or not os.path.exists(anoise_freq_name):
            if fgnoise_fac is None:
                random_freq_anoise = ReturnANoiseMap(anoise, nside, nonzero_len)
            else:
                random_freq_anoise = ReturnANoiseMap(noise / fgnoise_fac, nside, nonzero_len)
            if not os.path.exists(anoise_freq_name):
                numpy.save(anoise_freq_name, random_freq_anoise)
        else:
            random_freq_anoise = numpy.load(anoise_freq_name)
            print("read old anoise freq  map")

            random_anoise = ReturnANoiseMap(anoise, nside, nonzero_len)
            if not os.path.exists(anoise_name):
                numpy.save(anoise_name, random_anoise)

        if fgnoise_fac is None:
            random_freq_anoise = ReturnANoiseMap(anoise, nside, nonzero_len)
            # random_freq_anoise = numpy.random.randn(nonzero_len) * asigma
        else:
            random_freq_anoise = ReturnANoiseMap(noise / fgnoise_fac, nside, nonzero_len)
            # noise_sigma_freq = ReturnNoiseSigma( noise / fgnoise_fac, nside)
            # random_freq_anoise = numpy.random.randn(nonzero_len) * noise_sigma_freq
        mvec_each += random_freq_anoise

        mvec.append(mvec_each)
    return mvec


def TestFGWithNoiseCov(
    freq_list: Sequence[float],
    n_list: Sequence[float],
    fwhm_list: Sequence[float],
    maskname: str,
    map_parser: configparser.ConfigParser,
    nside: int = 4,
    fwhm: float = 2200.0,
    isdust: bool = True,
    issynch: bool = True,
    r: float = 1.0e-3,
    anoise: float = 2.0e-2,
    param_defs: Mapping[str, float] = {"beta_s": -3.0, "beta_d": 1.5, "T_d1": 20.9},
    dust_template: str | None = None,
    synch_template: str | None = None,
    uni: bool = False,
    fixTd: bool = False,
    fixbetad: bool = False,
    fgnoise_fac: float | None = None,
    fgfac: float = 1.0,
    dmp: DeltaMap | None = None,
    T_d1_mean: float = 20,
    beta_d_mean: float = 1.5,
    seed: int = 1,
    re_noise: bool = False,
    re_cmb: bool = False,
    isdust_map: bool | None = None,
    issynch_map: bool | None = None,
    input_dir: str | None = None,
) -> DeltaMap:
    nfac = 1

    tmpl = Templates()

    mbb1 = tmpl.ReturnMBB1()
    synch = tmpl.ReturnPowerLawSynch()
    if fixTd:
        mbb1 = mbb1.subs("T_d1", T_d1_mean)
    if fixbetad:
        mbb1 = mbb1.subs("beta_d", beta_d_mean)
    if isdust_map is None:
        isdust_map = isdust
    if issynch_map is None:
        issynch_map = issynch
    mvec = ReturnMapWithNoiseCov(
        map_parser,
        maskname,
        anoise,
        fwhm,
        nside,
        r,
        fgfac,
        nfac,
        freq_list,
        n_list,
        fwhm_list,
        mbb1,
        synch,
        param_defs,
        uni=uni,
        issynch=issynch_map,
        isdust=isdust_map,
        re_noise=re_noise,
        re_cmb=re_cmb,
        dust_template=dust_template,
        synch_template=synch_template,
        fgnoise_fac=fgnoise_fac,
        seed=seed,
        input_dir=input_dir,
    )
    if dmp is not None:
        dmp.SetMvec(mvec)
        dmp.initialise()
        return dmp

    dmt = DMatrix()
    if isdust:
        dmt.AddD(mbb1)
    if issynch:
        dmt.AddD(synch)

    dmp = DeltaMap(verbose=False)
    dmt.SetFreqs(freq_list, [None] * len(freq_list))
    if uni:
        dmt.PrepareUniformDMatrix()
    else:
        dmt.PrepareDMatrix()

    cov = Covariance(nside=nside, maskname=maskname, verbose=False, pixwin=True, lmax=nside * 2, fwhm=fwhm)
    cov.Initialise()

    S0_SM = cov.ReturnCovMatrix(True)
    S0_BSM = cov.ReturnCovMatrix(False)

    asigma = ReturnNoiseSigma(anoise, nside)
    aNoise_Cov = numpy.eye(S0_SM.shape[0]) * pow(asigma, 2)
    dmp.SetS0(S0_SM + aNoise_Cov, S0_BSM)
    dmp.SetFgDmatrix(dmt)

    # noise = numpy.array(n_list)
    # sigma = ReturnNoiseSigma(noise, nside)

    Noise_list = []
    for nu, noi, beam in zip(freq_list, n_list, fwhm_list):
        noise_cov = ReturnNoiseCov(noi, nside, beam, cov, pixwin=True)
        if fgnoise_fac is None:
            Noise_list.append(noise_cov + numpy.eye(S0_SM.shape[0]) * pow(asigma, 2))
        else:
            noise_sigma_freq = ReturnNoiseSigma(noi / fgnoise_fac, nside)
            Noise_list.append(noise_cov + numpy.eye(S0_SM.shape[0]) * pow(noise_sigma_freq, 2))
    # dmp.SetNoiseArray( sigma**2 )
    dmp.SetNoiseList(Noise_list)
    dmp.SetMvec(mvec)
    dmp.initialise()
    if isdust and issynch:
        if "T_d1" not in param_defs.keys():
            dmp.SetParameterInitial({"r": [0.0, (0.0, 2.0)], "beta_d": [1.5, (0.1, 10)], "beta_s": [-3.47, (-10.0, -0.01)]})
        else:
            dmp.SetParameterInitial(
                {
                    "r": [0.0, (0.0, 2.0)],
                    "T_d1": [20.0, (5.0, 40.0)],
                    "beta_d": [1.5, (0.1, 10)],
                    "beta_s": [-3.47, (-10.0, -0.01)],
                }
            )
    elif isdust:
        if fixTd:
            dmp.SetParameterInitial({"r": [0.0, (0.0, 2.0)], "beta_d": [1.5, (0.1, 10.0)]})
        elif fixbetad:
            dmp.SetParameterInitial({"r": [0.0, (0.0, 2.0)], "T_d1": [20.0, (5.0, 40.0)]})
        else:
            dmp.SetParameterInitial({"r": [0.0, (0.0, 2.0)], "T_d1": [20.0, (5.0, 40.0)], "beta_d": [1.5, (0.1, 10.0)]})

    elif issynch:
        dmp.SetParameterInitial(
            {
                "r": [0.0, (0.0, 2.0)],
                "beta_s": [-3.47, (-10.0, -0.01)],
            }
        )
    return dmp


def TestFGWithNoiseCovXRef(
    freq_list: Sequence[float],
    n_list: Sequence[float],
    fwhm_list: Sequence[float],
    maskname: str,
    map_parser: configparser.ConfigParser,
    nside: int = 4,
    fwhm: float = 2200.0,
    isdust: bool = True,
    issynch: bool = True,
    r: float = 1.0e-3,
    anoise: float = 2.0e-2,
    param_defs: Mapping[str, float] = {"beta_s": -3.0, "beta_d": 1.5, "x^R": 0.81},
    dust_template: str | None = None,
    synch_template: str | None = None,
    uni: bool = False,
    fixTd: bool = False,
    fixbetad: bool = False,
    fgnoise_fac: float | None = None,
    fgfac: float = 1.0,
    dmp: DeltaMap | None = None,
    T_d1_mean: float = 20,
    beta_d_mean: float = 1.5,
    seed: int = 1,
    re_noise: bool = False,
    re_cmb: bool = False,
    isdust_map: bool | None = None,
    issynch_map: bool | None = None,
    input_dir: str | None = None,
) -> DeltaMap:
    nfac = 1
    nu_ref = 353.0

    tmpl = Templates()
    mbb1 = tmpl.ReturnMBB1_xRef()
    synch = tmpl.ReturnPowerLawSynch()
    if fixTd:
        mbb1 = mbb1.subs("x^R", (constants.h * nu_ref * 1.0e9) / (T_d1_mean * constants.k))
    if fixbetad:
        mbb1 = mbb1.subs("beta_d", beta_d_mean)
    if isdust_map is None:
        isdust_map = isdust
    if issynch_map is None:
        issynch_map = issynch
    mvec = ReturnMapWithNoiseCov(
        map_parser,
        maskname,
        anoise,
        fwhm,
        nside,
        r,
        fgfac,
        nfac,
        freq_list,
        n_list,
        fwhm_list,
        mbb1,
        synch,
        param_defs,
        uni=uni,
        issynch=issynch_map,
        isdust=isdust_map,
        re_noise=re_noise,
        re_cmb=re_cmb,
        dust_template=dust_template,
        synch_template=synch_template,
        fgnoise_fac=fgnoise_fac,
        seed=seed,
        input_dir=input_dir,
    )

    if dmp is not None:
        dmp.SetMvec(mvec)
        dmp.initialise()
        return dmp

    dmt = DMatrix()
    if isdust:
        dmt.AddD(mbb1)
    if issynch:
        dmt.AddD(synch)

    dmp = DeltaMap(verbose=False)
    dmt.SetFreqs(freq_list, [None] * len(freq_list))
    if uni:
        dmt.PrepareUniformDMatrix()
    else:
        dmt.PrepareDMatrix()

    cov = Covariance(nside=nside, maskname=maskname, verbose=False, pixwin=True, lmax=nside * 2, fwhm=fwhm)
    cov.Initialise()

    S0_SM = cov.ReturnCovMatrix(True)
    S0_BSM = cov.ReturnCovMatrix(False)
    asigma = ReturnNoiseSigma(anoise, nside)
    # aNoise_Cov = numpy.eye( S0_SM.shape[0] )*pow(asigma,2)
    aNoise_Cov = numpy.eye(S0_SM.shape[0]) * pow(asigma, 2)
    dmp.SetS0(S0_SM + aNoise_Cov, S0_BSM)
    dmp.SetFgDmatrix(dmt)

    # noise = numpy.array(n_list)
    # sigma = ReturnNoiseSigma(noise, nside)

    Noise_list = []
    for nu, noi, beam in zip(freq_list, n_list, fwhm_list):
        noise_cov = ReturnNoiseCov(noi, nside, beam, cov, pixwin=True)
        if fgnoise_fac is None:
            Noise_list.append(noise_cov + numpy.eye(S0_SM.shape[0]) * pow(asigma, 2))
        else:
            noise_sigma_freq = ReturnNoiseSigma(noi / fgnoise_fac, nside)
            Noise_list.append(noise_cov + numpy.eye(S0_SM.shape[0]) * pow(noise_sigma_freq, 2))
    # dmp.SetNoiseArray( sigma**2 )
    dmp.SetNoiseList(Noise_list)
    dmp.SetMvec(mvec)
    dmp.initialise()
    if isdust and issynch:
        if "x^R" not in param_defs.keys():
            dmp.SetParameterInitial({"r": [0.0, (0.0, 2.0)], "beta_d": [1.5, (0.1, 10)], "beta_s": [-3.47, (-10.0, -0.01)]})
        else:
            dmp.SetParameterInitial(
                {
                    "r": [0.0, (0.0, 2.0)],
                    "x^R": [0.81, (0.1, 10.0)],
                    "beta_d": [1.5, (0.1, 10)],
                    "beta_s": [-3.47, (-10.0, -0.01)],
                }
            )
    elif isdust:
        if fixTd:
            dmp.SetParameterInitial({"r": [0.0, (0.0, 2.0)], "beta_d": [1.5, (0.1, 10.0)]})
        elif fixbetad:
            dmp.SetParameterInitial({"r": [0.0, (0.0, 2.0)], "x^R": [0.81, (0.1, 10.0)]})
        else:
            dmp.SetParameterInitial({"r": [0.0, (0.0, 2.0)], "x^R": [0.81, (0.1, 10.0)], "beta_d": [1.5, (0.1, 10.0)]})

    elif issynch:
        dmp.SetParameterInitial(
            {
                "r": [0.0, (0.0, 2.0)],
                "beta_s": [-3.47, (-10.0, -0.01)],
            }
        )
    return dmp


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="hoge")
    parser.add_argument("config", help="all config file", default="./LTD_config+M0.ini")
    parser.add_argument("fitconfig", help="fit config file", default="./configs/Dust_var_M5_4freq.ini")
    parser.add_argument("seed", help="fit config file", type=int)

    args = parser.parse_args(args=None if argv is None else list(argv)[1:])

    config_path = Path(args.config).resolve()
    fitconfig_path = Path(args.fitconfig).resolve()
    config_dir = config_path.parent
    fitconfig_dir = fitconfig_path.parent

    map_parser = configparser.ConfigParser()
    map_parser.read(config_path)
    fit_parser = configparser.ConfigParser()
    fit_parser.read(fitconfig_path)
    numpy.random.seed(args.seed)

    nu_list = numpy.array([float(i) for i in map_parser.get("par", "nu").split()])
    fwhm_list = numpy.array([float(i) for i in map_parser.get("par", "fwhm").split()])
    noise_list = numpy.array([float(i) for i in map_parser.get("par", "noise").split()])
    nside = fit_parser.getint("par", "nside")

    fwhm_norm = 2200.0
    if nside != 4:
        fwhm_norm = 2200.0 * pow(4.0 / nside, 2)

    dust_template_pattern = fit_parser.get("par", "dust_temp")
    synch_template_pattern = fit_parser.get("par", "synch_temp")
    dust_template = ResolvePath(fitconfig_dir, dust_template_pattern.format_map(SafeDict(nside=nside)))
    synch_template = ResolvePath(fitconfig_dir, synch_template_pattern.format_map(SafeDict(nside=nside)))

    dust_beta_env = "DELTAMAP_DUST_BETA_MAP"
    dust_temp_env = "DELTAMAP_DUST_TEMP_MAP"
    dust_beta_path = os.environ.get(dust_beta_env)
    dust_temp_path = os.environ.get(dust_temp_env)
    if dust_beta_path is None or dust_temp_path is None:
        raise FileNotFoundError(f"Set {dust_beta_env} and {dust_temp_env} to external template FITS files.")
    dust_beta_path = str(Path(dust_beta_path).expanduser().resolve())
    dust_temp_path = str(Path(dust_temp_path).expanduser().resolve())
    dust_beta_d = healpy.read_map(dust_beta_path, field=(0), dtype=numpy.float64)
    dust_Td1 = healpy.read_map(dust_temp_path, field=(0), dtype=numpy.float64)

    nu_list_fit = numpy.array([float(i) for i in fit_parser.get("par", "nu").split()])

    dust_beta_d = healpy.ud_grade(map_in=dust_beta_d, nside_out=nside, order_in="RING", order_out="RING")
    dust_Td1 = healpy.ud_grade(map_in=dust_Td1, nside_out=nside, order_in="RING", order_out="RING")

    temp_freqs = nu_list_fit
    temp_noise = noise_list
    anoise = fit_parser.getfloat("par", "anoise")
    if fit_parser.getfloat("par", "fgnoise_fac") < 0:
        fgnoise_fac = None
    else:
        fgnoise_fac = fit_parser.getfloat("par", "fgnoise_fac")

    fit_params = fit_parser.get("par", "params").split()
    fit_inits = numpy.array([float(i) for i in fit_parser.get("par", "inits").split()])
    params = {}
    param_defs = {}
    for par, ini in zip(fit_params, fit_inits):
        params[par] = ini
        if not par == "r":
            param_defs[par] = ini
        noise_comb = temp_noise[numpy.isin(nu_list, temp_freqs)]
    fwhm_comb = fwhm_list[numpy.isin(nu_list, temp_freqs)]

    input_dir_setting = map_parser.get("simpar", "input_dir")
    output_dir_setting = fit_parser.get("par", "odir")
    output_name_setting = fit_parser.get("par", "oname")

    input_dir = ResolvePath(config_dir, input_dir_setting)
    odir = Path(ResolvePath(fitconfig_dir, output_dir_setting.format(fitconfig_path.stem)))
    oname = odir / output_name_setting.format(args.seed)
    odir.mkdir(parents=True, exist_ok=True)
    if oname.exists():
        return 0

    maskname = ResolvePath(config_dir, map_parser.get("simpar", "maskname"))

    dmp = None

    results = []
    if "x^R" not in fit_params:
        dmp = TestFGWithNoiseCov(
            temp_freqs,
            noise_comb,
            fwhm_comb,
            maskname,
            map_parser,
            nside=nside,
            fwhm=fwhm_norm,
            isdust=fit_parser.getboolean("par", "isdust"),
            issynch=fit_parser.getboolean("par", "issynch"),
            r=fit_parser.getfloat("par", "r"),
            uni=fit_parser.getboolean("par", "uni"),
            param_defs=param_defs,
            dust_template=dust_template,
            synch_template=synch_template,
            anoise=anoise,
            fgnoise_fac=fgnoise_fac,
            fixTd=fit_parser.getboolean("par", "fixTd"),
            dmp=dmp,
            T_d1_mean=dust_Td1.mean(),
            beta_d_mean=dust_beta_d.mean(),
            seed=args.seed,
            input_dir=input_dir,
        )
    else:
        dmp = TestFGWithNoiseCovXRef(
            temp_freqs,
            noise_comb,
            fwhm_comb,
            maskname,
            map_parser,
            nside=nside,
            fwhm=fwhm_norm,
            isdust=fit_parser.getboolean("par", "isdust"),
            issynch=fit_parser.getboolean("par", "issynch"),
            r=fit_parser.getfloat("par", "r"),
            uni=fit_parser.getboolean("par", "uni"),
            param_defs=param_defs,
            dust_template=dust_template,
            synch_template=synch_template,
            anoise=anoise,
            fgnoise_fac=fgnoise_fac,
            fixTd=fit_parser.getboolean("par", "fixTd"),
            dmp=dmp,
            T_d1_mean=dust_Td1.mean(),
            beta_d_mean=dust_beta_d.mean(),
            seed=args.seed,
            input_dir=input_dir,
        )

    dmp.SetParameters(values=params)

    withTdPrior = False
    Tdsigma = None
    try:
        withTdPrior = fit_parser.getboolean("par", "Tdprior")
    except:
        pass
    try:
        Tdsigma = fit_parser.getfloat("par", "Tdsigma")
    except:
        Tdsigma = 3.0
    withxRPrior = False
    xRsigma = None
    try:
        withxRPrior = fit_parser.getboolean("par", "xRprior")
    except:
        pass

    try:
        xRsigma = fit_parser.getfloat("par", "xRsigma")
    except:
        xRsigma = 3.0

    if withTdPrior:
        mask_map = healpy.read_map(maskname, field=(0), nest=False, dtype=numpy.float64)
        mask_map = healpy.ud_grade(mask_map, nside_out=nside)
        dmp.withTdPrior = withTdPrior
        dmp.SetTdPrior(dust_Td1[mask_map == 1].mean(), dust_Td1[mask_map == 1].std() * Tdsigma)
    if withxRPrior:
        mask_map = healpy.read_map(maskname, field=(0), nest=False, dtype=numpy.float64)
        mask_map = healpy.ud_grade(mask_map, nside_out=nside)
        dmp.withxRPrior = withxRPrior
        nuRef = 353.0
        xRmean = constants.h * nuRef * 1.0e9 / (constants.k * dust_Td1[mask_map == 1].mean())
        xRstd = (
            constants.h
            * nuRef
            * 1.0e9
            / (constants.k * pow(dust_Td1[mask_map == 1].mean(), 2))
            * dust_Td1[mask_map == 1].std()
        )
        print("xR +/- = {0:.2f} +/- {1:.2f}".format(xRmean, xRstd))
        dmp.SetxRPrior(xRmean, xRstd * xRsigma)

    withmigrad = True
    try:
        withmigrad = fit_parser.getboolean("par", "migrad")
    except:
        pass

    dmp.migrad = withmigrad
    dmp.r_verbose = False

    simul = False
    try:
        simul = fit_parser.getboolean("par", "simul")
    except:
        simul = False
    if not simul:
        dmp.IterateMinimize()
    else:
        r_valid = False
        n_iter = 0
        while not r_valid and n_iter <= 10:
            parameter_initial = []
            limits = []
            err_params = []
            for param in dmp.params:
                print(param.name)
                parameter_initial.append(dmp.param_values[param.name])
                limits.append(dmp.inits[param.name][1])
                err_params.append(0.5)
            dmp.m = Minuit(dmp.MinimizeWithR, parameter_initial)
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
            for idx, param in enumerate(dmp.params):
                dmp.param_values[param.name] = dmp.m.params[idx].value
                dmp.param_errors[param.name] = dmp.m.params[idx].error
                dmp.lh = dmp.m.fmin.fval
            print(dmp.param_values)
            n_iter += 1

    tmp_res = []
    tmp_res.append(args.seed)
    tmp_res.append(dmp.lh)
    for key in dmp.param_values.keys():
        tmp_res.append(dmp.param_values[key])
        tmp_res.append(dmp.param_errors[key])
    results.append(tmp_res)
    numpy.save(oname, numpy.asarray(results))

    return 0


if "__main__" == __name__:
    sys.exit(main())
