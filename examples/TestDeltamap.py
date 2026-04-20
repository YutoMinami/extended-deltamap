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
    """Dictionary that leaves unknown format keys untouched."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def resolve_path(base_dir: Path, path_text: str) -> str:
    """Resolve a config-relative path to an absolute string path.

    Args:
        base_dir: Directory used as the reference point.
        path_text: Relative or absolute path text from configuration.

    Returns:
        The resolved absolute path.
    """
    return str((base_dir / path_text).resolve())


def read_cell(cl_s_name: str | Path, nside: int, is_scalar: bool = True) -> FloatArray:
    """Load CMB power spectra from a CAMB-style text file.

    Args:
        cl_s_name: Input spectrum filename.
        nside: HEALPix nside used by the caller.
        is_scalar: Whether the file stores scalar spectra with the expected
            column layout.

    Returns:
        A spectrum array padded for direct use with ``healpy.synfast``.
    """
    cl_s = numpy.loadtxt(cl_s_name)
    if len(cl_s[0]) in (4, 6) and is_scalar:
        cl_s = numpy.c_[cl_s[:, :3], numpy.zeros(len(cl_s))[:, numpy.newaxis], cl_s[:, 3]]
    cls, ls = cl_s.T[1:5], cl_s.T[0]
    cl = cls * 2.0 * numpy.pi / (ls * (ls + 1.0))
    cl = numpy.c_[numpy.zeros([cl.shape[0], 2]), cl]

    return cl


def return_bell(ell: FloatArray, fwhm: float) -> FloatArray:
    """Return the spin-2 Gaussian beam transfer function.

    Args:
        ell: Multipole values.
        fwhm: Beam full width at half maximum in arcminutes.

    Returns:
        Beam response evaluated at each multipole.
    """
    s = 2.0
    sigma_b = (fwhm * numpy.pi / 10800.0) / numpy.sqrt(8.0 * numpy.log(2))
    return numpy.exp(-(ell * (ell + 1) - s**2) * pow(sigma_b, 2) / 2)


def return_noise_sigma(noise: float, nside: int) -> float:
    """Convert map noise in uK-arcmin to pixel-domain sigma.

    Args:
        noise: Polarization noise level in uK-arcmin.
        nside: HEALPix nside of the target map.

    Returns:
        Per-pixel Gaussian sigma in map units.
    """
    npix = healpy.nside2npix(nside)
    pix_ster = 4.0 * numpy.pi / npix
    pix_amin = numpy.rad2deg(numpy.sqrt(pix_ster)) * 60.0
    sigma = noise / pix_amin
    return sigma


def return_cmb_map(r: float, nside: int, fwhm: float) -> FloatArray:
    """Simulate a smoothed CMB map with scalar and tensor contributions.

    Args:
        r: Tensor-to-scalar ratio used for the tensor spectrum amplitude.
        nside: Target HEALPix nside.
        fwhm: Smoothing scale in arcminutes.

    Returns:
        A three-component ``healpy.synfast`` map.
    """
    data_dir = importlib.resources.files("extended_deltamap").joinpath("files")
    cl_scalar = read_cell(data_dir / "test_lensedcls_49T7H5WT3X.dat", nside, True)
    cl_tensor = read_cell(data_dir / "test_tenscls_49T7H5WT3X.dat", nside, False)
    minlen = min(len(cl_scalar[1]), len(cl_tensor[1]))
    cmbmap = healpy.synfast(
        cl_scalar[:, :minlen] + cl_tensor[:, :minlen] * r,
        lmax=nside * 2,
        nside=nside,
        new=True,
        fwhm=fwhm * numpy.pi / 10800.0,
        pixwin=True,
    )
    return cmbmap


def return_anoise_map(anoise: float, nside: int, nonzero_len: int) -> FloatArray:
    """Generate additional white noise samples for unmasked pixels.

    Args:
        anoise: Additional noise level in uK-arcmin.
        nside: HEALPix nside of the target map.
        nonzero_len: Number of unmasked Q/U samples.

    Returns:
        One-dimensional Gaussian noise samples for concatenated Q/U data.
    """
    asigma = return_noise_sigma(anoise, nside)
    random_anoise = numpy.random.randn(nonzero_len) * asigma
    return random_anoise


def return_noise_cov(
    noi: float,
    nside: int,
    beam: float,
    cov: Covariance,
    pixwin: bool = True,
) -> FloatArray:
    """Build a pixel-space covariance matrix for white noise and beam smoothing.

    Args:
        noi: Frequency-channel noise level in uK-arcmin.
        nside: HEALPix nside of the covariance grid.
        beam: Instrument beam width in arcminutes.
        cov: Working covariance object used to project spectra into pixel space.
        pixwin: Whether to divide out the HEALPix pixel window before projection.

    Returns:
        Block covariance for concatenated Q/U pixels.
    """
    ell = numpy.arange(0, nside * 2 + 1, 1)
    bl_nominal = return_bell(ell, beam)
    pw = healpy.pixwin(lmax=nside * 2, nside=nside, pol=True)[1]
    cell = numpy.zeros((6, nside * 2 + 1))
    noise_level = pow(noi * numpy.pi / 10800.0, 2)
    if pixwin:
        bl_nominal[pw != 0] /= pw[pw != 0]
    cell[1] = numpy.full((1, nside * 2 + 1), noise_level) * pow(1.0 / bl_nominal, 2)
    cell[2] = numpy.full((1, nside * 2 + 1), noise_level) * pow(1.0 / bl_nominal, 2)
    cov.CalcCov02(cell[:, 2:])
    noise_cov = numpy.block([[cov.QQ, cov.QU], [cov.QU.T, cov.UU]])
    return noise_cov


def return_noise_map(noise: float, nside: int, beam: float, fwhm: float) -> FloatArray:
    """Generate a smoothed noise realization for Q/U maps.

    Args:
        noise: Frequency-channel noise level in uK-arcmin.
        nside: HEALPix nside of the target map.
        beam: Native beam width in arcminutes.
        fwhm: Final smoothing width in arcminutes.

    Returns:
        A two-component Q/U noise map.
    """
    npix = healpy.nside2npix(nside)
    noise_map = numpy.zeros(shape=(3, npix))
    sigma = return_noise_sigma(noise, nside)

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


def return_map_with_noise_cov(
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
    """Assemble simulated frequency maps for DeltaMap fitting.

    Args:
        map_parser: Simulation config parser.
        maskname: Mask FITS path.
        anoise: Additional shared noise level in uK-arcmin.
        fwhm: Common smoothing width in arcminutes.
        nside: Target HEALPix nside.
        r: Tensor-to-scalar ratio for the CMB simulation.
        fgfac: Foreground amplitude multiplier.
        nfac: Multiplicative factor for per-frequency noise maps.
        freqs: Frequency channels in GHz.
        noises: Per-channel noise levels in uK-arcmin.
        fwhm_list: Per-channel beam widths in arcminutes.
        dust_model: Symbolic dust scaling model.
        synch_model: Symbolic synchrotron scaling model.
        param_defs: Fixed foreground parameter values.
        uni: Whether to use template scaling instead of per-frequency templates.
        isdust: Whether to include dust emission.
        issynch: Whether to include synchrotron emission.
        re_noise: Whether to regenerate saved noise realizations.
        re_cmb: Whether to regenerate the saved CMB realization.
        dust_template: Dust template filename pattern.
        synch_template: Synchrotron template filename pattern.
        fixTd: Reserved compatibility flag for fixed dust temperature runs.
        fgnoise_fac: Optional rescaling for fitting-frequency additional noise.
        seed: Seed suffix used in cached filenames.
        input_dir: Optional override for the generated input directory.

    Returns:
        Concatenated masked Q/U vectors, one per frequency channel.
    """
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
        random_anoise = return_anoise_map(anoise, nside, nonzero_len)
        if not os.path.exists(anoise_name):
            numpy.save(anoise_name, random_anoise)
    else:
        print("read old anoise map")
        random_anoise = numpy.load(anoise_name)

    r_scale = "{0:.1e}".format(r).replace(".", "p")
    cmb_writename = cmb_template.format(nside, int(fwhm), r_scale, seed)
    if re_cmb or not os.path.exists(cmb_writename):
        cmbmap = return_cmb_map(r, nside, fwhm)
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
            noise_map = return_noise_map(noise, nside, beam, fwhm)
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
                random_freq_anoise = return_anoise_map(anoise, nside, nonzero_len)
            else:
                random_freq_anoise = return_anoise_map(noise / fgnoise_fac, nside, nonzero_len)
            if not os.path.exists(anoise_freq_name):
                numpy.save(anoise_freq_name, random_freq_anoise)
        else:
            random_freq_anoise = numpy.load(anoise_freq_name)
            print("read old anoise freq  map")

            random_anoise = return_anoise_map(anoise, nside, nonzero_len)
            if not os.path.exists(anoise_name):
                numpy.save(anoise_name, random_anoise)

        if fgnoise_fac is None:
            random_freq_anoise = return_anoise_map(anoise, nside, nonzero_len)
            # random_freq_anoise = numpy.random.randn(nonzero_len) * asigma
        else:
            random_freq_anoise = return_anoise_map(noise / fgnoise_fac, nside, nonzero_len)
            # noise_sigma_freq = return_noise_sigma(noise / fgnoise_fac, nside)
            # random_freq_anoise = numpy.random.randn(nonzero_len) * noise_sigma_freq
        mvec_each += random_freq_anoise

        mvec.append(mvec_each)
    return mvec


def test_fg_with_noise_cov(
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
    """Prepare a DeltaMap fit using the standard dust-temperature parameterization.

    Args:
        freq_list: Fitting frequencies in GHz.
        n_list: Per-frequency noise levels in uK-arcmin.
        fwhm_list: Per-frequency beam widths in arcminutes.
        maskname: Mask FITS path used to select valid pixels.
        map_parser: Simulation config parser.
        nside: HEALPix nside for simulation and fitting.
        fwhm: Common smoothing width in arcminutes.
        isdust: Whether the fit model includes dust.
        issynch: Whether the fit model includes synchrotron.
        r: Tensor-to-scalar ratio used for the simulated CMB signal.
        anoise: Shared additional noise level in uK-arcmin.
        param_defs: Fixed foreground parameter values substituted into the model.
        dust_template: Dust template filename pattern.
        synch_template: Synchrotron template filename pattern.
        uni: Whether to reuse scaled reference templates across frequencies.
        fixTd: Whether to fix the dust temperature during fitting.
        fixbetad: Whether to fix the dust spectral index during fitting.
        fgnoise_fac: Optional rescaling for fitting-frequency additional noise.
        fgfac: Overall foreground amplitude multiplier for simulations.
        dmp: Existing DeltaMap instance to reuse.
        T_d1_mean: Mean dust temperature used when ``fixTd`` is enabled.
        beta_d_mean: Mean dust spectral index used when ``fixbetad`` is enabled.
        seed: Seed suffix used in cached simulation products.
        re_noise: Whether to regenerate cached noise realizations.
        re_cmb: Whether to regenerate the cached CMB realization.
        isdust_map: Optional override for including dust in the simulated maps.
        issynch_map: Optional override for including synchrotron in the simulated maps.
        input_dir: Optional override for the cached input directory.

    Returns:
        An initialized ``DeltaMap`` instance ready for minimization.
    """
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
    mvec = return_map_with_noise_cov(
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

    s0_sm = cov.ReturnCovMatrix(True)
    s0_bsm = cov.ReturnCovMatrix(False)

    asigma = return_noise_sigma(anoise, nside)
    a_noise_cov = numpy.eye(s0_sm.shape[0]) * pow(asigma, 2)
    dmp.SetS0(s0_sm + a_noise_cov, s0_bsm)
    dmp.SetFgDmatrix(dmt)

    base_noise_diag = numpy.eye(s0_sm.shape[0]) * pow(asigma, 2)
    noise_list = []
    for nu, noi, beam in zip(freq_list, n_list, fwhm_list):
        noise_cov = return_noise_cov(noi, nside, beam, cov, pixwin=True)
        if fgnoise_fac is None:
            noise_total = noise_cov + base_noise_diag
        else:
            noise_sigma_freq = return_noise_sigma(noi / fgnoise_fac, nside)
            freq_noise_diag = numpy.eye(s0_sm.shape[0]) * pow(noise_sigma_freq, 2)
            noise_total = noise_cov + freq_noise_diag
        noise_list.append(noise_total)
    dmp.SetNoiseList(noise_list)
    dmp.SetMvec(mvec)
    dmp.initialise()
    initial_params: dict[str, list[float | tuple[float, float]]]
    if isdust and issynch:
        if "T_d1" not in param_defs.keys():
            initial_params = {"r": [0.0, (0.0, 2.0)], "beta_d": [1.5, (0.1, 10)], "beta_s": [-3.47, (-10.0, -0.01)]}
        else:
            initial_params = {
                "r": [0.0, (0.0, 2.0)],
                "T_d1": [20.0, (5.0, 40.0)],
                "beta_d": [1.5, (0.1, 10)],
                "beta_s": [-3.47, (-10.0, -0.01)],
            }
    elif isdust:
        if fixTd:
            initial_params = {"r": [0.0, (0.0, 2.0)], "beta_d": [1.5, (0.1, 10.0)]}
        elif fixbetad:
            initial_params = {"r": [0.0, (0.0, 2.0)], "T_d1": [20.0, (5.0, 40.0)]}
        else:
            initial_params = {"r": [0.0, (0.0, 2.0)], "T_d1": [20.0, (5.0, 40.0)], "beta_d": [1.5, (0.1, 10.0)]}
    elif issynch:
        initial_params = {"r": [0.0, (0.0, 2.0)], "beta_s": [-3.47, (-10.0, -0.01)]}
    else:
        initial_params = {"r": [0.0, (0.0, 2.0)]}
    dmp.SetParameterInitial(initial_params)
    return dmp


def test_fg_with_noise_cov_xref(
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
    """Prepare a DeltaMap fit using the ``x^R`` dust parameterization.

    Args:
        freq_list: Fitting frequencies in GHz.
        n_list: Per-frequency noise levels in uK-arcmin.
        fwhm_list: Per-frequency beam widths in arcminutes.
        maskname: Mask FITS path used to select valid pixels.
        map_parser: Simulation config parser.
        nside: HEALPix nside for simulation and fitting.
        fwhm: Common smoothing width in arcminutes.
        isdust: Whether the fit model includes dust.
        issynch: Whether the fit model includes synchrotron.
        r: Tensor-to-scalar ratio used for the simulated CMB signal.
        anoise: Shared additional noise level in uK-arcmin.
        param_defs: Fixed foreground parameter values substituted into the model.
        dust_template: Dust template filename pattern.
        synch_template: Synchrotron template filename pattern.
        uni: Whether to reuse scaled reference templates across frequencies.
        fixTd: Whether to fix the dust temperature prior through ``x^R``.
        fixbetad: Whether to fix the dust spectral index during fitting.
        fgnoise_fac: Optional rescaling for fitting-frequency additional noise.
        fgfac: Overall foreground amplitude multiplier for simulations.
        dmp: Existing DeltaMap instance to reuse.
        T_d1_mean: Mean dust temperature used when ``fixTd`` is enabled.
        beta_d_mean: Mean dust spectral index used when ``fixbetad`` is enabled.
        seed: Seed suffix used in cached simulation products.
        re_noise: Whether to regenerate cached noise realizations.
        re_cmb: Whether to regenerate the cached CMB realization.
        isdust_map: Optional override for including dust in the simulated maps.
        issynch_map: Optional override for including synchrotron in the simulated maps.
        input_dir: Optional override for the cached input directory.

    Returns:
        An initialized ``DeltaMap`` instance ready for minimization.
    """
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
    mvec = return_map_with_noise_cov(
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

    s0_sm = cov.ReturnCovMatrix(True)
    s0_bsm = cov.ReturnCovMatrix(False)
    asigma = return_noise_sigma(anoise, nside)
    # a_noise_cov = numpy.eye(s0_sm.shape[0]) * pow(asigma, 2)
    a_noise_cov = numpy.eye(s0_sm.shape[0]) * pow(asigma, 2)
    dmp.SetS0(s0_sm + a_noise_cov, s0_bsm)
    dmp.SetFgDmatrix(dmt)

    base_noise_diag = numpy.eye(s0_sm.shape[0]) * pow(asigma, 2)
    noise_list = []
    for nu, noi, beam in zip(freq_list, n_list, fwhm_list):
        noise_cov = return_noise_cov(noi, nside, beam, cov, pixwin=True)
        if fgnoise_fac is None:
            noise_total = noise_cov + base_noise_diag
        else:
            noise_sigma_freq = return_noise_sigma(noi / fgnoise_fac, nside)
            freq_noise_diag = numpy.eye(s0_sm.shape[0]) * pow(noise_sigma_freq, 2)
            noise_total = noise_cov + freq_noise_diag
        noise_list.append(noise_total)
    dmp.SetNoiseList(noise_list)
    dmp.SetMvec(mvec)
    dmp.initialise()
    initial_params: dict[str, list[float | tuple[float, float]]]
    if isdust and issynch:
        if "x^R" not in param_defs.keys():
            initial_params = {"r": [0.0, (0.0, 2.0)], "beta_d": [1.5, (0.1, 10)], "beta_s": [-3.47, (-10.0, -0.01)]}
        else:
            initial_params = {
                "r": [0.0, (0.0, 2.0)],
                "x^R": [0.81, (0.1, 10.0)],
                "beta_d": [1.5, (0.1, 10)],
                "beta_s": [-3.47, (-10.0, -0.01)],
            }
    elif isdust:
        if fixTd:
            initial_params = {"r": [0.0, (0.0, 2.0)], "beta_d": [1.5, (0.1, 10.0)]}
        elif fixbetad:
            initial_params = {"r": [0.0, (0.0, 2.0)], "x^R": [0.81, (0.1, 10.0)]}
        else:
            initial_params = {"r": [0.0, (0.0, 2.0)], "x^R": [0.81, (0.1, 10.0)], "beta_d": [1.5, (0.1, 10.0)]}
    elif issynch:
        initial_params = {"r": [0.0, (0.0, 2.0)], "beta_s": [-3.47, (-10.0, -0.01)]}
    else:
        initial_params = {"r": [0.0, (0.0, 2.0)]}
    dmp.SetParameterInitial(initial_params)
    return dmp


def main(argv: Sequence[str] | None = None) -> int:
    """Run the example DeltaMap fit workflow from configuration files.

    Args:
        argv: Command-line style arguments. When omitted, ``sys.argv`` is used.

    Returns:
        Zero when the fit completes or the output already exists.
    """
    parser = argparse.ArgumentParser(description="Run the DeltaMap fitting example")
    parser.add_argument("config", help="simulation config file", default="./LTD_config+M0.ini")
    parser.add_argument("fitconfig", help="fit config file", default="./configs/Dust_var_M5_4freq.ini")
    parser.add_argument("seed", help="random seed for this fit run", type=int)

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
    dust_template = resolve_path(fitconfig_dir, dust_template_pattern.format_map(SafeDict(nside=nside)))
    synch_template = resolve_path(fitconfig_dir, synch_template_pattern.format_map(SafeDict(nside=nside)))

    dust_beta_env = "DELTAMAP_DUST_BETA_MAP"
    dust_temp_env = "DELTAMAP_DUST_TEMP_MAP"
    dust_beta_path = os.environ.get(dust_beta_env)
    dust_temp_path = os.environ.get(dust_temp_env)
    if dust_beta_path is None or dust_temp_path is None:
        raise FileNotFoundError(f"Set {dust_beta_env} and {dust_temp_env} to external template FITS files.")
    dust_beta_path = str(Path(dust_beta_path).expanduser().resolve())
    dust_temp_path = str(Path(dust_temp_path).expanduser().resolve())
    dust_beta_d = healpy.read_map(dust_beta_path, field=(0), dtype=numpy.float64)
    dust_td1 = healpy.read_map(dust_temp_path, field=(0), dtype=numpy.float64)

    nu_list_fit = numpy.array([float(i) for i in fit_parser.get("par", "nu").split()])

    dust_beta_d = healpy.ud_grade(map_in=dust_beta_d, nside_out=nside, order_in="RING", order_out="RING")
    dust_td1 = healpy.ud_grade(map_in=dust_td1, nside_out=nside, order_in="RING", order_out="RING")

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

    input_dir = resolve_path(config_dir, input_dir_setting)
    odir = Path(resolve_path(fitconfig_dir, output_dir_setting.format(fitconfig_path.stem)))
    oname = odir / output_name_setting.format(args.seed)
    odir.mkdir(parents=True, exist_ok=True)
    if oname.exists():
        return 0

    maskname = resolve_path(config_dir, map_parser.get("simpar", "maskname"))

    dmp = None

    if "x^R" not in fit_params:
        dmp = test_fg_with_noise_cov(
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
            T_d1_mean=dust_td1.mean(),
            beta_d_mean=dust_beta_d.mean(),
            seed=args.seed,
            input_dir=input_dir,
        )
    else:
        dmp = test_fg_with_noise_cov_xref(
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
            T_d1_mean=dust_td1.mean(),
            beta_d_mean=dust_beta_d.mean(),
            seed=args.seed,
            input_dir=input_dir,
        )

    dmp.SetParameters(values=params)

    with_td_prior = False
    td_sigma = None
    try:
        with_td_prior = fit_parser.getboolean("par", "Tdprior")
    except (configparser.NoOptionError, ValueError):
        pass
    try:
        td_sigma = fit_parser.getfloat("par", "Tdsigma")
    except (configparser.NoOptionError, ValueError):
        td_sigma = 3.0
    with_xr_prior = False
    xr_sigma = None
    try:
        with_xr_prior = fit_parser.getboolean("par", "xRprior")
    except (configparser.NoOptionError, ValueError):
        pass

    try:
        xr_sigma = fit_parser.getfloat("par", "xRsigma")
    except (configparser.NoOptionError, ValueError):
        xr_sigma = 3.0

    if with_td_prior:
        mask_map = healpy.read_map(maskname, field=(0), nest=False, dtype=numpy.float64)
        mask_map = healpy.ud_grade(mask_map, nside_out=nside)
        dmp.withTdPrior = with_td_prior
        dmp.SetTdPrior(dust_td1[mask_map == 1].mean(), dust_td1[mask_map == 1].std() * td_sigma)
    if with_xr_prior:
        mask_map = healpy.read_map(maskname, field=(0), nest=False, dtype=numpy.float64)
        mask_map = healpy.ud_grade(mask_map, nside_out=nside)
        dmp.withxRPrior = with_xr_prior
        nu_ref = 353.0
        xr_mean = constants.h * nu_ref * 1.0e9 / (constants.k * dust_td1[mask_map == 1].mean())
        xr_std = (
            constants.h
            * nu_ref
            * 1.0e9
            / (constants.k * pow(dust_td1[mask_map == 1].mean(), 2))
            * dust_td1[mask_map == 1].std()
        )
        print("xR +/- = {0:.2f} +/- {1:.2f}".format(xr_mean, xr_std))
        dmp.SetxRPrior(xr_mean, xr_std * xr_sigma)

    with_migrad = True
    try:
        with_migrad = fit_parser.getboolean("par", "migrad")
    except (configparser.NoOptionError, ValueError):
        pass

    dmp.migrad = with_migrad
    dmp.r_verbose = False

    simul = False
    try:
        simul = fit_parser.getboolean("par", "simul")
    except (configparser.NoOptionError, ValueError):
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
            if with_migrad:
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

    result_row = [args.seed, dmp.lh]
    for key in dmp.param_values.keys():
        result_row.append(dmp.param_values[key])
        result_row.append(dmp.param_errors[key])
    numpy.save(oname, numpy.asarray([result_row]))

    return 0


if "__main__" == __name__:
    sys.exit(main())
