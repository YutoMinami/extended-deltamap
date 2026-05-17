from __future__ import annotations

import csv
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

from extended_deltamap import (
    Covariance,
    DeltaMap,
    DMatrix,
    Templates,
    expand_to_qu,
    validate_region_masks,
)

FloatArray = npt.NDArray[numpy.float64]


class SafeDict(dict[str, Any]):
    """Dictionary that leaves unknown format keys untouched."""

    def __missing__(self, key: str) -> str:
        """Return the original placeholder text for unknown format keys."""
        return "{" + key + "}"


def get_config_value(parser: configparser.ConfigParser, *candidates: tuple[str, str]) -> str:
    """Read the first available config entry from a list of section/key pairs."""
    for section, option in candidates:
        if parser.has_option(section, option):
            return parser.get(section, option)
    joined = ", ".join(f"[{section}] {option}" for section, option in candidates)
    raise configparser.NoOptionError(joined, candidates[0][0])


def parse_float_list(parser: configparser.ConfigParser, *candidates: tuple[str, str]) -> numpy.ndarray:
    """Read a whitespace-separated float list from the first matching config key."""
    return numpy.array([float(value) for value in get_config_value(parser, *candidates).split()])


def parse_optional_float_list(
    parser: configparser.ConfigParser,
    *candidates: tuple[str, str],
) -> numpy.ndarray | None:
    """Read an optional whitespace-separated float list."""
    for section, option in candidates:
        if parser.has_option(section, option):
            return numpy.array([float(value) for value in parser.get(section, option).split()])
    return None


def get_int_value(parser: configparser.ConfigParser, *candidates: tuple[str, str]) -> int:
    """Read an integer value from the first matching config key."""
    return int(get_config_value(parser, *candidates))


def get_float_value(parser: configparser.ConfigParser, *candidates: tuple[str, str]) -> float:
    """Read a float value from the first matching config key."""
    return float(get_config_value(parser, *candidates))


def get_bool_value(parser: configparser.ConfigParser, *candidates: tuple[str, str]) -> bool:
    """Read a boolean value from the first matching config key."""
    return parser._convert_to_boolean(get_config_value(parser, *candidates))


def get_optional_config_value(
    parser: configparser.ConfigParser,
    *candidates: tuple[str, str],
) -> str | None:
    """Read an optional config entry from a list of section/key pairs."""
    for section, option in candidates:
        if parser.has_option(section, option):
            return parser.get(section, option)
    return None


def count_component_terms(n_params: int, order: int) -> int:
    """Return the number of D-matrix columns for one component up to ``order``."""
    if order < 0 or order > 2:
        raise ValueError(f"Only D-matrix orders 0, 1, and 2 are currently supported, got {order}.")

    n_terms = 1
    if order >= 1:
        n_terms += n_params
    if order >= 2:
        n_terms += n_params * (n_params + 1) // 2
    return n_terms


def validate_fit_setup(
    *,
    map_parser: configparser.ConfigParser,
    fit_parser: configparser.ConfigParser,
    nu_list: numpy.ndarray,
    fwhm_list: numpy.ndarray,
    noise_list: numpy.ndarray,
    nu_list_fit: numpy.ndarray,
    fit_params: list[str],
    fit_inits: numpy.ndarray,
) -> None:
    """Raise a clear error when the example config is internally inconsistent."""
    if len(nu_list) != len(fwhm_list) or len(nu_list) != len(noise_list):
        raise ValueError("Map config must define the same number of nu, fwhm, and noise entries.")

    if len(fit_params) != len(fit_inits):
        raise ValueError("Fit config must define the same number of params and inits entries.")

    if len(nu_list_fit) == 0:
        raise ValueError("Fit config must define at least one fitting frequency in 'nu'.")

    if not numpy.all(numpy.isin(nu_list_fit, nu_list)):
        missing = [f"{freq:g}" for freq in nu_list_fit[~numpy.isin(nu_list_fit, nu_list)]]
        raise ValueError(
            "Fit frequencies must be a subset of the simulation frequencies from the map config. "
            f"Missing from map config: {', '.join(missing)}"
        )

    isdust = get_bool_value(fit_parser, ("fit", "isdust"), ("par", "isdust"))
    issynch = get_bool_value(fit_parser, ("fit", "issynch"), ("par", "issynch"))
    if not isdust and not issynch:
        raise ValueError("At least one foreground component must be enabled: set 'isdust' or 'issynch' to True.")

    if "r" not in fit_params:
        raise ValueError("Fit params must include 'r'.")

    order = get_int_value(fit_parser, ("fit", "order"), ("par", "order")) if (
        fit_parser.has_option("fit", "order") or fit_parser.has_option("par", "order")
    ) else 1

    dust_param_count = 0
    if isdust:
        dust_param_count = sum(param in fit_params for param in ("T_d1", "x^R", "beta_d"))
    synch_param_count = 0
    if issynch:
        synch_param_count = sum(param in fit_params for param in ("beta_s",))

    min_freq_count = 1
    if isdust:
        min_freq_count += count_component_terms(dust_param_count, order)
    if issynch:
        min_freq_count += count_component_terms(synch_param_count, order)
    if len(nu_list_fit) < min_freq_count:
        raise ValueError(
            "Not enough fitting frequencies for the selected model. "
            f"Need at least {min_freq_count}, got {len(nu_list_fit)}."
        )

    output_name = get_config_value(fit_parser, ("io", "oname"), ("par", "oname"))
    if not output_name.endswith(".csv"):
        raise ValueError("Fit outputs should use a '.csv' filename in 'oname'.")


def write_fit_result_csv(
    output_path: Path,
    seed: int,
    likelihood: float,
    param_values: Mapping[str, float],
    param_errors: Mapping[str, float],
) -> None:
    """Write one fit result row as a CSV file with explicit column names."""
    fieldnames = ["seed", "likelihood"]
    row: dict[str, float | int] = {"seed": seed, "likelihood": likelihood}

    for key in param_values.keys():
        value_field = key
        error_field = f"{key}_error"
        fieldnames.extend([value_field, error_field])
        row[value_field] = param_values[key]
        row[error_field] = param_errors[key]

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def resolve_path(base_dir: Path, path_text: str) -> str:
    """Resolve a config-relative path to an absolute string path.

    Args:
        base_dir: Directory used as the reference point.
        path_text: Relative or absolute path text from configuration.

    Returns:
        The resolved absolute path.
    """
    return str((base_dir / path_text).resolve())


def region_parameter_name(parameter_name: str, region_prefix: str, index: int) -> str:
    """Return the internal scalar name for one region-wise parameter."""
    return f"{parameter_name}_{region_prefix}{index}"


def expand_region_parameter_inits(
    initial_params: dict[str, list[float | tuple[float, float]]],
    parameter_name: str,
    region_prefix: str,
    region_count: int,
    region_values: Sequence[float] | None = None,
) -> dict[str, list[float | tuple[float, float]]]:
    """Expand one scalar parameter entry into region-wise scalar entries.

    This mutates ``initial_params`` in place and returns the same dictionary so
    callers can either use the return value or rely on the mutation.
    """
    if region_count <= 0:
        raise ValueError("region_count must be positive")
    if parameter_name not in initial_params:
        raise ValueError(f"Missing initial parameter entry for {parameter_name}")

    initial_value, limits = initial_params.pop(parameter_name)
    if region_values is None:
        values = [float(initial_value)] * region_count
    else:
        values = [float(value) for value in region_values]
        if len(values) != region_count:
            raise ValueError(
                f"{parameter_name} region initial value count must match "
                f"region count: {len(values)} != {region_count}"
            )

    for index, value in enumerate(values):
        initial_params[region_parameter_name(parameter_name, region_prefix, index)] = [
            value,
            limits,
        ]
    return initial_params


def make_synch_template(templates: Templates, beta_name: str):
    """Build a synchrotron template with a region-specific beta symbol."""
    return templates.ReturnPowerLawSynch(symbol_name=beta_name)


def make_dust_temperature_template(
    templates: Templates,
    beta_name: str,
    temperature_name: str,
):
    """Build the standard dust template with region-specific symbols."""
    return templates.ReturnMBB1(
        beta_symbol_name=beta_name,
        temperature_symbol_name=temperature_name,
    )


def make_dust_xref_template(templates: Templates, beta_name: str, xref_name: str):
    """Build the xRef dust template with region-specific symbols."""
    return templates.ReturnMBB1_xRef(
        beta_symbol_name=beta_name,
        xref_symbol_name=xref_name,
    )


def add_synch_components_to_dmatrix(
    dmt: DMatrix,
    templates: Templates,
    synch_region_masks: Sequence[numpy.ndarray] | None = None,
) -> None:
    """Add the uniform synchrotron component to a DMatrix."""
    if synch_region_masks is not None:
        raise ValueError(
            "column-mask synchrotron regions are retired from normal use; "
            "use add_spatial_synch_components_to_dmatrix instead"
        )
    dmt.AddD(templates.ReturnPowerLawSynch())


def use_spatial_synch_region_coefficients(
    synch_region_masks: Sequence[numpy.ndarray] | None,
    isdust: bool,
    issynch: bool,
    uni: bool,
    order: int,
) -> bool:
    """Return whether the fast spatial synchrotron region path should be used."""
    if synch_region_masks is None or not issynch:
        return False

    unsupported = []
    if isdust:
        unsupported.append("isdust=True")
    if uni:
        unsupported.append("uni=True")
    if order != 1:
        unsupported.append(f"order={order}")
    if unsupported:
        raise ValueError(
            "synch_region_masks currently require the spatial synchrotron-only "
            "first-order path; unsupported settings: "
            + ", ".join(unsupported)
        )
    return True


def add_spatial_synch_components_to_dmatrix(
    dmt: DMatrix,
    templates: Templates,
    synch_region_masks: Sequence[numpy.ndarray],
) -> None:
    """Add synchrotron terms using pixel-dependent region coefficients."""
    beta_symbols = [
        sympy.Symbol(region_parameter_name("beta_s", "sreg", index))
        for index in range(len(synch_region_masks))
    ]
    synch_terms = [
        make_synch_template(templates, beta_symbol.name)
        for beta_symbol in beta_symbols
    ]
    synch_derivatives = [
        sympy.diff(term, beta_symbol)
        for term, beta_symbol in zip(synch_terms, beta_symbols)
    ]
    template_terms = [
        sympy.simplify(sum(synch_terms)),
        sympy.simplify(sum(synch_derivatives)),
    ]
    nu_symbol = sympy.Symbol("nu")
    synch_funcs = [
        sympy.lambdify((nu_symbol, beta_symbol), term, "numpy")
        for term, beta_symbol in zip(synch_terms, beta_symbols)
    ]
    derivative_funcs = [
        sympy.lambdify((nu_symbol, beta_symbol), derivative, "numpy")
        for derivative, beta_symbol in zip(synch_derivatives, beta_symbols)
    ]
    region_masks = [
        numpy.asarray(region_mask, dtype=numpy.float64)
        for region_mask in synch_region_masks
    ]

    def coefficient_builder(freqs, param_values, size):
        coefficients = numpy.zeros((len(freqs), 2, size), dtype=numpy.float64)
        for region_mask in region_masks:
            if region_mask.shape != (size,):
                raise ValueError(
                    "Region mask length must match DeltaMap size for spatial "
                    f"coefficients: {region_mask.shape} != ({size},)"
                )
        for freq_index, nu_value in enumerate(freqs):
            for region_mask, term, derivative, beta_symbol in zip(
                region_masks,
                synch_funcs,
                derivative_funcs,
                beta_symbols,
            ):
                beta_value = param_values[beta_symbol.name]
                coefficients[freq_index, 0] += (
                    float(term(float(nu_value), beta_value)) * region_mask
                )
                coefficients[freq_index, 1] += (
                    float(derivative(float(nu_value), beta_value)) * region_mask
                )
        return coefficients

    dmt.SetSpatialCoefficients(template_terms, coefficient_builder)


def load_synch_region_masks(
    fit_parser: configparser.ConfigParser,
    fitconfig_dir: Path,
    maskname: str,
    nside: int,
) -> list[numpy.ndarray] | None:
    """Load optional Q/U-expanded synchrotron region masks from config."""
    mask_setting = get_optional_config_value(
        fit_parser,
        ("regions", "synch_region_masks"),
        ("par", "synch_region_masks"),
    )
    if mask_setting is None:
        return None

    analysis_mask = healpy.read_map(
        maskname,
        field=0,
        nest=False,
        dtype=numpy.float64,
    )
    analysis_mask = healpy.ud_grade(analysis_mask, nside_out=nside) != 0.0
    region_masks = []
    for mask_path in mask_setting.split():
        region_mask = numpy.load(resolve_path(fitconfig_dir, mask_path)).astype(bool)
        validate_region_mask_nside(region_mask, analysis_mask, nside)
        region_masks.append(
            restrict_region_mask_to_observed_qu(region_mask, analysis_mask)
        )
    observed_analysis_mask = numpy.ones(
        int(numpy.count_nonzero(analysis_mask)) * 2,
        dtype=bool,
    )
    validate_region_masks(region_masks, observed_analysis_mask)
    return region_masks


def validate_region_mask_nside(
    region_mask: numpy.ndarray,
    analysis_mask: numpy.ndarray,
    nside: int,
) -> None:
    """Raise if a region mask cannot match the configured nside."""
    n_pix = healpy.nside2npix(nside)
    if analysis_mask.shape != (n_pix,):
        raise ValueError(
            "Analysis mask shape is inconsistent with configured nside "
            f"{nside}: got {analysis_mask.shape}, expected {(n_pix,)}"
        )
    n_obs = int(numpy.count_nonzero(analysis_mask))
    valid_shapes = {(n_pix,), (2 * n_pix,), (2 * n_obs,)}
    if region_mask.shape not in valid_shapes:
        raise ValueError(
            "Region mask shape is inconsistent with configured nside "
            f"{nside}: got {region_mask.shape}, expected one of "
            f"{sorted(valid_shapes)}"
        )


def restrict_region_mask_to_observed_qu(
    region_mask: numpy.ndarray,
    analysis_mask: numpy.ndarray,
) -> numpy.ndarray:
    """Return a region mask in the observed [Q..., U...] vector layout."""
    n_pix = len(analysis_mask)
    n_obs = int(numpy.count_nonzero(analysis_mask))
    if region_mask.shape == (n_pix,):
        return expand_to_qu(region_mask[analysis_mask])
    if region_mask.shape == (2 * n_pix,):
        return numpy.concatenate(
            [
                region_mask[:n_pix][analysis_mask],
                region_mask[n_pix:][analysis_mask],
            ]
        )
    if region_mask.shape == (2 * n_obs,):
        return region_mask
    raise ValueError(
        "Region mask shape must match pixel, full Q/U, or observed Q/U layout: "
        f"{region_mask.shape} is incompatible with {n_pix} pixels and {n_obs} "
        "observed pixels"
    )


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

    input_dir_value = (
        input_dir
        if input_dir is not None
        else get_config_value(map_parser, ("simulation", "input_dir"), ("simpar", "input_dir"))
    )
    input_dir_path = Path(input_dir_value)
    input_dir_path.mkdir(parents=True, exist_ok=True)
    noise_template = str(input_dir_path / get_config_value(map_parser, ("simulation", "noise_name"), ("simpar", "noise_name")))
    anoise_freq_template = str(
        input_dir_path / get_config_value(map_parser, ("simulation", "anoise_freq_name"), ("simpar", "anoise_freq_name"))
    )
    anoise_template = str(input_dir_path / get_config_value(map_parser, ("simulation", "anoise_name"), ("simpar", "anoise_name")))
    cmb_template = str(input_dir_path / get_config_value(map_parser, ("simulation", "cmb_name"), ("simpar", "cmb_name")))

    synchmapname = synch_template
    dustmapname = dust_template

    nonzero_len = len(mask[mask != 0.0]) * 2

    anoise_scale = "{0:.1e}".format(anoise).replace(".", "p")
    anoise_name = anoise_template.format(nside, anoise_scale, seed)
    if re_noise or not os.path.exists(anoise_name):
        # Paper artificial noise: add a common CMB-side realization so the
        # shared covariance stays positive definite and invertible.
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
            # Paper artificial noise: add one independent instrument-side
            # realization per channel so each noise covariance remains
            # positive definite and invertible.
            if fgnoise_fac is None:
                random_freq_anoise = return_anoise_map(anoise, nside, nonzero_len)
            else:
                random_freq_anoise = return_anoise_map(noise / fgnoise_fac, nside, nonzero_len)
            if not os.path.exists(anoise_freq_name):
                numpy.save(anoise_freq_name, random_freq_anoise)
        else:
            random_freq_anoise = numpy.load(anoise_freq_name)
            print("read old anoise freq  map")
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
    order: int = 1,
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
    synch_region_masks: Sequence[numpy.ndarray] | None = None,
    beta_s_region_inits: Sequence[float] | None = None,
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

    use_spatial_synch_regions = use_spatial_synch_region_coefficients(
        synch_region_masks,
        isdust=isdust,
        issynch=issynch,
        uni=uni,
        order=order,
    )

    dmt = DMatrix()
    if isdust:
        dmt.AddD(mbb1)
    if issynch and not use_spatial_synch_regions:
        add_synch_components_to_dmatrix(dmt, tmpl, synch_region_masks)

    dmp = DeltaMap(verbose=False)
    dmt.SetFreqs(freq_list, [None] * len(freq_list))
    if use_spatial_synch_regions:
        add_spatial_synch_components_to_dmatrix(dmt, tmpl, synch_region_masks)
    elif uni:
        dmt.PrepareUniformDMatrix()
    else:
        dmt.PrepareDMatrix(order=order)

    cov = Covariance(nside=nside, maskname=maskname, verbose=False, pixwin=True, lmax=nside * 2, fwhm=fwhm)
    cov.Initialise()

    s0_sm = cov.ReturnCovMatrix(True)
    s0_bsm = cov.ReturnCovMatrix(False)

    asigma = return_noise_sigma(anoise, nside)
    # Match the common CMB-side artificial noise added in map generation.
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
    if synch_region_masks is not None and "beta_s" in initial_params:
        expand_region_parameter_inits(
            initial_params,
            "beta_s",
            "sreg",
            len(synch_region_masks),
            beta_s_region_inits,
        )
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
    order: int = 1,
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
    synch_region_masks: Sequence[numpy.ndarray] | None = None,
    beta_s_region_inits: Sequence[float] | None = None,
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

    use_spatial_synch_regions = use_spatial_synch_region_coefficients(
        synch_region_masks,
        isdust=isdust,
        issynch=issynch,
        uni=uni,
        order=order,
    )

    dmt = DMatrix()
    if isdust:
        dmt.AddD(mbb1)
    if issynch and not use_spatial_synch_regions:
        add_synch_components_to_dmatrix(dmt, tmpl, synch_region_masks)

    dmp = DeltaMap(verbose=False)
    dmt.SetFreqs(freq_list, [None] * len(freq_list))
    if use_spatial_synch_regions:
        add_spatial_synch_components_to_dmatrix(dmt, tmpl, synch_region_masks)
    elif uni:
        dmt.PrepareUniformDMatrix()
    else:
        dmt.PrepareDMatrix(order=order)

    cov = Covariance(nside=nside, maskname=maskname, verbose=False, pixwin=True, lmax=nside * 2, fwhm=fwhm)
    cov.Initialise()

    s0_sm = cov.ReturnCovMatrix(True)
    s0_bsm = cov.ReturnCovMatrix(False)
    asigma = return_noise_sigma(anoise, nside)
    # Match the common CMB-side artificial noise added in map generation.
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
    if synch_region_masks is not None and "beta_s" in initial_params:
        expand_region_parameter_inits(
            initial_params,
            "beta_s",
            "sreg",
            len(synch_region_masks),
            beta_s_region_inits,
        )
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

    nu_list = parse_float_list(map_parser, ("instrument", "nu"), ("par", "nu"))
    fwhm_list = parse_float_list(map_parser, ("instrument", "fwhm"), ("par", "fwhm"))
    noise_list = parse_float_list(map_parser, ("instrument", "noise"), ("par", "noise"))
    nside = get_int_value(fit_parser, ("fit", "nside"), ("par", "nside"))

    fwhm_norm = 2200.0
    if nside != 4:
        fwhm_norm = 2200.0 * pow(4.0 / nside, 2)

    dust_template_pattern = get_config_value(fit_parser, ("templates", "dust_temp"), ("par", "dust_temp"))
    synch_template_pattern = get_config_value(fit_parser, ("templates", "synch_temp"), ("par", "synch_temp"))
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

    nu_list_fit = parse_float_list(fit_parser, ("fit", "nu"), ("par", "nu"))

    dust_beta_d = healpy.ud_grade(map_in=dust_beta_d, nside_out=nside, order_in="RING", order_out="RING")
    dust_td1 = healpy.ud_grade(map_in=dust_td1, nside_out=nside, order_in="RING", order_out="RING")

    temp_freqs = nu_list_fit
    temp_noise = noise_list
    anoise = get_float_value(fit_parser, ("fit", "anoise"), ("par", "anoise"))
    if get_float_value(fit_parser, ("fit", "fgnoise_fac"), ("par", "fgnoise_fac")) < 0:
        fgnoise_fac = None
    else:
        fgnoise_fac = get_float_value(fit_parser, ("fit", "fgnoise_fac"), ("par", "fgnoise_fac"))
    fit_order = get_int_value(fit_parser, ("fit", "order"), ("par", "order")) if (
        fit_parser.has_option("fit", "order") or fit_parser.has_option("par", "order")
    ) else 1

    fit_params = get_config_value(fit_parser, ("fit", "params"), ("par", "params")).split()
    fit_inits = parse_float_list(fit_parser, ("fit", "inits"), ("par", "inits"))
    validate_fit_setup(
        map_parser=map_parser,
        fit_parser=fit_parser,
        nu_list=nu_list,
        fwhm_list=fwhm_list,
        noise_list=noise_list,
        nu_list_fit=nu_list_fit,
        fit_params=fit_params,
        fit_inits=fit_inits,
    )
    params = {}
    param_defs = {}
    for par, ini in zip(fit_params, fit_inits):
        params[par] = ini
        if not par == "r":
            param_defs[par] = ini
        noise_comb = temp_noise[numpy.isin(nu_list, temp_freqs)]
    fwhm_comb = fwhm_list[numpy.isin(nu_list, temp_freqs)]

    input_dir_setting = get_config_value(map_parser, ("simulation", "input_dir"), ("simpar", "input_dir"))
    output_dir_setting = get_config_value(fit_parser, ("io", "odir"), ("par", "odir"))
    output_name_setting = get_config_value(fit_parser, ("io", "oname"), ("par", "oname"))

    input_dir = resolve_path(config_dir, input_dir_setting)
    odir = Path(resolve_path(fitconfig_dir, output_dir_setting.format(fitconfig_path.stem)))
    oname = odir / output_name_setting.format(args.seed)
    odir.mkdir(parents=True, exist_ok=True)
    if oname.exists():
        return 0

    maskname = resolve_path(config_dir, get_config_value(map_parser, ("simulation", "maskname"), ("simpar", "maskname")))
    synch_region_masks = load_synch_region_masks(
        fit_parser,
        fitconfig_dir,
        maskname,
        nside,
    )
    beta_s_region_inits = parse_optional_float_list(
        fit_parser,
        ("regions", "beta_s_region_inits"),
        ("par", "beta_s_region_inits"),
    )

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
            isdust=get_bool_value(fit_parser, ("fit", "isdust"), ("par", "isdust")),
            issynch=get_bool_value(fit_parser, ("fit", "issynch"), ("par", "issynch")),
            r=get_float_value(fit_parser, ("fit", "r"), ("par", "r")),
            uni=get_bool_value(fit_parser, ("fit", "uni"), ("par", "uni")),
            param_defs=param_defs,
            dust_template=dust_template,
            synch_template=synch_template,
            anoise=anoise,
            fgnoise_fac=fgnoise_fac,
            fixTd=get_bool_value(fit_parser, ("fit", "fixTd"), ("par", "fixTd")),
            order=fit_order,
            dmp=dmp,
            T_d1_mean=dust_td1.mean(),
            beta_d_mean=dust_beta_d.mean(),
            seed=args.seed,
            input_dir=input_dir,
            synch_region_masks=synch_region_masks,
            beta_s_region_inits=beta_s_region_inits,
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
            isdust=get_bool_value(fit_parser, ("fit", "isdust"), ("par", "isdust")),
            issynch=get_bool_value(fit_parser, ("fit", "issynch"), ("par", "issynch")),
            r=get_float_value(fit_parser, ("fit", "r"), ("par", "r")),
            uni=get_bool_value(fit_parser, ("fit", "uni"), ("par", "uni")),
            param_defs=param_defs,
            dust_template=dust_template,
            synch_template=synch_template,
            anoise=anoise,
            fgnoise_fac=fgnoise_fac,
            fixTd=get_bool_value(fit_parser, ("fit", "fixTd"), ("par", "fixTd")),
            order=fit_order,
            dmp=dmp,
            T_d1_mean=dust_td1.mean(),
            beta_d_mean=dust_beta_d.mean(),
            seed=args.seed,
            input_dir=input_dir,
            synch_region_masks=synch_region_masks,
            beta_s_region_inits=beta_s_region_inits,
        )

    dmp.SetParameters(values=params)

    with_td_prior = False
    td_sigma = None
    try:
        with_td_prior = get_bool_value(fit_parser, ("priors", "Tdprior"), ("par", "Tdprior"))
    except (configparser.NoOptionError, ValueError):
        pass
    try:
        td_sigma = get_float_value(fit_parser, ("priors", "Tdsigma"), ("par", "Tdsigma"))
    except (configparser.NoOptionError, ValueError):
        td_sigma = 3.0
    with_xr_prior = False
    xr_sigma = None
    try:
        with_xr_prior = get_bool_value(fit_parser, ("priors", "xRprior"), ("par", "xRprior"))
    except (configparser.NoOptionError, ValueError):
        pass

    try:
        xr_sigma = get_float_value(fit_parser, ("priors", "xRsigma"), ("par", "xRsigma"))
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
        with_migrad = get_bool_value(fit_parser, ("fit", "migrad"), ("par", "migrad"))
    except (configparser.NoOptionError, ValueError):
        pass

    dmp.migrad = with_migrad
    dmp.r_verbose = False

    simul = False
    try:
        simul = get_bool_value(fit_parser, ("fit", "simul"), ("par", "simul"))
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

    write_fit_result_csv(
        oname,
        seed=args.seed,
        likelihood=dmp.lh,
        param_values=dmp.param_values,
        param_errors=dmp.param_errors,
    )

    return 0


if "__main__" == __name__:
    sys.exit(main())
