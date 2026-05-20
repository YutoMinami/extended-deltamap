from __future__ import annotations

import configparser
import sys
import warnings
from pathlib import Path
from typing import Sequence

import healpy as hp
import numpy as np
import numpy.typing as npt
import pysm3 as pysm
import pysm3.units as u

warnings.filterwarnings("ignore")


def get_config_value(parser: configparser.ConfigParser, *candidates: tuple[str, str]) -> str:
    """Read the first available config entry from a list of section/key pairs."""
    for section, option in candidates:
        if parser.has_option(section, option):
            return parser.get(section, option)
    joined = ", ".join(f"[{section}] {option}" for section, option in candidates)
    raise configparser.NoOptionError(joined, candidates[0][0])


def get_float_list(parser: configparser.ConfigParser, *candidates: tuple[str, str]) -> npt.NDArray[np.float64]:
    """Read a whitespace-separated float list from the first matching config key."""
    return np.array([float(value) for value in get_config_value(parser, *candidates).split()])


def get_int_value(parser: configparser.ConfigParser, *candidates: tuple[str, str]) -> int:
    """Read an integer value from the first matching config key."""
    return int(get_config_value(parser, *candidates))


def main(argv: Sequence[str] | None = None) -> int:
    """Generate PySM foreground maps from a config file.

    Args:
        argv: Command-line style arguments. When omitted, ``sys.argv`` is used.

    Returns:
        Zero when map generation completes successfully.
    """
    args = list(sys.argv if argv is None else argv)

    if len(args) == 1:
        filename = "./config.ini"
    else:
        filename = args[1]
    config_path = Path(filename).resolve()
    config_dir = config_path.parent

    parser = configparser.ConfigParser()
    parser.read(config_path)
    nu = get_float_list(parser, ("instrument", "nu"), ("par", "nu"))
    fwhm = get_float_list(parser, ("instrument", "fwhm"), ("par", "fwhm"))
    noise = get_float_list(parser, ("instrument", "noise"), ("par", "noise"))
    nside = get_int_value(parser, ("instrument", "nside"), ("par", "nside"))

    if len(nu) != len(fwhm) or len(nu) != len(noise):
        raise ValueError("Config must define matching lengths for nu, fwhm, and noise.")

    ofdir_setting = get_config_value(parser, ("foreground", "fg_dir"), ("fgpar", "fg_dir"))
    ofdir = (config_dir / ofdir_setting.format(nside)).resolve()
    ofdir.mkdir(parents=True, exist_ok=True)

    nu_template = "nu{0:07.2f}GHz"

    ofname_template = get_config_value(parser, ("foreground", "fg_name"), ("fgpar", "fg_name"))

    comp_names = get_config_value(parser, ("foreground", "components"), ("fgpar", "components")).split()

    for comp in comp_names:
        sky = pysm.Sky(nside=nside, preset_strings=[comp], output_unit="uK_CMB")
        for nu_i in nu:
            nu_name = nu_template.format(nu_i).replace(".", "p")
            ofname = ofdir / ofname_template.format(nu_name, comp, nside)
            if ofname.exists():
                continue
            hp.write_map(str(ofname), sky.get_emission(nu_i * u.GHz), nest=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
