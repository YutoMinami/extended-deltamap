#!/usr/bin/env python
"""Build two synchrotron regions from low-frequency polarization brightness."""

from __future__ import annotations

import argparse
from pathlib import Path

import healpy
import numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a synchrotron template into faint/bright regions using the "
            "median polarization brightness inside an analysis mask."
        )
    )
    parser.add_argument(
        "synch_map",
        type=Path,
        help="Input synchrotron FITS map with I,Q,U fields.",
    )
    parser.add_argument(
        "analysis_mask",
        type=Path,
        help="Analysis mask FITS map. Non-zero pixels are used for the median.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/regions"),
        help="Directory for output .npy masks.",
    )
    parser.add_argument(
        "--prefix",
        default="synch_brightness_median",
        help="Filename prefix for output masks.",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=None,
        help="Optional output nside. The map and mask are ud_graded if set.",
    )
    parser.add_argument(
        "--smooth-fwhm-deg",
        type=float,
        default=0.0,
        help="Optional smoothing FWHM in degrees before thresholding.",
    )
    return parser.parse_args()


def read_synch_pol_brightness(
    synch_map: Path,
    nside: int | None = None,
    smooth_fwhm_deg: float = 0.0,
) -> numpy.ndarray:
    """Return sqrt(Q^2 + U^2) from a synchrotron I,Q,U map."""
    sky_map = healpy.read_map(synch_map, field=(0, 1, 2), dtype=numpy.float64)
    if nside is not None:
        sky_map = healpy.ud_grade(sky_map, nside_out=nside)

    if smooth_fwhm_deg > 0.0:
        source_nside = healpy.npix2nside(sky_map.shape[-1])
        alm = healpy.map2alm(sky_map, lmax=source_nside * 2, pol=True)
        sky_map = healpy.alm2map(
            alm,
            nside=source_nside,
            lmax=source_nside * 2,
            pol=True,
            fwhm=numpy.deg2rad(smooth_fwhm_deg),
        )

    return numpy.sqrt(sky_map[1] ** 2 + sky_map[2] ** 2)


def read_analysis_mask(mask_path: Path, nside: int | None = None) -> numpy.ndarray:
    """Return a boolean analysis mask."""
    mask = healpy.read_map(mask_path, field=0, dtype=numpy.float64)
    if nside is not None:
        mask = healpy.ud_grade(mask, nside_out=nside)
    return mask != 0.0


def split_by_median(
    brightness: numpy.ndarray,
    analysis_mask: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray, float]:
    """Split valid pixels into faint and bright masks by median brightness."""
    if brightness.shape != analysis_mask.shape:
        raise ValueError(
            "brightness and analysis_mask must have the same shape: "
            f"{brightness.shape} != {analysis_mask.shape}"
        )

    valid_values = brightness[analysis_mask]
    if len(valid_values) == 0:
        raise ValueError("analysis_mask does not contain any valid pixels")

    threshold = float(numpy.median(valid_values))
    bright = analysis_mask & (brightness >= threshold)
    faint = analysis_mask & ~bright
    validate_region_masks([faint, bright], analysis_mask)
    return faint, bright, threshold


def expand_to_qu(mask: numpy.ndarray) -> numpy.ndarray:
    """Expand a pixel mask to the [Q pixels..., U pixels...] vector layout."""
    return numpy.concatenate([mask, mask])


def validate_region_masks(
    region_masks: list[numpy.ndarray],
    analysis_mask: numpy.ndarray,
) -> None:
    """Validate disjoint, full coverage of the analysis mask."""
    coverage = numpy.zeros_like(analysis_mask, dtype=bool)
    for index, region_mask in enumerate(region_masks):
        if region_mask.shape != analysis_mask.shape:
            raise ValueError(
                f"region {index} shape {region_mask.shape} does not match "
                f"analysis mask shape {analysis_mask.shape}"
            )
        if numpy.any(region_mask & ~analysis_mask):
            raise ValueError(f"region {index} contains pixels outside analysis_mask")
        if numpy.any(coverage & region_mask):
            raise ValueError(f"region {index} overlaps an earlier region")
        if not numpy.any(region_mask):
            raise ValueError(f"region {index} is empty")
        coverage |= region_mask

    if not numpy.array_equal(coverage, analysis_mask):
        missing = int(numpy.count_nonzero(analysis_mask & ~coverage))
        raise ValueError(f"region masks do not cover {missing} valid pixels")


def save_region_masks(
    output_dir: Path,
    prefix: str,
    faint: numpy.ndarray,
    bright: numpy.ndarray,
) -> None:
    """Save pixel and Q/U-expanded masks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    masks = {
        "sreg0_faint_pix": faint,
        "sreg1_bright_pix": bright,
        "sreg0_faint_qu": expand_to_qu(faint),
        "sreg1_bright_qu": expand_to_qu(bright),
    }
    for suffix, mask in masks.items():
        numpy.save(output_dir / f"{prefix}_{suffix}.npy", mask)


def main() -> None:
    args = parse_args()
    brightness = read_synch_pol_brightness(
        args.synch_map,
        nside=args.nside,
        smooth_fwhm_deg=args.smooth_fwhm_deg,
    )
    analysis_mask = read_analysis_mask(args.analysis_mask, nside=args.nside)
    faint, bright, threshold = split_by_median(brightness, analysis_mask)
    save_region_masks(args.output_dir, args.prefix, faint, bright)

    print(f"threshold: {threshold:.8e}")
    print(f"valid pixels: {int(numpy.count_nonzero(analysis_mask))}")
    print(f"sreg0 faint pixels: {int(numpy.count_nonzero(faint))}")
    print(f"sreg1 bright pixels: {int(numpy.count_nonzero(bright))}")
    print(f"output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
