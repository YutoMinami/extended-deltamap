#!/usr/bin/env python
"""Build synchrotron regions from low-frequency polarization brightness."""

from __future__ import annotations

import argparse
from pathlib import Path

import healpy
import numpy

from extended_deltamap.regions import expand_to_qu, validate_region_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a synchrotron template into faint/bright regions using the "
            "polarization brightness quantiles inside an analysis mask."
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
    parser.add_argument(
        "--region-count",
        type=int,
        default=2,
        help="Number of brightness-quantile regions to create.",
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


def split_by_quantiles(
    brightness: numpy.ndarray,
    analysis_mask: numpy.ndarray,
    region_count: int = 2,
) -> tuple[list[numpy.ndarray], list[float]]:
    """Split valid pixels into brightness-quantile masks."""
    if brightness.shape != analysis_mask.shape:
        raise ValueError(
            "brightness and analysis_mask must have the same shape: "
            f"{brightness.shape} != {analysis_mask.shape}"
        )
    if region_count < 2:
        raise ValueError("region_count must be at least 2")

    valid_values = brightness[analysis_mask]
    if len(valid_values) == 0:
        raise ValueError("analysis_mask does not contain any valid pixels")
    if region_count > len(valid_values):
        raise ValueError(
            "region_count cannot exceed the number of valid pixels: "
            f"{region_count} > {len(valid_values)}"
        )

    thresholds = [
        float(value)
        for value in numpy.quantile(
            valid_values,
            numpy.arange(1, region_count) / region_count,
        )
    ]
    region_ids = numpy.searchsorted(thresholds, brightness, side="right")
    regions = [
        analysis_mask & (region_ids == region_index)
        for region_index in range(region_count)
    ]
    validate_region_masks(regions, analysis_mask)
    return regions, thresholds


def split_by_median(
    brightness: numpy.ndarray,
    analysis_mask: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray, float]:
    """Split valid pixels into faint and bright masks by median brightness."""
    regions, thresholds = split_by_quantiles(
        brightness,
        analysis_mask,
        region_count=2,
    )
    return regions[0], regions[1], thresholds[0]


def region_suffix(index: int, region_count: int, layout: str) -> str:
    """Return a stable filename suffix for one region mask."""
    if region_count == 2:
        labels = ["faint", "bright"]
        return f"sreg{index}_{labels[index]}_{layout}"
    if region_count == 4:
        labels = ["q00_25", "q25_50", "q50_75", "q75_100"]
        return f"sreg{index}_{labels[index]}_{layout}"
    return f"sreg{index}_q{index:02d}_{layout}"


def save_region_masks(
    output_dir: Path,
    prefix: str,
    regions: list[numpy.ndarray],
) -> None:
    """Save pixel and Q/U-expanded masks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    region_count = len(regions)
    for index, region in enumerate(regions):
        numpy.save(
            output_dir / f"{prefix}_{region_suffix(index, region_count, 'pix')}.npy",
            region,
        )
        numpy.save(
            output_dir / f"{prefix}_{region_suffix(index, region_count, 'qu')}.npy",
            expand_to_qu(region),
        )


def main() -> None:
    args = parse_args()
    brightness = read_synch_pol_brightness(
        args.synch_map,
        nside=args.nside,
        smooth_fwhm_deg=args.smooth_fwhm_deg,
    )
    analysis_mask = read_analysis_mask(args.analysis_mask, nside=args.nside)
    regions, thresholds = split_by_quantiles(
        brightness,
        analysis_mask,
        region_count=args.region_count,
    )
    save_region_masks(args.output_dir, args.prefix, regions)

    for index, threshold in enumerate(thresholds):
        print(f"threshold {index}: {threshold:.8e}")
    print(f"valid pixels: {int(numpy.count_nonzero(analysis_mask))}")
    for index, region in enumerate(regions):
        print(f"sreg{index} pixels: {int(numpy.count_nonzero(region))}")
    print(f"output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
