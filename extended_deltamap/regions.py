"""Utilities for region-wise foreground masks."""

import numpy


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
