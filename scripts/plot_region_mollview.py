#!/usr/bin/env python
"""Plot saved HEALPix region masks with healpy.mollview."""

from __future__ import annotations

import argparse
from pathlib import Path

import healpy
import matplotlib
import numpy

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        default="combined_feature_kmeans_ns8",
        help="Region mask filename prefix.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of regions in the saved mask set.",
    )
    parser.add_argument(
        "--region-dir",
        type=Path,
        default=Path("data/regions"),
        help="Directory containing *_regXX_pix.npy masks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/plots"),
        help="Directory for PNG outputs.",
    )
    parser.add_argument(
        "--coord",
        default="G",
        help="healpy coordinate flag, e.g. G, C, or E.",
    )
    return parser.parse_args()


def load_masks(region_dir: Path, prefix: str, k: int) -> list[numpy.ndarray]:
    masks = []
    for index in range(k):
        path = region_dir / f"{prefix}_k{k:02d}_reg{index:02d}_pix.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        mask = numpy.load(path).astype(bool)
        masks.append(mask)
    return masks


def build_label_map(masks: list[numpy.ndarray]) -> numpy.ndarray:
    npix = len(masks[0])
    label_map = numpy.full(npix, healpy.UNSEEN, dtype=float)
    for index, mask in enumerate(masks):
        if len(mask) != npix:
            raise ValueError("all masks must have the same npix")
        label_map[mask] = index
    return label_map


def plot_combined(label_map: numpy.ndarray, k: int, output_path: Path, coord: str) -> None:
    cmap = plt.get_cmap("tab20", k)
    healpy.mollview(
        label_map,
        title=f"Region labels: k={k}",
        coord=coord,
        min=-0.5,
        max=k - 0.5,
        cmap=cmap,
        badcolor="lightgray",
        cbar=True,
    )
    healpy.graticule(verbose=False)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_individual(masks: list[numpy.ndarray], output_path: Path, coord: str) -> None:
    cols = 2
    rows = int(numpy.ceil(len(masks) / cols))
    plt.figure(figsize=(10, 4.2 * rows))
    for index, mask in enumerate(masks):
        one_region = numpy.full(len(mask), healpy.UNSEEN, dtype=float)
        one_region[mask] = 1.0
        healpy.mollview(
            one_region,
            fig=plt.gcf().number,
            sub=(rows, cols, index + 1),
            title=f"region {index:02d} ({int(mask.sum())} pix)",
            coord=coord,
            min=0.0,
            max=1.0,
            cmap="viridis",
            badcolor="lightgray",
            cbar=False,
        )
        healpy.graticule(verbose=False)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    masks = load_masks(args.region_dir, args.prefix, args.k)
    label_map = build_label_map(masks)
    combined_path = args.output_dir / f"{args.prefix}_k{args.k:02d}_mollview.png"
    individual_path = args.output_dir / f"{args.prefix}_k{args.k:02d}_mollview_regions.png"
    plot_combined(label_map, args.k, combined_path, args.coord)
    plot_individual(masks, individual_path, args.coord)
    print(f"saved combined: {combined_path}")
    print(f"saved individual: {individual_path}")


if __name__ == "__main__":
    main()
