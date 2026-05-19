#!/usr/bin/env python
"""Build spatially connected region masks from HEALPix-neighbor feature edges."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import healpy
import numpy
from scipy import sparse
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

from extended_deltamap.regions import expand_to_qu, validate_region_masks
from make_combined_feature_regions import (
    FEATURE_NAMES,
    SED_FEATURE_INDICES,
    build_feature_matrix,
    read_analysis_mask,
    stable_relabel,
    standardize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create spatially connected region masks by combining HEALPix "
            "neighbor proximity with standardized SED-tracer feature distances."
        )
    )
    parser.add_argument(
        "--map-dir",
        type=Path,
        default=Path("examples/output_pysm3_ns8"),
        help="Directory containing PySM I,Q,U maps at nside=8.",
    )
    parser.add_argument(
        "--analysis-mask",
        type=Path,
        default=Path("examples/files/mask_p06_Nside4.v2.fits"),
        help="Analysis mask FITS map.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/regions"),
        help="Directory for output .npy masks and summary CSV.",
    )
    parser.add_argument(
        "--prefix",
        default="spatial_feature_mst_ns8",
        help="Filename prefix for output masks.",
    )
    parser.add_argument("--nside", type=int, default=8, help="Output nside.")
    parser.add_argument("--k", type=int, nargs="+", default=[4], help="Region counts to generate.")
    parser.add_argument(
        "--feature-lambda",
        type=float,
        default=1.0,
        help="Relative weight for standardized SED-feature distance.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=50,
        help="Minimum pixels allowed in each output region.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0e-30,
        help="Positive floor applied before logarithms.",
    )
    return parser.parse_args()


def build_neighbor_graph(
    nside: int,
    analysis_mask: numpy.ndarray,
    standardized_features: numpy.ndarray,
    feature_lambda: float,
) -> tuple[sparse.csr_matrix, numpy.ndarray]:
    """Return a sparse valid-pixel graph with angular+SED edge weights."""
    valid_pixels = numpy.flatnonzero(analysis_mask)
    compact_index = numpy.full(analysis_mask.size, -1, dtype=int)
    compact_index[valid_pixels] = numpy.arange(valid_pixels.size)
    vectors = numpy.asarray(healpy.pix2vec(nside, valid_pixels)).T

    theta_samples = []
    edge_i = []
    edge_j = []
    edge_theta = []
    for pixel in valid_pixels:
        i = compact_index[pixel]
        for neighbor in healpy.get_all_neighbours(nside, int(pixel)):
            neighbor = int(neighbor)
            if neighbor < 0 or compact_index[neighbor] < 0:
                continue
            j = compact_index[neighbor]
            if j <= i:
                continue
            dot = float(numpy.clip(numpy.dot(vectors[i], vectors[j]), -1.0, 1.0))
            theta = float(numpy.arccos(dot))
            theta_samples.append(theta)
            edge_i.append(i)
            edge_j.append(j)
            edge_theta.append(theta)

    if not edge_i:
        raise ValueError("no valid HEALPix-neighbor edges found")
    theta0 = float(numpy.median(theta_samples))
    sed_features = standardized_features[:, SED_FEATURE_INDICES]

    weights = []
    for i, j, theta in zip(edge_i, edge_j, edge_theta, strict=True):
        feature_distance = float(numpy.mean((sed_features[i] - sed_features[j]) ** 2))
        weights.append((theta / theta0) ** 2 + feature_lambda * feature_distance)

    rows = numpy.asarray(edge_i + edge_j, dtype=int)
    cols = numpy.asarray(edge_j + edge_i, dtype=int)
    data = numpy.asarray(weights + weights, dtype=float)
    graph = sparse.csr_matrix((data, (rows, cols)), shape=(valid_pixels.size, valid_pixels.size))
    return graph, valid_pixels


def component_size_after_cut(
    adjacency: list[set[int]],
    start: int,
    blocked: tuple[int, int],
) -> int:
    """Return the component size reachable from start if one edge is blocked."""
    stack = [start]
    seen = {start}
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if (node, neighbor) == blocked or (neighbor, node) == blocked:
                continue
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen)


def cut_mst_to_labels(
    graph: sparse.csr_matrix,
    cluster_count: int,
    min_pixels: int,
) -> numpy.ndarray:
    """Cut the largest MST edges until the requested number of components appears."""
    base_components, _ = connected_components(graph, directed=False)
    if cluster_count < base_components:
        raise ValueError(
            f"k={cluster_count} is smaller than the analysis mask's "
            f"{base_components} connected components"
        )

    forest = minimum_spanning_tree(graph).tocoo()
    edge_count_to_cut = cluster_count - base_components
    keep = numpy.ones(forest.data.size, dtype=bool)
    adjacency = [set() for _ in range(graph.shape[0])]
    for row, col in zip(forest.row, forest.col, strict=True):
        row = int(row)
        col = int(col)
        adjacency[row].add(col)
        adjacency[col].add(row)

    cut_edges = 0
    for edge_index in numpy.argsort(forest.data)[::-1]:
        if cut_edges == edge_count_to_cut:
            break
        row = int(forest.row[edge_index])
        col = int(forest.col[edge_index])
        if col not in adjacency[row]:
            continue
        size_a = component_size_after_cut(adjacency, row, (row, col))
        size_b = component_size_after_cut(adjacency, col, (row, col))
        if size_a < min_pixels or size_b < min_pixels:
            continue
        adjacency[row].remove(col)
        adjacency[col].remove(row)
        keep[edge_index] = False
        cut_edges += 1

    if cut_edges != edge_count_to_cut:
        raise RuntimeError(
            f"could only cut {cut_edges} balanced edges; "
            f"needed {edge_count_to_cut}. Try lowering --min-pixels."
        )

    rows = forest.row[keep]
    cols = forest.col[keep]
    data = forest.data[keep]

    kept = sparse.csr_matrix((data, (rows, cols)), shape=graph.shape)
    kept = kept + kept.T
    actual_count, labels = connected_components(kept, directed=False)
    if actual_count != cluster_count:
        raise RuntimeError(f"expected {cluster_count} components, got {actual_count}")
    return labels


def labels_to_masks(
    labels: numpy.ndarray,
    valid_pixels: numpy.ndarray,
    npix: int,
    cluster_count: int,
) -> list[numpy.ndarray]:
    """Expand compact valid-pixel labels to full-sky boolean masks."""
    masks = []
    for label in range(cluster_count):
        mask = numpy.zeros(npix, dtype=bool)
        mask[valid_pixels[labels == label]] = True
        masks.append(mask)
    return masks


def save_masks(output_dir: Path, prefix: str, cluster_count: int, masks: list[numpy.ndarray]) -> None:
    """Save pixel and Q/U masks for one clustering result."""
    for index, mask in enumerate(masks):
        stem = f"{prefix}_k{cluster_count:02d}_reg{index:02d}"
        numpy.save(output_dir / f"{stem}_pix.npy", mask)
        numpy.save(output_dir / f"{stem}_qu.npy", expand_to_qu(mask))


def component_sizes(nside: int, mask: numpy.ndarray) -> list[int]:
    """Return HEALPix-neighbor connected-component sizes inside one mask."""
    remaining = set(numpy.flatnonzero(mask).tolist())
    sizes = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        size = 1
        while stack:
            pixel = stack.pop()
            for neighbor in healpy.get_all_neighbours(nside, int(pixel)):
                neighbor = int(neighbor)
                if neighbor >= 0 and neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    size += 1
        sizes.append(size)
    return sorted(sizes, reverse=True)


def write_summary(
    output_dir: Path,
    prefix: str,
    rows: list[dict[str, int | float | str]],
) -> None:
    """Write a CSV summary for all requested cluster counts."""
    path = output_dir / f"{prefix}_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "k",
                "region",
                "npix",
                "component_count",
                "largest_components",
                *[f"mean_{name}" for name in FEATURE_NAMES],
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    analysis_mask = read_analysis_mask(args.analysis_mask, args.nside)
    features, _ = build_feature_matrix(args.map_dir, analysis_mask, args.nside, args.epsilon)
    standardized_features, _, _ = standardize(features)
    graph, valid_pixels = build_neighbor_graph(
        args.nside,
        analysis_mask,
        standardized_features,
        args.feature_lambda,
    )

    summary_rows: list[dict[str, int | float | str]] = []
    for cluster_count in args.k:
        labels = cut_mst_to_labels(graph, cluster_count, args.min_pixels)
        labels = stable_relabel(labels, features, cluster_count)
        masks = labels_to_masks(labels, valid_pixels, analysis_mask.size, cluster_count)
        validate_region_masks(masks, analysis_mask)
        save_masks(args.output_dir, args.prefix, cluster_count, masks)

        counts = [int(mask.sum()) for mask in masks]
        print(f"k={cluster_count} counts={counts}")
        for region, mask in enumerate(masks):
            region_features = features[labels == region]
            components = component_sizes(args.nside, mask)
            row: dict[str, int | float | str] = {
                "k": cluster_count,
                "region": region,
                "npix": int(mask.sum()),
                "component_count": len(components),
                "largest_components": " ".join(str(size) for size in components[:8]),
            }
            for feature_index, name in enumerate(FEATURE_NAMES):
                row[f"mean_{name}"] = float(region_features[:, feature_index].mean())
            summary_rows.append(row)
            print(
                f"  reg{region:02d}: npix={row['npix']} "
                f"components={row['component_count']} largest={row['largest_components']}"
            )

    write_summary(args.output_dir, args.prefix, summary_rows)


if __name__ == "__main__":
    main()
