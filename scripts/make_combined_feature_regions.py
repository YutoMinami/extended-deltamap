#!/usr/bin/env python
"""Build region masks from combined dust/synchrotron SED-shape features."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import healpy
import numpy
from scipy.cluster.vq import kmeans2

from extended_deltamap.regions import expand_to_qu, validate_region_masks


FEATURE_NAMES = (
    "log_P40_s",
    "log_P60_over_P40_s",
    "log_P78_over_P60_s",
    "log_P337_d",
    "log_P337_over_P235_d",
    "log_P402_over_P280_d",
)
SED_FEATURE_INDICES = (1, 2, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create k-means region masks from standardized combined "
            "synchrotron/dust amplitude and SED-ratio features."
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
        default="combined_feature_kmeans_ns8",
        help="Filename prefix for output masks.",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=8,
        help="Output nside for maps and masks.",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[4, 8, 16],
        help="Cluster counts to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for k-means initialisation.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0e-30,
        help="Positive floor applied before logarithms.",
    )
    return parser.parse_args()


def freq_token(freq: int) -> str:
    """Return the frequency token used in generated PySM filenames."""
    return f"{freq:04d}p00"


def map_path(map_dir: Path, freq: int, component: str, nside: int) -> Path:
    """Return the expected PySM map path for one component and frequency."""
    return map_dir / f"test01_nu{freq_token(freq)}GHz_{component}1_nside{nside:04d}.fits"


def read_pol_amplitude(
    map_dir: Path,
    freq: int,
    component: str,
    nside: int,
) -> numpy.ndarray:
    """Return sqrt(Q^2 + U^2) from an I,Q,U map."""
    path = map_path(map_dir, freq, component, nside)
    sky_map = healpy.read_map(path, field=(0, 1, 2), dtype=numpy.float64)
    if healpy.npix2nside(sky_map.shape[-1]) != nside:
        sky_map = healpy.ud_grade(sky_map, nside_out=nside)
    return numpy.sqrt(sky_map[1] ** 2 + sky_map[2] ** 2)


def read_analysis_mask(mask_path: Path, nside: int) -> numpy.ndarray:
    """Return a boolean analysis mask at the requested nside."""
    mask = healpy.read_map(mask_path, field=0, dtype=numpy.float64)
    if healpy.npix2nside(mask.shape[-1]) != nside:
        mask = healpy.ud_grade(mask, nside_out=nside)
    return mask != 0.0


def safe_log(values: numpy.ndarray, epsilon: float) -> numpy.ndarray:
    """Return log(values) after clipping to a positive floor."""
    return numpy.log(numpy.clip(values, epsilon, None))


def build_feature_matrix(
    map_dir: Path,
    analysis_mask: numpy.ndarray,
    nside: int,
    epsilon: float,
) -> tuple[numpy.ndarray, dict[str, numpy.ndarray]]:
    """Return feature matrix for valid pixels and full-sky raw feature maps."""
    p40_s = read_pol_amplitude(map_dir, 40, "s", nside)
    p60_s = read_pol_amplitude(map_dir, 60, "s", nside)
    p78_s = read_pol_amplitude(map_dir, 78, "s", nside)
    p235_d = read_pol_amplitude(map_dir, 235, "d", nside)
    p280_d = read_pol_amplitude(map_dir, 280, "d", nside)
    p337_d = read_pol_amplitude(map_dir, 337, "d", nside)
    p402_d = read_pol_amplitude(map_dir, 402, "d", nside)

    feature_maps = {
        "log_P40_s": safe_log(p40_s, epsilon),
        "log_P60_over_P40_s": safe_log(p60_s, epsilon) - safe_log(p40_s, epsilon),
        "log_P78_over_P60_s": safe_log(p78_s, epsilon) - safe_log(p60_s, epsilon),
        "log_P337_d": safe_log(p337_d, epsilon),
        "log_P337_over_P235_d": safe_log(p337_d, epsilon) - safe_log(p235_d, epsilon),
        "log_P402_over_P280_d": safe_log(p402_d, epsilon) - safe_log(p280_d, epsilon),
    }
    features = numpy.column_stack([feature_maps[name][analysis_mask] for name in FEATURE_NAMES])
    if not numpy.all(numpy.isfinite(features)):
        raise ValueError("feature matrix contains non-finite values")
    return features, feature_maps


def standardize(features: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Return standardized features plus per-column means and standard deviations."""
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    if numpy.any(stds == 0.0):
        names = [FEATURE_NAMES[index] for index in numpy.flatnonzero(stds == 0.0)]
        raise ValueError(f"cannot standardize zero-variance features: {names}")
    return (features - means) / stds, means, stds


def stable_relabel(
    labels: numpy.ndarray,
    features: numpy.ndarray,
    cluster_count: int,
) -> numpy.ndarray:
    """Relabel clusters in a stable order using raw synch and dust amplitudes."""
    sort_keys = []
    for label in range(cluster_count):
        cluster_features = features[labels == label]
        sort_keys.append(
            (
                float(cluster_features[:, 0].mean()),
                float(cluster_features[:, 3].mean()),
                label,
            )
        )
    ordered_labels = [label for _, _, label in sorted(sort_keys)]
    relabel = {old_label: new_label for new_label, old_label in enumerate(ordered_labels)}
    return numpy.array([relabel[int(label)] for label in labels], dtype=int)


def labels_to_masks(
    labels: numpy.ndarray,
    analysis_mask: numpy.ndarray,
    cluster_count: int,
) -> list[numpy.ndarray]:
    """Expand compact valid-pixel labels to full-sky boolean masks."""
    valid_indices = numpy.flatnonzero(analysis_mask)
    masks = []
    for label in range(cluster_count):
        mask = numpy.zeros_like(analysis_mask, dtype=bool)
        mask[valid_indices[labels == label]] = True
        masks.append(mask)
    validate_region_masks(masks, analysis_mask)
    return masks


def save_masks(output_dir: Path, prefix: str, cluster_count: int, masks: list[numpy.ndarray]) -> None:
    """Save pixel and Q/U masks for one clustering result."""
    for index, mask in enumerate(masks):
        stem = f"{prefix}_k{cluster_count:02d}_reg{index:02d}"
        numpy.save(output_dir / f"{stem}_pix.npy", mask)
        numpy.save(output_dir / f"{stem}_qu.npy", expand_to_qu(mask))


def summarize_clusters(
    cluster_count: int,
    labels: numpy.ndarray,
    features: numpy.ndarray,
    standardized_features: numpy.ndarray,
    inertia: float,
) -> list[dict[str, float | int | str]]:
    """Return summary rows for one k-means result."""
    rows: list[dict[str, float | int | str]] = []
    for label in range(cluster_count):
        cluster = features[labels == label]
        cluster_std = standardized_features[labels == label]
        sed_spread = float(cluster_std[:, SED_FEATURE_INDICES].std(axis=0).mean())
        row: dict[str, float | int | str] = {
            "k": cluster_count,
            "region": label,
            "pixels": int(cluster.shape[0]),
            "inertia": inertia,
            "mean_sed_std": sed_spread,
        }
        for index, name in enumerate(FEATURE_NAMES):
            row[f"{name}_mean"] = float(cluster[:, index].mean())
            row[f"{name}_std"] = float(cluster[:, index].std())
        rows.append(row)
    return rows


def run_kmeans(
    features: numpy.ndarray,
    standardized_features: numpy.ndarray,
    cluster_count: int,
    seed: int,
) -> tuple[numpy.ndarray, float]:
    """Run k-means and return stable labels plus inertia."""
    if cluster_count < 2:
        raise ValueError("cluster count must be at least 2")
    if cluster_count > standardized_features.shape[0]:
        raise ValueError(
            "cluster count cannot exceed number of valid pixels: "
            f"{cluster_count} > {standardized_features.shape[0]}"
        )
    _, labels = kmeans2(
        standardized_features,
        cluster_count,
        minit="++",
        seed=seed,
        iter=100,
    )
    labels = stable_relabel(labels, features, cluster_count)
    inertia = float(
        sum(
            numpy.sum((standardized_features[labels == label] - standardized_features[labels == label].mean(axis=0)) ** 2)
            for label in range(cluster_count)
        )
    )
    return labels, inertia


def write_summary(output_dir: Path, prefix: str, rows: list[dict[str, float | int | str]]) -> None:
    """Write cluster diagnostics to CSV."""
    fieldnames = ["k", "region", "pixels", "inertia", "mean_sed_std"]
    for name in FEATURE_NAMES:
        fieldnames.extend([f"{name}_mean", f"{name}_std"])
    with (output_dir / f"{prefix}_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_mask = read_analysis_mask(args.analysis_mask, args.nside)
    features, _ = build_feature_matrix(
        args.map_dir,
        analysis_mask,
        args.nside,
        args.epsilon,
    )
    standardized_features, means, stds = standardize(features)

    summary_rows: list[dict[str, float | int | str]] = []
    print(f"valid pixels: {int(numpy.count_nonzero(analysis_mask))}")
    for index, name in enumerate(FEATURE_NAMES):
        print(f"feature {name}: mean={means[index]:.8e} std={stds[index]:.8e}")

    for cluster_count in args.k:
        labels, inertia = run_kmeans(
            features,
            standardized_features,
            cluster_count,
            args.seed,
        )
        masks = labels_to_masks(labels, analysis_mask, cluster_count)
        save_masks(args.output_dir, args.prefix, cluster_count, masks)
        rows = summarize_clusters(
            cluster_count,
            labels,
            features,
            standardized_features,
            inertia,
        )
        summary_rows.extend(rows)
        counts = [int(numpy.count_nonzero(mask)) for mask in masks]
        mean_sed_std = numpy.mean([row["mean_sed_std"] for row in rows])
        print(
            f"k={cluster_count}: inertia={inertia:.8e} "
            f"counts={counts} mean_sed_std={mean_sed_std:.8e}"
        )

    write_summary(args.output_dir, args.prefix, summary_rows)
    print(f"output dir: {args.output_dir}")
    print(f"summary: {args.output_dir / f'{args.prefix}_summary.csv'}")


if __name__ == "__main__":
    main()
