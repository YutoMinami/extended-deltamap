#!/usr/bin/env python
"""Plot foreground-parameter correlation heatmaps from saved Minuit matrices."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a heatmap from a *_fg_corr.csv matrix written by TestDeltamap.py."
    )
    parser.add_argument("corr_csv", type=Path, help="Input foreground correlation CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to data/plots/<input-stem>.png.",
    )
    parser.add_argument(
        "--strong-output",
        type=Path,
        default=None,
        help="CSV for strongest off-diagonal correlations.",
    )
    parser.add_argument(
        "--strong-threshold",
        type=float,
        default=0.8,
        help="Absolute-correlation threshold for the strongest-pairs CSV.",
    )
    parser.add_argument(
        "--legacy-k4-combined-names",
        action="store_true",
        help="Map legacy x0..x11 names to k=4 combined-feature foreground names.",
    )
    return parser.parse_args()


def read_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    """Read a named square matrix CSV."""
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        names = header[1:]
        rows = []
        row_names = []
        for row in reader:
            row_names.append(row[0])
            rows.append([float(value) for value in row[1:]])
    if row_names != names:
        # Older files can have the same generic names in both places; reject only
        # true shape/name mismatches.
        if len(row_names) != len(names):
            raise ValueError("row and column names have different lengths")
    matrix = np.array(rows, dtype=float)
    if matrix.shape != (len(names), len(names)):
        raise ValueError(f"matrix shape {matrix.shape} does not match names {len(names)}")
    return names, matrix


def legacy_k4_combined_names(names: list[str]) -> list[str]:
    """Map x0..x11 to the k=4 combined-feature foreground parameter order."""
    expected = [f"x{index}" for index in range(12)]
    if names != expected:
        return names
    return [
        "beta_s_sreg0",
        "beta_s_sreg1",
        "beta_s_sreg2",
        "beta_s_sreg3",
        "beta_d_dreg0",
        "beta_d_dreg1",
        "beta_d_dreg2",
        "beta_d_dreg3",
        "T_d1_dreg0",
        "T_d1_dreg1",
        "T_d1_dreg2",
        "T_d1_dreg3",
    ]


def parameter_group(name: str) -> str:
    """Return a broad foreground-parameter group name."""
    if name.startswith("beta_s"):
        return "synch beta"
    if name.startswith("beta_d"):
        return "dust beta"
    if name.startswith("T_d1"):
        return "dust T"
    return "other"


def group_boundaries(names: list[str]) -> list[int]:
    """Return indices where broad parameter groups change."""
    boundaries = []
    previous = parameter_group(names[0])
    for index, name in enumerate(names[1:], start=1):
        group = parameter_group(name)
        if group != previous:
            boundaries.append(index)
            previous = group
    return boundaries


def write_strong_pairs(
    output_path: Path,
    names: list[str],
    matrix: np.ndarray,
    threshold: float,
) -> None:
    """Write strong off-diagonal correlations sorted by |corr|."""
    rows = []
    for i, name_i in enumerate(names):
        for j in range(i + 1, len(names)):
            corr = float(matrix[i, j])
            if abs(corr) >= threshold:
                name_j = names[j]
                rows.append(
                    {
                        "param_i": name_i,
                        "param_j": name_j,
                        "group_i": parameter_group(name_i),
                        "group_j": parameter_group(name_j),
                        "corr": corr,
                        "abs_corr": abs(corr),
                    }
                )
    rows.sort(key=lambda row: row["abs_corr"], reverse=True)
    with output_path.open("w", newline="") as stream:
        fieldnames = ["param_i", "param_j", "group_i", "group_j", "corr", "abs_corr"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmap(output_path: Path, names: list[str], matrix: np.ndarray) -> None:
    """Plot and save the correlation heatmap."""
    fig_size = max(7.0, 0.55 * len(names) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("correlation")

    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("Final foreground Minuit correlation")

    for boundary in group_boundaries(names):
        ax.axhline(boundary - 0.5, color="black", linewidth=1.4)
        ax.axvline(boundary - 0.5, color="black", linewidth=1.4)

    ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    names, matrix = read_matrix(args.corr_csv)
    if args.legacy_k4_combined_names:
        names = legacy_k4_combined_names(names)

    output = args.output
    if output is None:
        output = Path("data/plots") / f"{args.corr_csv.stem}_heatmap.png"

    strong_output = args.strong_output
    if strong_output is None:
        strong_output = output.with_name(f"{output.stem}_strong_pairs.csv")

    plot_heatmap(output, names, matrix)
    write_strong_pairs(strong_output, names, matrix, args.strong_threshold)
    print(f"saved heatmap: {output}")
    print(f"saved strong pairs: {strong_output}")


if __name__ == "__main__":
    main()
