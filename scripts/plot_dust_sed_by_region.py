"""Visualize dust SED shape (normalized polarization amplitude) for bright vs. faint regions.

Usage:
    uv run python scripts/plot_dust_sed_by_region.py
"""

import re
import sys
from pathlib import Path

import healpy
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
MAP_DIR = REPO / "examples" / "output_pysm3_ns8"
MASK_DIR = REPO / "data" / "regions"
MASK_NS4 = REPO / "examples" / "files" / "mask_p06_Nside4.v2.fits"
REF_FREQ = 337.0
NSIDE = 8


def load_dust_maps():
    """Return dict {freq_GHz: P_map} for all nside=8 dust map files."""
    maps = {}
    for f in sorted(MAP_DIR.glob("test01_nu*_d1_nside0008.fits")):
        m = re.search(r"nu(\d+)p(\d+)GHz", f.name)
        if m:
            freq = float(f"{m.group(1)}.{m.group(2)}")
            q, u = healpy.read_map(str(f), field=(1, 2), verbose=False)
            maps[freq] = np.sqrt(q**2 + u**2)
    return maps


def load_analysis_mask():
    """Return boolean pixel mask at nside=8."""
    m4 = healpy.read_map(str(MASK_NS4), verbose=False)
    m8 = healpy.ud_grade(m4, NSIDE)
    return m8 > 0.5


def main():
    print("Loading maps...")
    dust_maps = load_dust_maps()
    freqs = np.array(sorted(dust_maps.keys()))
    analysis_mask = load_analysis_mask()

    faint_mask = np.load(MASK_DIR / "dust337_median_ns8_sreg0_faint_pix.npy")
    bright_mask = np.load(MASK_DIR / "dust337_median_ns8_sreg1_bright_pix.npy")

    ref_map = dust_maps[REF_FREQ]

    regions = {"faint": faint_mask, "bright": bright_mask}
    colors = {"faint": "steelblue", "bright": "tomato"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: normalized SED (shape only)
    ax = axes[0]
    for label, mask in regions.items():
        # Per-pixel normalized SED: shape (n_freq, n_pix_in_region)
        sed = np.array([dust_maps[nu][mask] / ref_map[mask] for nu in freqs])
        mean = sed.mean(axis=1)
        err = sed.std(axis=1) / np.sqrt(mask.sum())
        ax.errorbar(freqs, mean, yerr=err, fmt="o-", color=colors[label],
                    label=f"{label} (N={mask.sum()})", capsize=3)

    ax.axvline(REF_FREQ, color="gray", linestyle="--", alpha=0.5, label=f"ref {REF_FREQ} GHz")
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel(f"P(ν) / P({REF_FREQ:.0f} GHz)  [mean ± SE over pixels]")
    ax.set_title("Normalized dust SED shape")
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: absolute amplitude per region
    ax = axes[1]
    for label, mask in regions.items():
        amps = np.array([dust_maps[nu][mask].mean() for nu in freqs])
        errs = np.array([dust_maps[nu][mask].std() / np.sqrt(mask.sum()) for nu in freqs])
        ax.errorbar(freqs, amps, yerr=errs, fmt="o-", color=colors[label],
                    label=f"{label} (N={mask.sum()})", capsize=3)

    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Mean P(ν)  [K_CMB, mean ± SE over pixels]")
    ax.set_title("Absolute dust polarization amplitude")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    out = REPO / "data" / "plots" / "dust_sed_by_region_ns8.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")

    # Print summary numbers
    print(f"\nNormalized SED at each freq (mean ± SE), ref={REF_FREQ} GHz")
    print(f"{'freq':>8}  {'faint mean':>12}  {'faint SE':>10}  {'bright mean':>12}  {'bright SE':>10}  {'ratio b/f':>10}")
    ref_faint = dust_maps[REF_FREQ][faint_mask].mean()
    ref_bright = dust_maps[REF_FREQ][bright_mask].mean()
    print(f"  ref amp at {REF_FREQ} GHz: faint={ref_faint:.4e}  bright={ref_bright:.4e}  ratio={ref_bright/ref_faint:.2f}x")
    for nu in freqs:
        f_sed = dust_maps[nu][faint_mask] / dust_maps[REF_FREQ][faint_mask]
        b_sed = dust_maps[nu][bright_mask] / dust_maps[REF_FREQ][bright_mask]
        f_m, f_e = f_sed.mean(), f_sed.std() / np.sqrt(faint_mask.sum())
        b_m, b_e = b_sed.mean(), b_sed.std() / np.sqrt(bright_mask.sum())
        print(f"  {nu:8.1f}  {f_m:12.4f}  {f_e:10.4f}  {b_m:12.4f}  {b_e:10.4f}  {b_m/f_m:10.4f}")


if __name__ == "__main__":
    main()
