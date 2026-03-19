#!/usr/bin/env python3
"""
plot_cache_miss_rates.py
========================
Reproduces Fig. 2 from:
  Delestrac et al., "Analyzing GPU Energy Consumption in Data Movement
  and Storage", ASAP 2024 — Section V-A: Parameter Identification Phase.

Reads  results/cache_miss_rates.csv  (output of cache_probe.cu) and
produces  results/cache_miss_rates.png  (and .pdf).

Usage
-----
    python3 plot_cache_miss_rates.py
    python3 plot_cache_miss_rates.py --csv results/cache_miss_rates.csv
    python3 plot_cache_miss_rates.py --no-pdf --dpi 150
"""

import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")                          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

# ── colour palette (close to the paper's greyscale-friendly style) ──────────
C_L1_LINE   = "#1f77b4"   # blue  — L1 miss rate line
C_L2_LINE   = "#d62728"   # red   — L2 miss rate line
C_L1_SHADE  = "#aec7e8"   # light blue  — L1 range band
C_L2_SHADE  = "#ffbb78"   # light orange — L2 range band
C_DRAM_SHADE= "#c7c7c7"   # light grey  — DRAM range band
C_VLINE     = "#333333"   # dark grey   — boundary markers

# ── helpers ──────────────────────────────────────────────────────────────────

def human_bytes(n: int) -> str:
    """Return a tidy human-readable size string (e.g. '192 KB', '40 MB')."""
    if n >= 1 << 30: return f"{n/(1<<30):.0f} GB"
    if n >= 1 << 20: return f"{n/(1<<20):.0f} MB"
    if n >= 1 << 10: return f"{n/(1<<10):.0f} KB"
    return f"{n} B"


def log2_tick_formatter(value, pos):
    """Format x-axis ticks as human-readable byte sizes."""
    n = int(round(value))
    return human_bytes(n)


def parse_metadata(path: str) -> dict:
    """Extract key:value pairs from the # comment header of the CSV."""
    meta = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#"):
                break
            body = line.lstrip("# ").strip()
            if ":" in body:
                k, _, v = body.partition(":")
                meta[k.strip()] = v.strip()
    return meta


def load_data(path: str):
    """
    Returns parallel lists:
        sizes        – array size in bytes (int)
        l1_miss      – L1 miss rate 0–100 (float)
        l2_miss      – L2 miss rate 0–100 (float)
        regions      – memory_region string per row
    """
    sizes, l1_miss, l2_miss, regions = [], [], [], []
    with open(path, newline="") as f:
        # skip comment lines
        for line in f:
            if not line.startswith("#"):
                break          # consumed the header line
        reader = csv.DictReader(f, fieldnames=[
            "array_size_bytes", "array_size_kb", "array_size_mb",
            "l1_miss_rate_pct", "l2_miss_rate_pct",
            "l1_hit_rate_pct",  "l2_hit_rate_pct",
            "memory_region",
        ])
        for row in reader:
            try:
                sizes.append(int(row["array_size_bytes"]))
                l1_miss.append(float(row["l1_miss_rate_pct"]))
                l2_miss.append(float(row["l2_miss_rate_pct"]))
                regions.append(row["memory_region"].strip())
            except (ValueError, KeyError):
                continue
    return sizes, l1_miss, l2_miss, regions


def detect_boundary(sizes, miss_rates, threshold=50.0):
    """
    Linear-interpolated first crossing of `threshold` percent.
    Returns the interpolated size in bytes, or None.
    """
    for i in range(1, len(miss_rates)):
        if miss_rates[i] >= threshold and miss_rates[i - 1] < threshold:
            r0, r1 = miss_rates[i - 1], miss_rates[i]
            s0, s1 = sizes[i - 1], sizes[i]
            t = (threshold - r0) / (r1 - r0) if r1 != r0 else 0.5
            return s0 + t * (s1 - s0)
    return None


# ── main plot ────────────────────────────────────────────────────────────────

def make_plot(csv_path: str, out_dir: str, dpi: int, save_pdf: bool) -> None:

    # ── load ──
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        print("Run cache_probe first (or `make run` / `make dry-run`).",
              file=sys.stderr)
        sys.exit(1)

    meta              = parse_metadata(csv_path)
    sizes, l1, l2, _ = load_data(csv_path)

    gpu_name  = meta.get("GPU", "GPU")
    n_pts     = len(sizes)

    if n_pts == 0:
        print("Error: no data rows in CSV.", file=sys.stderr)
        sys.exit(1)

    # ── boundaries ──
    l1_b = detect_boundary(sizes, l1, 50.0)
    l2_b = detect_boundary(sizes, l2, 50.0)

    x_min = sizes[0]
    x_max = sizes[-1]

    # ── figure setup — match the paper's compact single-column style ──
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.18)

    # ── background region shading ──────────────────────────────────────────
    # L1 range: x_min  →  l1_b  (or first point where l1 > 50%)
    # L2 range: l1_b   →  l2_b
    # DRAM range: l2_b →  x_max

    shade_l1_end  = l1_b  if l1_b  else x_max
    shade_l2_end  = l2_b  if l2_b  else x_max
    shade_l2_start= l1_b  if l1_b  else x_min

    ax.axvspan(x_min,         shade_l1_end,  alpha=0.18, color=C_L1_SHADE,  zorder=0)
    if l1_b and l2_b:
        ax.axvspan(shade_l2_start, shade_l2_end,  alpha=0.18, color=C_L2_SHADE,  zorder=0)
        ax.axvspan(shade_l2_end,   x_max,         alpha=0.18, color=C_DRAM_SHADE, zorder=0)
    elif l1_b:
        ax.axvspan(shade_l2_start, x_max,         alpha=0.18, color=C_L2_SHADE,  zorder=0)

    # ── region label annotations (centred in each band, at the top) ───────
    def region_label(x_lo, x_hi, text):
        x_mid = math.exp((math.log(x_lo) + math.log(x_hi)) / 2)
        ax.text(x_mid, 97, text, ha="center", va="top",
                fontsize=8.5, color="#444444",
                fontstyle="italic", zorder=5)

    if l1_b and l2_b:
        region_label(x_min,   l1_b,  "L1 range")
        region_label(l1_b,    l2_b,  "L2 range")
        region_label(l2_b,    x_max, "DRAM range")
    elif l1_b:
        region_label(x_min,   l1_b,  "L1 range")
        region_label(l1_b,    x_max, "L2/DRAM range")

    # ── vertical boundary lines ────────────────────────────────────────────
    vline_kw = dict(color=C_VLINE, linewidth=1.0, linestyle="--", zorder=3)

    if l1_b:
        ax.axvline(l1_b, **vline_kw)
        ax.text(l1_b * 1.05, 54,
                f"≈ {human_bytes(int(l1_b))}",
                fontsize=7.5, color=C_VLINE, va="bottom", zorder=5)

    if l2_b:
        ax.axvline(l2_b, **vline_kw)
        ax.text(l2_b * 1.05, 54,
                f"≈ {human_bytes(int(l2_b))}",
                fontsize=7.5, color=C_VLINE, va="bottom", zorder=5)

    # ── miss rate lines ────────────────────────────────────────────────────
    ax.plot(sizes, l1, color=C_L1_LINE, linewidth=1.8,
            label="L1 miss rate", zorder=4)
    ax.plot(sizes, l2, color=C_L2_LINE, linewidth=1.8,
            label="L2 miss rate", zorder=4)

    # ── axes formatting ────────────────────────────────────────────────────
    ax.set_xscale("log", base=2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-3, 103)

    # Y axis: 0 / 25 / 50 / 75 / 100  with % sign — exactly like the paper
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{int(v)}%"))

    # X axis: human-readable powers-of-two labels
    # Identify "round" tick positions matching the paper's Fig 2 x-axis
    # (1k, 4k, 16k, 64k, 256k, 1M, 4M, 16M, 64M, 256M, 1G)
    nice_ticks = [
        1024, 4*1024, 16*1024, 64*1024, 256*1024,
        1024**2, 4*1024**2, 16*1024**2, 64*1024**2, 256*1024**2,
        1024**3,
    ]
    nice_ticks = [t for t in nice_ticks if x_min <= t <= x_max]
    ax.set_xticks(nice_ticks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(log2_tick_formatter))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=8)

    # Grid
    ax.grid(axis="y", linestyle=":", linewidth=0.6, color="#cccccc", zorder=0)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, color="#dddddd", zorder=0)

    ax.set_xlabel("Array size (bytes)", fontsize=9, labelpad=4)
    ax.set_ylabel("Miss rate", fontsize=9, labelpad=4)

    # ── title ──────────────────────────────────────────────────────────────
    title = (f"Miss rate of L1 and L2 caches — {gpu_name}\n"
             f"array sizes {human_bytes(x_min)} – {human_bytes(x_max)}"
             f"  ({n_pts} measurement points)")
    ax.set_title(title, fontsize=8.5, pad=6)

    # ── legend ──────────────────────────────────────────────────────────────
    line_handles, line_labels = ax.get_legend_handles_labels()
    shade_handles = [
        Patch(facecolor=C_L1_SHADE,   alpha=0.6, label="L1 range"),
        Patch(facecolor=C_L2_SHADE,   alpha=0.6, label="L2 range"),
        Patch(facecolor=C_DRAM_SHADE, alpha=0.6, label="DRAM range"),
    ]
    ax.legend(
        handles=line_handles + shade_handles,
        labels =line_labels  + [h.get_label() for h in shade_handles],
        fontsize=8, loc="center left",
        framealpha=0.9, edgecolor="#bbbbbb",
        handlelength=1.6, handletextpad=0.5,
        borderpad=0.5, labelspacing=0.35,
    )

    # ── save ──────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "cache_miss_rates.png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {png_path}")

    if save_pdf:
        pdf_path = os.path.join(out_dir, "cache_miss_rates.pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved: {pdf_path}")

    plt.close(fig)

    # ── console summary ───────────────────────────────────────────────────
    print("\n--- Detected Cache Boundaries (50% miss-rate threshold) ---")
    print(f"  L1 : {human_bytes(int(l1_b)) if l1_b else 'not detected'}")
    print(f"  L2 : {human_bytes(int(l2_b)) if l2_b else 'not detected'}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot L1/L2 cache miss-rate sweep — Fig. 2 replica")
    parser.add_argument("--csv",    default="results/cache_miss_rates.csv",
                        help="Input CSV (default: results/cache_miss_rates.csv)")
    parser.add_argument("--outdir", default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--dpi",    type=int, default=200,
                        help="PNG resolution in DPI (default: 200)")
    parser.add_argument("--no-pdf", action="store_true",
                        help="Skip PDF output")
    args = parser.parse_args()

    make_plot(
        csv_path = args.csv,
        out_dir  = args.outdir,
        dpi      = args.dpi,
        save_pdf = not args.no_pdf,
    )


if __name__ == "__main__":
    main()