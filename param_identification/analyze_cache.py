#!/usr/bin/env python3
"""
analyze_cache.py
================
Post-processing companion to cache_probe.cu.

Reads  results/cache_miss_rates.csv  (produced by the CUDA profiler),
detects cache-size boundaries with three complementary methods, and
writes an extended summary CSV + a human-readable report.

Usage
-----
    python3 analyze_cache.py [--csv PATH] [--threshold 50] [--verbose]

Outputs
-------
    results/cache_miss_rates.csv          — as written by cache_probe.cu
    results/cache_analysis_summary.csv    — per-cache boundary table
    results/cache_analysis_report.txt     — human-readable report

Reference
---------
Delestrac et al., "Analyzing GPU Energy Consumption in Data Movement and
Storage", ASAP 2024, Section V-A: Parameter Identification Phase.
"""

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ──────────────────────────── data structures ──────────────────────────────

@dataclass
class MeasurementPoint:
    array_size_bytes: int
    array_size_kb:    float
    array_size_mb:    float
    l1_miss_pct:      float
    l2_miss_pct:      float
    l1_hit_pct:       float
    l2_hit_pct:       float
    memory_region:    str   # filled in by analysis


@dataclass
class CacheBoundary:
    name:          str
    miss_rates:    List[float]
    sizes:         List[int]
    # Results
    threshold_50:  Optional[int]   = None   # first size where miss ≥ 50 %
    threshold_100: Optional[int]   = None   # first size where miss ≥ 99 %
    inflection:    Optional[int]   = None   # size of maximum derivative
    confidence:    str             = "N/A"


# ──────────────────────────── CSV loading ──────────────────────────────────

def load_csv(path: str) -> Tuple[List[MeasurementPoint], dict]:
    """
    Parse the CSV written by cache_probe.cu.
    Lines starting with '#' are treated as metadata.
    Returns (data_points, metadata_dict).
    """
    points:   List[MeasurementPoint] = []
    metadata: dict = {}

    with open(path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # extract key: value from comment lines
                body = line.lstrip("# ").strip()
                if ":" in body:
                    k, _, v = body.partition(":")
                    metadata[k.strip()] = v.strip()
                continue

            # First non-comment, non-empty line is the header
            break

        reader = csv.DictReader(f,
            fieldnames=[
                "array_size_bytes", "array_size_kb", "array_size_mb",
                "l1_miss_rate_pct", "l2_miss_rate_pct",
                "l1_hit_rate_pct",  "l2_hit_rate_pct",
                "memory_region",
            ])

        # The file pointer is now past the header; just read data rows
        for row in reader:
            try:
                points.append(MeasurementPoint(
                    array_size_bytes = int(row["array_size_bytes"]),
                    array_size_kb    = float(row["array_size_kb"]),
                    array_size_mb    = float(row["array_size_mb"]),
                    l1_miss_pct      = float(row["l1_miss_rate_pct"]),
                    l2_miss_pct      = float(row["l2_miss_rate_pct"]),
                    l1_hit_pct       = float(row["l1_hit_rate_pct"]),
                    l2_hit_pct       = float(row["l2_hit_rate_pct"]),
                    memory_region    = row["memory_region"].strip(),
                ))
            except (ValueError, KeyError):
                continue   # skip malformed rows

    return points, metadata


# ──────────────────────────── boundary detection ───────────────────────────

def _threshold_boundary(sizes: List[int], rates: List[float],
                        pct: float) -> Optional[int]:
    """First array size where miss_rate >= pct%."""
    for i in range(1, len(rates)):
        if rates[i] >= pct and rates[i - 1] < pct:
            # Linear interpolation for a more precise estimate
            r0, r1 = rates[i - 1], rates[i]
            s0, s1 = sizes[i - 1], sizes[i]
            t = (pct - r0) / (r1 - r0) if r1 != r0 else 0.5
            return int(s0 + t * (s1 - s0))
    return None


def _inflection_boundary(sizes: List[int], rates: List[float]) -> Optional[int]:
    """
    Point of maximum gradient in log-size space.
    Corresponds to the steepest part of the miss-rate transition curve —
    the best single-point estimate of the cache capacity.
    """
    if len(sizes) < 3:
        return None
    log_sizes = [math.log2(s) for s in sizes]
    max_grad  = 0.0
    best_idx  = None
    for i in range(1, len(rates) - 1):
        grad = abs(rates[i + 1] - rates[i - 1]) / (log_sizes[i + 1] - log_sizes[i - 1] + 1e-12)
        if grad > max_grad:
            max_grad = grad
            best_idx = i
    return sizes[best_idx] if best_idx is not None else None


def detect_boundaries(points: List[MeasurementPoint]) -> Tuple[CacheBoundary, CacheBoundary]:
    """
    Run all three detection methods on L1 and L2 miss-rate series.
    Returns (l1_boundary, l2_boundary).
    """
    sizes   = [p.array_size_bytes for p in points]
    l1_miss = [p.l1_miss_pct      for p in points]
    l2_miss = [p.l2_miss_pct      for p in points]

    l1 = CacheBoundary(name="L1", miss_rates=l1_miss, sizes=sizes)
    l2 = CacheBoundary(name="L2", miss_rates=l2_miss, sizes=sizes)

    for cb, miss in [(l1, l1_miss), (l2, l2_miss)]:
        cb.threshold_50  = _threshold_boundary(sizes, miss, 50.0)
        cb.threshold_100 = _threshold_boundary(sizes, miss, 99.0)
        cb.inflection    = _inflection_boundary(sizes, miss)

        # Confidence heuristic: high if all three methods agree within 2×
        estimates = [e for e in
                     [cb.threshold_50, cb.inflection] if e is not None]
        if len(estimates) >= 2:
            ratio = max(estimates) / (min(estimates) + 1)
            cb.confidence = "HIGH" if ratio < 2.0 else "MEDIUM"
        elif len(estimates) == 1:
            cb.confidence = "LOW (only one method found a boundary)"
        else:
            cb.confidence = "UNDETECTED"

    return l1, l2


# ──────────────────────────── region labelling ─────────────────────────────

def label_regions(points: List[MeasurementPoint],
                  l1: CacheBoundary, l2: CacheBoundary) -> None:
    """
    Re-label each measurement point with its memory region
    (L1_range / L2_range / DRAM_range) using detected boundaries.
    """
    l1b = l1.threshold_50
    l2b = l2.threshold_50

    for p in points:
        if l1b and p.array_size_bytes < l1b:
            p.memory_region = "L1_range"
        elif l2b and p.array_size_bytes < l2b:
            p.memory_region = "L2_range"
        else:
            p.memory_region = "DRAM_range"


# ──────────────────────────── human-readable size ──────────────────────────

def human(n: Optional[int]) -> str:
    if n is None:
        return "not detected"
    if n >= 1 << 30: return f"{n / (1<<30):.3f} GB  ({n:,} bytes)"
    if n >= 1 << 20: return f"{n / (1<<20):.3f} MB  ({n:,} bytes)"
    if n >= 1 << 10: return f"{n / (1<<10):.1f} KB  ({n:,} bytes)"
    return f"{n} B"


# ──────────────────────────── report writer ────────────────────────────────

def write_report(path: str, points: List[MeasurementPoint],
                 l1: CacheBoundary, l2: CacheBoundary,
                 metadata: dict) -> None:
    lines = []
    sep   = "=" * 60

    lines += [
        sep,
        " GPU Cache Size Identification Report",
        " Method: pointer-chasing miss-rate sweep (Section V-A)",
        " Ref: Delestrac et al. ASAP 2024",
        sep,
    ]

    if metadata:
        lines.append("\n[Device Metadata]")
        for k, v in metadata.items():
            lines.append(f"  {k}: {v}")

    lines += [
        "",
        "[Detected Cache Boundaries]",
        "",
        "  Three detection methods used:",
        "    (1) Threshold-50  : first size where miss rate >= 50%",
        "    (2) Threshold-99  : first size where miss rate >= 99%",
        "    (3) Inflection    : size of maximum gradient (log-scale)",
        "",
    ]

    for cb in (l1, l2):
        lines += [
            f"  {cb.name} Cache:",
            f"    Threshold-50  : {human(cb.threshold_50)}",
            f"    Threshold-99  : {human(cb.threshold_100)}",
            f"    Inflection pt : {human(cb.inflection)}",
            f"    Confidence    : {cb.confidence}",
            "",
        ]

    lines += [
        "[Region Classification]",
        "  (based on Threshold-50 boundaries)",
        "",
    ]

    region_counts = {}
    for p in points:
        region_counts[p.memory_region] = region_counts.get(p.memory_region, 0) + 1

    for region, count in sorted(region_counts.items()):
        lines.append(f"  {region}: {count} measurement points")

    lines += [
        "",
        "[Summary Table]",
        "",
        f"  {'Array Size':>14}  {'L1 miss%':>10}  {'L2 miss%':>10}  {'Region':<12}",
        f"  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*12}",
    ]

    for p in points:
        if p.array_size_bytes >= 1 << 20:
            sz = f"{p.array_size_mb:.2f} MB"
        elif p.array_size_bytes >= 1 << 10:
            sz = f"{p.array_size_kb:.1f} KB"
        else:
            sz = f"{p.array_size_bytes} B"

        lines.append(
            f"  {sz:>14}  {p.l1_miss_pct:>10.1f}  {p.l2_miss_pct:>10.1f}  {p.memory_region:<12}"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Report written to: {path}")


# ──────────────────────────── summary CSV ──────────────────────────────────

def write_summary_csv(path: str, l1: CacheBoundary, l2: CacheBoundary) -> None:
    """
    Write a compact CSV with one row per cache level summarising the
    detected size estimates from each method.
    """
    rows = [
        [
            "cache_level",
            "threshold_50_bytes", "threshold_50_kb", "threshold_50_mb",
            "threshold_99_bytes", "threshold_99_kb", "threshold_99_mb",
            "inflection_bytes",   "inflection_kb",   "inflection_mb",
            "confidence",
        ]
    ]

    for cb in (l1, l2):
        def parts(n):
            if n is None:
                return ["", "", ""]
            return [str(n), f"{n/1024:.3f}", f"{n/1048576:.6f}"]

        rows.append(
            [cb.name]
            + parts(cb.threshold_50)
            + parts(cb.threshold_100)
            + parts(cb.inflection)
            + [cb.confidence]
        )

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Summary CSV written to: {path}")


# ──────────────────────────── main ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse cache miss-rate sweep CSV from cache_probe.cu")
    parser.add_argument("--csv",       default="results/cache_miss_rates.csv",
                        help="Input CSV path")
    parser.add_argument("--threshold", type=float, default=50.0,
                        help="Miss-rate threshold for boundary detection (default 50)")
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}", file=sys.stderr)
        print("Run cache_probe first to generate results.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.csv} ...")
    points, metadata = load_csv(args.csv)
    print(f"  Loaded {len(points)} measurement points.")

    if not points:
        print("Error: no data rows found in CSV.", file=sys.stderr)
        sys.exit(1)

    l1, l2 = detect_boundaries(points)
    label_regions(points, l1, l2)

    print("\n--- Detected Boundaries ---")
    for cb in (l1, l2):
        print(f"  {cb.name}  threshold-50={human(cb.threshold_50)}"
              f"  inflection={human(cb.inflection)}"
              f"  confidence={cb.confidence}")

    out_dir = os.path.dirname(args.csv) or "results"
    os.makedirs(out_dir, exist_ok=True)

    write_summary_csv(os.path.join(out_dir, "cache_analysis_summary.csv"), l1, l2)
    write_report(os.path.join(out_dir, "cache_analysis_report.txt"),
                 points, l1, l2, metadata)

    if args.verbose:
        print("\n--- Full Data ---")
        print(f"{'Size':>12}  {'L1 miss':>8}  {'L2 miss':>8}  {'Region'}")
        for p in points:
            print(f"{p.array_size_bytes:>12}  {p.l1_miss_pct:>8.1f}  "
                  f"{p.l2_miss_pct:>8.1f}  {p.memory_region}")


if __name__ == "__main__":
    main()
