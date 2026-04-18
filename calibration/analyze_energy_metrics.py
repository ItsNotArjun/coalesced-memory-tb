#!/usr/bin/env python3
"""
analyze_energy_metrics.py

Combine energy sweep data with Nsight Compute counters and produce paper-ready
metrics:
- amplification vs coalesced baseline
- efficiency = useful bytes / DRAM bytes transferred
- energy per useful byte and per DRAM byte
"""

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


MODE_LABEL = {
    0: "coalesced",
    1: "strided",
    2: "random",
}


@dataclass
class EnergyPoint:
    mode: int
    stride: int
    threads: int
    blocks: int
    array_bytes: int
    kernel_seconds: float
    dynamic_power_w: float
    dynamic_energy_j: float
    total_accesses: int
    useful_bytes: int
    energy_per_access_pj: float
    energy_per_useful_byte_pj: float


@dataclass
class NcuPoint:
    dram_bytes_total: float
    l2_read_sectors: float
    l2_read_requests: float
    status: str


def parse_int(row: dict, key: str, default: int = 0) -> int:
    value = row.get(key, "").strip()
    if not value:
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def parse_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_energy(path: str) -> List[EnergyPoint]:
    points: List[EnergyPoint] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "Mode",
            "Stride",
            "ThreadsPerBlock",
            "Blocks",
            "ArrayBytes",
            "KernelSeconds",
            "DynamicPowerW",
            "DynamicEnergyJ",
            "TotalAccesses",
            "UsefulBytes",
            "EnergyPerAccessPJ",
            "EnergyPerUsefulBytePJ",
        }
        if not required.issubset(set(reader.fieldnames or [])):
            missing = sorted(required.difference(set(reader.fieldnames or [])))
            raise ValueError(f"Energy CSV is missing columns: {', '.join(missing)}")

        for row in reader:
            points.append(
                EnergyPoint(
                    mode=parse_int(row, "Mode"),
                    stride=parse_int(row, "Stride"),
                    threads=parse_int(row, "ThreadsPerBlock"),
                    blocks=parse_int(row, "Blocks"),
                    array_bytes=parse_int(row, "ArrayBytes"),
                    kernel_seconds=parse_float(row, "KernelSeconds"),
                    dynamic_power_w=parse_float(row, "DynamicPowerW"),
                    dynamic_energy_j=parse_float(row, "DynamicEnergyJ"),
                    total_accesses=parse_int(row, "TotalAccesses"),
                    useful_bytes=parse_int(row, "UsefulBytes"),
                    energy_per_access_pj=parse_float(row, "EnergyPerAccessPJ"),
                    energy_per_useful_byte_pj=parse_float(row, "EnergyPerUsefulBytePJ"),
                )
            )
    return points


def load_ncu(path: str) -> Dict[Tuple[int, int, int], NcuPoint]:
    table: Dict[Tuple[int, int, int], NcuPoint] = {}
    if not path or not os.path.exists(path):
        return table

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = parse_int(row, "Mode")
            stride = parse_int(row, "Stride")
            threads = parse_int(row, "ThreadsPerBlock")
            table[(mode, stride, threads)] = NcuPoint(
                dram_bytes_total=parse_float(row, "DramBytesTotal"),
                l2_read_sectors=parse_float(row, "L2ReadSectors"),
                l2_read_requests=parse_float(row, "L2ReadRequests"),
                status=row.get("Status", "missing").strip() or "missing",
            )
    return table


def choose_baseline(points: List[EnergyPoint], baseline_stride: Optional[int]) -> EnergyPoint:
    candidates = [p for p in points if p.mode == 0]
    if not candidates:
        raise ValueError("No MODE=0 row found in energy CSV; cannot compute amplification")

    if baseline_stride is not None:
        stride_matches = [p for p in candidates if p.stride == baseline_stride]
        if stride_matches:
            return stride_matches[0]

    candidates.sort(key=lambda p: (p.stride, p.threads))
    return candidates[0]


def fmt(value: Optional[float], ndigits: int = 6) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return ""
    return f"{value:.{ndigits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute amplification and efficiency metrics")
    parser.add_argument("--energy", default="results.csv", help="Energy CSV path")
    parser.add_argument("--ncu", default="ncu_metrics.csv", help="NCU CSV path")
    parser.add_argument("--out", default="paper_metrics.csv", help="Output metrics CSV")
    parser.add_argument(
        "--baseline-stride",
        type=int,
        default=None,
        help="Preferred MODE=0 stride for baseline (default: first MODE=0 row)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.energy):
        print(f"Error: energy CSV not found: {args.energy}", file=sys.stderr)
        return 1

    energy_points = load_energy(args.energy)
    ncu_points = load_ncu(args.ncu)

    baseline = choose_baseline(energy_points, args.baseline_stride)
    baseline_e_useful = baseline.energy_per_useful_byte_pj

    baseline_ncu = ncu_points.get((baseline.mode, baseline.stride, baseline.threads))
    baseline_dram_bytes = baseline_ncu.dram_bytes_total if baseline_ncu else 0.0

    rows: List[List[str]] = []
    header = [
        "Mode",
        "ModeLabel",
        "Stride",
        "ThreadsPerBlock",
        "DynamicEnergyJ",
        "UsefulBytes",
        "EnergyPerAccessPJ",
        "EnergyPerUsefulBytePJ",
        "AmplificationVsCoalesced",
        "DramBytesTotal",
        "EfficiencyUsefulOverDram",
        "EnergyPerDramBytePJ",
        "DramByteAmplificationVsCoalesced",
        "L2ReadSectors",
        "L2ReadRequests",
        "ProfilerStatus",
    ]
    rows.append(header)

    for p in energy_points:
        key = (p.mode, p.stride, p.threads)
        ncu = ncu_points.get(key)

        amplification = None
        if baseline_e_useful != 0.0:
            amplification = p.energy_per_useful_byte_pj / baseline_e_useful

        dram_bytes_total = ncu.dram_bytes_total if ncu else 0.0
        efficiency = None
        energy_per_dram_byte_pj = None
        dram_byte_amp = None

        if dram_bytes_total > 0.0:
            efficiency = p.useful_bytes / dram_bytes_total
            energy_per_dram_byte_pj = (p.dynamic_energy_j * 1e12) / dram_bytes_total
            if baseline_dram_bytes > 0.0:
                dram_byte_amp = dram_bytes_total / baseline_dram_bytes

        rows.append(
            [
                str(p.mode),
                MODE_LABEL.get(p.mode, f"mode_{p.mode}"),
                str(p.stride),
                str(p.threads),
                f"{p.dynamic_energy_j:.9e}",
                str(p.useful_bytes),
                f"{p.energy_per_access_pj:.9e}",
                f"{p.energy_per_useful_byte_pj:.9e}",
                fmt(amplification, 6),
                f"{dram_bytes_total:.6f}" if ncu else "",
                fmt(efficiency, 6),
                fmt(energy_per_dram_byte_pj, 6),
                fmt(dram_byte_amp, 6),
                f"{ncu.l2_read_sectors:.6f}" if ncu else "",
                f"{ncu.l2_read_requests:.6f}" if ncu else "",
                ncu.status if ncu else "missing",
            ]
        )

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    random_points = [p for p in energy_points if p.mode == 2]
    max_strided = max((p for p in energy_points if p.mode == 1), key=lambda x: x.stride, default=None)

    print("--- Paper Metrics Summary ---")
    print(
        "Baseline (coalesced):"
        f" stride={baseline.stride}, threads={baseline.threads},"
        f" E_useful={baseline.energy_per_useful_byte_pj:.6f} pJ/byte"
    )
    if random_points and baseline_e_useful != 0.0:
        rand = random_points[0]
        rand_amp = rand.energy_per_useful_byte_pj / baseline_e_useful
        print(
            "Random amplification:"
            f" {rand_amp:.4f}x"
            f" (mode=2, stride={rand.stride})"
        )
    if max_strided and baseline_e_useful != 0.0:
        strided_amp = max_strided.energy_per_useful_byte_pj / baseline_e_useful
        print(
            "Max-stride amplification:"
            f" {strided_amp:.4f}x"
            f" (mode=1, stride={max_strided.stride})"
        )
    print(f"Wrote: {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
