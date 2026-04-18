#!/usr/bin/env python3
"""
collect_ncu_metrics.py

Run Nsight Compute for the benchmark binary and append a compact metrics row.
This script is resilient to missing counters: it records a status field instead
of failing hard, so long sweeps can continue.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Nsight Compute DRAM/L2 metrics")
    parser.add_argument("--binary", required=True, help="Path to benchmark binary")
    parser.add_argument("--mode", type=int, required=True, help="Benchmark MODE value")
    parser.add_argument("--stride", type=int, required=True, help="SPATIAL_STRIDE value")
    parser.add_argument("--threads", type=int, required=True, help="THREADS_PER_BLOCK value")
    parser.add_argument("--out", default="ncu_metrics.csv", help="Output CSV path")
    parser.add_argument("--raw-log", default="", help="Optional path to write raw NCU output")
    parser.add_argument(
        "--metrics",
        default=(
            "dram__bytes_read.sum,dram__bytes_write.sum,"
            "dram__sectors_read.sum,dram__sectors_write.sum,"
            "lts__t_sectors_op_read.sum,lts__t_requests_op_read.sum"
        ),
        help="Comma-separated NCU metrics",
    )
    parser.add_argument(
        "--kernel-name",
        default="",
        help="Optional kernel-name filter for Nsight Compute (leave empty to profile by launch index)",
    )
    parser.add_argument(
        "--launch-skip",
        type=int,
        default=1,
        help="Number of kernel launches to skip before profiling (default: 1 to skip warmup_data)",
    )
    parser.add_argument(
        "--launch-count",
        type=int,
        default=1,
        help="Number of launches to profile after skip (default: 1)",
    )
    return parser.parse_args()


def parse_number(text: str) -> Optional[float]:
    cleaned = text.strip().strip('"').replace(",", "")
    if not cleaned or cleaned.upper() == "N/A":
        return None
    try:
        return float(cleaned)
    except ValueError:
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
        if m:
            return float(m.group(0))
    return None


def parse_metric_rows(raw_output: str, metric_names: List[str]) -> Dict[str, float]:
    values = {name: 0.0 for name in metric_names}

    reader = csv.reader(raw_output.splitlines())
    for row in reader:
        if not row:
            continue
        row_joined = ",".join(row)
        for metric in metric_names:
            if metric in row_joined:
                numeric = parse_number(row[-1])
                if numeric is not None:
                    values[metric] += numeric

    return values


def append_row(path: str, header: List[str], row: List[object]) -> None:
    needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow(header)
        writer.writerow(row)


def main() -> int:
    args = parse_args()

    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]

    cmd = [
        "ncu",
        "--csv",
        "--page",
        "raw",
        "--target-processes",
        "all",
        "--launch-skip",
        str(args.launch_skip),
        "--launch-count",
        str(args.launch_count),
        "--metrics",
        ",".join(metric_names),
    ]
    if args.kernel_name:
        cmd.extend(["--kernel-name-base", "demangled", "--kernel-name", args.kernel_name])
    cmd.append(args.binary)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    raw_output = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")

    if args.raw_log:
        os.makedirs(os.path.dirname(args.raw_log) or ".", exist_ok=True)
        with open(args.raw_log, "w") as f:
            f.write(raw_output)

    status = "ok"
    if proc.returncode != 0:
        status = f"ncu_failed_{proc.returncode}"
    if status == "ok" and "no kernels were profiled" in raw_output.lower():
        status = "no_kernels_profiled"

    metrics = parse_metric_rows(raw_output, metric_names)

    dram_bytes_read = metrics.get("dram__bytes_read.sum", 0.0)
    dram_bytes_write = metrics.get("dram__bytes_write.sum", 0.0)

    # Fallback to sector estimates if byte counters are unavailable.
    if dram_bytes_read == 0.0:
        dram_bytes_read = metrics.get("dram__sectors_read.sum", 0.0) * 32.0
    if dram_bytes_write == 0.0:
        dram_bytes_write = metrics.get("dram__sectors_write.sum", 0.0) * 32.0

    dram_bytes_total = dram_bytes_read + dram_bytes_write

    l2_read_sectors = (
        metrics.get("lts__t_sectors_op_read.sum", 0.0)
        + metrics.get("lts__t_sectors_srcunit_tex_op_read.sum", 0.0)
    )
    l2_read_requests = (
        metrics.get("lts__t_requests_op_read.sum", 0.0)
        + metrics.get("lts__t_requests_srcunit_tex_op_read.sum", 0.0)
    )

    if dram_bytes_total <= 0.0 and status == "ok":
        status = "missing_dram_metrics"

    header = [
        "Mode",
        "Stride",
        "ThreadsPerBlock",
        "DramBytesRead",
        "DramBytesWrite",
        "DramBytesTotal",
        "DramSectorsRead",
        "DramSectorsWrite",
        "L2ReadSectors",
        "L2ReadRequests",
        "Status",
    ]

    row = [
        args.mode,
        args.stride,
        args.threads,
        f"{dram_bytes_read:.6f}",
        f"{dram_bytes_write:.6f}",
        f"{dram_bytes_total:.6f}",
        f"{metrics.get('dram__sectors_read.sum', 0.0):.6f}",
        f"{metrics.get('dram__sectors_write.sum', 0.0):.6f}",
        f"{l2_read_sectors:.6f}",
        f"{l2_read_requests:.6f}",
        status,
    ]

    append_row(args.out, header, row)

    print(
        "Collected NCU metrics"
        f" | mode={args.mode} stride={args.stride} threads={args.threads}"
        f" | dram_bytes_total={dram_bytes_total:.0f}"
        f" | status={status}"
    )

    # Never hard-fail the sweep solely because profiler counters were unavailable.
    return 0


if __name__ == "__main__":
    sys.exit(main())
