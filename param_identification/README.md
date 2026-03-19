# GPU Cache Size Identification Tool

> Implements **Section V-A – Parameter Identification Phase** from:  
> Delestrac et al., *"Analyzing GPU Energy Consumption in Data Movement and Storage"*, ASAP 2024.

---

## What This Tool Does

Determines the L1 and L2 cache sizes of a modern GPU by running a
**pointer-chasing** microbenchmark across a logarithmic sweep of array
sizes (default: 1 KB → 1 GB) and measuring the L1 / L2 miss rate at
each size using NVIDIA Nsight Compute performance counters.

As described in the paper (and shown in Fig. 2):

- **L1 miss rate** rises steeply once the working set exceeds the L1 capacity.
- **L2 miss rate** rises steeply once the working set exceeds the L2 capacity.
- Below each threshold, accesses are **hits**; above it, they are **misses** → DRAM accesses.

### Outputs

| File | Description |
|---|---|
| `results/cache_miss_rates.csv` | Full sweep data — one row per array size |
| `results/cache_analysis_summary.csv` | Detected cache boundaries (one row per cache level) |
| `results/cache_analysis_report.txt` | Human-readable boundary report |

---

## File Structure

```
param_identification/
├── cache_probe.cu                  # CUDA microbenchmark + ncu orchestrator
├── analyze_cache.py                # Python post-processor (boundary detection)
├── plot_cache_miss_rates.py        # creates graph from given data
├── Makefile                        # Build + run shortcuts
└── README.md                       # This file
```

---

## Requirements

| Tool | Version | Purpose |
|---|---|---|
| NVIDIA CUDA Toolkit | ≥ 11.0 | Compile `cache_probe.cu` |
| NVIDIA Nsight Compute (`ncu`) | ≥ 2021.x | Hardware performance counters |
| Python | ≥ 3.8 | `analyze_cache.py` |
| GPU | Any NVIDIA (Volta+) | Target device |

**matplotlib** python package is required to create the graph.

---

## Quick Start

### 1. Build

```bash
# Default: compiles for A100 (sm_80)
make

# For other GPUs, set ARCH:
ARCH=sm_86 make      # RTX 30xx / A6000
ARCH=sm_70 make      # Tesla V100
ARCH=sm_75 make      # Turing (RTX 20xx, T4)
ARCH=sm_89 make      # Ada Lovelace (RTX 40xx)
ARCH=sm_90 make      # Hopper H100
```

### 2. Run the sweep

```bash
# Standard run (uses ncu for hardware counters)
make run

# A100-specific sweep (wider range, 1024 threads)
make run-a100

# Dry-run: synthetic data, no GPU or ncu required (for testing)
make dry-run
```

### 3. Analyse the CSV

```bash
make analyze
```

### 4. Full pipeline

```bash
make pipeline
```

---

## CSV Format

### `results/cache_miss_rates.csv`

```
# GPU Cache Size Identification
# Method: pointer-chasing miss-rate sweep (Section V-A)
# Reference: Delestrac et al. ASAP 2024
# GPU: NVIDIA A100-SXM4-80GB
# Stride (bytes): 128
# ...
# Detected boundaries (miss_rate >= 50%):
#   L1 cache size estimate : 196608 bytes  (192.0 KB  /  0.188 MB)
#   L2 cache size estimate : 41943040 bytes  (40960.0 KB  /  40.000 MB)
#
array_size_bytes,array_size_kb,array_size_mb,l1_miss_rate_pct,l2_miss_rate_pct,l1_hit_rate_pct,l2_hit_rate_pct,memory_region
1024,1.000,0.000977,0.00,0.00,100.00,100.00,L1_range
...
```

#### Columns

| Column | Type | Description |
|---|---|---|
| `array_size_bytes` | int | Working-set size in bytes |
| `array_size_kb` | float | Same, in kilobytes |
| `array_size_mb` | float | Same, in megabytes |
| `l1_miss_rate_pct` | float | L1 cache miss rate 0–100% |
| `l2_miss_rate_pct` | float | L2 cache miss rate 0–100% |
| `l1_hit_rate_pct` | float | 100 − l1_miss_rate_pct |
| `l2_hit_rate_pct` | float | 100 − l2_miss_rate_pct |
| `memory_region` | string | `L1_range` / `L2_range` / `DRAM_range` |

### `results/cache_analysis_summary.csv`

One row per cache level with size estimates from three methods:

| Column | Description |
|---|---|
| `cache_level` | L1 or L2 |
| `threshold_50_bytes/kb/mb` | First size where miss rate ≥ 50% |
| `threshold_99_bytes/kb/mb` | First size where miss rate ≥ 99% (full saturation) |
| `inflection_bytes/kb/mb` | Size at maximum miss-rate gradient (log scale) |
| `confidence` | HIGH / MEDIUM / LOW / UNDETECTED |

---

## How It Works (Paper Methodology)

### Step 1 – Parameter Identification (Section V-A)

The tool runs the pointer-chasing kernel for each array size in the sweep.
The stride between pointer links is set to `STRIDE_BYTES = 128` (≥ max cache
line size) to ensure every access lands on a fresh cache line, preventing
false hits.

```
array[i] = (i + stride_elems) % n_elems   ← chain initialisation
pos = array[pos]                            ← dependent load in kernel
```

### Step 2 – Counter Reading (via ncu)

For each array size, the binary re-invokes `ncu` targeting the
`pointer_chase_kernel` and reads:

```
l1tex__t_sector_hit_rate.pct    → L1 hit rate
lts__t_sector_hit_rate.pct      → L2 hit rate
```

Miss rate = 100 − hit rate.

### Step 3 – Boundary Detection (Python post-processor)

Three methods are applied:

1. **Threshold-50**: first size where miss rate ≥ 50% — matches Fig. 2 of the paper.
2. **Threshold-99**: first size where miss rate ≥ 99% — full saturation point.
3. **Inflection**: size of steepest miss-rate slope in log-size space.

Expected results on an A100 (from the paper):
- L1: miss rate rises sharply above ~150 KB, saturates at ~190 KB.
- L2: miss rate rises sharply above ~15 MB, saturates at ~22 MB.

---

## Performance Counters Reference

### Ampere (A100, sm_80)
```
l1tex__t_sector_hit_rate.pct     — L1 cache hit rate
lts__t_sector_hit_rate.pct       — L2 cache hit rate
```

### Volta / Turing (V100, T4, RTX 20xx)
```
tex__t_sector_hit_rate.pct       — L1 hit rate
l2__t_sector_hit_rate.pct        — L2 hit rate
```

Check available metrics on your GPU:
```bash
ncu --query-metrics 2>&1 | grep -i "hit_rate"
```

---

## Advanced Usage

```bash
# Custom sweep range and density
./cache_probe --min-kb 1 --max-gb 2 --steps 80 --iters 10

# More threads for better signal-to-noise ratio
./cache_probe --threads 1024 --blocks 8

# Verbose: print ncu command for each point
./cache_probe --verbose
```

---

## Expected A100 Output

```
=================================================
 GPU Cache Size Identification Tool
 Based on: Delestrac et al. ASAP 2024, Section V-A
=================================================
 Device  : NVIDIA A100-SXM4-80GB
 SM count: 108
 VRAM    : 79.2 GiB
 L2 size : 40.0 MiB (reported by driver)
-------------------------------------------------
 Sweep   : [1024 B .. 1073741824 B], 60 steps
 Threads : 256 per block, 4 blocks
 Iters   : 5
=================================================

Step    Array size      L1 miss%   L2 miss%
------  --------------  ---------- ----------
1       1.0 KB          0.0        0.0
...
24      150.0 KB        12.4       0.0
25      170.0 KB        65.8       0.0
26      190.0 KB        99.9       0.0
...
48      15.0 MB         99.9       8.1
49      18.0 MB         99.9       51.3
50      22.0 MB         99.9       99.8
...

-------------------------------------------------
 CACHE SIZE ESTIMATES (miss rate ≥ 50% threshold)
-------------------------------------------------
 L1: 196608 B  =  192.0 KB  =  0.19 MB
 L2: 41943040 B  =  40960.0 KB  =  40.00 MB
 L2 (driver-reported): 40.0 MB
-------------------------------------------------
```
