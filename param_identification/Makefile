# ──────────────────────────────────────────────────────────────────────────
# Makefile — GPU Cache Size Identification Tool
# Ref: Delestrac et al. ASAP 2024, Section V-A
# ──────────────────────────────────────────────────────────────────────────

# ── Compiler ───────────────────────────────────────────────────────────────
NVCC        := nvcc
PYTHON      := python3

# ── GPU Architecture ───────────────────────────────────────────────────────
# Change to match your GPU:
#   sm_70  = Tesla V100
#   sm_75  = Turing (RTX 20xx, T4)
#   sm_80  = Ampere A100
#   sm_86  = Ampere RTX 30xx / A6000
#   sm_89  = Ada Lovelace RTX 40xx
#   sm_90  = Hopper H100
ARCH        := sm_80

# ── Flags ──────────────────────────────────────────────────────────────────
NVCC_FLAGS  := -O2 -arch=$(ARCH) -Xcompiler -Wall
NVCC_FLAGS  += -lineinfo                     # keep line numbers for ncu

# ── Targets ────────────────────────────────────────────────────────────────
TARGET      := cache_probe
SRC         := cache_probe.cu

.PHONY: all run analyze clean help

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $< -o $@
	@echo "Built: $(TARGET)  (arch=$(ARCH))"

# ── Default run: full sweep, 60 logarithmic steps ──────────────────────────
run: $(TARGET)
	@mkdir -p results
	./$(TARGET) --steps 60 --iters 5 --threads 256 --blocks 4

# ── Quick smoke-test (dry-run, synthetic data, no GPU required) ────────────
dry-run: $(TARGET)
	@mkdir -p results
	./$(TARGET) --dry-run --steps 40 --iters 3

# ── Analyse CSV with Python ────────────────────────────────────────────────
analyze:
	$(PYTHON) analyze_cache.py --csv results/cache_miss_rates.csv --verbose
	$(PYTHON) plot_cache_miss_rates.py --no-pdf --dpi 150

# ── Full pipeline ──────────────────────────────────────────────────────────
pipeline: run analyze

# ── A100-specific run (192 kB L1, 40 MB L2, up to 1 GB DRAM) ─────────────
run-a100: $(TARGET)
	@mkdir -p results
	./$(TARGET) \
	    --min-kb 1    \
	    --max-gb 1    \
	    --steps  80   \
	    --iters  5    \
	    --threads 1024 \
	    --blocks  4

# ── Clean ──────────────────────────────────────────────────────────────────
clean:
	rm -f $(TARGET) results/*.csv results/*.txt

# ── Help ───────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  make              Build the CUDA profiler"
	@echo "  make run          Build and run (real ncu profiling)"
	@echo "  make dry-run      Run with synthetic data (no GPU needed)"
	@echo "  make analyze      Post-process CSV with Python"
	@echo "  make pipeline     run + analyze"
	@echo "  make run-a100     A100-optimised sweep"
	@echo "  make clean        Remove binary and results"
	@echo ""
	@echo "  ARCH=sm_86 make   Override GPU architecture"
	@echo ""
