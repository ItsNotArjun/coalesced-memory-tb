# ─────────────────────────────────────────────────────────────────────────
# Makefile — GPU Energy Model Calibration Tool
# Ref: Delestrac et al. ASAP 2024, Section V-B
# ─────────────────────────────────────────────────────────────────────────

NVCC        := nvcc
ARCH        := sm_80     # A100; change for your GPU (sm_70/sm_86/sm_89/sm_90)
TARGET      := calibrate
SRC         := calibrate.cu

# -lnvidia-ml links NVML for the internal power sensor
NVCC_FLAGS  := -O2 -arch=$(ARCH) -lineinfo -Xcompiler -Wall,-pthread
LIBS        := -lnvidia-ml -lpthread -lm

.PHONY: all run-l1 run-l2 run-dram run-shared run-all dry-run clean help

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LIBS)
	@echo "Built: $(TARGET)  (arch=$(ARCH))"

# ── Real-GPU runs ──────────────────────────────────────────────────────────
# Uses paper defaults: 1024 threads, granularity=32, stride=32 (A100 sector)

run-l1: $(TARGET)
	@mkdir -p results
	./$(TARGET) --mem l1 \
	    --min-acc 15 --max-acc 3000 --acc-steps 20 \
	    --threads 1024 --blocks 4 \
	    --granularity 32 --stride 32 \
	    --n-runs 5 --n-iters 200

run-l2: $(TARGET)
	@mkdir -p results
	./$(TARGET) --mem l2 \
	    --min-acc 100 --max-acc 5000 --acc-steps 20 \
	    --threads 1024 --blocks 4 \
	    --granularity 32 --stride 32 \
	    --n-runs 5 --n-iters 200

run-dram: $(TARGET)
	@mkdir -p results
	./$(TARGET) --mem dram \
	    --min-acc 100 --max-acc 3000 --acc-steps 20 \
	    --threads 1024 --blocks 4 \
	    --granularity 32 --stride 32 \
	    --n-runs 5 --n-iters 100

run-shared: $(TARGET)
	@mkdir -p results
	./$(TARGET) --mem shared \
	    --min-acc 15 --max-acc 3000 --acc-steps 20 \
	    --threads 1024 --blocks 4 \
	    --granularity 32 --stride 32 \
	    --n-runs 5 --n-iters 200

run-all: run-l1 run-l2 run-dram run-shared

# ── Dry-run: synthetic data, no GPU needed ─────────────────────────────────
dry-run: $(TARGET)
	@mkdir -p results
	./$(TARGET) --dry-run --mem l1 \
	    --min-acc 15 --max-acc 3000 --acc-steps 20 \
	    --threads 1024 --blocks 4 \
	    --granularity 32 --stride 32

dry-run-all: $(TARGET)
	@mkdir -p results
	for m in l1 l2 dram shared; do \
	    ./$(TARGET) --dry-run --mem $$m \
	        --min-acc 15 --max-acc 3000 --acc-steps 20 \
	        --threads 1024 --blocks 4 \
	        --granularity 32 --stride 32; \
	done

# ── Custom granularity / stride example ────────────────────────────────────
run-l2-128b: $(TARGET)
	./$(TARGET) --mem l2 \
	    --granularity 128 --stride 128 \
	    --threads 1024 --blocks 4 \
	    --min-acc 100 --max-acc 5000 --acc-steps 20

clean:
	rm -f $(TARGET) results/calibration_*.csv

help:
	@echo ""
	@echo "  make                Build the calibration binary"
	@echo "  make run-l1         Calibrate L1 cache (LOAD + STORE)"
	@echo "  make run-l2         Calibrate L2 cache"
	@echo "  make run-dram       Calibrate DRAM"
	@echo "  make run-shared     Calibrate shared memory"
	@echo "  make run-all        Calibrate all memory levels"
	@echo "  make dry-run        Synthetic data (no GPU needed)"
	@echo "  make dry-run-all    Synthetic data for all levels"
	@echo "  make clean          Remove binary and CSVs"
	@echo ""
	@echo "  ARCH=sm_86 make     Override GPU architecture"
	@echo ""
	@echo "Key runtime flags:"
	@echo "  --granularity <B>   Access granularity in bytes (e.g. 32, 64, 128)"
	@echo "  --stride      <B>   Pointer-chase stride in bytes"
	@echo "  --threads     <N>   Threads per block (1..1024)"
	@echo "  --load-only         Skip STORE benchmark"
	@echo "  --store-only        Skip LOAD benchmark"
	@echo ""
