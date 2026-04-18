# --- H100 HOPPER DEFAULTS ---
# Base clock for this specific H100 node is 1065 MHz (from nvidia-smi -q -d CLOCK)
CLOCK_SPEED ?= 1065
PYTHON ?= python3
NVCC = nvcc
CFLAGS = -O3 -arch=sm_90 -lnvidia-ml 
TARGET = benchmark

# --- PHYSICS & TIMING SETTINGS ---
# 132 SMs on the H100 (SXM). 264 blocks guarantees 100% chip saturation.
BLOCKS ?= 264

# Keep thread count fixed to isolate coalescing behavior from occupancy effects.
THREADS ?= 256

# Disorder sweep for MODE=1 (continuous degradation experiment).
DISORDER_STRIDES ?= 1 2 4 8 16 32 64
BASELINE_STRIDE ?= 1
RANDOM_STRIDE ?= 1

# Keep loops tiny for NCU so the profiler doesn't crash (TDR timeout)
NCU_LOOPS ?= 100
NCU_LAUNCH_SKIP ?= 1
NCU_LAUNCH_COUNT ?= 1

# Make loops MASSIVE for CSV so NVML has 15+ seconds to average out the 700W background noise
CSV_LOOPS ?= 500000

NCU_METRICS ?= dram__bytes_read.sum,dram__bytes_write.sum,dram__sectors_read.sum,dram__sectors_write.sum,lts__t_sectors_op_read.sum,lts__t_requests_op_read.sum

.PHONY: all compile lock unlock sweep sweep_csv sweep_ncu analyze_metrics pipeline clean

all: compile

compile:
	@echo "Compiling..."
	$(NVCC) $(CFLAGS) benchmark.cu -o $(TARGET)

lock:
	@echo "Enabling Persistence Mode and Locking GPU clock to $(CLOCK_SPEED) MHz..."
	nvidia-smi -pm 1
# 	sudo nvidia-smi --lock-gpu-clocks=$(CLOCK_SPEED),$(CLOCK_SPEED)

unlock:
	@echo "Unlocking GPU clock..."
# 	sudo nvidia-smi --reset-gpu-clocks

# Hardware Counter Verification Sweep (Human Readable, FAST)
sweep: lock
	@echo "Starting Sweeps..."
	@for mode in 0 2; do \
		if [ $$mode -eq 0 ]; then stride=$(BASELINE_STRIDE); else stride=$(RANDOM_STRIDE); fi; \
		echo "====================================================="; \
		echo "Testing MODE=$$mode | THREADS=$(THREADS) | STRIDE=$$stride"; \
		$(NVCC) $(CFLAGS) -DMODE=$$mode -DTHREADS_PER_BLOCK=$(THREADS) -DSPATIAL_STRIDE=$$stride -DN_BLOCKS=$(BLOCKS) -DREPEAT_SECOND_LOOP=$(NCU_LOOPS) benchmark.cu -o $(TARGET); \
		echo "\n[1] Running Energy Profiling..."; \
		./$(TARGET); \
		echo "\n[2] Running NCU counter capture..."; \
		ncu --metrics $(NCU_METRICS) --page raw ./$(TARGET) | grep -E "dram__|lts__"; \
	done
	@for stride in $(DISORDER_STRIDES); do \
		echo "====================================================="; \
		echo "Testing MODE=1 | THREADS=$(THREADS) | STRIDE=$$stride"; \
		$(NVCC) $(CFLAGS) -DMODE=1 -DTHREADS_PER_BLOCK=$(THREADS) -DSPATIAL_STRIDE=$$stride -DN_BLOCKS=$(BLOCKS) -DREPEAT_SECOND_LOOP=$(NCU_LOOPS) benchmark.cu -o $(TARGET); \
		echo "\n[1] Running Energy Profiling..."; \
		./$(TARGET); \
		echo "\n[2] Running NCU counter capture..."; \
		ncu --metrics $(NCU_METRICS) --page raw ./$(TARGET) | grep -E "dram__|lts__"; \
	done
	@$(MAKE) unlock

# Dedicated target for clean CSV generation (LONG execution for stable power averages)
sweep_csv: lock
	@echo "Starting CSV Sweep..."
	@echo "Mode,Stride,ThreadsPerBlock,Blocks,ArrayBytes,KernelSeconds,DynamicPowerW,DynamicEnergyJ,TotalAccesses,UsefulBytes,EnergyPerAccessPJ,EnergyPerUsefulBytePJ" > results.csv
	@for mode in 0 2; do \
		if [ $$mode -eq 0 ]; then stride=$(BASELINE_STRIDE); else stride=$(RANDOM_STRIDE); fi; \
		echo "Running MODE=$$mode | THREADS=$(THREADS) | STRIDE=$$stride..."; \
		$(NVCC) $(CFLAGS) -DMODE=$$mode -DTHREADS_PER_BLOCK=$(THREADS) -DSPATIAL_STRIDE=$$stride -DCSV_OUTPUT -DN_BLOCKS=$(BLOCKS) -DREPEAT_SECOND_LOOP=$(CSV_LOOPS) benchmark.cu -o $(TARGET); \
		./$(TARGET) >> results.csv; \
	done
	@for stride in $(DISORDER_STRIDES); do \
		echo "Running MODE=1 | THREADS=$(THREADS) | STRIDE=$$stride..."; \
		$(NVCC) $(CFLAGS) -DMODE=1 -DTHREADS_PER_BLOCK=$(THREADS) -DSPATIAL_STRIDE=$$stride -DCSV_OUTPUT -DN_BLOCKS=$(BLOCKS) -DREPEAT_SECOND_LOOP=$(CSV_LOOPS) benchmark.cu -o $(TARGET); \
		./$(TARGET) >> results.csv; \
	done
	@$(MAKE) unlock
	@echo "Sweep complete. Data saved to results.csv"

# Profile DRAM and L2 counters for each point and store a structured CSV.
sweep_ncu: lock
	@echo "Starting NCU Sweep..."
	@rm -f ncu_metrics.csv
	@mkdir -p ncu_raw
	@for mode in 0 2; do \
		if [ $$mode -eq 0 ]; then stride=$(BASELINE_STRIDE); else stride=$(RANDOM_STRIDE); fi; \
		echo "Profiling MODE=$$mode | THREADS=$(THREADS) | STRIDE=$$stride..."; \
		$(NVCC) $(CFLAGS) -DMODE=$$mode -DTHREADS_PER_BLOCK=$(THREADS) -DSPATIAL_STRIDE=$$stride -DN_BLOCKS=$(BLOCKS) -DREPEAT_SECOND_LOOP=$(NCU_LOOPS) benchmark.cu -o $(TARGET); \
		$(PYTHON) collect_ncu_metrics.py --binary ./$(TARGET) --mode $$mode --stride $$stride --threads $(THREADS) --metrics "$(NCU_METRICS)" --launch-skip $(NCU_LAUNCH_SKIP) --launch-count $(NCU_LAUNCH_COUNT) --out ncu_metrics.csv --raw-log ncu_raw/mode$${mode}_stride$${stride}.csv; \
	done
	@for stride in $(DISORDER_STRIDES); do \
		echo "Profiling MODE=1 | THREADS=$(THREADS) | STRIDE=$$stride..."; \
		$(NVCC) $(CFLAGS) -DMODE=1 -DTHREADS_PER_BLOCK=$(THREADS) -DSPATIAL_STRIDE=$$stride -DN_BLOCKS=$(BLOCKS) -DREPEAT_SECOND_LOOP=$(NCU_LOOPS) benchmark.cu -o $(TARGET); \
		$(PYTHON) collect_ncu_metrics.py --binary ./$(TARGET) --mode 1 --stride $$stride --threads $(THREADS) --metrics "$(NCU_METRICS)" --launch-skip $(NCU_LAUNCH_SKIP) --launch-count $(NCU_LAUNCH_COUNT) --out ncu_metrics.csv --raw-log ncu_raw/mode1_stride$${stride}.csv; \
	done
	@$(MAKE) unlock
	@echo "NCU sweep complete. Data saved to ncu_metrics.csv"

analyze_metrics:
	$(PYTHON) analyze_energy_metrics.py --energy results.csv --ncu ncu_metrics.csv --out paper_metrics.csv

pipeline: sweep_csv sweep_ncu analyze_metrics

clean:
	rm -f $(TARGET) results.csv ncu_metrics.csv paper_metrics.csv
	rm -rf ncu_raw
