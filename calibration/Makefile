# --- H100 HOPPER DEFAULTS ---
# Base clock for this specific H100 node is 1065 MHz (from nvidia-smi -q -d CLOCK)
CLOCK_SPEED ?= 1065
NVCC = nvcc
CFLAGS = -O3 -arch=sm_90 -lnvidia-ml 
TARGET = benchmark

# --- PHYSICS & TIMING SETTINGS ---
# 132 SMs on the H100 (SXM). 264 blocks guarantees 100% chip saturation.
BLOCKS ?= 264

# Keep loops tiny for NCU so the profiler doesn't crash (TDR timeout)
NCU_LOOPS ?= 100

# Make loops MASSIVE for CSV so NVML has 15+ seconds to average out the 700W background noise
CSV_LOOPS ?= 500000

.PHONY: all compile lock unlock sweep sweep_csv clean

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

# Hardware Counter Verification & Sweep (Human Readable, FAST)
sweep: lock
	@echo "Starting Sweeps..."
	@for stride in 1 4 16; do \
		for threads in 1 32 128 512 1024; do \
			echo "====================================================="; \
			echo "Testing THREADS=$$threads | STRIDE=$$stride"; \
			$(NVCC) $(CFLAGS) -DTHREADS_PER_BLOCK=$$threads -DSPATIAL_STRIDE=$$stride -DN_BLOCKS=$(BLOCKS) -DREPEAT_SECOND_LOOP=$(NCU_LOOPS) benchmark.cu -o $(TARGET); \
			echo "\n[1] Running Energy Profiling..."; \
			./$(TARGET); \
			echo "\n[2] Running NCU Hardware Counter Verification..."; \
			ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__hit_rate_hwctls.avg --page raw ./$(TARGET) | grep -E "l1tex__"; \
		done \
	done
	@$(MAKE) unlock

# Dedicated target for clean CSV generation (LONG execution for stable power averages)
sweep_csv: lock
	@echo "Starting CSV Sweep..."
	@echo "Stride,ThreadsPerBlock,EnergyPerAccess" > results.csv
	@for stride in 1 4 16; do \
		for threads in 1 32 128 512 1024; do \
			echo "Running THREADS=$$threads | STRIDE=$$stride..."; \
			$(NVCC) $(CFLAGS) -DTHREADS_PER_BLOCK=$$threads -DSPATIAL_STRIDE=$$stride -DCSV_OUTPUT -DN_BLOCKS=$(BLOCKS) -DREPEAT_SECOND_LOOP=$(CSV_LOOPS) benchmark.cu -o $(TARGET); \
			./$(TARGET) >> results.csv; \
		done \
	done
	@$(MAKE) unlock
	@echo "Sweep complete. Data saved to results.csv"

clean:
	rm -f $(TARGET) results.csv
