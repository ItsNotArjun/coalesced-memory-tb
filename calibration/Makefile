# Default RTX 5000 Ada Base Clock (Adjust if needed for your specific setup)
CLOCK_SPEED ?= 1155
NVCC = nvcc
CFLAGS = -O3 -arch=sm_89 -lnvidia-ml 
TARGET = benchmark

.PHONY: all compile lock unlock sweep sweep_csv clean

all: compile

compile:
	@echo "Compiling..."
	$(NVCC) $(CFLAGS) benchmark.cu -o $(TARGET)

lock:
	@echo "Locking GPU clock to $(CLOCK_SPEED) MHz..."
	sudo nvidia-smi --lock-gpu-clocks=$(CLOCK_SPEED),$(CLOCK_SPEED)

unlock:
	@echo "Unlocking GPU clock..."
	sudo nvidia-smi --reset-gpu-clocks

power_mode:
	sudo nvidia-smi -pm 1

# Hardware Counter Verification & Sweep (Human Readable)
sweep:
	power_mode
	lock
	@echo "Starting Sweeps..."
	@for stride in 1 4 16; do \
		for threads in 1 32 128 512 1024; do \
			echo "====================================================="; \
			echo "Testing THREADS=$$threads | STRIDE=$$stride"; \
			$(NVCC) $(CFLAGS) -DTHREADS_PER_BLOCK=$$threads -DSPATIAL_STRIDE=$$stride benchmark.cu -o $(TARGET); \
			echo "\n[1] Running Energy Profiling..."; \
			./$(TARGET); \
			echo "\n[2] Running NCU Hardware Counter Verification..."; \
			ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__hit_rate_hwctls.avg --page raw ./$(TARGET) | grep -E "l1tex__"; \
		done \
	done
	@$(MAKE) unlock

# Dedicated target for clean CSV generation (No NCU, no formatting)
sweep_csv:
	power_mode
	lock
	@echo "Starting CSV Sweep..."
	@echo "Stride,ThreadsPerBlock,EnergyPerAccess" > results.csv
	@for stride in 1 4 16; do \
		for threads in 1 32 128 512 1024; do \
			echo "Running THREADS=$$threads | STRIDE=$$stride..."; \
			$(NVCC) $(CFLAGS) -DTHREADS_PER_BLOCK=$$threads -DSPATIAL_STRIDE=$$stride -DCSV_OUTPUT benchmark.cu -o $(TARGET); \
			./$(TARGET) >> results.csv; \
		done \
	done
	@$(MAKE) unlock
	@echo "Sweep complete. Data saved to results.csv"

clean:
	rm -f $(TARGET) results.csv