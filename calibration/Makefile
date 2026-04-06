# Default RTX 5000 Ada Base Clock (Adjust if needed for your specific setup)
CLOCK_SPEED ?= 1155
NVCC = nvcc
CFLAGS = -O3 -arch=sm_89 -lnvidia-ml 
TARGET = benchmark

.PHONY: all compile lock unlock sweep clean

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

# Hardware Counter Verification & Sweep
# Sweeps over different thread counts and spatial strides (coalescence levels)
sweep: lock
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

clean:
	rm -f $(TARGET)