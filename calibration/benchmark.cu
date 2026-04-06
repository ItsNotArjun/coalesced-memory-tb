#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <thread>
#include <atomic>
#include <cuda_runtime.h>
#include <nvml.h>

// --- RTX 5000 Ada Defaults ---
// L1 Cache is 128KB per SM. We use 100KB to guarantee we don't spill into L2.
#ifndef SIZE_BYTES
#define SIZE_BYTES (100 * 1024) 
#endif

// The pointer-chasing jump size (4 uint64_t = 32 bytes = 1 sector)
#ifndef CHASE_JUMP
#define CHASE_JUMP 4
#endif

// Determines Coalescence. 
// 1 = Fully coalesced (adjacent threads access adjacent 8-byte blocks)
// 4 = Uncoalesced (adjacent threads access memory 32-bytes apart)
#ifndef SPATIAL_STRIDE
#define SPATIAL_STRIDE 1 
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 32
#endif

#ifndef N_BLOCKS
#define N_BLOCKS 1
#endif

#ifndef REPEAT_SECOND_LOOP
#define REPEAT_SECOND_LOOP 10000 // High iteration count for stable power reading
#endif

#ifndef GPU_ID
#define GPU_ID 0
#endif

// NVML Power Monitoring Variables
std::atomic<bool> keep_monitoring(false);
std::atomic<unsigned long long> total_active_power(0);
std::atomic<unsigned int> active_samples(0);

// Background thread function to poll NVML power during kernel execution
void monitor_power(nvmlDevice_t device) {
    unsigned int power;
    while (keep_monitoring) {
        nvmlDeviceGetPowerUsage(device, &power); // Returned in milliwatts
        total_active_power += power;
        active_samples++;
        usleep(5000); // Poll every 5ms
    }
}

__global__ void load_data(uint64_t *my_array, uint64_t subtabSize) {
    uint64_t tid = threadIdx.x;
    
    // CPU initialized the pointers. We just offset the starting point to vary coalescence.
    // The modulo ensures we never access memory outside the 100KB L1 footprint.
    uint64_t start_idx = (tid * SPATIAL_STRIDE) % subtabSize;
    uint64_t *start_addr = my_array + start_idx;

    asm volatile(
        "\n{\n"
        ".reg .pred %p;\n"
        ".reg .u64 %tmp;\n"
        "mov.u64 %tmp, %0;\n\n"

        // WARMUP: Pull data into L1 Cache
        "$warmup_loop:\n"
        "ld.global.ca.u64 %tmp, [%tmp];\n"
        "setp.ne.u64 %p, %tmp, %0;\n" 
        "@%p bra $warmup_loop;\n"
        "bar.sync 0;\n" 

        // MEASUREMENT: All accesses here should be L1 Hits
        ".reg .u32 %k;\n"             
        "mov.u32 %k, 0;\n"
        "mov.u64 %tmp, %0;\n\n"

        "\n$start:\n"
        // Unrolled 16 times
        "ld.global.ca.u64 %tmp, [%tmp];\n" "ld.global.ca.u64 %tmp, [%tmp];\n"
        "ld.global.ca.u64 %tmp, [%tmp];\n" "ld.global.ca.u64 %tmp, [%tmp];\n"
        "ld.global.ca.u64 %tmp, [%tmp];\n" "ld.global.ca.u64 %tmp, [%tmp];\n"
        "ld.global.ca.u64 %tmp, [%tmp];\n" "ld.global.ca.u64 %tmp, [%tmp];\n"
        "ld.global.ca.u64 %tmp, [%tmp];\n" "ld.global.ca.u64 %tmp, [%tmp];\n"
        "ld.global.ca.u64 %tmp, [%tmp];\n" "ld.global.ca.u64 %tmp, [%tmp];\n"
        "ld.global.ca.u64 %tmp, [%tmp];\n" "ld.global.ca.u64 %tmp, [%tmp];\n"
        "ld.global.ca.u64 %tmp, [%tmp];\n" "ld.global.ca.u64 %tmp, [%tmp];\n"
        
        "setp.ne.u64 %p, %tmp, %0;\n" 
        "@%p bra $start;\n"
        "add.u32 %k, %k, 1;\n"
        "setp.lt.u32 %p, %k, %1;\n"
        "@%p bra $start;\n"
        "}" : "+l"(start_addr) : "n"(REPEAT_SECOND_LOOP));
}

int main() {
    cudaSetDevice(GPU_ID);
    nvmlInit();
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(GPU_ID, &device);

    // 1. Set Access Granularities (Minimum 32 Bytes)
    cudaFuncSetAttribute(load_data, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 32);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);

    uint64_t subtabSize = SIZE_BYTES / sizeof(uint64_t);
    uint64_t *h_a = (uint64_t *)malloc(SIZE_BYTES);
    uint64_t *d_a;
    cudaMalloc((void **)&d_a, SIZE_BYTES);

    // 2. CPU-Side Pointer Initialization (Bake the Chase Jump)
    for (uint64_t i = 0; i < subtabSize; i++) {
        h_a[i] = (uint64_t)(d_a + ((i + CHASE_JUMP) % subtabSize));
    }
    cudaMemcpy(d_a, h_a, SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

#ifndef CSV_OUTPUT
    printf("Array Size: %d KB | Threads: %d | Spatial Stride: %d\n", SIZE_BYTES/1024, THREADS_PER_BLOCK, SPATIAL_STRIDE);
    printf("Sleeping for 3 seconds to establish Idle Baseline...\n");
#endif

    // 3. The Idle Power Baseline
    usleep(3000000);
    unsigned int idle_power_mw = 0;
    nvmlDeviceGetPowerUsage(device, &idle_power_mw);

    // 4. Launch Kernel and Monitor Active Power
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    keep_monitoring = true;
    std::thread monitor(monitor_power, device);

    dim3 Db(THREADS_PER_BLOCK, 1, 1);
    dim3 Dg(N_BLOCKS, 1, 1);

    cudaEventRecord(start);
    load_data<<<Dg, Db>>>(d_a, subtabSize);
    cudaEventRecord(stop);
    
    cudaDeviceSynchronize();

    keep_monitoring = false;
    monitor.join();

    // 5. Calculate Time and Total Accesses
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double seconds = milliseconds / 1000.0;

    // 16 unrolled loads * REPEAT_SECOND_LOOP * THREADS_PER_BLOCK * N_BLOCKS
    uint64_t total_accesses = (uint64_t)16 * REPEAT_SECOND_LOOP * THREADS_PER_BLOCK * N_BLOCKS;

    // 6. Calculate Dynamic Energy
    double avg_active_power_w = (double)total_active_power / active_samples / 1000.0;
    double dynamic_power_w = avg_active_power_w - (idle_power_mw / 1000.0);
    
    // Power (W) * Time (s) = Energy (Joules)
    double dynamic_energy_j = dynamic_power_w * seconds;
    double energy_per_access_j = dynamic_energy_j / total_accesses;
    double energy_per_access_pj = energy_per_access_j * 1e12; // Picojoules

#ifdef CSV_OUTPUT
    // Print ONLY CSV data for the python script
    printf("%d,%d,%e\n", SPATIAL_STRIDE, THREADS_PER_BLOCK, energy_per_access_pj);
#else
    // Print Human Readable output
    printf("Idle Power: %.2f W\n", idle_power_mw / 1000.0);
    printf("Kernel Time: %.4f seconds\n", seconds);
    printf("Avg Active Power: %.2f W\n", avg_active_power_w);
    printf("Dynamic Power (Isolated): %.2f W\n", dynamic_power_w);
    printf("Total Accesses: %llu\n", (unsigned long long)total_accesses);
    printf("Dynamic Energy: %e J\n", dynamic_energy_j);
    printf("Energy per Access: %e J (%.2f pJ)\n", energy_per_access_j, energy_per_access_pj);
#endif

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    free(h_a);
    nvmlShutdown();
    return 0;
}