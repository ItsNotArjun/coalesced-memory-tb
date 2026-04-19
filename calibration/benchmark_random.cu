#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <thread>
#include <atomic>
#include <time.h>
#include <cuda_runtime.h>
#include <nvml.h>

// 1 = L1 Cache, 2 = L2 Cache, 3 = DRAM
#ifndef MEM_LEVEL
#define MEM_LEVEL 1 
#endif

// --- H100 Architecture Memory Sizing ---
#if MEM_LEVEL == 1
    #define SIZE_BYTES (100ULL * 1024ULL)             // 100 KB
    #define MEM_NAME "L1"
    #define LOOP_SCALE 1                              
#elif MEM_LEVEL == 2
    #define SIZE_BYTES (20ULL * 1024ULL * 1024ULL)    // 20 MB
    #define MEM_NAME "L2"
    #define LOOP_SCALE 200                            
#elif MEM_LEVEL == 3
    #define SIZE_BYTES (150ULL * 1024ULL * 1024ULL)   // 150 MB
    #define MEM_NAME "DRAM"
    #define LOOP_SCALE 1500                           
#else
    #error "Invalid MEM_LEVEL. Must be 1, 2, or 3."
#endif

// The pointer-chasing jump size (4 uint64_t = 32 bytes = 1 sector)
#ifndef CHASE_JUMP
#define CHASE_JUMP 4
#endif

// Determines Coalescence. 
#ifndef SPATIAL_STRIDE
#define SPATIAL_STRIDE 1 
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 32
#endif

#ifndef N_BLOCKS
#define N_BLOCKS 264
#endif

#ifndef REPEAT_SECOND_LOOP
#define REPEAT_SECOND_LOOP 100
#endif
#define ACTUAL_LOOPS ((REPEAT_SECOND_LOOP / LOOP_SCALE) > 0 ? (REPEAT_SECOND_LOOP / LOOP_SCALE) : 1)

#ifndef GPU_ID
#define GPU_ID 0
#endif

// The size of our randomized "Pages". 512 uint64_t = 4096 Bytes.
// This is perfectly sized so Thread 31 (Stride 16) stays inside the chunk boundary.
#define CHUNK_ELEMS 512

// NVML Power Monitoring Variables
std::atomic<bool> keep_monitoring(false);
std::atomic<unsigned long long> total_active_power(0);
std::atomic<unsigned int> active_samples(0);

// Background thread function to poll NVML power during kernel execution
void monitor_power(nvmlDevice_t device) {
    unsigned int power;
    while (keep_monitoring) {
        nvmlDeviceGetPowerUsage(device, &power); 
        total_active_power += power;
        active_samples++;
        usleep(5000); 
    }
}

__global__ void load_data(uint64_t *my_array, uint64_t subtabSize) {
    uint64_t tid = threadIdx.x;
    
    // CPU initialized the pointers. We just offset the starting point to vary coalescence.
    uint64_t start_idx = (tid * SPATIAL_STRIDE) % subtabSize;
    uint64_t *start_addr = my_array + start_idx;

    asm volatile(
        "\n{\n"
        ".reg .pred %p;\n"
        ".reg .u64 %tmp;\n"
        "mov.u64 %tmp, %0;\n\n"

        // WARMUP: Pull data into highest possible cache level
        "$warmup_loop:\n"
        "ld.global.ca.u64 %tmp, [%tmp];\n"
        "setp.ne.u64 %p, %tmp, %0;\n" 
        "@%p bra $warmup_loop;\n"
        "bar.sync 0;\n" 

        // MEASUREMENT: All accesses here target the intended MEM_LEVEL
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
        "}" : "+l"(start_addr) : "n"(ACTUAL_LOOPS));
}

int main() {
    cudaSetDevice(GPU_ID);
    nvmlInit();
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(GPU_ID, &device);

    cudaFuncSetAttribute(load_data, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 32);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);

    uint64_t subtabSize = SIZE_BYTES / sizeof(uint64_t);
    uint64_t *h_a = (uint64_t *)malloc(SIZE_BYTES);
    uint64_t *d_a;
    cudaMalloc((void **)&d_a, SIZE_BYTES);

    // =========================================================================
    // 2. CPU-Side Pointer Initialization: "PAGE HOPPING" DEFEATS THE OPTIMIZER
    // =========================================================================
    
    uint64_t num_chunks = subtabSize / CHUNK_ELEMS;
    
    // Create sequential array of chunks
    uint64_t *chunk_order = (uint64_t *)malloc(num_chunks * sizeof(uint64_t));
    for (uint64_t i = 0; i < num_chunks; i++) {
        chunk_order[i] = i;
    }
    
    // Fisher-Yates Shuffle the Chunks Macroscopically
    srand(time(NULL));
    for (uint64_t i = num_chunks - 1; i > 0; i--) {
        uint64_t j = rand() % (i + 1);
        uint64_t temp = chunk_order[i];
        chunk_order[i] = chunk_order[j];
        chunk_order[j] = temp;
    }
    
    // Build the pointer chain microscopically inside the shuffled chunks
    for (uint64_t c = 0; c < num_chunks; c++) {
        uint64_t curr_chunk = chunk_order[c];
        uint64_t next_chunk = chunk_order[(c + 1) % num_chunks];
        
        for (uint64_t j = 0; j < CHUNK_ELEMS; j++) {
            uint64_t curr_idx = curr_chunk * CHUNK_ELEMS + j;
            uint64_t next_idx;
            
            if (j + CHASE_JUMP < CHUNK_ELEMS) {
                // Step forward mathematically inside the same chunk
                next_idx = curr_chunk * CHUNK_ELEMS + j + CHASE_JUMP;
            } else {
                // Instantly jump to the random next chunk
                // We modulo the remainder to ensure the 32-byte sector alignment holds
                next_idx = next_chunk * CHUNK_ELEMS + (j % CHASE_JUMP);
            }
            
            h_a[curr_idx] = (uint64_t)(d_a + next_idx);
        }
    }
    free(chunk_order);
    
    // =========================================================================

    cudaMemcpy(d_a, h_a, SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

#ifndef CSV_OUTPUT
    printf("Target: %s | Array Size: %llu MB | Threads: %d | Spatial Stride: %d\n", MEM_NAME, SIZE_BYTES/(1024*1024), THREADS_PER_BLOCK, SPATIAL_STRIDE);
    printf("Sleeping for 3 seconds to establish Idle Baseline...\n");
#endif

    usleep(3000000);
    unsigned int idle_power_mw = 0;
    nvmlDeviceGetPowerUsage(device, &idle_power_mw);

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

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double seconds = milliseconds / 1000.0;

    uint64_t cycle_length = subtabSize / CHASE_JUMP;
    uint64_t total_accesses = cycle_length * ACTUAL_LOOPS * THREADS_PER_BLOCK * N_BLOCKS;

    double avg_active_power_w = (double)total_active_power / active_samples / 1000.0;
    double dynamic_power_w = avg_active_power_w - (idle_power_mw / 1000.0);
    
    double dynamic_energy_j = dynamic_power_w * seconds;
    double energy_per_access_j = dynamic_energy_j / total_accesses;
    double energy_per_access_pj = energy_per_access_j * 1e12; 

#ifdef CSV_OUTPUT
    printf("%s,%d,%d,%e\n", MEM_NAME, SPATIAL_STRIDE, THREADS_PER_BLOCK, energy_per_access_pj);
#else
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