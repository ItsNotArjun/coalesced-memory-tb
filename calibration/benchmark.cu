#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <thread>
#include <atomic>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <limits>
#include <cuda_runtime.h>
#include <nvml.h>

// --- DRAM-Focused Defaults ---
// 128 MB working set is large enough to move beyond private cache capacity.
#ifndef SIZE_BYTES
#define SIZE_BYTES (128ULL * 1024ULL * 1024ULL)
#endif

// The pointer-chasing jump size (4 uint64_t = 32 bytes = 1 sector)
#ifndef CHASE_JUMP
#define CHASE_JUMP 4
#endif

// Used only in MODE=1 (strided) to create a deterministic partial-coalescing pattern.
#ifndef SPATIAL_STRIDE
#define SPATIAL_STRIDE 4
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

#ifndef N_BLOCKS
#define N_BLOCKS 264
#endif

#ifndef REPEAT_SECOND_LOOP
#define REPEAT_SECOND_LOOP 100 // High iteration count for stable power reading
#endif

#ifndef WARMUP_REPEAT_LOOP
#define WARMUP_REPEAT_LOOP 1
#endif

#ifndef GPU_ID
#define GPU_ID 0
#endif

// 0 = coalesced baseline, 1 = strided, 2 = random (worst-case)
#ifndef MODE
#define MODE 2
#endif

#ifndef RANDOM_SEED
#define RANDOM_SEED 12345ULL
#endif

#if MODE < 0 || MODE > 2
#error "MODE must be 0 (coalesced), 1 (strided), or 2 (random)"
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

static const char *mode_name() {
#if MODE == 0
    return "coalesced";
#elif MODE == 1
    return "strided";
#else
    return "random";
#endif
}

static uint64_t gcd_u64(uint64_t a, uint64_t b) {
    while (b != 0) {
        uint64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static uint64_t lcm_u64(uint64_t a, uint64_t b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    return (a / gcd_u64(a, b)) * b;
}

static void build_structured_chain(uint64_t *h_a, uint64_t *d_a, uint64_t subtabSize) {
    for (uint64_t i = 0; i < subtabSize; i++) {
        h_a[i] = (uint64_t)(d_a + ((i + CHASE_JUMP) % subtabSize));
    }
}

static void build_random_cycle_chain(uint64_t *h_a, uint64_t *d_a, uint64_t subtabSize) {
    if (subtabSize > (uint64_t)std::numeric_limits<uint32_t>::max()) {
        fprintf(stderr, "subtabSize too large for random cycle builder: %llu\n",
                (unsigned long long)subtabSize);
        exit(EXIT_FAILURE);
    }

    std::vector<uint32_t> perm((size_t)subtabSize);
    std::iota(perm.begin(), perm.end(), 0U);

    std::mt19937 rng((uint32_t)RANDOM_SEED);
    std::shuffle(perm.begin(), perm.end(), rng);

    for (size_t i = 0; i < perm.size(); ++i) {
        uint32_t cur = perm[i];
        uint32_t nxt = perm[(i + 1) % perm.size()];
        h_a[cur] = (uint64_t)(d_a + nxt);
    }
}

static void build_pointer_chain(uint64_t *h_a, uint64_t *d_a, uint64_t subtabSize) {
    if (MODE == 2) {
        build_random_cycle_chain(h_a, d_a, subtabSize);
    } else {
        build_structured_chain(h_a, d_a, subtabSize);
    }
}

__device__ __forceinline__ uint64_t compute_start_idx(uint64_t tid, uint64_t subtabSize) {
#if MODE == 0
    return tid % subtabSize;
#elif MODE == 1
    return (tid * SPATIAL_STRIDE) % subtabSize;
#else
    return (tid * 9973ULL + 17ULL) % subtabSize;
#endif
}

__global__ void warmup_data(uint64_t *my_array, uint64_t subtabSize) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start_idx = compute_start_idx(tid, subtabSize);
    uint64_t *start_addr = my_array + start_idx;

    asm volatile(
        "\n{\n"
        ".reg .pred %p;\n"
        ".reg .u64 %tmp;\n"
        ".reg .u32 %k;\n"
        "mov.u64 %tmp, %0;\n"
        "mov.u32 %k, 0;\n"

        // Warmup runs outside timed region to avoid synchronization noise.
        "$warmup_loop:\n"
        "ld.global.cg.u64 %tmp, [%tmp];\n"
        "setp.ne.u64 %p, %tmp, %0;\n"
        "@%p bra $warmup_loop;\n"
        "add.u32 %k, %k, 1;\n"
        "setp.lt.u32 %p, %k, %1;\n"
        "@%p bra $warmup_loop;\n"
        "}"
        : "+l"(start_addr)
        : "n"(WARMUP_REPEAT_LOOP));
}

__global__ void measure_data(uint64_t *my_array, uint64_t subtabSize) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start_idx = compute_start_idx(tid, subtabSize);
    uint64_t *start_addr = my_array + start_idx;

    asm volatile(
        "\n{\n"
        ".reg .pred %p;\n"
        ".reg .u64 %tmp;\n"
        ".reg .u32 %k;\n"
        "mov.u32 %k, 0;\n"
        "mov.u64 %tmp, %0;\n\n"

        "\n$start:\n"
        // Timed memory stream: 16 dependent DRAM-biased loads per checkpoint.
        "ld.global.cg.u64 %tmp, [%tmp];\n" "ld.global.cg.u64 %tmp, [%tmp];\n"
        "ld.global.cg.u64 %tmp, [%tmp];\n" "ld.global.cg.u64 %tmp, [%tmp];\n"
        "ld.global.cg.u64 %tmp, [%tmp];\n" "ld.global.cg.u64 %tmp, [%tmp];\n"
        "ld.global.cg.u64 %tmp, [%tmp];\n" "ld.global.cg.u64 %tmp, [%tmp];\n"
        "ld.global.cg.u64 %tmp, [%tmp];\n" "ld.global.cg.u64 %tmp, [%tmp];\n"
        "ld.global.cg.u64 %tmp, [%tmp];\n" "ld.global.cg.u64 %tmp, [%tmp];\n"
        "ld.global.cg.u64 %tmp, [%tmp];\n" "ld.global.cg.u64 %tmp, [%tmp];\n"
        "ld.global.cg.u64 %tmp, [%tmp];\n" "ld.global.cg.u64 %tmp, [%tmp];\n"
        
        "setp.ne.u64 %p, %tmp, %0;\n"
        "@%p bra $start;\n"
        "add.u32 %k, %k, 1;\n"
        "setp.lt.u32 %p, %k, %1;\n"
        "@%p bra $start;\n"
        "}"
        : "+l"(start_addr)
        : "n"(REPEAT_SECOND_LOOP));
}

int main() {
    cudaSetDevice(GPU_ID);
    nvmlInit();
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(GPU_ID, &device);

    // 1. Avoid persisting L2 cache state between runs.
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);

    uint64_t subtabSize = SIZE_BYTES / sizeof(uint64_t);
    uint64_t *h_a = (uint64_t *)malloc(SIZE_BYTES);
    uint64_t *d_a;
    cudaMalloc((void **)&d_a, SIZE_BYTES);

    // 2. Build the pointer chain on CPU then upload to GPU.
    build_pointer_chain(h_a, d_a, subtabSize);
    cudaMemcpy(d_a, h_a, SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

#ifndef CSV_OUTPUT
    printf("Mode: %s | Array: %.1f MB | Threads: %d | Blocks: %d | Spatial Stride: %d\n",
           mode_name(),
           SIZE_BYTES / (1024.0 * 1024.0),
           THREADS_PER_BLOCK,
           N_BLOCKS,
           SPATIAL_STRIDE);
    printf("Warmup Repeats: %d | Timed Repeats: %d\n", WARMUP_REPEAT_LOOP, REPEAT_SECOND_LOOP);
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

    total_active_power = 0;
    active_samples = 0;
    keep_monitoring = true;
    std::thread monitor(monitor_power, device);

    dim3 Db(THREADS_PER_BLOCK, 1, 1);
    dim3 Dg(N_BLOCKS, 1, 1);

    // Untimed warmup to stabilize cache/TLB state before measurement.
    warmup_data<<<Dg, Db>>>(d_a, subtabSize);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    measure_data<<<Dg, Db>>>(d_a, subtabSize);
    cudaEventRecord(stop);
    
    cudaDeviceSynchronize();

    keep_monitoring = false;
    monitor.join();

    // 5. Calculate Time and Total Accesses
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double seconds = milliseconds / 1000.0;

#if MODE == 2
    // Random permutation is a single Hamiltonian cycle.
    uint64_t cycle_length = subtabSize;
#else
    uint64_t cycle_length = subtabSize / gcd_u64(subtabSize, CHASE_JUMP);
#endif
    uint64_t loads_per_repeat = lcm_u64(cycle_length, 16ULL);
    uint64_t total_accesses = loads_per_repeat
                            * (uint64_t)REPEAT_SECOND_LOOP
                            * (uint64_t)THREADS_PER_BLOCK
                            * (uint64_t)N_BLOCKS;

    // 6. Calculate Dynamic Energy
    double avg_active_power_w = (active_samples > 0)
                              ? (double)total_active_power / active_samples / 1000.0
                              : (double)idle_power_mw / 1000.0;
    double dynamic_power_w = avg_active_power_w - (idle_power_mw / 1000.0);
    
    // Power (W) * Time (s) = Energy (Joules)
    double dynamic_energy_j = dynamic_power_w * seconds;
    double energy_per_access_j = dynamic_energy_j / total_accesses;
    double energy_per_access_pj = energy_per_access_j * 1e12; // Picojoules
    double useful_bytes = (double)total_accesses * sizeof(uint64_t);
    double energy_per_useful_byte_j = dynamic_energy_j / useful_bytes;
    double energy_per_useful_byte_pj = energy_per_useful_byte_j * 1e12;

#ifdef CSV_OUTPUT
    // CSV fields are consumed by analysis scripts for amplification/efficiency reporting.
    printf("%d,%d,%d,%d,%llu,%.6f,%.6f,%.9e,%llu,%llu,%.9e,%.9e\n",
           MODE,
           SPATIAL_STRIDE,
           THREADS_PER_BLOCK,
           N_BLOCKS,
           (unsigned long long)SIZE_BYTES,
           seconds,
           dynamic_power_w,
           dynamic_energy_j,
           (unsigned long long)total_accesses,
           (unsigned long long)(total_accesses * sizeof(uint64_t)),
           energy_per_access_pj,
           energy_per_useful_byte_pj);
#else
    // Print Human Readable output
    printf("Idle Power: %.2f W\n", idle_power_mw / 1000.0);
    printf("Kernel Time: %.4f seconds\n", seconds);
    printf("Avg Active Power: %.2f W\n", avg_active_power_w);
    printf("Dynamic Power (Isolated): %.2f W\n", dynamic_power_w);
    printf("Cycle Length: %llu | Loads per Repeat: %llu\n",
           (unsigned long long)cycle_length,
           (unsigned long long)loads_per_repeat);
    printf("Total Accesses: %llu\n", (unsigned long long)total_accesses);
    printf("Useful Bytes: %.0f\n", useful_bytes);
    printf("Dynamic Energy: %e J\n", dynamic_energy_j);
    printf("Energy per Access: %e J (%.2f pJ)\n", energy_per_access_j, energy_per_access_pj);
    printf("Energy per Useful Byte: %e J/byte (%.2f pJ/byte)\n",
           energy_per_useful_byte_j,
           energy_per_useful_byte_pj);
#endif

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    free(h_a);
    nvmlShutdown();
    return 0;
}