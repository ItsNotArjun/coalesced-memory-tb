/**
 * calibrate.cu  —  GPU Energy Model Calibration Tool
 * =====================================================
 * Implements Section V-B of:
 *   Delestrac et al., "Analyzing GPU Energy Consumption in Data Movement
 *   and Storage", ASAP 2024.
 *
 * Build
 * -----
 *   nvcc -O2 -arch=sm_86 calibrate.cu -o calibrate -lnvidia-ml -lpthread -lm
 *   (adjust -arch: sm_70=V100, sm_75=T4/RTX20xx, sm_80=A100, sm_86=RTX30xx,
 *                  sm_89=RTX40xx, sm_90=H100)
 *
 * Usage
 * -----
 *   ./calibrate --mem <l1|l2|dram|shared> [options]
 *
 *   --mem         <l1|l2|dram|shared>  Target memory level     (default: l1)
 *   --min-acc     <N>  Min accesses (millions)                  (default: 15)
 *   --max-acc     <N>  Max accesses (millions)                  (default: 3000)
 *   --acc-steps   <N>  Logarithmic sweep points                 (default: 20)
 *   --n-runs      <N>  Kernel repeats per sweep point           (default: 5)
 *   --threads     <N>  Threads per block                        (default: 1024)
 *   --blocks      <N>  Thread blocks                            (default: 4)
 *   --granularity <B>  Access granularity in bytes              (default: 32)
 *   --stride      <B>  Pointer-chase stride in bytes            (default: 32)
 *   --n-iters     <N>  Measurement loop iterations              (default: 200)
 *   --array-bytes <B>  Override working-set size                (auto by level)
 *   --outdir      <P>  Output directory                         (default: results)
 *   --load-only        Run LOAD benchmark only
 *   --store-only       Run STORE benchmark only
 *   --dry-run          Synthetic data mode (no GPU required)
 *   --verbose          Print per-run energy values
 *   --help
 *
 * Outputs
 * -------
 *   <outdir>/calibration_load_<mem>.csv    — per-point LOAD sweep
 *   <outdir>/calibration_store_<mem>.csv   — per-point STORE sweep
 *   <outdir>/calibration_summary_<mem>.csv — linear regression summary
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * Bug-fix notes (vs. first version)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * BUG 1 — All-zero energy on real GPU
 *   Root cause: the power sampler stored timestamps as absolute
 *   CLOCK_MONOTONIC µs, but compute_energy() was called with
 *   (IDLE_DELAY_MS * 1000) as kernel_start_us — a value ~500 000 µs,
 *   far smaller than any absolute timestamp (billions of µs since boot).
 *   No samples ever fell inside the bogus [500ms, ~501ms] window → zeros.
 *   Additionally, there was a race: timestamps_us[0] was read before the
 *   sampler thread had written it, giving undefined behaviour.
 *
 *   Fix: record kernel_start and kernel_end with the same mono_us()
 *   function used by the sampler thread.  Both are now absolute
 *   CLOCK_MONOTONIC µs.  No offset arithmetic, no subtraction, no race.
 *
 * BUG 2 — Illegal memory access in STORE kernel
 *   Root cause: a single array was used for both pointer navigation and
 *   writes.  With 4 blocks × 1024 threads = 4096 threads sharing
 *   ~19 200 elements, thread A writes array[pos] = arbitrary_value.
 *   Thread B then reads that element expecting a valid index in
 *   [0, n_elems), gets arbitrary_value ≥ n_elems, and crashes.
 *
 *   Fix: allocate two separate GPU buffers of equal size:
 *     d_chain[] — read-only, holds pointer indices, NEVER written
 *     d_data[]  — write-only scratchpad, receives the store values
 *   The store kernel reads next index from d_chain[] (always valid),
 *   then writes to d_data[pos] (harmless).  No corruption possible.
 *
 * BUG 3 — Per-step output missing in --dry-run mode
 *   Root cause: the printf for each sweep row was inside an
 *   if (!cfg.dry_run) block.
 *   Fix: print the row unconditionally after each point is computed.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <pthread.h>
#include <nvml.h>

/* ═══════════════════════ error macros ════════════════════════════════════ */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_e));              \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while(0)

#define NVML_CHECK(call)                                                     \
    do {                                                                     \
        nvmlReturn_t _r = (call);                                            \
        if (_r != NVML_SUCCESS)                                              \
            fprintf(stderr, "NVML warning %s:%d  %s\n",                     \
                    __FILE__, __LINE__, nvmlErrorString(_r));                 \
    } while(0)

/* ═══════════════════════ constants ═══════════════════════════════════════ */
#define ELEM_SIZE         8ULL
#define UNROLL            8
#define POWER_SAMPLE_MS   5
#define MAX_POWER_SAMPLES 200000
#define IDLE_DELAY_MS     500
#define IDLE_WINDOW_MS    400

typedef enum { MEM_L1=0, MEM_L2, MEM_DRAM, MEM_SHARED } MemLevel;
static const char *MEM_NAMES[] = { "l1", "l2", "dram", "shared" };
typedef enum { OP_LOAD=0, OP_STORE } OpType;
static const char *OP_NAMES[]  = { "load", "store" };

/* ═══════════════════════ config ══════════════════════════════════════════ */
typedef struct {
    MemLevel mem_level;
    uint64_t min_acc_M, max_acc_M;
    int      acc_steps, n_runs, threads, blocks;
    uint32_t granularity, stride, n_iters;
    uint64_t array_bytes;
    char     outdir[256];
    int      load_only, store_only, dry_run, verbose;
} Config;

/* ═══════════════════════ monotonic clock ════════════════════════════════
 * Single source of truth for every timestamp in this program.
 * The sampler thread and the main thread both call mono_us(); no
 * conversions or offsets are needed when comparing their values.
 * ════════════════════════════════════════════════════════════════════════ */
static uint64_t mono_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)(ts.tv_nsec / 1000);
}

static void sleep_ms(int ms)
{
    struct timespec ts = { ms/1000, (long)(ms%1000)*1000000L };
    nanosleep(&ts, NULL);
}

/* ═══════════════════════ power sampler ══════════════════════════════════
 * BUG 1 FIX: timestamps_us[] is filled with mono_us() — the same clock
 * used in the main thread.  No race on timestamps_us[0]: the index is
 * incremented only after both the timestamp and power sample are written.
 * ════════════════════════════════════════════════════════════════════════ */
typedef struct {
    nvmlDevice_t  device;
    volatile int  running;
    unsigned int *samples_mw;
    uint64_t     *timestamps_us;
    volatile int  n_samples;
    int           max_samples;
    int           interval_ms;
} PowerSampler;

static void *power_thread(void *arg)
{
    PowerSampler *ps = (PowerSampler *)arg;
    while (ps->running && ps->n_samples < ps->max_samples) {
        int idx = ps->n_samples;
        ps->timestamps_us[idx] = mono_us();        /* absolute µs — same clock */
        unsigned int pw = 0;
        nvmlDeviceGetPowerUsage(ps->device, &pw);
        ps->samples_mw[idx] = pw;
        __sync_synchronize();                      /* publish before increment */
        ps->n_samples = idx + 1;
        struct timespec t = { 0, (long)ps->interval_ms * 1000000L };
        nanosleep(&t, NULL);
    }
    return NULL;
}

static PowerSampler *ps_create(nvmlDevice_t dev, int interval_ms)
{
    PowerSampler *ps  = (PowerSampler *)calloc(1, sizeof(PowerSampler));
    ps->device        = dev;
    ps->interval_ms   = interval_ms;
    ps->max_samples   = MAX_POWER_SAMPLES;
    ps->samples_mw    = (unsigned int *)malloc(MAX_POWER_SAMPLES * sizeof(unsigned int));
    ps->timestamps_us = (uint64_t *)    malloc(MAX_POWER_SAMPLES * sizeof(uint64_t));
    return ps;
}

static void ps_start(PowerSampler *ps, pthread_t *tid)
{ ps->running = 1; ps->n_samples = 0; pthread_create(tid,NULL,power_thread,ps); }

static void ps_stop(PowerSampler *ps, pthread_t tid)
{ ps->running = 0; pthread_join(tid, NULL); }

static void ps_free(PowerSampler *ps)
{ free(ps->samples_mw); free(ps->timestamps_us); free(ps); }

/* ═══════════════════════ energy computation (Fig. 4) ════════════════════
 * BUG 1 FIX: kernel_start_us / kernel_end_us are now absolute mono_us()
 * values stamped immediately before/after cudaDeviceSynchronize().
 * compute_energy simply scans for samples whose absolute timestamp falls
 * in [kernel_start_us, kernel_end_us].  No offsets needed.
 * ════════════════════════════════════════════════════════════════════════ */
typedef struct {
    double static_power_w;
    double total_energy_j;
    double dynamic_energy_j;
    double duration_s;
    int    n_idle_samples;
    int    n_kernel_samples;
} EnergyResult;

static EnergyResult compute_energy(const PowerSampler *ps,
                                    uint64_t kernel_start_us,
                                    uint64_t kernel_end_us)
{
    EnergyResult r = {0};
    int n = ps->n_samples;

    /* static power: mean of samples in IDLE_WINDOW_MS before kernel start */
    uint64_t idle_lo = kernel_start_us - (uint64_t)(IDLE_WINDOW_MS * 1000);
    double   idle_sum = 0.0;
    for (int i=0; i<n; ++i) {
        uint64_t t = ps->timestamps_us[i];
        if (t >= idle_lo && t < kernel_start_us) {
            idle_sum += ps->samples_mw[i];
            r.n_idle_samples++;
        }
    }
    r.static_power_w = r.n_idle_samples > 0
                     ? (idle_sum / r.n_idle_samples) / 1000.0
                     : 0.0;

    /* trapezoid integral over kernel execution window */
    double prev_t = -1.0, prev_p = 0.0;
    for (int i=0; i<n; ++i) {
        uint64_t t = ps->timestamps_us[i];
        if (t < kernel_start_us || t > kernel_end_us) continue;
        double t_s = (double)t / 1e6;
        double p_w = ps->samples_mw[i] / 1000.0;
        if (prev_t >= 0.0)
            r.total_energy_j += (p_w + prev_p) * 0.5 * (t_s - prev_t);
        prev_t = t_s;
        prev_p = p_w;
        r.n_kernel_samples++;
    }

    r.duration_s = (double)(kernel_end_us - kernel_start_us) / 1e6;
    r.dynamic_energy_j = r.total_energy_j - r.static_power_w * r.duration_s;
    if (r.dynamic_energy_j < 0.0) r.dynamic_energy_j = 0.0;
    return r;
}

/* ═══════════════════════ chain init ═════════════════════════════════════ */
static void build_chain(uint64_t *a, uint64_t n_elems, uint64_t stride_e)
{ for (uint64_t i=0; i<n_elems; ++i) a[i] = (i + stride_e) % n_elems; }

/* ═══════════════════════ LOAD kernel ════════════════════════════════════
 * Pointer-chasing LOAD microbenchmark (Listing 1 of the paper).
 * Warm-up loop: one full chain traversal → all MISSes (fills cache).
 * Measurement loop: n_iters×UNROLL chased LOADs → all HITs after warmup.
 * ════════════════════════════════════════════════════════════════════════ */
__global__ void load_kernel(
    const uint64_t * __restrict__ array,
    uint64_t  n_elems,
    uint64_t  stride_elems,
    uint32_t  n_iters,
    int       do_warmup_only,
    uint64_t *out)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t pos = tid % n_elems;
    uint64_t acc = 0;

    /* warm-up */
    uint64_t steps = n_elems / stride_elems;
    for (uint64_t s=0; s<steps; ++s) { pos = array[pos]; acc ^= pos; }

    if (do_warmup_only) { if (tid==0) *out=acc; return; }

    /* measurement: all HITs, unrolled ×8 */
    pos = tid % n_elems;
    for (uint32_t it=0; it<n_iters; ++it) {
        pos=array[pos]; acc^=pos;
        pos=array[pos]; acc^=pos;
        pos=array[pos]; acc^=pos;
        pos=array[pos]; acc^=pos;
        pos=array[pos]; acc^=pos;
        pos=array[pos]; acc^=pos;
        pos=array[pos]; acc^=pos;
        pos=array[pos]; acc^=pos;
    }
    if (tid==0) *out=acc;
}

/* ═══════════════════════ STORE kernel ═══════════════════════════════════
 * BUG 2 FIX: two separate buffers.
 *
 *   chain[]  read-only pointer-chasing indices — never written by this kernel
 *   data[]   write-only scratchpad — receives store values
 *
 * Why the crash happened:
 *   Single-buffer version: thread A writes array[pos] = X (arbitrary).
 *   Thread B reads array[pos] expecting index in [0,n_elems), gets X
 *   which may be >> n_elems → out-of-bounds access → illegal memory error.
 *
 * Why two buffers fix it:
 *   chain[] is never written, so every index read from it is always valid.
 *   data[] is written freely; those values are never used as indices.
 *
 * Warm-up: reads chain[] to pull working set into cache (same cache lines
 *          as the subsequent store targets, since both arrays are the same
 *          size and indexed identically).
 * Measure: reads next index from chain[], writes sentinel to data[pos].
 * ════════════════════════════════════════════════════════════════════════ */
__global__ void store_kernel(
    const uint64_t * __restrict__ chain,
    uint64_t *data,
    uint64_t  n_elems,
    uint64_t  stride_elems,
    uint32_t  n_iters,
    int       do_warmup_only,
    uint64_t *out)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t pos = tid % n_elems;
    uint64_t acc = 0;

    /* warm-up: read chain[] to fill cache */
    uint64_t steps = n_elems / stride_elems;
    for (uint64_t s=0; s<steps; ++s) { pos = chain[pos]; acc ^= pos; }

    if (do_warmup_only) { if (tid==0) *out=acc; return; }

    /* measurement: navigate via chain[], write to data[] — unrolled ×8 */
    pos = tid % n_elems;
    uint64_t val = tid + 1;
    for (uint32_t it=0; it<n_iters; ++it) {
        uint64_t nxt;
        nxt=chain[pos]; data[pos]=val^pos; acc^=pos; pos=nxt;
        nxt=chain[pos]; data[pos]=val^pos; acc^=pos; pos=nxt;
        nxt=chain[pos]; data[pos]=val^pos; acc^=pos; pos=nxt;
        nxt=chain[pos]; data[pos]=val^pos; acc^=pos; pos=nxt;
        nxt=chain[pos]; data[pos]=val^pos; acc^=pos; pos=nxt;
        nxt=chain[pos]; data[pos]=val^pos; acc^=pos; pos=nxt;
        nxt=chain[pos]; data[pos]=val^pos; acc^=pos; pos=nxt;
        nxt=chain[pos]; data[pos]=val^pos; acc^=pos; pos=nxt;
    }
    if (tid==0) *out=acc;
}

/* ═══════════════════════ #accesses (Eq. 7) ══════════════════════════════ */
static uint64_t calc_accesses(uint64_t array_bytes, int blocks, int threads,
                               uint32_t stride, uint32_t granularity)
{
    return (array_bytes * (uint64_t)blocks * (uint64_t)threads)
         / ((uint64_t)stride * (uint64_t)granularity);
}

/* ═══════════════════════ linear regression ══════════════════════════════ */
typedef struct {
    double slope, intercept, r_squared, slope_pj, slope_nj;
} LinReg;

static LinReg do_linreg(const double *x, const double *y, int n)
{
    LinReg r = {0};
    if (n < 2) return r;
    double sx=0,sy=0,sxy=0,sxx=0;
    for (int i=0;i<n;++i){sx+=x[i];sy+=y[i];sxy+=x[i]*y[i];sxx+=x[i]*x[i];}
    double d = (double)n*sxx - sx*sx;
    if (fabs(d)<1e-30) return r;
    r.slope     = ((double)n*sxy - sx*sy)/d;
    r.intercept = (sy - r.slope*sx)/(double)n;
    double ym=sy/(double)n, ss_tot=0, ss_res=0;
    for (int i=0;i<n;++i){
        double dt=y[i]-ym, dr=y[i]-(r.slope*x[i]+r.intercept);
        ss_tot+=dt*dt; ss_res+=dr*dr;
    }
    r.r_squared = ss_tot>1e-30 ? 1.0-ss_res/ss_tot : 0.0;
    r.slope_pj  = r.slope*1e12;
    r.slope_nj  = r.slope*1e9;
    return r;
}

static double arr_stddev(const double *a, int n, double mean)
{
    if (n<2) return 0.0;
    double s=0;
    for (int i=0;i<n;++i){double d=a[i]-mean;s+=d*d;}
    return sqrt(s/(n-1));
}

/* ═══════════════════════ measurement point ══════════════════════════════ */
typedef struct {
    uint64_t n_accesses;
    double avg_dynamic_energy_j, std_dynamic_energy_j;
    double min_dynamic_energy_j, max_dynamic_energy_j;
    double avg_warmup_energy_j,  avg_total_energy_j;
    double avg_static_power_w,   avg_duration_s;
    int    n_valid_runs;
} MeasPoint;

/* ═══════════════════════ run_point ══════════════════════════════════════
 * Per-run procedure:
 *   (a) start sampler  →  sleep IDLE_DELAY_MS  →
 *       stamp t0 (absolute mono_us)  →  launch warmup-only kernel  →
 *       stamp t1  →  sleep IDLE_DELAY_MS  →  stop sampler
 *       → E_wu = compute_energy(ps, t0, t1)
 *
 *   (b) reset d_data for STORE  →  start sampler  →  sleep IDLE_DELAY_MS →
 *       stamp t2  →  launch warmup+measurement kernel  →
 *       stamp t3  →  sleep IDLE_DELAY_MS  →  stop sampler
 *       → E_full = compute_energy(ps, t2, t3)
 *
 *   (c) E_meas = E_full.dynamic - E_wu.dynamic
 *
 * BUG 1 FIX: t0/t1/t2/t3 are all mono_us() — the same clock as the
 * sampler thread.  compute_energy scans for samples with absolute
 * timestamps in [t_start, t_end].  No offset arithmetic anywhere.
 * ════════════════════════════════════════════════════════════════════════ */
static MeasPoint run_point(
    uint64_t *d_chain, uint64_t *d_data,
    uint64_t  array_bytes, uint64_t n_elems, uint64_t stride_elems,
    uint32_t  n_iters, int threads, int blocks,
    uint32_t  stride, uint32_t granularity,
    OpType op, nvmlDevice_t nvml_dev,
    PowerSampler *ps, int n_runs, int verbose)
{
    MeasPoint mp = {0};
    mp.min_dynamic_energy_j =  1e30;
    mp.max_dynamic_energy_j = -1e30;
    double *run_e = (double *)malloc(n_runs * sizeof(double));

    uint64_t *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(uint64_t)));

    mp.n_accesses = calc_accesses(array_bytes, blocks, threads, stride, granularity)
                  * (uint64_t)n_iters * (uint64_t)UNROLL;

    for (int run=0; run<n_runs; ++run) {

        /* ── (a) warm-up only ── */
        if (op == OP_STORE) CUDA_CHECK(cudaMemset(d_data, 0, array_bytes));

        pthread_t ptid;
        ps->n_samples = 0;
        ps_start(ps, &ptid);

        sleep_ms(IDLE_DELAY_MS);
        uint64_t t0 = mono_us();               /* kernel_start — absolute µs  */

        if (op == OP_LOAD)
            load_kernel<<<blocks,threads>>>(d_chain, n_elems, stride_elems,
                                            n_iters, 1, d_out);
        else
            store_kernel<<<blocks,threads>>>(d_chain, d_data, n_elems,
                                             stride_elems, n_iters, 1, d_out);
        CUDA_CHECK(cudaDeviceSynchronize());

        uint64_t t1 = mono_us();               /* kernel_end */
        sleep_ms(IDLE_DELAY_MS);
        ps_stop(ps, ptid);

        EnergyResult e_wu = compute_energy(ps, t0, t1);

        /* ── (b) warm-up + measurement ── */
        if (op == OP_STORE) CUDA_CHECK(cudaMemset(d_data, 0, array_bytes));

        ps->n_samples = 0;
        ps_start(ps, &ptid);

        sleep_ms(IDLE_DELAY_MS);
        uint64_t t2 = mono_us();

        if (op == OP_LOAD)
            load_kernel<<<blocks,threads>>>(d_chain, n_elems, stride_elems,
                                            n_iters, 0, d_out);
        else
            store_kernel<<<blocks,threads>>>(d_chain, d_data, n_elems,
                                             stride_elems, n_iters, 0, d_out);
        CUDA_CHECK(cudaDeviceSynchronize());

        uint64_t t3 = mono_us();
        sleep_ms(IDLE_DELAY_MS);
        ps_stop(ps, ptid);

        EnergyResult e_full = compute_energy(ps, t2, t3);

        /* ── (c) isolate measurement energy ── */
        double e_meas = e_full.dynamic_energy_j - e_wu.dynamic_energy_j;
        if (e_meas < 0.0) e_meas = 0.0;

        run_e[run] = e_meas;
        mp.avg_warmup_energy_j += e_wu.dynamic_energy_j;
        mp.avg_total_energy_j  += e_full.dynamic_energy_j;
        mp.avg_static_power_w  += e_full.static_power_w;
        mp.avg_duration_s      += e_full.duration_s;
        if (e_meas < mp.min_dynamic_energy_j) mp.min_dynamic_energy_j = e_meas;
        if (e_meas > mp.max_dynamic_energy_j) mp.max_dynamic_energy_j = e_meas;
        mp.n_valid_runs++;

        if (verbose)
            printf("    run %d/%d  E_wu=%.4fJ  E_full=%.4fJ  E_meas=%.5fJ"
                   "  idle_n=%d  kern_n=%d\n",
                   run+1, n_runs,
                   e_wu.dynamic_energy_j, e_full.dynamic_energy_j, e_meas,
                   e_full.n_idle_samples, e_full.n_kernel_samples);
    }

    double sum=0.0;
    for (int i=0;i<n_runs;++i) sum+=run_e[i];
    mp.avg_dynamic_energy_j = sum/n_runs;
    mp.std_dynamic_energy_j = arr_stddev(run_e, n_runs, mp.avg_dynamic_energy_j);
    mp.avg_warmup_energy_j /= n_runs;
    mp.avg_total_energy_j  /= n_runs;
    mp.avg_static_power_w  /= n_runs;
    mp.avg_duration_s      /= n_runs;

    CUDA_CHECK(cudaFree(d_out));
    free(run_e);
    return mp;
}

/* ═══════════════════════ synthetic mode ══════════════════════════════════ */
static const double SYNTH_EPS_PJ[4][2] = {
    {107.0, 123.0}, {378.0, 435.0}, {2090.0, 2407.0}, {82.1, 94.5}
};
static const double SYNTH_DELTA_J[4] = {20.0, 30.0, 40.0, 15.0};

static MeasPoint synth_point(uint64_t n_acc, MemLevel m, OpType op, int n_runs)
{
    static unsigned seed = 1234;
    double eps   = SYNTH_EPS_PJ[m][op] * 1e-12;
    double delta = SYNTH_DELTA_J[m];
    if (n_runs > 64) n_runs = 64;
    double run_e[64];
    double sum=0.0;
    for (int i=0;i<n_runs;++i){
        double noise = 1.0 + 0.015*(((double)(rand_r(&seed)%1000)/500.0)-1.0);
        run_e[i] = (eps*n_acc+delta)*noise; sum+=run_e[i];
    }
    MeasPoint mp = {0};
    mp.n_accesses           = n_acc;
    mp.avg_dynamic_energy_j = sum/n_runs;
    mp.std_dynamic_energy_j = arr_stddev(run_e, n_runs, mp.avg_dynamic_energy_j);
    mp.min_dynamic_energy_j = run_e[0]; mp.max_dynamic_energy_j = run_e[0];
    for (int i=1;i<n_runs;++i){
        if (run_e[i]<mp.min_dynamic_energy_j) mp.min_dynamic_energy_j=run_e[i];
        if (run_e[i]>mp.max_dynamic_energy_j) mp.max_dynamic_energy_j=run_e[i];
    }
    mp.avg_warmup_energy_j  = delta*0.45;
    mp.avg_total_energy_j   = mp.avg_dynamic_energy_j+delta;
    mp.avg_static_power_w   = 30.0;
    mp.avg_duration_s       = mp.avg_total_energy_j/80.0;
    mp.n_valid_runs         = n_runs;
    return mp;
}

/* ═══════════════════════ sweep generation ══════════════════════════════ */
static void gen_acc_sweep(uint64_t *acc, int n, uint64_t min_M, uint64_t max_M)
{
    double lo=log((double)(min_M*1000000ULL)), hi=log((double)(max_M*1000000ULL));
    for (int i=0;i<n;++i)
        acc[i]=(uint64_t)exp(lo+(double)i/(double)(n-1)*(hi-lo));
}

static uint64_t default_array_bytes(MemLevel m)
{
    switch(m){
        case MEM_L1:     return 150ULL*1024;
        case MEM_L2:     return 250ULL*1024;
        case MEM_DRAM:   return 50ULL*1024*1024;
        case MEM_SHARED: return 48ULL*1024;
        default:         return 150ULL*1024;
    }
}

/* ═══════════════════════ CSV writers ════════════════════════════════════ */
static void write_sweep_csv(const char *path,
                             const MeasPoint *pts, int n,
                             MemLevel mem, OpType op,
                             const Config *cfg, const char *gpu,
                             const LinReg *lr)
{
    FILE *f = fopen(path,"w"); if(!f){perror(path);return;}
    fprintf(f,"# GPU Energy Model Calibration -- Sweep Data\n");
    fprintf(f,"# Reference: Delestrac et al. ASAP 2024, Section V-B\n");
    fprintf(f,"# GPU: %s\n",gpu);
    fprintf(f,"# Memory level: %s\n",MEM_NAMES[mem]);
    fprintf(f,"# Operation: %s\n",OP_NAMES[op]);
    fprintf(f,"# Stride (bytes): %u\n",cfg->stride);
    fprintf(f,"# Access granularity (bytes): %u\n",cfg->granularity);
    fprintf(f,"# Threads per block: %d\n",cfg->threads);
    fprintf(f,"# Thread blocks: %d\n",cfg->blocks);
    fprintf(f,"# Kernel iterations: %u\n",cfg->n_iters);
    fprintf(f,"# Runs per point: %d\n",cfg->n_runs);
    fprintf(f,"# Array size (bytes): %llu\n",(unsigned long long)cfg->array_bytes);
    fprintf(f,"#\n");
    fprintf(f,"# Linear regression:\n");
    fprintf(f,"#   slope  (e_MEM) = %.10f J/access = %.4f pJ/access\n",
            lr->slope, lr->slope_pj);
    fprintf(f,"#   intercept (D_MEM) = %.6f J\n",lr->intercept);
    fprintf(f,"#   R-squared = %.6f\n",lr->r_squared);
    fprintf(f,"#\n");
    fprintf(f,"n_accesses,"
              "avg_dynamic_energy_j,std_dynamic_energy_j,"
              "min_dynamic_energy_j,max_dynamic_energy_j,"
              "avg_warmup_energy_j,avg_total_energy_j,"
              "avg_static_power_w,avg_kernel_duration_s,"
              "energy_per_access_pj,linreg_predicted_j,residual_j,"
              "n_valid_runs\n");
    for(int i=0;i<n;++i){
        const MeasPoint *p=&pts[i];
        double pred=lr->slope*(double)p->n_accesses+lr->intercept;
        double eps_pj=p->n_accesses>0
                     ? p->avg_dynamic_energy_j/(double)p->n_accesses*1e12 : 0.0;
        fprintf(f,"%llu,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.4f,%.6f,%.4f,%.8f,%.8f,%d\n",
                (unsigned long long)p->n_accesses,
                p->avg_dynamic_energy_j, p->std_dynamic_energy_j,
                p->min_dynamic_energy_j, p->max_dynamic_energy_j,
                p->avg_warmup_energy_j,  p->avg_total_energy_j,
                p->avg_static_power_w,   p->avg_duration_s,
                eps_pj, pred, p->avg_dynamic_energy_j-pred, p->n_valid_runs);
    }
    fclose(f);
    printf("  Saved: %s\n",path);
}

static void write_summary_csv(const char *path,
                               const LinReg *lrl, int lok,
                               const LinReg *lrs, int sok,
                               MemLevel mem, const Config *cfg, const char *gpu)
{
    FILE *f = fopen(path,"w"); if(!f){perror(path);return;}
    fprintf(f,"# GPU Energy Model Calibration -- Summary\n");
    fprintf(f,"# Reference: Delestrac et al. ASAP 2024, Section V-B\n");
    fprintf(f,"# GPU: %s\n",gpu);
    fprintf(f,"# Memory level: %s\n",MEM_NAMES[mem]);
    fprintf(f,"# Stride (bytes): %u\n",cfg->stride);
    fprintf(f,"# Access granularity (bytes): %u\n",cfg->granularity);
    fprintf(f,"# Threads per block: %d\n",cfg->threads);
    fprintf(f,"# Thread blocks: %d\n",cfg->blocks);
    fprintf(f,"#\n");
    fprintf(f,"operation,epsilon_mem_j_per_access,epsilon_mem_pj_per_access,"
              "epsilon_mem_nj_per_access,delta_mem_offset_j,r_squared,"
              "granularity_bytes,stride_bytes,threads_per_block,blocks\n");
    if(lok) fprintf(f,"load,%.12f,%.6f,%.9f,%.6f,%.6f,%u,%u,%d,%d\n",
            lrl->slope,lrl->slope_pj,lrl->slope_nj,lrl->intercept,lrl->r_squared,
            cfg->granularity,cfg->stride,cfg->threads,cfg->blocks);
    if(sok) fprintf(f,"store,%.12f,%.6f,%.9f,%.6f,%.6f,%u,%u,%d,%d\n",
            lrs->slope,lrs->slope_pj,lrs->slope_nj,lrs->intercept,lrs->r_squared,
            cfg->granularity,cfg->stride,cfg->threads,cfg->blocks);
    fclose(f);
    printf("  Saved: %s\n",path);
}

/* ═══════════════════════ ncu verification (optional) ════════════════════ */
static void verify_counters(const char *binary, MemLevel mem, OpType op,
                              uint64_t array_bytes, int threads, int blocks,
                              uint32_t stride, uint32_t gran, uint32_t n_iters)
{
    FILE *fp=popen("ncu --version 2>&1","r"); if(!fp) return;
    char buf[128]={0};
    int ok=(fgets(buf,sizeof(buf),fp)&&strstr(buf,"NVIDIA"));
    pclose(fp);
    if(!ok){printf("  [ncu] not found -- skipping counter verification\n");return;}

    const char *metric;
    switch(mem){
        case MEM_L1: case MEM_SHARED: metric="l1tex__t_sector_hit_rate.pct"; break;
        case MEM_L2:                  metric="lts__t_sector_hit_rate.pct";   break;
        case MEM_DRAM: printf("  [ncu] DRAM: no hit-rate metric, skipping\n"); return;
        default:                      metric="l1tex__t_sector_hit_rate.pct"; break;
    }

    char cmd[2048];
    snprintf(cmd,sizeof(cmd),
        "ncu --metrics %s --csv --quiet --kernel-name %s_kernel "
        "%s --mem %s --array-bytes %llu --threads %d --blocks %d "
        "--stride %u --granularity %u --n-iters %u --load-only --dry-run 2>&1",
        metric, OP_NAMES[op], binary, MEM_NAMES[mem],
        (unsigned long long)array_bytes, threads, blocks, stride, gran, n_iters);

    fp=popen(cmd,"r"); if(!fp) return;
    char line[512]; double hr=-1.0; int hdr=0;
    while(fgets(line,sizeof(line),fp)){
        if(strstr(line,"Metric Name")){hdr=1;continue;}
        if(!hdr) continue;
        if(strstr(line,metric)){char *p=strrchr(line,',');if(p)hr=atof(p+1);}
    }
    pclose(fp);
    if(hr<0.0) printf("  [ncu] could not parse hit rate\n");
    else        printf("  [ncu] %s %s hit rate = %.1f%%  %s\n",
                       MEM_NAMES[mem],OP_NAMES[op],hr,
                       hr>=95.0?"PASS":"FAIL -- check array size/stride");
}

/* ═══════════════════════ CLI ════════════════════════════════════════════ */
static void usage(const char *p){
    printf("Usage: %s [options]\n\n"
           "  --mem         <l1|l2|dram|shared>  (default: l1)\n"
           "  --min-acc     <N>  millions         (default: 15)\n"
           "  --max-acc     <N>  millions         (default: 3000)\n"
           "  --acc-steps   <N>                   (default: 20)\n"
           "  --n-runs      <N>                   (default: 5)\n"
           "  --threads     <N>                   (default: 1024)\n"
           "  --blocks      <N>                   (default: 4)\n"
           "  --granularity <B>  bytes             (default: 32)\n"
           "  --stride      <B>  bytes             (default: 32)\n"
           "  --n-iters     <N>                   (default: 200)\n"
           "  --array-bytes <B>  override\n"
           "  --outdir      <P>                   (default: results)\n"
           "  --load-only   --store-only\n"
           "  --dry-run     --verbose  --help\n",p);}

static Config parse_args(int argc, char **argv){
    Config c={.mem_level=MEM_L1,.min_acc_M=15,.max_acc_M=3000,
              .acc_steps=20,.n_runs=5,.threads=1024,.blocks=4,
              .granularity=32,.stride=32,.n_iters=200,.array_bytes=0,
              .load_only=0,.store_only=0,.dry_run=0,.verbose=0};
    strncpy(c.outdir,"results",sizeof(c.outdir)-1);
    for(int i=1;i<argc;++i){
        if     (!strcmp(argv[i],"--help"     )){usage(argv[0]);exit(0);}
        else if(!strcmp(argv[i],"--dry-run"  )) c.dry_run   =1;
        else if(!strcmp(argv[i],"--load-only")) c.load_only =1;
        else if(!strcmp(argv[i],"--store-only"))c.store_only=1;
        else if(!strcmp(argv[i],"--verbose"  )) c.verbose   =1;
        else if(!strcmp(argv[i],"--mem")&&i+1<argc){
            const char *m=argv[++i];
            if     (!strcmp(m,"l1"    )) c.mem_level=MEM_L1;
            else if(!strcmp(m,"l2"    )) c.mem_level=MEM_L2;
            else if(!strcmp(m,"dram"  )) c.mem_level=MEM_DRAM;
            else if(!strcmp(m,"shared")) c.mem_level=MEM_SHARED;
            else{fprintf(stderr,"Unknown mem: %s\n",m);exit(1);}
        }
        else if(!strcmp(argv[i],"--min-acc"    )&&i+1<argc) c.min_acc_M  =(uint64_t)atoll(argv[++i]);
        else if(!strcmp(argv[i],"--max-acc"    )&&i+1<argc) c.max_acc_M  =(uint64_t)atoll(argv[++i]);
        else if(!strcmp(argv[i],"--acc-steps"  )&&i+1<argc) c.acc_steps  =atoi(argv[++i]);
        else if(!strcmp(argv[i],"--n-runs"     )&&i+1<argc) c.n_runs     =atoi(argv[++i]);
        else if(!strcmp(argv[i],"--threads"    )&&i+1<argc) c.threads    =atoi(argv[++i]);
        else if(!strcmp(argv[i],"--blocks"     )&&i+1<argc) c.blocks     =atoi(argv[++i]);
        else if(!strcmp(argv[i],"--granularity")&&i+1<argc) c.granularity=(uint32_t)atoi(argv[++i]);
        else if(!strcmp(argv[i],"--stride"     )&&i+1<argc) c.stride     =(uint32_t)atoi(argv[++i]);
        else if(!strcmp(argv[i],"--n-iters"    )&&i+1<argc) c.n_iters    =(uint32_t)atoi(argv[++i]);
        else if(!strcmp(argv[i],"--array-bytes")&&i+1<argc) c.array_bytes=(uint64_t)atoll(argv[++i]);
        else if(!strcmp(argv[i],"--outdir"     )&&i+1<argc) strncpy(c.outdir,argv[++i],sizeof(c.outdir)-1);
    }
    if(c.stride<c.granularity) c.stride=c.granularity;
    if(c.array_bytes==0)       c.array_bytes=default_array_bytes(c.mem_level);
    return c;
}

/* ═══════════════════════ main ═══════════════════════════════════════════ */
int main(int argc, char **argv)
{
    Config cfg = parse_args(argc, argv);

    printf("=======================================================\n");
    printf(" GPU Energy Model Calibration Tool\n");
    printf(" Ref: Delestrac et al. ASAP 2024, Section V-B\n");
    printf("=======================================================\n");

    nvmlDevice_t  nvml_dev = NULL;
    char          gpu_name[256] = "SYNTHETIC";
    PowerSampler *ps = NULL;

    if (!cfg.dry_run) {
        int dc=0;
        CUDA_CHECK(cudaGetDeviceCount(&dc));
        if(dc==0){fprintf(stderr,"No CUDA devices.\n");return 1;}
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
        strncpy(gpu_name,prop.name,sizeof(gpu_name)-1);

        NVML_CHECK(nvmlInit());
        NVML_CHECK(nvmlDeviceGetHandleByIndex(0,&nvml_dev));

        /* lock to base clock for reproducibility */
        unsigned int clks[64]; unsigned int nclk=64;
        if(nvmlDeviceGetSupportedGraphicsClocks(nvml_dev,
                prop.clockRate/1000,&nclk,clks)==NVML_SUCCESS && nclk>0){
            nvmlDeviceSetApplicationsClocks(nvml_dev,clks[nclk-1],clks[nclk-1]);
            printf(" Clock locked to base: %u MHz\n",clks[nclk-1]);
        }
        ps = ps_create(nvml_dev, POWER_SAMPLE_MS);
    }

    printf(" Device      : %s\n",   gpu_name);
    printf(" Memory      : %s\n",   MEM_NAMES[cfg.mem_level]);
    printf(" Array       : %llu bytes (%.1f KB)\n",
           (unsigned long long)cfg.array_bytes, cfg.array_bytes/1024.0);
    printf(" Stride      : %u bytes\n",   cfg.stride);
    printf(" Granularity : %u bytes\n",   cfg.granularity);
    printf(" Threads     : %d per block, %d blocks\n",cfg.threads,cfg.blocks);
    printf(" Sweep       : %llu M -> %llu M, %d steps\n",
           (unsigned long long)cfg.min_acc_M,
           (unsigned long long)cfg.max_acc_M, cfg.acc_steps);
    printf(" Runs/point  : %d\n", cfg.n_runs);
    printf("=======================================================\n\n");

    struct stat st={0};
    if(stat(cfg.outdir,&st)==-1) mkdir(cfg.outdir,0755);

    /* ── GPU buffers ──────────────────────────────────────────────────────
     * d_chain: read-only chain — allocated once, never written by kernels
     * d_data:  write scratchpad for STORE kernel — reset before each run
     * ──────────────────────────────────────────────────────────────────── */
    uint64_t  n_elems  = cfg.array_bytes / ELEM_SIZE;
    uint64_t  stride_e = cfg.stride      / ELEM_SIZE;
    uint64_t *h_chain  = NULL;
    uint64_t *d_chain  = NULL;
    uint64_t *d_data   = NULL;

    if (!cfg.dry_run) {
        h_chain = (uint64_t *)malloc(cfg.array_bytes);
        CUDA_CHECK(cudaMalloc(&d_chain, cfg.array_bytes));
        CUDA_CHECK(cudaMalloc(&d_data,  cfg.array_bytes));
        build_chain(h_chain, n_elems, stride_e);
        CUDA_CHECK(cudaMemcpy(d_chain, h_chain, cfg.array_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_data, 0, cfg.array_bytes));

        uint64_t *d_tmp; CUDA_CHECK(cudaMalloc(&d_tmp,sizeof(uint64_t)));
        load_kernel<<<cfg.blocks,cfg.threads>>>(d_chain,n_elems,stride_e,10,0,d_tmp);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_tmp));
        printf("GPU warmed up.\n\n");
    }

    uint64_t *acc_vals = (uint64_t *)malloc(cfg.acc_steps * sizeof(uint64_t));
    gen_acc_sweep(acc_vals, cfg.acc_steps, cfg.min_acc_M, cfg.max_acc_M);

    MeasPoint *pts_load  = (MeasPoint *)calloc(cfg.acc_steps, sizeof(MeasPoint));
    MeasPoint *pts_store = (MeasPoint *)calloc(cfg.acc_steps, sizeof(MeasPoint));
    LinReg lr_load={0}, lr_store={0};
    int    load_ok=0,   store_ok=0;

    /* ════ LOAD sweep ════════════════════════════════════════════════════ */
    if (!cfg.store_only) {
        printf("--------------------------------------------------\n");
        printf(" LOAD sweep  (%s, %d points)\n",MEM_NAMES[cfg.mem_level],cfg.acc_steps);
        printf("--------------------------------------------------\n");
        printf("  %-6s  %-16s  %-14s  %-12s\n","Step","#accesses","E_dyn (J)","std (J)");

        for (int i=0; i<cfg.acc_steps; ++i) {
            uint64_t base = calc_accesses(cfg.array_bytes,cfg.blocks,cfg.threads,
                                          cfg.stride,cfg.granularity)*(uint64_t)UNROLL;
            uint32_t ni = base>0 ? (uint32_t)((acc_vals[i]+base-1)/base) : cfg.n_iters;
            if(ni<1) ni=1;

            if (cfg.dry_run) {
                uint64_t actual = calc_accesses(cfg.array_bytes,cfg.blocks,cfg.threads,
                                                cfg.stride,cfg.granularity)
                                * (uint64_t)ni*(uint64_t)UNROLL;
                pts_load[i] = synth_point(actual, cfg.mem_level, OP_LOAD, cfg.n_runs);
            } else {
                pts_load[i] = run_point(d_chain, d_data,
                                         cfg.array_bytes, n_elems, stride_e, ni,
                                         cfg.threads, cfg.blocks,
                                         cfg.stride, cfg.granularity,
                                         OP_LOAD, nvml_dev, ps, cfg.n_runs, cfg.verbose);
            }
            /* FIX BUG 3: unconditional print */
            printf("  %-6d  %-16llu  %-14.6f  %-12.6f\n",
                   i+1,(unsigned long long)pts_load[i].n_accesses,
                   pts_load[i].avg_dynamic_energy_j,
                   pts_load[i].std_dynamic_energy_j);
        }

        double *rx=(double*)malloc(cfg.acc_steps*sizeof(double));
        double *ry=(double*)malloc(cfg.acc_steps*sizeof(double));
        for(int i=0;i<cfg.acc_steps;++i){rx[i]=(double)pts_load[i].n_accesses;ry[i]=pts_load[i].avg_dynamic_energy_j;}
        lr_load=do_linreg(rx,ry,cfg.acc_steps); load_ok=1;
        free(rx); free(ry);

        printf("\n  LOAD regression:\n");
        printf("    e_MEM = %.4f pJ/access  (%.6f nJ/access)\n",lr_load.slope_pj,lr_load.slope_nj);
        printf("    D_MEM = %.4f J\n",lr_load.intercept);
        printf("    R2    = %.6f\n\n",lr_load.r_squared);

        char path[512];
        snprintf(path,sizeof(path),"%s/calibration_load_%s.csv",cfg.outdir,MEM_NAMES[cfg.mem_level]);
        write_sweep_csv(path,pts_load,cfg.acc_steps,cfg.mem_level,OP_LOAD,&cfg,gpu_name,&lr_load);

        if(!cfg.dry_run)
            verify_counters(argv[0],cfg.mem_level,OP_LOAD,cfg.array_bytes,
                            cfg.threads,cfg.blocks,cfg.stride,cfg.granularity,cfg.n_iters);
    }

    /* ════ STORE sweep ═══════════════════════════════════════════════════ */
    if (!cfg.load_only) {
        printf("--------------------------------------------------\n");
        printf(" STORE sweep  (%s, %d points)\n",MEM_NAMES[cfg.mem_level],cfg.acc_steps);
        printf("--------------------------------------------------\n");
        printf("  %-6s  %-16s  %-14s  %-12s\n","Step","#accesses","E_dyn (J)","std (J)");

        for (int i=0; i<cfg.acc_steps; ++i) {
            uint64_t base = calc_accesses(cfg.array_bytes,cfg.blocks,cfg.threads,
                                          cfg.stride,cfg.granularity)*(uint64_t)UNROLL;
            uint32_t ni = base>0 ? (uint32_t)((acc_vals[i]+base-1)/base) : cfg.n_iters;
            if(ni<1) ni=1;

            if (cfg.dry_run) {
                uint64_t actual = calc_accesses(cfg.array_bytes,cfg.blocks,cfg.threads,
                                                cfg.stride,cfg.granularity)
                                * (uint64_t)ni*(uint64_t)UNROLL;
                pts_store[i] = synth_point(actual, cfg.mem_level, OP_STORE, cfg.n_runs);
            } else {
                pts_store[i] = run_point(d_chain, d_data,
                                          cfg.array_bytes, n_elems, stride_e, ni,
                                          cfg.threads, cfg.blocks,
                                          cfg.stride, cfg.granularity,
                                          OP_STORE, nvml_dev, ps, cfg.n_runs, cfg.verbose);
            }
            /* FIX BUG 3: unconditional print */
            printf("  %-6d  %-16llu  %-14.6f  %-12.6f\n",
                   i+1,(unsigned long long)pts_store[i].n_accesses,
                   pts_store[i].avg_dynamic_energy_j,
                   pts_store[i].std_dynamic_energy_j);
        }

        double *rx=(double*)malloc(cfg.acc_steps*sizeof(double));
        double *ry=(double*)malloc(cfg.acc_steps*sizeof(double));
        for(int i=0;i<cfg.acc_steps;++i){rx[i]=(double)pts_store[i].n_accesses;ry[i]=pts_store[i].avg_dynamic_energy_j;}
        lr_store=do_linreg(rx,ry,cfg.acc_steps); store_ok=1;
        free(rx); free(ry);

        printf("\n  STORE regression:\n");
        printf("    e_MEM = %.4f pJ/access  (%.6f nJ/access)\n",lr_store.slope_pj,lr_store.slope_nj);
        printf("    D_MEM = %.4f J\n",lr_store.intercept);
        printf("    R2    = %.6f\n\n",lr_store.r_squared);

        char path[512];
        snprintf(path,sizeof(path),"%s/calibration_store_%s.csv",cfg.outdir,MEM_NAMES[cfg.mem_level]);
        write_sweep_csv(path,pts_store,cfg.acc_steps,cfg.mem_level,OP_STORE,&cfg,gpu_name,&lr_store);

        if(!cfg.dry_run)
            verify_counters(argv[0],cfg.mem_level,OP_STORE,cfg.array_bytes,
                            cfg.threads,cfg.blocks,cfg.stride,cfg.granularity,cfg.n_iters);
    }

    /* ── summary ── */
    {
        char path[512];
        snprintf(path,sizeof(path),"%s/calibration_summary_%s.csv",
                 cfg.outdir,MEM_NAMES[cfg.mem_level]);
        write_summary_csv(path,&lr_load,load_ok,&lr_store,store_ok,
                          cfg.mem_level,&cfg,gpu_name);
    }

    /* ── cleanup ── */
    if(!cfg.dry_run){
        if(d_chain) CUDA_CHECK(cudaFree(d_chain));
        if(d_data)  CUDA_CHECK(cudaFree(d_data));
        if(h_chain) free(h_chain);
        if(ps)      ps_free(ps);
        nvmlDeviceResetApplicationsClocks(nvml_dev);
        nvmlShutdown();
    }
    free(pts_load); free(pts_store); free(acc_vals);

    printf("=======================================================\n");
    printf(" Done. Results in: %s/\n",cfg.outdir);
    printf("=======================================================\n");
    return 0;
}