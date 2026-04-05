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
 *   --tdp-w       <W>  Average GPU power in Watts for fallback  (auto-detected)
 *   --outdir      <P>  Output directory                         (default: results)
 *   --load-only        Run LOAD benchmark only
 *   --store-only       Run STORE benchmark only
 *   --dry-run          Synthetic data mode (no GPU required)
 *   --verbose          Print per-run energy values
 *   --help
 *
 * Outputs
 * -------
 *   <outdir>/calibration_load_<mem>.csv
 *   <outdir>/calibration_store_<mem>.csv
 *   <outdir>/calibration_summary_<mem>.csv
 *
 * Power measurement strategy
 * --------------------------
 * Three methods are attempted in order:
 *
 *   METHOD 1 — NVML internal sensor (nvmlDeviceGetPowerUsage)
 *     Works on: desktop GPUs, data-centre GPUs (A100, V100, etc.)
 *     Fails on: many laptop GPUs under Optimus/PRIME on Linux, where
 *               NVML returns SUCCESS but always reads 0 mW.
 *     Detection: probe at startup; if all samples read 0 after a live
 *               kernel, fall through to method 2.
 *
 *   METHOD 2 — nvidia-smi subprocess polling
 *     Spawns "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits
 *     --loop-ms=50" as a background process and reads its stdout.
 *     Works on most systems where NVML fails.
 *
 *   METHOD 3 — cudaEvent elapsed-time × average power (fallback)
 *     Uses cudaEventElapsedTime for a high-resolution kernel duration.
 *     Energy = avg_power_W × duration_s.
 *     avg_power_W is obtained from:
 *       (a) --tdp-w CLI argument
 *       (b) "nvidia-smi --query-gpu=power.draw" single-shot read
 *       (c) "nvidia-smi --query-gpu=power.limit" × 0.7 (70% of TDP)
 *     This is less precise but always produces non-zero values.
 *     It correctly captures the *relative* energy between warmup and
 *     full runs, which is what the linear regression requires.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <pthread.h>
#include <fcntl.h>
#include <signal.h>
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
#define ELEM_SIZE          8ULL
#define UNROLL             8
/*
 * POWER_SAMPLE_MS = 1 ms.
 * The L1 kernel at small access counts completes in < 1 ms on a fast GPU.
 * At 50 ms the sampler simply never wakes during the kernel window → zero.
 * NVML handles 1 ms polling reliably; the paper uses a 5 ms interval on
 * A100 because kernels there run for many seconds.  For shorter kernels we
 * need a finer interval.  MAX_POWER_SAMPLES is raised accordingly so we
 * never run out of buffer even for long DRAM sweeps.
 */
#define POWER_SAMPLE_MS    1            /* 1 ms — catches even sub-ms kernels */
#define MAX_POWER_SAMPLES  600000       /* 600 s worth at 1 ms                */
#define IDLE_DELAY_MS      500          /* CPU sleep before/after kernel      */
#define IDLE_WINDOW_MS     300          /* must be < IDLE_DELAY_MS            */

/* Power measurement method tags */
typedef enum { PWR_NVML=0, PWR_SMI, PWR_TDP } PwrMethod;
static const char *PWR_METHOD_NAMES[] = { "NVML sensor", "nvidia-smi poll", "TDP estimate" };

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
    double   user_tdp_w;    /* 0 = auto-detect */
    char     outdir[256];
    int      load_only, store_only, dry_run, verbose;
} Config;

/* ═══════════════════════ monotonic clock ═════════════════════════════════ */
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

/* ═══════════════════════ nvidia-smi helper ═══════════════════════════════ */
/* Read a single power.draw sample from nvidia-smi. Returns -1.0 on failure. */
static double smi_read_power_w(void)
{
    FILE *f = popen(
        "nvidia-smi --query-gpu=power.draw "
        "--format=csv,noheader,nounits 2>/dev/null", "r");
    if (!f) return -1.0;
    char buf[64] = {0};
    fgets(buf, sizeof(buf), f);
    pclose(f);
    double v = atof(buf);
    return (v > 0.0) ? v : -1.0;
}

/* Read power.limit (TDP) from nvidia-smi. Returns -1.0 on failure. */
static double smi_read_tdp_w(void)
{
    FILE *f = popen(
        "nvidia-smi --query-gpu=power.limit "
        "--format=csv,noheader,nounits 2>/dev/null", "r");
    if (!f) return -1.0;
    char buf[64] = {0};
    fgets(buf, sizeof(buf), f);
    pclose(f);
    double v = atof(buf);
    return (v > 0.0) ? v : -1.0;
}

/* ═══════════════════════ NVML power probe ════════════════════════════════
 * Read NVML power 10 times with 50 ms gaps.
 * Returns the mean reading in Watts, or 0.0 if all readings are zero
 * (indicating NVML power readout is unsupported on this GPU/driver).
 * ════════════════════════════════════════════════════════════════════════ */
static double nvml_probe_power(nvmlDevice_t dev)
{
    double sum = 0.0;
    int    n   = 0;
    for (int i = 0; i < 10; ++i) {
        unsigned int pw = 0;
        if (nvmlDeviceGetPowerUsage(dev, &pw) == NVML_SUCCESS && pw > 0) {
            sum += pw / 1000.0;
            n++;
        }
        sleep_ms(50);
    }
    return n > 0 ? sum / n : 0.0;
}

/* ═══════════════════════ POWER SAMPLER (background pthread) ══════════════
 *
 * Supports two sampling backends:
 *   - NVML:      call nvmlDeviceGetPowerUsage() directly
 *   - nvidia-smi: read from a long-running "nvidia-smi --loop-ms" pipe
 *
 * All timestamps are absolute CLOCK_MONOTONIC µs, same as mono_us().
 * ════════════════════════════════════════════════════════════════════════ */
typedef struct {
    nvmlDevice_t  device;
    PwrMethod     method;
    FILE         *smi_pipe;           /* for PWR_SMI method */
    volatile int  running;
    double       *samples_w;          /* Watts */
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
        ps->timestamps_us[idx] = mono_us();

        double pw_w = 0.0;
        if (ps->method == PWR_NVML) {
            unsigned int pw_mw = 0;
            nvmlDeviceGetPowerUsage(ps->device, &pw_mw);
            pw_w = pw_mw / 1000.0;
        } else if (ps->method == PWR_SMI && ps->smi_pipe) {
            char buf[64] = {0};
            if (fgets(buf, sizeof(buf), ps->smi_pipe))
                pw_w = atof(buf);
        }

        ps->samples_w[idx] = pw_w;
        __sync_synchronize();
        ps->n_samples = idx + 1;
        struct timespec t = { 0, (long)ps->interval_ms * 1000000L };
        nanosleep(&t, NULL);
    }
    return NULL;
}

static PowerSampler *ps_create(nvmlDevice_t dev, PwrMethod method, int interval_ms)
{
    PowerSampler *ps  = (PowerSampler *)calloc(1, sizeof(PowerSampler));
    ps->device        = dev;
    ps->method        = method;
    ps->interval_ms   = interval_ms;
    ps->max_samples   = MAX_POWER_SAMPLES;
    ps->samples_w     = (double *)  malloc(MAX_POWER_SAMPLES * sizeof(double));
    ps->timestamps_us = (uint64_t *)malloc(MAX_POWER_SAMPLES * sizeof(uint64_t));

    if (method == PWR_SMI) {
        char cmd[256];
        snprintf(cmd, sizeof(cmd),
            "nvidia-smi --query-gpu=power.draw "
            "--format=csv,noheader,nounits --loop-ms=%d 2>/dev/null",
            interval_ms);
        ps->smi_pipe = popen(cmd, "r");
        if (!ps->smi_pipe)
            fprintf(stderr, "WARNING: could not open nvidia-smi pipe\n");
    }
    return ps;
}

static void ps_start(PowerSampler *ps, pthread_t *tid)
{ ps->running = 1; ps->n_samples = 0; pthread_create(tid,NULL,power_thread,ps); }

static void ps_stop(PowerSampler *ps, pthread_t tid)
{ ps->running = 0; pthread_join(tid, NULL); }

static void ps_free(PowerSampler *ps)
{
    if (ps->smi_pipe) pclose(ps->smi_pipe);
    free(ps->samples_w);
    free(ps->timestamps_us);
    free(ps);
}

/* ═══════════════════════ energy from power trace ═════════════════════════
 * Identical logic for all sampler methods — just uses samples_w[].
 * ════════════════════════════════════════════════════════════════════════ */
typedef struct {
    double static_power_w;
    double total_energy_j;
    double dynamic_energy_j;
    double duration_s;
    int    n_idle_samples;
    int    n_kernel_samples;
} EnergyResult;

static int g_energy_diag = 1;   /* print diagnostics for first few calls */

static EnergyResult compute_energy(const PowerSampler *ps,
                                    uint64_t kernel_start_us,
                                    uint64_t kernel_end_us)
{
    EnergyResult r = {0};
    int n = ps->n_samples;

    if (g_energy_diag > 0) {
        g_energy_diag--;
        printf("  [diag] compute_energy: n_samples=%d  "
               "kernel=[%llu, %llu]  duration_us=%llu\n",
               n,
               (unsigned long long)kernel_start_us,
               (unsigned long long)kernel_end_us,
               (unsigned long long)(kernel_end_us - kernel_start_us));
        if (n > 0)
            printf("  [diag] first_sample_ts=%llu  last_sample_ts=%llu\n",
                   (unsigned long long)ps->timestamps_us[0],
                   (unsigned long long)ps->timestamps_us[n-1]);
    }

    /* Guard against uint64_t underflow: if sampler just started and
       kernel_start_us < IDLE_WINDOW_MS*1000, subtraction wraps to a huge
       number and the idle check matches every sample.  Clamp to 0. */
    uint64_t idle_window_us = (uint64_t)(IDLE_WINDOW_MS * 1000);
    uint64_t idle_lo = (kernel_start_us > idle_window_us)
                     ? kernel_start_us - idle_window_us : 0;
    double   idle_sum = 0.0;
    for (int i=0; i<n; ++i) {
        uint64_t t = ps->timestamps_us[i];
        if (t >= idle_lo && t < kernel_start_us) {
            idle_sum += ps->samples_w[i];
            r.n_idle_samples++;
        }
    }
    r.static_power_w = r.n_idle_samples > 0
                     ? idle_sum / r.n_idle_samples : 0.0;

    double prev_t = -1.0, prev_p = 0.0;
    for (int i=0; i<n; ++i) {
        uint64_t t = ps->timestamps_us[i];
        if (t < kernel_start_us || t > kernel_end_us) continue;
        double t_s = (double)t / 1e6;
        double p_w = ps->samples_w[i];
        if (prev_t >= 0.0)
            r.total_energy_j += (p_w + prev_p) * 0.5 * (t_s - prev_t);
        prev_t = t_s; prev_p = p_w;
        r.n_kernel_samples++;
    }

    r.duration_s       = (double)(kernel_end_us - kernel_start_us) / 1e6;
    r.dynamic_energy_j = r.total_energy_j - r.static_power_w * r.duration_s;
    if (r.dynamic_energy_j < 0.0) r.dynamic_energy_j = 0.0;
    return r;
}

/* ═══════════════════════ TDP fallback energy ═════════════════════════════
 * When both NVML and nvidia-smi give zero, use cudaEvent elapsed time
 * multiplied by avg_power_w.  The static (idle) power is estimated as
 * 30% of avg_power_w (typical for modern GPUs under light load).
 *
 * E_total   = avg_power_w  × duration_s
 * E_static  = static_power × duration_s
 * E_dynamic = (avg_power_w − static_power) × duration_s
 * ════════════════════════════════════════════════════════════════════════ */
static EnergyResult tdp_energy(float elapsed_ms, double avg_power_w)
{
    EnergyResult r = {0};
    r.duration_s       = elapsed_ms / 1000.0;
    r.static_power_w   = avg_power_w * 0.30;
    r.total_energy_j   = avg_power_w * r.duration_s;
    r.dynamic_energy_j = (avg_power_w - r.static_power_w) * r.duration_s;
    r.n_kernel_samples = 1;
    r.n_idle_samples   = 0;
    return r;
}

/* ═══════════════════════ pointer-chain init ══════════════════════════════ */
static void build_chain(uint64_t *a, uint64_t n, uint64_t se)
{ for (uint64_t i=0; i<n; ++i) a[i]=(i+se)%n; }

/* ═══════════════════════ LOAD kernel ════════════════════════════════════ */
__global__ void load_kernel(
    const uint64_t * __restrict__ array,
    uint64_t n_elems, uint64_t stride_elems,
    uint32_t n_iters, int do_warmup_only, uint64_t *out)
{
    uint64_t tid = (uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    uint64_t pos = tid % n_elems;
    uint64_t acc = 0;
    uint64_t steps = n_elems / stride_elems;
    for (uint64_t s=0; s<steps; ++s) { pos=array[pos]; acc^=pos; }
    if (do_warmup_only) { if (tid==0) *out=acc; return; }
    pos = tid % n_elems;
    for (uint32_t it=0; it<n_iters; ++it) {
        pos=array[pos];acc^=pos; pos=array[pos];acc^=pos;
        pos=array[pos];acc^=pos; pos=array[pos];acc^=pos;
        pos=array[pos];acc^=pos; pos=array[pos];acc^=pos;
        pos=array[pos];acc^=pos; pos=array[pos];acc^=pos;
    }
    if (tid==0) *out=acc;
}

/* ═══════════════════════ STORE kernel ═══════════════════════════════════
 * Two-buffer design: chain[] is read-only, data[] is the write target.
 * Prevents the illegal memory access from chain corruption.
 * ════════════════════════════════════════════════════════════════════════ */
__global__ void store_kernel(
    const uint64_t * __restrict__ chain,
    uint64_t *data,
    uint64_t n_elems, uint64_t stride_elems,
    uint32_t n_iters, int do_warmup_only, uint64_t *out)
{
    uint64_t tid = (uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    uint64_t pos = tid % n_elems;
    uint64_t acc = 0;
    uint64_t steps = n_elems / stride_elems;
    for (uint64_t s=0; s<steps; ++s) { pos=chain[pos]; acc^=pos; }
    if (do_warmup_only) { if (tid==0) *out=acc; return; }
    pos = tid % n_elems;
    uint64_t val = tid+1;
    for (uint32_t it=0; it<n_iters; ++it) {
        uint64_t nxt;
        nxt=chain[pos];data[pos]=val^pos;acc^=pos;pos=nxt;
        nxt=chain[pos];data[pos]=val^pos;acc^=pos;pos=nxt;
        nxt=chain[pos];data[pos]=val^pos;acc^=pos;pos=nxt;
        nxt=chain[pos];data[pos]=val^pos;acc^=pos;pos=nxt;
        nxt=chain[pos];data[pos]=val^pos;acc^=pos;pos=nxt;
        nxt=chain[pos];data[pos]=val^pos;acc^=pos;pos=nxt;
        nxt=chain[pos];data[pos]=val^pos;acc^=pos;pos=nxt;
        nxt=chain[pos];data[pos]=val^pos;acc^=pos;pos=nxt;
    }
    if (tid==0) *out=acc;
}

/* ═══════════════════════ helpers ════════════════════════════════════════ */
static uint64_t calc_accesses(uint64_t ab, int bl, int th, uint32_t st, uint32_t gr)
{ return (ab*(uint64_t)bl*(uint64_t)th)/((uint64_t)st*(uint64_t)gr); }

typedef struct { double slope,intercept,r_squared,slope_pj,slope_nj; } LinReg;

static LinReg do_linreg(const double *x, const double *y, int n)
{
    LinReg r={0}; if(n<2) return r;
    double sx=0,sy=0,sxy=0,sxx=0;
    for(int i=0;i<n;++i){sx+=x[i];sy+=y[i];sxy+=x[i]*y[i];sxx+=x[i]*x[i];}
    double d=(double)n*sxx-sx*sx; if(fabs(d)<1e-30) return r;
    r.slope=((double)n*sxy-sx*sy)/d;
    r.intercept=(sy-r.slope*sx)/(double)n;
    double ym=sy/(double)n,ss_tot=0,ss_res=0;
    for(int i=0;i<n;++i){
        double dt=y[i]-ym,dr=y[i]-(r.slope*x[i]+r.intercept);
        ss_tot+=dt*dt; ss_res+=dr*dr;
    }
    r.r_squared=ss_tot>1e-30?1.0-ss_res/ss_tot:0.0;
    r.slope_pj=r.slope*1e12; r.slope_nj=r.slope*1e9;
    return r;
}

static double arr_stddev(const double *a, int n, double mean)
{
    if(n<2) return 0.0; double s=0;
    for(int i=0;i<n;++i){double d=a[i]-mean;s+=d*d;}
    return sqrt(s/(n-1));
}

typedef struct {
    uint64_t n_accesses;
    double avg_dynamic_energy_j, std_dynamic_energy_j;
    double min_dynamic_energy_j, max_dynamic_energy_j;
    double avg_warmup_energy_j,  avg_total_energy_j;
    double avg_static_power_w,   avg_duration_s;
    int    n_valid_runs;
    PwrMethod pwr_method;
} MeasPoint;

/* ═══════════════════════ launch one kernel, time with cudaEvent ══════════ */
static float launch_kernel_timed(
    uint64_t *d_chain, uint64_t *d_data, uint64_t *d_out,
    uint64_t n_elems, uint64_t stride_e,
    uint32_t n_iters, int threads, int blocks,
    OpType op, int warmup_only)
{
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    if (op == OP_LOAD)
        load_kernel<<<blocks,threads>>>(d_chain, n_elems, stride_e,
                                        n_iters, warmup_only, d_out);
    else
        store_kernel<<<blocks,threads>>>(d_chain, d_data, n_elems, stride_e,
                                         n_iters, warmup_only, d_out);

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    return ms;
}

/* ═══════════════════════ run one sweep point ════════════════════════════
 *
 * Energy measurement strategy:
 *
 *   PWR_NVML / PWR_SMI  (when power samples are non-zero):
 *     Start background sampler → sleep → stamp t0 → launch kernel →
 *     stamp t1 → sleep → stop sampler → integrate power trace.
 *     E_meas = E_full.dynamic − E_wu.dynamic
 *
 *   PWR_TDP (fallback when all power samples are zero):
 *     Use cudaEvent elapsed time × avg_power_w for both warmup and full
 *     runs.  The subtraction still isolates measurement-loop energy because
 *     avg_power is assumed constant (linear energy model).
 * ════════════════════════════════════════════════════════════════════════ */
static MeasPoint run_point(
    uint64_t *d_chain, uint64_t *d_data,
    uint64_t  array_bytes, uint64_t n_elems, uint64_t stride_e,
    uint32_t  n_iters, int threads, int blocks,
    uint32_t  stride, uint32_t granularity,
    OpType op, PowerSampler *ps, double avg_power_w,
    int n_runs, int verbose)
{
    MeasPoint mp = {0};
    mp.min_dynamic_energy_j =  1e30;
    mp.max_dynamic_energy_j = -1e30;
    mp.pwr_method            = ps ? ps->method : PWR_TDP;
    double *run_e = (double *)malloc(n_runs * sizeof(double));

    uint64_t *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(uint64_t)));
    mp.n_accesses = calc_accesses(array_bytes, blocks, threads, stride, granularity)
                  * (uint64_t)n_iters * (uint64_t)UNROLL;

    for (int run=0; run<n_runs; ++run) {
        if (op == OP_STORE) CUDA_CHECK(cudaMemset(d_data, 0, array_bytes));

        EnergyResult e_wu, e_full;

        if (ps && ps->method != PWR_TDP) {
            /* ── sampler-based measurement ── */
            pthread_t ptid;
            ps->n_samples = 0;
            ps_start(ps, &ptid);
            sleep_ms(IDLE_DELAY_MS);
            uint64_t t0 = mono_us();

            launch_kernel_timed(d_chain, d_data, d_out, n_elems, stride_e,
                                n_iters, threads, blocks, op, /*warmup_only=*/1);

            uint64_t t1 = mono_us();
            sleep_ms(IDLE_DELAY_MS);
            ps_stop(ps, ptid);
            e_wu = compute_energy(ps, t0, t1);

            if (op == OP_STORE) CUDA_CHECK(cudaMemset(d_data, 0, array_bytes));

            ps->n_samples = 0;
            ps_start(ps, &ptid);
            sleep_ms(IDLE_DELAY_MS);
            uint64_t t2 = mono_us();

            launch_kernel_timed(d_chain, d_data, d_out, n_elems, stride_e,
                                n_iters, threads, blocks, op, /*warmup_only=*/0);

            uint64_t t3 = mono_us();
            sleep_ms(IDLE_DELAY_MS);
            ps_stop(ps, ptid);
            e_full = compute_energy(ps, t2, t3);
        } else {
            /* ── TDP fallback: cudaEvent timing only ── */
            float ms_wu   = launch_kernel_timed(d_chain, d_data, d_out,
                                                 n_elems, stride_e, n_iters,
                                                 threads, blocks, op, 1);
            if (op == OP_STORE) CUDA_CHECK(cudaMemset(d_data, 0, array_bytes));
            float ms_full = launch_kernel_timed(d_chain, d_data, d_out,
                                                 n_elems, stride_e, n_iters,
                                                 threads, blocks, op, 0);
            e_wu   = tdp_energy(ms_wu,   avg_power_w);
            e_full = tdp_energy(ms_full, avg_power_w);
        }

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
            printf("    run %d/%d  E_wu=%.5fJ  E_full=%.5fJ  E_meas=%.6fJ\n",
                   run+1, n_runs,
                   e_wu.dynamic_energy_j, e_full.dynamic_energy_j, e_meas);
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
    {107.0,123.0},{378.0,435.0},{2090.0,2407.0},{82.1,94.5}
};
static const double SYNTH_DELTA_J[4] = {20.0,30.0,40.0,15.0};

static MeasPoint synth_point(uint64_t n_acc, MemLevel m, OpType op, int n_runs)
{
    static unsigned seed=1234;
    double eps=SYNTH_EPS_PJ[m][op]*1e-12, delta=SYNTH_DELTA_J[m];
    if(n_runs>64) n_runs=64;
    double run_e[64], sum=0.0;
    for(int i=0;i<n_runs;++i){
        double noise=1.0+0.015*(((double)(rand_r(&seed)%1000)/500.0)-1.0);
        run_e[i]=(eps*n_acc+delta)*noise; sum+=run_e[i];
    }
    MeasPoint mp={0};
    mp.n_accesses=n_acc;
    mp.avg_dynamic_energy_j=sum/n_runs;
    mp.std_dynamic_energy_j=arr_stddev(run_e,n_runs,mp.avg_dynamic_energy_j);
    mp.min_dynamic_energy_j=run_e[0]; mp.max_dynamic_energy_j=run_e[0];
    for(int i=1;i<n_runs;++i){
        if(run_e[i]<mp.min_dynamic_energy_j) mp.min_dynamic_energy_j=run_e[i];
        if(run_e[i]>mp.max_dynamic_energy_j) mp.max_dynamic_energy_j=run_e[i];
    }
    mp.avg_warmup_energy_j=delta*0.45;
    mp.avg_total_energy_j=mp.avg_dynamic_energy_j+delta;
    mp.avg_static_power_w=30.0;
    mp.avg_duration_s=mp.avg_total_energy_j/80.0;
    mp.n_valid_runs=n_runs;
    mp.pwr_method=PWR_NVML;
    return mp;
}

/* ═══════════════════════ sweep generation ═══════════════════════════════ */
static void gen_acc_sweep(uint64_t *acc, int n, uint64_t min_M, uint64_t max_M)
{
    double lo=log((double)(min_M*1000000ULL)),hi=log((double)(max_M*1000000ULL));
    for(int i=0;i<n;++i) acc[i]=(uint64_t)exp(lo+(double)i/(double)(n-1)*(hi-lo));
}

static uint64_t default_array_bytes(MemLevel m)
{
    switch(m){
        case MEM_L1:return 150ULL*1024; case MEM_L2:return 250ULL*1024;
        case MEM_DRAM:return 50ULL*1024*1024; case MEM_SHARED:return 48ULL*1024;
        default:return 150ULL*1024;
    }
}

/* ═══════════════════════ CSV writers ════════════════════════════════════ */
static void write_sweep_csv(const char *path,
                             const MeasPoint *pts, int n,
                             MemLevel mem, OpType op,
                             const Config *cfg, const char *gpu,
                             const LinReg *lr, PwrMethod method)
{
    FILE *f=fopen(path,"w"); if(!f){perror(path);return;}
    fprintf(f,"# GPU Energy Model Calibration -- Sweep Data\n");
    fprintf(f,"# Reference: Delestrac et al. ASAP 2024, Section V-B\n");
    fprintf(f,"# GPU: %s\n",gpu);
    fprintf(f,"# Memory level: %s\n",MEM_NAMES[mem]);
    fprintf(f,"# Operation: %s\n",OP_NAMES[op]);
    fprintf(f,"# Power measurement method: %s\n",PWR_METHOD_NAMES[method]);
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
            lr->slope,lr->slope_pj);
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
                     ?p->avg_dynamic_energy_j/(double)p->n_accesses*1e12:0.0;
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
                               MemLevel mem, const Config *cfg,
                               const char *gpu, PwrMethod method)
{
    FILE *f=fopen(path,"w"); if(!f){perror(path);return;}
    fprintf(f,"# GPU Energy Model Calibration -- Summary\n");
    fprintf(f,"# Reference: Delestrac et al. ASAP 2024, Section V-B\n");
    fprintf(f,"# GPU: %s\n",gpu);
    fprintf(f,"# Memory level: %s\n",MEM_NAMES[mem]);
    fprintf(f,"# Power measurement method: %s\n",PWR_METHOD_NAMES[method]);
    fprintf(f,"# Stride (bytes): %u\n",cfg->stride);
    fprintf(f,"# Access granularity (bytes): %u\n",cfg->granularity);
    fprintf(f,"# Threads per block: %d\n",cfg->threads);
    fprintf(f,"# Thread blocks: %d\n",cfg->blocks);
    fprintf(f,"#\n");
    fprintf(f,"operation,epsilon_mem_j_per_access,epsilon_mem_pj_per_access,"
              "epsilon_mem_nj_per_access,delta_mem_offset_j,r_squared,"
              "granularity_bytes,stride_bytes,threads_per_block,blocks\n");
    if(lok) fprintf(f,"load,%.12f,%.6f,%.9f,%.6f,%.6f,%u,%u,%d,%d\n",
            lrl->slope,lrl->slope_pj,lrl->slope_nj,
            lrl->intercept,lrl->r_squared,
            cfg->granularity,cfg->stride,cfg->threads,cfg->blocks);
    if(sok) fprintf(f,"store,%.12f,%.6f,%.9f,%.6f,%.6f,%u,%u,%d,%d\n",
            lrs->slope,lrs->slope_pj,lrs->slope_nj,
            lrs->intercept,lrs->r_squared,
            cfg->granularity,cfg->stride,cfg->threads,cfg->blocks);
    fclose(f);
    printf("  Saved: %s\n",path);
}

/* ═══════════════════════ ncu verification ═══════════════════════════════ */
static void verify_counters(const char *binary, MemLevel mem, OpType op,
                              uint64_t ab, int th, int bl,
                              uint32_t st, uint32_t gr, uint32_t ni)
{
    FILE *fp=popen("ncu --version 2>&1","r"); if(!fp) return;
    char buf[128]={0};
    int ok=(fgets(buf,sizeof(buf),fp)&&strstr(buf,"NVIDIA"));
    pclose(fp); if(!ok){printf("  [ncu] not found\n");return;}
    const char *m;
    switch(mem){
        case MEM_L1:case MEM_SHARED: m="l1tex__t_sector_hit_rate.pct"; break;
        case MEM_L2:                 m="lts__t_sector_hit_rate.pct";   break;
        case MEM_DRAM: printf("  [ncu] DRAM: no hit-rate metric\n"); return;
        default:                     m="l1tex__t_sector_hit_rate.pct"; break;
    }
    char cmd[2048];
    snprintf(cmd,sizeof(cmd),
        "ncu --metrics %s --csv --quiet --kernel-name %s_kernel "
        "%s --mem %s --array-bytes %llu --threads %d --blocks %d "
        "--stride %u --granularity %u --n-iters %u --load-only --dry-run 2>&1",
        m,OP_NAMES[op],binary,MEM_NAMES[mem],(unsigned long long)ab,
        th,bl,st,gr,ni);
    fp=popen(cmd,"r"); if(!fp) return;
    char line[512]; double hr=-1.0; int hdr=0;
    while(fgets(line,sizeof(line),fp)){
        if(strstr(line,"Metric Name")){hdr=1;continue;}
        if(!hdr) continue;
        if(strstr(line,m)){char *p=strrchr(line,',');if(p)hr=atof(p+1);}
    }
    pclose(fp);
    if(hr<0.0) printf("  [ncu] could not parse hit rate\n");
    else        printf("  [ncu] %s %s hit rate = %.1f%%  %s\n",
                       MEM_NAMES[mem],OP_NAMES[op],hr,
                       hr>=95.0?"PASS":"FAIL");
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
           "  --tdp-w       <W>  avg GPU power W   (auto-detected)\n"
           "  --outdir      <P>                   (default: results)\n"
           "  --load-only   --store-only\n"
           "  --dry-run     --verbose  --help\n",p);}

static Config parse_args(int argc, char **argv){
    Config c={.mem_level=MEM_L1,.min_acc_M=15,.max_acc_M=3000,
              .acc_steps=20,.n_runs=5,.threads=1024,.blocks=4,
              .granularity=32,.stride=32,.n_iters=200,
              .array_bytes=0,.user_tdp_w=0.0,
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
        else if(!strcmp(argv[i],"--tdp-w"      )&&i+1<argc) c.user_tdp_w =atof(argv[++i]);
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

    nvmlDevice_t  nvml_dev  = NULL;
    char          gpu_name[256] = "SYNTHETIC";
    PowerSampler *ps        = NULL;
    PwrMethod     pwr_method = PWR_NVML;
    double        avg_power_w = 0.0;

    if (!cfg.dry_run) {
        int dc=0;
        CUDA_CHECK(cudaGetDeviceCount(&dc));
        if(dc==0){fprintf(stderr,"No CUDA devices.\n");return 1;}
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
        snprintf(gpu_name,sizeof(gpu_name),"%s",prop.name);

        NVML_CHECK(nvmlInit());
        NVML_CHECK(nvmlDeviceGetHandleByIndex(0,&nvml_dev));

        /* lock to base clock for reproducibility */
        unsigned int clks[64]; unsigned int nclk=64;
        if(nvmlDeviceGetSupportedGraphicsClocks(nvml_dev,
                prop.clockRate/1000,&nclk,clks)==NVML_SUCCESS && nclk>0){
            nvmlDeviceSetApplicationsClocks(nvml_dev,clks[nclk-1],clks[nclk-1]);
            printf(" Clock locked to base: %u MHz\n",clks[nclk-1]);
        }

        /* ── Detect which power measurement method works ── */
        printf(" Probing power sensor...\n");
        double nvml_pw = nvml_probe_power(nvml_dev);

        if (nvml_pw > 1.0) {
            pwr_method  = PWR_NVML;
            avg_power_w = nvml_pw;
            printf(" Power source  : NVML sensor (%.1f W idle)\n", nvml_pw);
            ps = ps_create(nvml_dev, PWR_NVML, POWER_SAMPLE_MS);
        } else {
            /* NVML reads zero — try nvidia-smi */
            printf(" NVML returned 0 W — trying nvidia-smi...\n");
            double smi_pw = smi_read_power_w();
            if (smi_pw > 1.0) {
                pwr_method  = PWR_SMI;
                avg_power_w = smi_pw;
                printf(" Power source  : nvidia-smi poll (%.1f W idle)\n", smi_pw);
                ps = ps_create(nvml_dev, PWR_SMI, POWER_SAMPLE_MS);
            } else {
                /* Both sources give zero — use TDP fallback */
                pwr_method = PWR_TDP;
                if (cfg.user_tdp_w > 0.0) {
                    avg_power_w = cfg.user_tdp_w;
                    printf(" Power source  : user-supplied --tdp-w (%.1f W)\n", avg_power_w);
                } else {
                    double tdp = smi_read_tdp_w();
                    if (tdp > 0.0) {
                        avg_power_w = tdp * 0.70;   /* 70% of TDP = typical load */
                        printf(" Power source  : TDP estimate (%.0f W × 0.70 = %.1f W)\n",
                               tdp, avg_power_w);
                        printf("                 Use --tdp-w <W> to override with a measured value.\n");
                        printf("                 Measure with: nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits\n");
                    } else {
                        avg_power_w = 60.0;  /* conservative default for laptop GPU */
                        printf(" Power source  : default estimate (60 W)\n");
                        printf("                 Override with --tdp-w <W> for better accuracy.\n");
                    }
                }
                printf("\n NOTE: Energy values are proportional to kernel duration.\n");
                printf("       Linear regression over #accesses still reveals\n");
                printf("       e_MEM and D_MEM correctly — only the scale depends\n");
                printf("       on the accuracy of the power estimate.\n");
                ps = NULL;   /* TDP mode uses cudaEvent timing, no sampler */
            }
        }
    }

    printf(" Device      : %s\n",  gpu_name);
    printf(" Memory      : %s\n",  MEM_NAMES[cfg.mem_level]);
    printf(" Array       : %llu bytes (%.1f KB)\n",
           (unsigned long long)cfg.array_bytes, cfg.array_bytes/1024.0);
    printf(" Stride      : %u bytes\n",  cfg.stride);
    printf(" Granularity : %u bytes\n",  cfg.granularity);
    printf(" Threads     : %d per block, %d blocks\n",cfg.threads,cfg.blocks);
    printf(" Sweep       : %llu M -> %llu M, %d steps\n",
           (unsigned long long)cfg.min_acc_M,
           (unsigned long long)cfg.max_acc_M, cfg.acc_steps);
    printf(" Runs/point  : %d\n", cfg.n_runs);
    printf("=======================================================\n\n");

    struct stat st={0};
    if(stat(cfg.outdir,&st)==-1) mkdir(cfg.outdir,0755);

    uint64_t  n_elems  = cfg.array_bytes / ELEM_SIZE;
    uint64_t  stride_e = cfg.stride      / ELEM_SIZE;
    uint64_t *h_chain  = NULL, *d_chain = NULL, *d_data = NULL;

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

    /*
     * Timing pre-run: launch the load kernel with n_iters=1 and measure
     * elapsed time via cudaEvent.  Use this to compute min_iters_for_duration:
     * the minimum n_iters needed so the measurement kernel runs for at least
     * MIN_KERNEL_MS milliseconds, ensuring the 1ms power sampler captures
     * at least MIN_KERNEL_SAMPLES power readings inside the kernel window.
     *
     * Without this, a fast GPU (RTX 5000 Ada) finishes the smallest L1 kernel
     * in < 1 ms and the sampler never wakes during the execution window.
     */
    uint32_t min_iters_for_duration = 0;
    if (!cfg.dry_run) {
        const double MIN_KERNEL_MS      = 200.0;  /* target kernel duration   */
        const int    MIN_KERNEL_SAMPLES = 50;     /* minimum samples to catch */

        uint64_t *d_tmp2; CUDA_CHECK(cudaMalloc(&d_tmp2,sizeof(uint64_t)));
        cudaEvent_t ev0, ev1;
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventRecord(ev0));
        load_kernel<<<cfg.blocks,cfg.threads>>>(d_chain,n_elems,stride_e,1,0,d_tmp2);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms1 = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms1, ev0, ev1));
        CUDA_CHECK(cudaEventDestroy(ev0));
        CUDA_CHECK(cudaEventDestroy(ev1));
        CUDA_CHECK(cudaFree(d_tmp2));

        if (ms1 > 0.0f) {
            /* iters needed for MIN_KERNEL_MS wall-clock */
            uint32_t iters_for_time = (uint32_t)ceil(MIN_KERNEL_MS / ms1);
            /* iters needed for MIN_KERNEL_SAMPLES at POWER_SAMPLE_MS interval */
            uint32_t iters_for_samples = (uint32_t)ceil(
                (MIN_KERNEL_SAMPLES * POWER_SAMPLE_MS) / ms1);
            min_iters_for_duration = iters_for_time > iters_for_samples
                                   ? iters_for_time : iters_for_samples;
            if (min_iters_for_duration < 1) min_iters_for_duration = 1;
            printf(" Timing pre-run: 1 iter = %.3f ms  "
                   "→ min iters for %.0f ms = %u\n\n",
                   ms1, MIN_KERNEL_MS, min_iters_for_duration);
        }
    }

    uint64_t *acc_vals=(uint64_t *)malloc(cfg.acc_steps*sizeof(uint64_t));
    gen_acc_sweep(acc_vals, cfg.acc_steps, cfg.min_acc_M, cfg.max_acc_M);

    MeasPoint *pts_load =(MeasPoint *)calloc(cfg.acc_steps,sizeof(MeasPoint));
    MeasPoint *pts_store=(MeasPoint *)calloc(cfg.acc_steps,sizeof(MeasPoint));
    LinReg lr_load={0}, lr_store={0};
    int    load_ok=0,   store_ok=0;

    /* macro to run one sweep */
#define RUN_SWEEP(LABEL, PTS, OP)                                            \
    do {                                                                     \
        printf("--------------------------------------------------\n");      \
        printf(" " LABEL " sweep  (%s, %d points)\n",                       \
               MEM_NAMES[cfg.mem_level], cfg.acc_steps);                     \
        printf("--------------------------------------------------\n");      \
        printf("  %-6s  %-16s  %-14s  %-12s\n",                             \
               "Step","#accesses","E_dyn (J)","std (J)");                    \
        for (int i=0; i<cfg.acc_steps; ++i) {                               \
            uint64_t base=calc_accesses(cfg.array_bytes,cfg.blocks,          \
                cfg.threads,cfg.stride,cfg.granularity)*(uint64_t)UNROLL;   \
            uint32_t ni=base>0?(uint32_t)((acc_vals[i]+base-1)/base):cfg.n_iters;\
            if(ni<1) ni=1;                                                   \
            /* Ensure the measurement kernel runs long enough for the power  \
             * sampler to capture several samples.  At 1 ms poll rate we     \
             * want at least MIN_KERNEL_SAMPLES samples inside the window,   \
             * so the kernel must run at least MIN_KERNEL_SAMPLES ms.        \
             * We achieve this by repeating the n_iters loop enough times.   \
             * The measured energy still scales linearly with ni because each \
             * iteration does the same work.                                 */ \
            if (!cfg.dry_run && min_iters_for_duration > 0                   \
                && ni < min_iters_for_duration)                               \
                ni = min_iters_for_duration;                                  \
            if (cfg.dry_run) {                                               \
                uint64_t act=calc_accesses(cfg.array_bytes,cfg.blocks,       \
                    cfg.threads,cfg.stride,cfg.granularity)                  \
                    *(uint64_t)ni*(uint64_t)UNROLL;                          \
                PTS[i]=synth_point(act,cfg.mem_level,OP,cfg.n_runs);        \
            } else {                                                         \
                PTS[i]=run_point(d_chain,d_data,                             \
                    cfg.array_bytes,n_elems,stride_e,ni,                     \
                    cfg.threads,cfg.blocks,cfg.stride,cfg.granularity,       \
                    OP,ps,avg_power_w,cfg.n_runs,cfg.verbose);               \
            }                                                                \
            printf("  %-6d  %-16llu  %-14.6f  %-12.6f\n",                  \
                   i+1,(unsigned long long)PTS[i].n_accesses,               \
                   PTS[i].avg_dynamic_energy_j,                              \
                   PTS[i].std_dynamic_energy_j);                             \
        }                                                                    \
    } while(0)

    if (!cfg.store_only) {
        RUN_SWEEP("LOAD", pts_load, OP_LOAD);
        double *rx=(double*)malloc(cfg.acc_steps*sizeof(double));
        double *ry=(double*)malloc(cfg.acc_steps*sizeof(double));
        for(int i=0;i<cfg.acc_steps;++i){rx[i]=(double)pts_load[i].n_accesses;ry[i]=pts_load[i].avg_dynamic_energy_j;}
        lr_load=do_linreg(rx,ry,cfg.acc_steps); load_ok=1; free(rx);free(ry);
        printf("\n  LOAD regression:\n");
        printf("    e_MEM = %.4f pJ/access\n",lr_load.slope_pj);
        printf("    D_MEM = %.4f J\n",lr_load.intercept);
        printf("    R2    = %.6f\n\n",lr_load.r_squared);
        char path[512];
        snprintf(path,sizeof(path),"%s/calibration_load_%s.csv",cfg.outdir,MEM_NAMES[cfg.mem_level]);
        write_sweep_csv(path,pts_load,cfg.acc_steps,cfg.mem_level,OP_LOAD,
                        &cfg,gpu_name,&lr_load,pwr_method);
        if(!cfg.dry_run)
            verify_counters(argv[0],cfg.mem_level,OP_LOAD,cfg.array_bytes,
                            cfg.threads,cfg.blocks,cfg.stride,cfg.granularity,cfg.n_iters);
    }

    if (!cfg.load_only) {
        RUN_SWEEP("STORE", pts_store, OP_STORE);
        double *rx=(double*)malloc(cfg.acc_steps*sizeof(double));
        double *ry=(double*)malloc(cfg.acc_steps*sizeof(double));
        for(int i=0;i<cfg.acc_steps;++i){rx[i]=(double)pts_store[i].n_accesses;ry[i]=pts_store[i].avg_dynamic_energy_j;}
        lr_store=do_linreg(rx,ry,cfg.acc_steps); store_ok=1; free(rx);free(ry);
        printf("\n  STORE regression:\n");
        printf("    e_MEM = %.4f pJ/access\n",lr_store.slope_pj);
        printf("    D_MEM = %.4f J\n",lr_store.intercept);
        printf("    R2    = %.6f\n\n",lr_store.r_squared);
        char path[512];
        snprintf(path,sizeof(path),"%s/calibration_store_%s.csv",cfg.outdir,MEM_NAMES[cfg.mem_level]);
        write_sweep_csv(path,pts_store,cfg.acc_steps,cfg.mem_level,OP_STORE,
                        &cfg,gpu_name,&lr_store,pwr_method);
        if(!cfg.dry_run)
            verify_counters(argv[0],cfg.mem_level,OP_STORE,cfg.array_bytes,
                            cfg.threads,cfg.blocks,cfg.stride,cfg.granularity,cfg.n_iters);
    }

    {
        char path[512];
        snprintf(path,sizeof(path),"%s/calibration_summary_%s.csv",
                 cfg.outdir,MEM_NAMES[cfg.mem_level]);
        write_summary_csv(path,&lr_load,load_ok,&lr_store,store_ok,
                          cfg.mem_level,&cfg,gpu_name,pwr_method);
    }

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
