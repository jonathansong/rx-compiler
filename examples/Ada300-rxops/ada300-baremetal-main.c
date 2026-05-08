/*
 * ada300-baremetal-main.c - Ada300 rx-ops inference demo (baremetal, QEMU)
 *
 * Baremetal C port of ada300-rxops-main.cpp for the RISC-V Ada300 target.
 *
 * Boot assumptions (provided by ada300_snpu_sdk BSP):
 *   startup.S  : sets sp/gp, clears .bss, enables FP/V, calls bsp_main()
 *   bsp_main.c : uart_init() + mr_heap_init(), then calls main()
 *   LPDDR heap : mr_alloc(n, MR_REGION_LPDDR) gives 8 GiB-scale allocations
 *   mini_printf: printf() mapped to the BSP UART; no floating-point support
 *                (format floats as integer_part.fractional_part manually)
 *
 * MLIR runtime symbols provided here:
 *   malloc / free / calloc  : backed by the LPDDR mr_heap pool
 *   memrefCopy              : flat memcpy for contiguous 2-D tensors
 *
 * Subgraph function:
 *   void _mlir_ciface_subgraph0(
 *       StridedMemRef2D *result,      // out [1 x 64]  (malloc'd by callee)
 *       StridedMemRef2D *fc1_weight,  // in  [64 x 128]
 *       StridedMemRef2D *fc1_bias,    // in  [1 x 128]
 *       StridedMemRef2D *fc2_weight,  // in  [128 x 64]
 *       StridedMemRef2D *fc2_bias,    // in  [1 x 64]
 *       StridedMemRef2D *input);      // in  [1 x 64]
 *
 * Build:
 *   See CMakeLists.txt ADA300_BAREMETAL target.
 *
 * Run:
 *   /opt/ada300-qemu/bin/qemu-system-riscv64 \
 *       -machine ada300s_evk -cpu ada300s \
 *       -nographic -serial mon:stdio \
 *       -kernel build/ada300-baremetal.elf
 *   (Exit QEMU: Ctrl-A, then x)
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "mini_printf.h"
#include "mr_heap.h"

/* =========================================================================
 * MLIR baremetal runtime: malloc / free / calloc
 *
 * The generated subgraph0_rxops.ll calls @malloc to allocate intermediate
 * and output tensor buffers.  We back them with the BSP LPDDR heap pool.
 * =========================================================================*/
void *malloc(size_t n)           { return mr_alloc(n, MR_REGION_LPDDR); }
void  free(void *p)              { mr_free(p); }
void *calloc(size_t nmemb, size_t size) {
    return mr_calloc(nmemb, size, MR_REGION_LPDDR);
}

/* =========================================================================
 * rxnn_mem_alloc / rxnn_mem_free — Ada300 rx-ops backend heap
 *
 * rx_ops_memory.c in the Ada300 backend defines these as __attribute__((weak))
 * and falls back to a static 18 MiB TCM bump allocator.  Override them here
 * with strong definitions that use the BSP LPDDR heap so large model scratch
 * buffers (e.g. fp16 conversion temporaries in rxops_bridge_matmul_f32) do
 * not overflow the TCM arena.
 * =========================================================================*/
void *rxnn_mem_alloc(int64_t size) { return mr_alloc((size_t)size, MR_REGION_LPDDR); }
void  rxnn_mem_free(void *ptr)     { mr_free(ptr); }

/* =========================================================================
 * memrefCopy — required by MLIR CRunnerUtils.
 *
 * Called by the -one-shot-bufferize pass when copying ranked memref results.
 * The Ada300 model uses contiguous row-major buffers throughout, so only
 * the flat-copy fast path is exercised.
 *
 * Signature (matches MLIR's CRunnerUtils):
 *   void memrefCopy(int64_t elemSize,
 *                   UnrankedMemRef *src,
 *                   UnrankedMemRef *dst);
 *
 * UnrankedMemRef layout: { int64_t rank; void *descriptor }
 * Descriptor layout:     { void *base; void *data; int64_t offset;
 *                          int64_t sizes[rank]; int64_t strides[rank] }
 * =========================================================================*/
typedef struct {
    int64_t  rank;
    void    *descriptor;
} UnrankedMemRef;

void memrefCopy(int64_t elem_size,
                UnrankedMemRef *src_u,
                UnrankedMemRef *dst_u)
{
    typedef struct { void *base; void *data; int64_t offset; } DescHdr;

    int64_t  rank = src_u->rank;
    DescHdr *sh   = (DescHdr *)src_u->descriptor;
    DescHdr *dh   = (DescHdr *)dst_u->descriptor;
    /* sizes/strides follow immediately after the three pointer/offset fields */
    const int64_t *ss  = (const int64_t *)(sh + 1);          /* src sizes   */
    const int64_t *sst = ss + rank;                           /* src strides */
    const int64_t *ds  = (const int64_t *)(dh + 1);          /* dst sizes   */
    const int64_t *dst = ds + rank;                           /* dst strides */

    /* Count total elements. */
    int64_t total = 1;
    for (int64_t i = 0; i < rank; ++i)
        total *= ss[i];

    /* Fast path: both src and dst are contiguous (row-major). */
    int contiguous = 1;
    int64_t expected = 1;
    for (int64_t i = rank - 1; i >= 0 && contiguous; --i) {
        if (sst[i] != expected || dst[i] != expected)
            contiguous = 0;
        expected *= ss[i];
    }
    if (contiguous) {
        const char *s = (const char *)sh->data + sh->offset * elem_size;
        char       *d = (char *)dh->data + dh->offset * elem_size;
        memcpy(d, s, (size_t)(total * elem_size));
        return;
    }

    /* Slow path: general strided copy. */
    const char *sbase = (const char *)sh->data;
    char       *dbase = (char *)dh->data;
    for (int64_t idx = 0; idx < total; ++idx) {
        int64_t rem = idx, soff = sh->offset, doff = dh->offset;
        for (int64_t dim = rank - 1; dim >= 0; --dim) {
            int64_t coord = rem % ss[dim];
            rem /= ss[dim];
            soff += coord * sst[dim];
            doff += coord * dst[dim];
        }
        memcpy(dbase + doff * elem_size,
               sbase + soff * elem_size,
               (size_t)elem_size);
    }
}

/* =========================================================================
 * 2-D StridedMemRef descriptor
 *
 * Mirrors MLIR's StridedMemRefType<T, 2>.  Must match the layout that
 * -llvm-request-c-wrappers expects for the _mlir_ciface_* wrappers.
 * Layout: { base_ptr, data_ptr, offset, sizes[2], strides[2] }
 * =========================================================================*/
typedef struct {
    float   *base_ptr;
    float   *data_ptr;
    int64_t  offset;
    int64_t  sizes[2];
    int64_t  strides[2];
} StridedMemRef2D;

/* Build a row-major 2-D descriptor over an existing buffer. */
static StridedMemRef2D make_memref2d(float *data, int64_t rows, int64_t cols)
{
    StridedMemRef2D m;
    m.base_ptr   = data;
    m.data_ptr   = data;
    m.offset     = 0;
    m.sizes[0]   = rows;
    m.sizes[1]   = cols;
    m.strides[0] = cols;  /* row-major: inner stride = #cols */
    m.strides[1] = 1;
    return m;
}

/* =========================================================================
 * _mlir_ciface_subgraph0 declaration
 *
 * Generated by -llvm-request-c-wrappers from subgraph0_rxops_finished.mlir.
 * The callee malloc's the output buffer; the caller is responsible for
 * free(result.data_ptr) after use.
 *
 * Arg order (matches subgraph0_top.mlir produced by ada300-import.py):
 *   result      [BATCH x FC2_OUT]  output (malloc'd by callee)
 *   fc1_weight  [FC1_IN x HIDDEN]
 *   fc1_bias    [BATCH  x HIDDEN]
 *   fc2_weight  [HIDDEN x FC2_OUT]
 *   fc2_bias    [BATCH  x FC2_OUT]
 *   input       [BATCH  x FC1_IN]
 * =========================================================================*/
extern void _mlir_ciface_subgraph0(
    StridedMemRef2D *result,
    StridedMemRef2D *fc1_weight,
    StridedMemRef2D *fc1_bias,
    StridedMemRef2D *fc2_weight,
    StridedMemRef2D *fc2_bias,
    StridedMemRef2D *input);

/* =========================================================================
 * Model dimensions (must match ada300-import.py defaults)
 * =========================================================================*/
#define BATCH   1
#define FC1_IN  64
#define HIDDEN  128
#define FC2_OUT 64

/* =========================================================================
 * RISC-V timer helper
 *
 * Reads the `time` CSR (mtime alias exposed to S/U-mode by CLINT).
 * On QEMU ada300s_evk the CLINT frequency is 10 MHz, so one tick = 100 ns.
 * =========================================================================*/
static inline uint64_t rdtime_csr(void)
{
    uint64_t t;
    __asm__ volatile ("rdtime %0" : "=r"(t));
    return t;
}

/* Print an fp32 value as integer + 4 decimal places using mini_printf.
 * mini_printf has no %f support; this avoids a libm dependency. */
static void print_f32(float v)
{
    /* Extract sign */
    int neg = (v < 0.0f);
    if (neg) v = -v;

    /* Integer part */
    int64_t whole = (int64_t)v;
    /* Fractional part × 10000 */
    int64_t frac  = (int64_t)((v - (float)whole) * 10000.0f + 0.5f);
    if (frac >= 10000) { whole++; frac = 0; }

    if (neg) printf("-");
    printf("%lld.%04lld", (long long)whole, (long long)frac);
}

/* =========================================================================
 * main
 * =========================================================================*/
int main(void)
{
    printf("[Ada300] Baremetal rx-ops inference demo\n");
    printf("[Ada300] Model: fc1(%dx%d) -> exp -> sqrt -> fc2(%dx%d)\n",
           (int)FC1_IN, (int)HIDDEN, (int)HIDDEN, (int)FC2_OUT);

    /* --- Allocate weight / bias / input buffers from the LPDDR heap ------- */
    float *fc1w = (float *)mr_alloc(FC1_IN  * HIDDEN  * sizeof(float), MR_REGION_LPDDR);
    float *fc1b = (float *)mr_alloc(BATCH   * HIDDEN  * sizeof(float), MR_REGION_LPDDR);
    float *inp  = (float *)mr_alloc(BATCH   * FC1_IN  * sizeof(float), MR_REGION_LPDDR);
    float *fc2w = (float *)mr_alloc(HIDDEN  * FC2_OUT * sizeof(float), MR_REGION_LPDDR);
    float *fc2b = (float *)mr_alloc(BATCH   * FC2_OUT * sizeof(float), MR_REGION_LPDDR);

    if (!fc1w || !fc1b || !inp || !fc2w || !fc2b) {
        printf("[Ada300] ERROR: heap allocation failed\n");
        return 1;
    }

    /* --- Initialise model weights (0.01), biases (0.0), input (1.0) ------- */
    for (int i = 0; i < FC1_IN  * HIDDEN;  i++) fc1w[i] = 0.01f;
    for (int i = 0; i < BATCH   * HIDDEN;  i++) fc1b[i] = 0.0f;
    for (int i = 0; i < BATCH   * FC1_IN;  i++) inp[i]  = 1.0f;
    for (int i = 0; i < HIDDEN  * FC2_OUT; i++) fc2w[i] = 0.01f;
    for (int i = 0; i < BATCH   * FC2_OUT; i++) fc2b[i] = 0.0f;

    /* --- Build StridedMemRef2D descriptors --------------------------------- */
    StridedMemRef2D mFc1w = make_memref2d(fc1w, FC1_IN,  HIDDEN);
    StridedMemRef2D mFc1b = make_memref2d(fc1b, BATCH,   HIDDEN);
    StridedMemRef2D mInp  = make_memref2d(inp,  BATCH,   FC1_IN);
    StridedMemRef2D mFc2w = make_memref2d(fc2w, HIDDEN,  FC2_OUT);
    StridedMemRef2D mFc2b = make_memref2d(fc2b, BATCH,   FC2_OUT);

    /* Result descriptor: callee allocates output buffer via malloc(). */
    StridedMemRef2D result;
    memset(&result, 0, sizeof(result));

    /* --- Run inference ------------------------------------------------------ */
    printf("[Ada300] Running inference (input=1.0, weights=0.01)...\n");
    uint64_t t0 = rdtime_csr();

    _mlir_ciface_subgraph0(&result, &mFc1w, &mFc1b, &mFc2w, &mFc2b, &mInp);

    uint64_t t1 = rdtime_csr();

    /*
     * QEMU ada300s_evk CLINT frequency = 10 MHz (100 ns per tick).
     * Multiply ticks by 100 to get nanoseconds.
     */
    uint64_t ticks = t1 - t0;
    uint64_t ns    = ticks * 100ULL;
    printf("[Ada300] Inference: %llu ticks (%llu ns ~= %llu us)\n",
           (unsigned long long)ticks,
           (unsigned long long)ns,
           (unsigned long long)(ns / 1000));

    /* --- Print output ------------------------------------------------------- */
    float *out = result.data_ptr;
    printf("[Ada300] Output (1x%d):\n  [", (int)FC2_OUT);
    for (int i = 0; i < FC2_OUT; i++) {
        print_f32(out[i]);
        if (i + 1 < FC2_OUT)
            printf(", ");
    }
    printf("]\n");

    /*
     * Expected output with input=1.0, weights=0.01, biases=0.0:
     *   FC1 matmul: 64 * 1.0 * 0.01 = 0.64  per hidden unit
     *   exp(0.64)  ≈ 1.8965
     *   sqrt(1.8965) ≈ 1.3771
     *   FC2 matmul: 128 * 1.3771 * 0.01 ≈ 1.763  per output
     */
    printf("[Ada300] Expected output ≈ 1.7629 per element\n");

    /* --- Release malloc'd output buffer ------------------------------------ */
    free(result.data_ptr);

    /* --- Release weight buffers -------------------------------------------- */
    mr_free(fc1w);
    mr_free(fc1b);
    mr_free(inp);
    mr_free(fc2w);
    mr_free(fc2b);

    printf("[Ada300] Done.\n");
    return 0;
}
