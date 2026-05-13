/*
 * rx_ops_bridge.c - Flat C ABI wrappers for rx-ops functions.
 *
 * Which bridge symbol gets called is decided entirely by ada300-import.py:
 *   plain  rxops_bridge_<op>_f32        → RXOPS_C  (host CPU)
 *   prefixed rxops_bridge_ada300_<op>_f32 → Ada300  (device or ivshmem)
 *
 * The implementation of the ada300_* symbols differs by compile target,
 * detected automatically from the target architecture — no explicit flag
 * needed:
 *
 *   #ifndef __riscv  (x86 host):
 *     rxops_bridge_ada300_* → ivshmem dispatch to QEMU RISC-V device
 *
 *   #ifdef __riscv   (RISC-V device / bare-metal):
 *     rxops_bridge_ada300_* → RXOPS_ADA300 hardware backend directly
 *     rxops_bridge_*        → RXOPS_C  (soft fallback, normally unused)
 *
 * Op table:
 *   rxops_bridge_exp_f32           → RXOPS_C
 *   rxops_bridge_sqrt_f32          → RXOPS_C
 *   rxops_bridge_log_f32           → RXOPS_C
 *   rxops_bridge_rsqrt_f32         → RXOPS_C
 *   rxops_bridge_matmul_f32        → RXOPS_C   (fp32, no conversion)
 *   rxops_bridge_add_f32           → RXOPS_C
 *   rxops_bridge_ada300_exp_f32    → Ada300 SNPU
 *   rxops_bridge_ada300_sqrt_f32   → Ada300 SNPU
 *   rxops_bridge_ada300_matmul_f32 → Ada300 SNPU  (fp16 path: fp32→fp16→fp32)
 *   rxops_bridge_ada300_add_f32    → Ada300 SNPU
 *
 * Compile with:
 *   -I<rx-ops>/include
 *   -I<rx-ops>/include/interface
 *   (host: also -I<hetero>/include  for hetero_shmem.h)
 */

#include "rx_ops_bridge.h"

#include <string.h>
#include <stdlib.h>   /* malloc / free — backed by mr_alloc on baremetal */

#include "rx_ops.h"
#include "rx_ops_data_structure.h"
#include "rx_ops_data_type.h"   /* __fp16_t = uint16_t */
#include "rx_ops_register.h"

/* -------------------------------------------------------------------------
 * ivshmem dispatch declarations (x86 host only).
 *
 * On the x86 host the ada300_* bridge functions forward calls to the
 * QEMU-hosted RISC-V device via shared memory.  On RISC-V the hardware
 * backend is called directly.
 * -------------------------------------------------------------------------*/
#ifndef __riscv
#include "include/hetero_shmem.h"

extern void *g_hetero_base;

extern int hetero_dispatch_matmul(void *base,
                                   const float *A, const float *B,
                                   float *C,
                                   int32_t M, int32_t N, int32_t K);
extern int hetero_dispatch_sqrt(void *base,
                                 const float *in, float *out, int32_t n);
extern int hetero_dispatch_exp(void *base,
                                const float *in, float *out, int32_t n);
extern int hetero_dispatch_add(void *base,
                                const float *in0, const float *in1,
                                float *out, int32_t n);
#endif /* !__riscv */

/* -------------------------------------------------------------------------
 * One-time backend initialisation:
 *   - rxnn_c_init()      : registers RXOPS_C callback table
 *   - rxnn_ada300_init() : registers RXOPS_ADA300 callback table
 *                          (skipped when using ivshmem dispatch)
 * -------------------------------------------------------------------------*/
/* On RISC-V (bare-metal device): provide a weak no-op stub for rxnn_c_init
 * because the device only links the Ada300 backend (no RXOPS_C library).
 *
 * On x86 (host): forward-declare the real rxnn_c_init() from librx_ops.a.
 * Do NOT define a stub here — rx_ops_bridge.o is placed before
 * -Wl,--whole-archive librx_ops.a in the link command, so any definition
 * (even weak) would shadow the real registration and leave RXOPS_C callbacks
 * empty, causing ops to silently produce zeros. */
#ifdef __riscv
void __attribute__((weak)) rxnn_c_init(void) { }
extern void rxnn_ada300_init(void);
#else
extern void rxnn_c_init(void);  /* real impl in librx_ops.a */
#endif

static int g_rxops_initialised = 0;

static void ensure_rxops_init(void)
{
    if (!g_rxops_initialised) {
        rxnn_c_init();              /* no-op stub on bare-metal device     */
#ifdef __riscv
        rxnn_ada300_init();         /* register RXOPS_ADA300 (Ada300 NPU)  */
#endif
        g_rxops_initialised = 1;
    }
}

/* -------------------------------------------------------------------------
 * Internal helpers: fill rxops_tensor descriptors on the stack.
 * -------------------------------------------------------------------------*/
static void fill_tensor_1d(struct rxops_tensor *t, void *data,
                            enum rxops_dtype_enum dtype, int64_t n)
{
    memset(t, 0, sizeof(*t));
    t->data      = data;
    t->dtype     = dtype;
    t->mtype     = RXOPS_MEM_TYPE_CPU;
    t->dim[0]    = (int32_t)n;
    t->dim_count = 1;
    t->layout    = RXOPS_LAYOUT_N;
}

static void fill_tensor_2d(struct rxops_tensor *t, void *data,
                            enum rxops_dtype_enum dtype,
                            int64_t rows, int64_t cols)
{
    memset(t, 0, sizeof(*t));
    t->data      = data;
    t->dtype     = dtype;
    t->mtype     = RXOPS_MEM_TYPE_CPU;
    t->dim[0]    = (int32_t)rows;
    t->dim[1]    = (int32_t)cols;
    t->dim_count = 2;
    t->layout    = RXOPS_LAYOUT_NC;
}

/* -------------------------------------------------------------------------
 * siso_params helper with embedded callback buffer (avoids heap alloc).
 * -------------------------------------------------------------------------*/
typedef struct {
    struct rxops_siso_params p;
    struct rxops_callback    cb;
} siso_params_with_cb;

static void init_siso_c(siso_params_with_cb *pwcb)
{
    memset(pwcb, 0, sizeof(*pwcb));
    pwcb->p.base.api = RXOPS_C;        /* reference C fallback */
    pwcb->p.base.cb  = &pwcb->cb;
}

#ifdef __riscv
static void init_siso_ada300(siso_params_with_cb *pwcb)
{
    memset(pwcb, 0, sizeof(*pwcb));
    pwcb->p.base.api = RXOPS_ADA300;   /* Ada300 NPU path */
    pwcb->p.base.cb  = &pwcb->cb;
}

/* =========================================================================
 * fp32 ↔ fp16 conversion helpers (RXOPS_ADA300 matmul path only)
 *
 * On riscv64 with -march=...zfh... the Ada300 LLVM toolchain supports
 * the __fp16 type and emits fcvt.h.s / fcvt.s.h instructions.
 * __fp16_t (from rx_ops_data_type.h) is a uint16_t storage alias; we
 * memcpy through __fp16 to get the hardware conversion without UB.
 * =========================================================================*/

/* float32 → fp16 bits (uses fcvt.h.s on riscv64-zfh) */
static inline __fp16_t f32_to_f16(__fp16_t *dst, const float *src, int64_t n)
{
    for (int64_t i = 0; i < n; i++) {
        __fp16 h = (__fp16)src[i];   /* fcvt.h.s */
        __fp16_t bits;
        __builtin_memcpy(&bits, &h, sizeof(bits));
        dst[i] = bits;
    }
    return 0;
}

/* fp16 bits → float32 (uses fcvt.s.h on riscv64-zfh) */
static inline void f16_to_f32(float *dst, const __fp16_t *src, int64_t n)
{
    for (int64_t i = 0; i < n; i++) {
        __fp16 h;
        __builtin_memcpy(&h, &src[i], sizeof(h));
        dst[i] = (float)h;           /* fcvt.s.h */
    }
}
#endif /* __riscv */

/* =========================================================================
 * exp  →  RXOPS_C reference backend
 * =========================================================================*/
int rxops_bridge_exp_f32(float *out, const float *in, int64_t n)
{
    struct rxops_tensor t_in, t_out;
    siso_params_with_cb pwcb;

    ensure_rxops_init();
    fill_tensor_1d(&t_in,  (void *)in,  RXOPS_DTYPE_FLOAT32, n);
    fill_tensor_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);
    init_siso_c(&pwcb);

    rxops_exp_init(&t_in, &t_out, &pwcb.p);
    int rc = rxops_exp(&t_in, &t_out, &pwcb.p);
    return (rc == 1 /* RXOPS_TRUE */) ? 0 : -1;
}

/* =========================================================================
 * sqrt  →  RXOPS_C reference backend
 * =========================================================================*/
int rxops_bridge_sqrt_f32(float *out, const float *in, int64_t n)
{
    struct rxops_tensor t_in, t_out;
    siso_params_with_cb pwcb;

    ensure_rxops_init();
    fill_tensor_1d(&t_in,  (void *)in,  RXOPS_DTYPE_FLOAT32, n);
    fill_tensor_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);
    init_siso_c(&pwcb);

    rxops_sqrt_init(&t_in, &t_out, &pwcb.p);
    int rc = rxops_sqrt(&t_in, &t_out, &pwcb.p);
    return (rc == 1 /* RXOPS_TRUE */) ? 0 : -1;
}

/* =========================================================================
 * log  →  RXOPS_C reference backend
 * =========================================================================*/
int rxops_bridge_log_f32(float *out, const float *in, int64_t n)
{
    struct rxops_tensor t_in, t_out;
    siso_params_with_cb pwcb;

    ensure_rxops_init();
    fill_tensor_1d(&t_in,  (void *)in,  RXOPS_DTYPE_FLOAT32, n);
    fill_tensor_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);
    init_siso_c(&pwcb);

    rxops_log_init(&t_in, &t_out, &pwcb.p);
    int rc = rxops_log(&t_in, &t_out, &pwcb.p);
    return (rc == 1 /* RXOPS_TRUE */) ? 0 : -1;
}

/* =========================================================================
 * rsqrt  →  RXOPS_C reference backend
 * =========================================================================*/
int rxops_bridge_rsqrt_f32(float *out, const float *in, int64_t n)
{
    struct rxops_tensor t_in, t_out;
    siso_params_with_cb pwcb;

    ensure_rxops_init();
    fill_tensor_1d(&t_in,  (void *)in,  RXOPS_DTYPE_FLOAT32, n);
    fill_tensor_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);
    init_siso_c(&pwcb);

    rxops_rsqrt_init(&t_in, &t_out, &pwcb.p);
    int rc = rxops_rsqrt(&t_in, &t_out, &pwcb.p);
    return (rc == 1 /* RXOPS_TRUE */) ? 0 : -1;
}

/* =========================================================================
 * matmul  C[M,N] = A[M,K] * B[K,N]  →  RXOPS_C reference backend (fp32)
 * =========================================================================*/
int rxops_bridge_matmul_f32(float *C, const float *A, const float *B,
                             int64_t M, int64_t N, int64_t K)
{
    struct rxops_tensor t_a, t_b, t_c;
    struct rxops_matmul_params params;
    struct rxops_callback cb;

    ensure_rxops_init();
    fill_tensor_2d(&t_a, (void *)A, RXOPS_DTYPE_FLOAT32, M, K);
    fill_tensor_2d(&t_b, (void *)B, RXOPS_DTYPE_FLOAT32, K, N);
    fill_tensor_2d(&t_c, (void *)C, RXOPS_DTYPE_FLOAT32, M, N);

    memset(&params, 0, sizeof(params));
    memset(&cb, 0, sizeof(cb));
    params.base.api = RXOPS_C;
    params.base.cb  = &cb;
    params.trans_a  = false;
    params.trans_b  = false;

    rxops_matmul_init(&t_a, &t_b, &t_c, &params);
    int rc = rxops_matmul(&t_a, &t_b, &t_c, &params);
    return (rc == 1 /* RXOPS_TRUE */) ? 0 : -1;
}

/* =========================================================================
 * add  out[i] = in0[i] + in1[i]  →  RXOPS_C reference backend
 * =========================================================================*/
int rxops_bridge_add_f32(float *out, const float *in0, const float *in1,
                          int64_t n)
{
    struct rxops_tensor t_in0, t_in1, t_out;
    struct rxops_diso_params params;
    struct rxops_callback cb;

    ensure_rxops_init();
    fill_tensor_1d(&t_in0, (void *)in0, RXOPS_DTYPE_FLOAT32, n);
    fill_tensor_1d(&t_in1, (void *)in1, RXOPS_DTYPE_FLOAT32, n);
    fill_tensor_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);

    memset(&params, 0, sizeof(params));
    memset(&cb, 0, sizeof(cb));
    params.base.api = RXOPS_C;       /* reference C path */
    params.base.cb  = &cb;

    rxops_add_init(&t_in0, &t_in1, &t_out, &params);
    int rc = rxops_add(&t_in0, &t_in1, &t_out, &params);
    return (rc == 1 /* RXOPS_TRUE */) ? 0 : -1;
}

/* =========================================================================
 * Ada300 SNPU variants  (rxops_bridge_ada300_*)
 *
 * #ifndef __riscv  (x86 host): dispatch via ivshmem to QEMU RISC-V device.
 * #ifdef  __riscv  (RISC-V device / bare-metal): call RXOPS_ADA300 directly.
 *   Matmul uses the fp16 hardware path: fp32→fp16→matmul→fp32.
 * =========================================================================*/

#ifndef __riscv

int rxops_bridge_ada300_exp_f32(float *out, const float *in, int64_t n)
{
    return hetero_dispatch_exp(g_hetero_base, in, out, (int32_t)n);
}

int rxops_bridge_ada300_sqrt_f32(float *out, const float *in, int64_t n)
{
    return hetero_dispatch_sqrt(g_hetero_base, in, out, (int32_t)n);
}

int rxops_bridge_ada300_matmul_f32(float *C, const float *A, const float *B,
                                    int64_t M, int64_t N, int64_t K)
{
    return hetero_dispatch_matmul(g_hetero_base,
                                   A, B, C,
                                   (int32_t)M, (int32_t)N, (int32_t)K);
}

int rxops_bridge_ada300_add_f32(float *out, const float *in0,
                                 const float *in1, int64_t n)
{
    return hetero_dispatch_add(g_hetero_base, in0, in1, out, (int32_t)n);
}

#else /* __riscv — RXOPS_ADA300 hardware backend */

int rxops_bridge_ada300_exp_f32(float *out, const float *in, int64_t n)
{
    struct rxops_tensor t_in, t_out;
    siso_params_with_cb pwcb;

    ensure_rxops_init();
    fill_tensor_1d(&t_in,  (void *)in,  RXOPS_DTYPE_FLOAT32, n);
    fill_tensor_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);
    init_siso_ada300(&pwcb);

    rxops_exp_init(&t_in, &t_out, &pwcb.p);
    int rc = rxops_exp(&t_in, &t_out, &pwcb.p);
    return (rc == 1 /* RXOPS_TRUE */) ? 0 : -1;
}

int rxops_bridge_ada300_sqrt_f32(float *out, const float *in, int64_t n)
{
    struct rxops_tensor t_in, t_out;
    siso_params_with_cb pwcb;

    ensure_rxops_init();
    fill_tensor_1d(&t_in,  (void *)in,  RXOPS_DTYPE_FLOAT32, n);
    fill_tensor_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);
    init_siso_ada300(&pwcb);

    rxops_sqrt_init(&t_in, &t_out, &pwcb.p);
    int rc = rxops_sqrt(&t_in, &t_out, &pwcb.p);
    return (rc == 1 /* RXOPS_TRUE */) ? 0 : -1;
}

int rxops_bridge_ada300_matmul_f32(float *C, const float *A, const float *B,
                                    int64_t M, int64_t N, int64_t K)
{
    ensure_rxops_init();

    int64_t szA = M * K, szB = K * N, szC = M * N;
    __fp16_t *A16 = (__fp16_t *)malloc((size_t)(szA * (int64_t)sizeof(__fp16_t)));
    __fp16_t *B16 = (__fp16_t *)malloc((size_t)(szB * (int64_t)sizeof(__fp16_t)));
    __fp16_t *C16 = (__fp16_t *)malloc((size_t)(szC * (int64_t)sizeof(__fp16_t)));
    if (!A16 || !B16 || !C16) {
        free(A16); free(B16); free(C16);
        return -1;
    }

    f32_to_f16(A16, A, szA);
    f32_to_f16(B16, B, szB);

    struct rxops_tensor t_a, t_b, t_c;
    struct rxops_matmul_params params;
    struct rxops_callback cb;

    memset(&t_a, 0, sizeof(t_a));
    t_a.data = A16; t_a.dtype = RXOPS_DTYPE_FLOAT16;
    t_a.mtype = RXOPS_MEM_TYPE_CPU;
    t_a.dim[0] = (int32_t)M; t_a.dim[1] = (int32_t)K;
    t_a.dim_count = 2; t_a.layout = RXOPS_LAYOUT_NC;

    memset(&t_b, 0, sizeof(t_b));
    t_b.data = B16; t_b.dtype = RXOPS_DTYPE_FLOAT16;
    t_b.mtype = RXOPS_MEM_TYPE_CPU;
    t_b.dim[0] = (int32_t)K; t_b.dim[1] = (int32_t)N;
    t_b.dim_count = 2; t_b.layout = RXOPS_LAYOUT_NC;

    memset(&t_c, 0, sizeof(t_c));
    t_c.data = C16; t_c.dtype = RXOPS_DTYPE_FLOAT16;
    t_c.mtype = RXOPS_MEM_TYPE_CPU;
    t_c.dim[0] = (int32_t)M; t_c.dim[1] = (int32_t)N;
    t_c.dim_count = 2; t_c.layout = RXOPS_LAYOUT_NC;

    memset(&params, 0, sizeof(params));
    memset(&cb, 0, sizeof(cb));
    params.base.api = RXOPS_ADA300;
    params.base.cb  = &cb;
    params.trans_a  = false;
    params.trans_b  = false;

    rxops_matmul_init(&t_a, &t_b, &t_c, &params);
    int rc = rxops_matmul(&t_a, &t_b, &t_c, &params);

    f16_to_f32(C, C16, szC);
    free(A16); free(B16); free(C16);
    return (rc == 1 /* RXOPS_TRUE */) ? 0 : -1;
}

int rxops_bridge_ada300_add_f32(float *out, const float *in0,
                                 const float *in1, int64_t n)
{
    struct rxops_tensor t_in0, t_in1, t_out;
    struct rxops_diso_params params;
    struct rxops_callback cb;

    ensure_rxops_init();
    fill_tensor_1d(&t_in0, (void *)in0, RXOPS_DTYPE_FLOAT32, n);
    fill_tensor_1d(&t_in1, (void *)in1, RXOPS_DTYPE_FLOAT32, n);
    fill_tensor_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);

    memset(&params, 0, sizeof(params));
    memset(&cb, 0, sizeof(cb));
    params.base.api = RXOPS_ADA300;
    params.base.cb  = &cb;

    rxops_add_init(&t_in0, &t_in1, &t_out, &params);
    int rc = rxops_add(&t_in0, &t_in1, &t_out, &params);
    return (rc == 1 /* RXOPS_TRUE */) ? 0 : -1;
}

#endif /* !__riscv */
