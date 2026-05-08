/*
 * rx_ops_bridge_ada300.c - Ada300 NPU backend wrappers for the rx-ops bridge.
 *
 * Baremetal variant of rx_ops_bridge.c targeting the Ada300 RISC-V NPU.
 *
 * Backend selection per operator:
 *   op.exp    → RXOPS_ADA300  (ada300_exp_f32.S: vectorised Ada300 SNPU)
 *   op.sqrt   → RXOPS_ADA300  (ada300_sqrt_f32.S)
 *   op.add    → RXOPS_ADA300  (ada300_add_f32.S)
 *   op.matmul → RXOPS_ADA300  via fp16 path (ada300_matmul_f16.S):
 *                 1. convert fp32 inputs A, B → fp16 (__fp16_t)
 *                 2. run Ada300 fp16 matmul kernel
 *                 3. convert fp16 result C → fp32
 *               The Ada300 backend has NO fp32 matmul kernel; fp16 is the
 *               only available hardware path.  Conversion uses the RISC-V
 *               Zfh instructions (fcvt.h.s / fcvt.s.h) available on
 *               -march=rv64gcv_zfh_... (Ada300 LLVM toolchain).
 *               Scratch fp16 buffers are heap-allocated via malloc().
 *
 * Both rxnn_c_init() and rxnn_ada300_init() are called once on first use so
 * that the RXOPS_C table (fallback) and RXOPS_ADA300 table are both registered.
 *
 * Compile with:
 *   -DRXNN_BUILD_RTOS           (suppress dirent.h / clock_gettime in log.c)
 *   -I<rx-ops>/include
 *   -I<rx-ops>/include/interface
 *   --target=riscv64-unknown-elf -march=rv64gcv_zfh_... (Ada300 LLVM toolchain)
 */

#include "rx_ops_bridge.h"

#include <string.h>
#include <stdlib.h>   /* malloc / free — backed by mr_alloc on baremetal */

#include "rx_ops.h"
#include "rx_ops_data_structure.h"
#include "rx_ops_data_type.h"   /* __fp16_t = uint16_t */
#include "rx_ops_register.h"

/* -------------------------------------------------------------------------
 * One-time backend initialisation:
 *   - rxnn_c_init()      : registers RXOPS_C callback table (for matmul)
 *   - rxnn_ada300_init() : registers RXOPS_ADA300 callback table (exp/sqrt/add)
 * -------------------------------------------------------------------------*/
/* rxnn_c_init() lives in backend/reference/setup.c which is NOT compiled
 * for baremetal Ada300 builds (RXNN_BUILD_RTOS=1).  Calling an undefined
 * symbol would cause a link error, so guard it out on RTOS/baremetal. */
#ifndef RXNN_BUILD_RTOS
extern void rxnn_c_init(void);
#endif
extern void rxnn_ada300_init(void);

static int g_rxops_initialised = 0;

static void ensure_rxops_init(void)
{
    if (!g_rxops_initialised) {
#ifndef RXNN_BUILD_RTOS
        rxnn_c_init();       /* register RXOPS_C table (reference C backend) */
#endif
        rxnn_ada300_init();  /* register RXOPS_ADA300 table (Ada300 NPU)      */
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

static void init_siso_ada300(siso_params_with_cb *pwcb)
{
    memset(pwcb, 0, sizeof(*pwcb));
    pwcb->p.base.api = RXOPS_ADA300;   /* Ada300 NPU path */
    pwcb->p.base.cb  = &pwcb->cb;
}

static void init_siso_c(siso_params_with_cb *pwcb)
{
    memset(pwcb, 0, sizeof(*pwcb));
    pwcb->p.base.api = RXOPS_C;        /* reference C fallback */
    pwcb->p.base.cb  = &pwcb->cb;
}

/* =========================================================================
 * fp32 ↔ fp16 conversion helpers
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

/* =========================================================================
 * exp  →  Ada300 NPU (ada300_exp_f32.S)
 * =========================================================================*/
int rxops_bridge_exp_f32(float *out, const float *in, int64_t n)
{
    struct rxops_tensor t_in, t_out;
    siso_params_with_cb pwcb;

    ensure_rxops_init();
    fill_tensor_1d(&t_in,  (void *)in,  RXOPS_DTYPE_FLOAT32, n);
    fill_tensor_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);
    init_siso_ada300(&pwcb);

    rxops_exp_init(&t_in, &t_out, &pwcb.p);
    return rxops_exp(&t_in, &t_out, &pwcb.p);
}

/* =========================================================================
 * sqrt  →  Ada300 NPU (ada300_sqrt_f32.S)
 * =========================================================================*/
int rxops_bridge_sqrt_f32(float *out, const float *in, int64_t n)
{
    struct rxops_tensor t_in, t_out;
    siso_params_with_cb pwcb;

    ensure_rxops_init();
    fill_tensor_1d(&t_in,  (void *)in,  RXOPS_DTYPE_FLOAT32, n);
    fill_tensor_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);
    init_siso_ada300(&pwcb);

    rxops_sqrt_init(&t_in, &t_out, &pwcb.p);
    return rxops_sqrt(&t_in, &t_out, &pwcb.p);
}

/* =========================================================================
 * log  →  Ada300 NPU (ada300_log_f32.S)
 * =========================================================================*/
int rxops_bridge_log_f32(float *out, const float *in, int64_t n)
{
    struct rxops_tensor t_in, t_out;
    siso_params_with_cb pwcb;

    ensure_rxops_init();
    fill_tensor_1d(&t_in,  (void *)in,  RXOPS_DTYPE_FLOAT32, n);
    fill_tensor_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);
    init_siso_ada300(&pwcb);

    rxops_log_init(&t_in, &t_out, &pwcb.p);
    return rxops_log(&t_in, &t_out, &pwcb.p);
}

/* =========================================================================
 * rsqrt  →  Ada300 NPU (ada300_rsqrt_f32.S)
 * =========================================================================*/
int rxops_bridge_rsqrt_f32(float *out, const float *in, int64_t n)
{
    struct rxops_tensor t_in, t_out;
    siso_params_with_cb pwcb;

    ensure_rxops_init();
    fill_tensor_1d(&t_in,  (void *)in,  RXOPS_DTYPE_FLOAT32, n);
    fill_tensor_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);
    init_siso_ada300(&pwcb);

    rxops_rsqrt_init(&t_in, &t_out, &pwcb.p);
    return rxops_rsqrt(&t_in, &t_out, &pwcb.p);
}

/* =========================================================================
 * matmul  C[M,N] = A[M,K] * B[K,N]
 *
 * Ada300 backend has NO fp32 matmul; only fp16 is available via
 * ada300_matmul_f16.S / rxnn_npu_matmul_fp16.
 *
 * Strategy:
 *   1. Allocate fp16 scratch buffers: A16[M×K], B16[K×N], C16[M×N]
 *   2. Convert A (fp32) → A16 (fp16) using fcvt.h.s
 *   3. Convert B (fp32) → B16 (fp16) using fcvt.h.s
 *   4. Run Ada300 fp16 matmul (RXOPS_ADA300, FLOAT16)
 *   5. Convert C16 (fp16) → C (fp32) using fcvt.s.h
 *   6. Free scratch buffers
 *
 * For the Ada300 example model the largest scratch is B16[64×128] = 16 KB,
 * which is well within the LPDDR heap.  malloc() is backed by mr_alloc()
 * on baremetal (see ada300-baremetal-main.c).
 * =========================================================================*/
int rxops_bridge_matmul_f32(float *C, const float *A, const float *B,
                             int64_t M, int64_t N, int64_t K)
{
    ensure_rxops_init();

    /* Allocate fp16 scratch. */
    int64_t szA = M * K, szB = K * N, szC = M * N;
    __fp16_t *A16 = (__fp16_t *)malloc((size_t)(szA * (int64_t)sizeof(__fp16_t)));
    __fp16_t *B16 = (__fp16_t *)malloc((size_t)(szB * (int64_t)sizeof(__fp16_t)));
    __fp16_t *C16 = (__fp16_t *)malloc((size_t)(szC * (int64_t)sizeof(__fp16_t)));
    if (!A16 || !B16 || !C16) {
        free(A16); free(B16); free(C16);
        return -1;   /* allocation failure */
    }

    /* fp32 → fp16 */
    f32_to_f16(A16, A, szA);
    f32_to_f16(B16, B, szB);

    /* Build fp16 tensor descriptors. */
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
    params.base.api = RXOPS_ADA300;   /* Ada300 fp16 matmul kernel */
    params.base.cb  = &cb;
    params.trans_a  = false;
    params.trans_b  = false;

    rxops_matmul_init(&t_a, &t_b, &t_c, &params);
    int rc = rxops_matmul(&t_a, &t_b, &t_c, &params);

    /* fp16 → fp32 */
    f16_to_f32(C, C16, szC);

    free(A16); free(B16); free(C16);
    return rc;
}

/* =========================================================================
 * add  out[i] = in0[i] + in1[i]  →  Ada300 NPU (ada300_add_f32.S)
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
    params.base.api = RXOPS_ADA300;  /* Ada300 NPU path */
    params.base.cb  = &cb;

    rxops_add_init(&t_in0, &t_in1, &t_out, &params);
    return rxops_add(&t_in0, &t_in1, &t_out, &params);
}
