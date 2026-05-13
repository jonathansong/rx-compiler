/*
 * rx_ops_bridge_hetero.c - Heterogeneous rx-ops bridge for the Ada300 demo.
 *
 * Provides the same flat C ABI declared in rx_ops_bridge.h, in two
 * families:
 *
 *   rxops_bridge_<op>_f32         — reference RXOPS_C backend (host x86)
 *   rxops_bridge_ada300_<op>_f32  — Ada300 SNPU via ivshmem (QEMU)
 *
 * Generic symbols (RXOPS_C):
 *   rxops_bridge_matmul_f32   → rxops_matmul  (RXOPS_C)
 *   rxops_bridge_exp_f32      → rxops_exp     (RXOPS_C)
 *   rxops_bridge_sqrt_f32     → rxops_sqrt    (RXOPS_C)
 *   rxops_bridge_add_f32      → rxops_add     (RXOPS_C)
 *   rxops_bridge_log_f32      → rxops_log     (RXOPS_C)
 *   rxops_bridge_rsqrt_f32    → rxops_rsqrt   (RXOPS_C)
 *
 * Ada300 SNPU symbols (ivshmem → QEMU):
 *   rxops_bridge_ada300_matmul_f32 → hetero_dispatch_matmul
 *   rxops_bridge_ada300_sqrt_f32   → hetero_dispatch_sqrt
 *   rxops_bridge_ada300_add_f32    → hetero_dispatch_add
 *   rxops_bridge_ada300_exp_f32    → hetero_dispatch_exp
 *
 * TopToRxOps emits ada300_* symbols for Top ops carrying device="ada300";
 * plain symbols are emitted for ops with no device attribute.
 *
 * The global g_hetero_base (set by hetero_shmem_open in hetero_shmem_host.c)
 * must be valid before any ada300_* bridge function is called.
 *
 * Compile with:
 *   -I<ada300-hetero>/include
 *   -I<rx-ops>/include
 *   -I<rx-ops>/include/interface
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../rx_ops_bridge.h"
#include "../include/hetero_shmem.h"

#include <string.h>
#include <stdint.h>

#include "rx_ops.h"
#include "rx_ops_data_structure.h"
#include "rx_ops_register.h"

/* Shared memory base pointer — defined in hetero_shmem_host.c. */
extern void *g_hetero_base;

/* hetero_dispatch_matmul — defined in hetero_shmem_host.c. */
extern int hetero_dispatch_matmul(void *base,
                                   const float *A, const float *B,
                                   float *C,
                                   int32_t M, int32_t N, int32_t K);

/* hetero_dispatch_sqrt — defined in hetero_shmem_host.c. */
extern int hetero_dispatch_sqrt(void *base,
                                 const float *in, float *out, int32_t n);

/* hetero_dispatch_exp — defined in hetero_shmem_host.c. */
extern int hetero_dispatch_exp(void *base,
                                const float *in, float *out, int32_t n);

/* hetero_dispatch_add — defined in hetero_shmem_host.c. */
extern int hetero_dispatch_add(void *base,
                                const float *in0, const float *in1,
                                float *out, int32_t n);

/* -------------------------------------------------------------------------
 * One-time RXOPS_C initialisation (for the local ops: exp, sqrt, add, etc.)
 * -------------------------------------------------------------------------*/
static int g_rxops_c_init = 0;

static void ensure_rxops_c(void)
{
    if (!g_rxops_c_init) {
        extern void rxnn_c_init(void) __attribute__((weak));
        if (rxnn_c_init)
            rxnn_c_init();
        g_rxops_c_init = 1;
    }
}

/* -------------------------------------------------------------------------
 * Internal helpers
 * -------------------------------------------------------------------------*/
static void fill_1d(struct rxops_tensor *t, void *data,
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

static void fill_2d(struct rxops_tensor *t, void *data,
                    enum rxops_dtype_enum dtype, int64_t rows, int64_t cols)
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

typedef struct {
    struct rxops_siso_params p;
    struct rxops_callback    cb;
} siso_with_cb;

static void init_siso_c(siso_with_cb *s)
{
    memset(s, 0, sizeof(*s));
    s->p.base.api = RXOPS_C;
    s->p.base.cb  = &s->cb;
}

/* =========================================================================
 * MatMul — reference C backend (RXOPS_C)
 *
 * C[M×N] = A[M×K] * B[K×N]
 * =========================================================================*/
int rxops_bridge_matmul_f32(float *C, const float *A, const float *B,
                             int64_t M, int64_t N, int64_t K)
{
    struct rxops_tensor t_A, t_B, t_C;
    struct rxops_matmul_params params;
    struct rxops_callback cb;

    ensure_rxops_c();
    fill_2d(&t_A, (void *)A, RXOPS_DTYPE_FLOAT32, M, K);
    fill_2d(&t_B, (void *)B, RXOPS_DTYPE_FLOAT32, K, N);
    fill_2d(&t_C, (void *)C, RXOPS_DTYPE_FLOAT32, M, N);

    memset(&params, 0, sizeof(params));
    memset(&cb,     0, sizeof(cb));
    params.base.api = RXOPS_C;
    params.base.cb  = &cb;

    rxops_matmul_init(&t_A, &t_B, &t_C, &params);
    return rxops_matmul(&t_A, &t_B, &t_C, &params);
}

/* =========================================================================
 * Exp — runs locally on the host via RXOPS_C
 * =========================================================================*/
int rxops_bridge_exp_f32(float *out, const float *in, int64_t n)
{
    struct rxops_tensor t_in, t_out;
    siso_with_cb s;

    ensure_rxops_c();
    fill_1d(&t_in,  (void *)in,  RXOPS_DTYPE_FLOAT32, n);
    fill_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);
    init_siso_c(&s);
    rxops_exp_init(&t_in, &t_out, &s.p);
    return rxops_exp(&t_in, &t_out, &s.p);
}

/* =========================================================================
 * Sqrt — reference C backend (RXOPS_C)
 * =========================================================================*/
int rxops_bridge_sqrt_f32(float *out, const float *in, int64_t n)
{
    struct rxops_tensor t_in, t_out;
    siso_with_cb s;

    ensure_rxops_c();
    fill_1d(&t_in,  (void *)in,  RXOPS_DTYPE_FLOAT32, n);
    fill_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);
    init_siso_c(&s);
    rxops_sqrt_init(&t_in, &t_out, &s.p);
    return rxops_sqrt(&t_in, &t_out, &s.p);
}

/* =========================================================================
 * Log — runs locally on the host via RXOPS_C
 * =========================================================================*/
int rxops_bridge_log_f32(float *out, const float *in, int64_t n)
{
    struct rxops_tensor t_in, t_out;
    siso_with_cb s;

    ensure_rxops_c();
    fill_1d(&t_in,  (void *)in,  RXOPS_DTYPE_FLOAT32, n);
    fill_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);
    init_siso_c(&s);
    rxops_log_init(&t_in, &t_out, &s.p);
    return rxops_log(&t_in, &t_out, &s.p);
}

/* =========================================================================
 * Rsqrt — runs locally on the host via RXOPS_C
 * =========================================================================*/
int rxops_bridge_rsqrt_f32(float *out, const float *in, int64_t n)
{
    struct rxops_tensor t_in, t_out;
    siso_with_cb s;

    ensure_rxops_c();
    fill_1d(&t_in,  (void *)in,  RXOPS_DTYPE_FLOAT32, n);
    fill_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);
    init_siso_c(&s);
    rxops_rsqrt_init(&t_in, &t_out, &s.p);
    return rxops_rsqrt(&t_in, &t_out, &s.p);
}

/* =========================================================================
 * Add — runs locally on the host via RXOPS_C
 * =========================================================================*/
int rxops_bridge_add_f32(float *out, const float *in0, const float *in1,
                          int64_t n)
{
    struct rxops_tensor t_in0, t_in1, t_out;
    struct rxops_diso_params params;
    struct rxops_callback cb;

    ensure_rxops_c();
    fill_1d(&t_in0, (void *)in0, RXOPS_DTYPE_FLOAT32, n);
    fill_1d(&t_in1, (void *)in1, RXOPS_DTYPE_FLOAT32, n);
    fill_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);

    memset(&params, 0, sizeof(params));
    memset(&cb,     0, sizeof(cb));
    params.base.api = RXOPS_C;
    params.base.cb  = &cb;

    rxops_add_init(&t_in0, &t_in1, &t_out, &params);
    return rxops_add(&t_in0, &t_in1, &t_out, &params);
}

/* =========================================================================
 * Ada300 SNPU backend variants
 *
 * These symbols are emitted by TopToRxOps when a Top op carries the
 * attribute  device = "ada300".  They are the compiler-dispatch equivalents
 * of the runtime-dispatch logic that was previously inside the generic
 * rxops_bridge_matmul_f32 / rxops_bridge_sqrt_f32 implementations above.
 * =========================================================================*/

int rxops_bridge_ada300_matmul_f32(float *C, const float *A, const float *B,
                                    int64_t M, int64_t N, int64_t K)
{
    return hetero_dispatch_matmul(g_hetero_base,
                                   A, B, C,
                                   (int32_t)M, (int32_t)N, (int32_t)K);
}

int rxops_bridge_ada300_sqrt_f32(float *out, const float *in, int64_t n)
{
    return hetero_dispatch_sqrt(g_hetero_base, in, out, (int32_t)n);
}

int rxops_bridge_ada300_exp_f32(float *out, const float *in, int64_t n)
{
    return hetero_dispatch_exp(g_hetero_base, in, out, (int32_t)n);
}

int rxops_bridge_ada300_add_f32(float *out, const float *in0,
                                 const float *in1, int64_t n)
{
    return hetero_dispatch_add(g_hetero_base, in0, in1, out, (int32_t)n);
}
