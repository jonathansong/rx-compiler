/*
 * rx_ops_bridge.c - Flat C ABI wrappers around the rx-ops API.
 *
 * See rx_ops_bridge.h for the API contract.
 *
 * Each wrapper:
 *   1. Builds rxops_tensor structs on the stack (no heap allocation).
 *   2. Builds the corresponding rxops_*_params struct on the stack.
 *   3. Calls the init function once (no-op if no pre-packing is needed).
 *   4. Calls the execute function and returns its status.
 *
 * The `api` field in params->base is set to RXOPS_C (the reference C backend)
 * so that the bridge can be tested on any host (x86, etc.) without Ada300
 * hardware.  To target Ada300 hardware, change RXOPS_C to RXOPS_ADA300 and
 * call rxnn_ada300_init() in the application before invoking inference.
 *
 * Compile with:
 *   -I<buddy-mlir>/thirdparty/rx-ops/include
 */

#include "rx_ops_bridge.h"

#include <string.h>

#include "rx_ops.h"
#include "rx_ops_data_structure.h"
#include "rx_ops_register.h"

/* -------------------------------------------------------------------------
 * rxnn_c_init is declared weak inside librx_ops so that the application can
 * override it.  We provide a strong definition here that registers the
 * reference (CPU) callback table, which is sufficient for host (x86) testing.
 * When running on real Ada300 hardware, replace RXOPS_C with RXOPS_ADA300 and
 * call rxnn_ada300_init() instead.
 * -------------------------------------------------------------------------*/
extern void rxnn_register_op_callback(int api, void *cb);

/* This symbol is expected by rx_ops internals; we initialise on first use. */
static int g_rxops_initialised = 0;

static void ensure_rxops_c_init(void) {
    if (!g_rxops_initialised) {
        /* Call the weak rxnn_c_init which registers the reference backend. */
        extern void __attribute__((weak)) rxnn_c_init(void);
        if (rxnn_c_init) rxnn_c_init();
        g_rxops_initialised = 1;
    }
}

/* -------------------------------------------------------------------------
 * Internal helper: fill a 1-D rxops_tensor view over a flat buffer.
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

/* -------------------------------------------------------------------------
 * Internal helper: fill a 2-D rxops_tensor view over a contiguous buffer.
 * -------------------------------------------------------------------------*/
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
 * Internal helper: initialise a siso_params for the ADA300 backend.
 * A local rxops_callback buffer is embedded alongside the params so that
 * rxnn_op_callback_map can memcpy into base->cb without a null deref.
 * -------------------------------------------------------------------------*/
typedef struct {
    struct rxops_siso_params p;
    struct rxops_callback    cb;
} siso_params_with_cb;

static void init_siso_params(siso_params_with_cb *pwcb)
{
    memset(pwcb, 0, sizeof(*pwcb));
    pwcb->p.base.api = RXOPS_C;
    pwcb->p.base.cb  = &pwcb->cb;   /* point into the embedded buffer */
}

/* =========================================================================
 * exp
 * =========================================================================*/
int rxops_bridge_exp_f32(float *out, const float *in, int64_t n)
{
    struct rxops_tensor t_in, t_out;
    siso_params_with_cb pwcb;

    ensure_rxops_c_init();
    fill_tensor_1d(&t_in,  (void *)in,  RXOPS_DTYPE_FLOAT32, n);
    fill_tensor_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);
    init_siso_params(&pwcb);

    rxops_exp_init(&t_in, &t_out, &pwcb.p);
    return rxops_exp(&t_in, &t_out, &pwcb.p);
}

/* =========================================================================
 * sqrt
 * =========================================================================*/
int rxops_bridge_sqrt_f32(float *out, const float *in, int64_t n)
{
    struct rxops_tensor t_in, t_out;
    siso_params_with_cb pwcb;

    ensure_rxops_c_init();
    fill_tensor_1d(&t_in,  (void *)in,  RXOPS_DTYPE_FLOAT32, n);
    fill_tensor_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);
    init_siso_params(&pwcb);

    rxops_sqrt_init(&t_in, &t_out, &pwcb.p);
    return rxops_sqrt(&t_in, &t_out, &pwcb.p);
}

/* =========================================================================
 * log
 * =========================================================================*/
int rxops_bridge_log_f32(float *out, const float *in, int64_t n)
{
    struct rxops_tensor t_in, t_out;
    siso_params_with_cb pwcb;

    ensure_rxops_c_init();
    fill_tensor_1d(&t_in,  (void *)in,  RXOPS_DTYPE_FLOAT32, n);
    fill_tensor_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);
    init_siso_params(&pwcb);

    rxops_log_init(&t_in, &t_out, &pwcb.p);
    return rxops_log(&t_in, &t_out, &pwcb.p);
}

/* =========================================================================
 * rsqrt
 * =========================================================================*/
int rxops_bridge_rsqrt_f32(float *out, const float *in, int64_t n)
{
    struct rxops_tensor t_in, t_out;
    siso_params_with_cb pwcb;

    ensure_rxops_c_init();
    fill_tensor_1d(&t_in,  (void *)in,  RXOPS_DTYPE_FLOAT32, n);
    fill_tensor_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);
    init_siso_params(&pwcb);

    rxops_rsqrt_init(&t_in, &t_out, &pwcb.p);
    return rxops_rsqrt(&t_in, &t_out, &pwcb.p);
}

/* =========================================================================
 * matmul  C[M,N] = A[M,K] * B[K,N]
 * =========================================================================*/
int rxops_bridge_matmul_f32(float *C, const float *A, const float *B,
                             int64_t M, int64_t N, int64_t K)
{
    struct rxops_tensor t_a, t_b, t_c;
    struct rxops_matmul_params params;
    struct rxops_callback cb;

    ensure_rxops_c_init();
    fill_tensor_2d(&t_a, (void *)A, RXOPS_DTYPE_FLOAT32, M, K);
    fill_tensor_2d(&t_b, (void *)B, RXOPS_DTYPE_FLOAT32, K, N);
    fill_tensor_2d(&t_c, (void *)C, RXOPS_DTYPE_FLOAT32, M, N);

    memset(&params, 0, sizeof(params));
    memset(&cb, 0, sizeof(cb));
    params.base.api  = RXOPS_C;
    params.base.cb   = &cb;          /* writable buffer for callback ptrs */
    params.trans_a   = false;
    params.trans_b   = false;

    rxops_matmul_init(&t_a, &t_b, &t_c, &params);
    return rxops_matmul(&t_a, &t_b, &t_c, &params);
}

/* =========================================================================
 * add  out[i] = in0[i] + in1[i]
 * =========================================================================*/
int rxops_bridge_add_f32(float *out, const float *in0, const float *in1,
                          int64_t n)
{
    struct rxops_tensor t_in0, t_in1, t_out;
    struct rxops_diso_params params;
    struct rxops_callback cb;

    ensure_rxops_c_init();
    fill_tensor_1d(&t_in0, (void *)in0, RXOPS_DTYPE_FLOAT32, n);
    fill_tensor_1d(&t_in1, (void *)in1, RXOPS_DTYPE_FLOAT32, n);
    fill_tensor_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);

    memset(&params, 0, sizeof(params));
    memset(&cb, 0, sizeof(cb));
    params.base.api = RXOPS_C;
    params.base.cb  = &cb;

    rxops_add_init(&t_in0, &t_in1, &t_out, &params);
    return rxops_add(&t_in0, &t_in1, &t_out, &params);
}
