/*
 * rx_ops_bridge.h - Flat C ABI wrappers for rx-ops functions.
 *
 * The MLIR compiler pipeline works with memrefs that are lowered to bare
 * (pointer, sizes, strides) tuples by -finalize-memref-to-llvm / C-wrapper
 * generation.  The native rx-ops API takes `struct rxops_tensor *` which
 * requires heap allocation and struct initialisation that cannot be expressed
 * directly in MLIR.
 *
 * This header declares thin wrappers with flat signatures that:
 *   1. Accept the raw buffer pointer and dimension integers that MLIR emits.
 *   2. Build the required `rxops_tensor` and `rxops_*_params` structs on the
 *      stack.
 *   3. Forward to the corresponding `rxops_*` function.
 *
 * Naming convention:
 *   rxops_bridge_<op>_<dtype>(...)
 *
 * The wrapper functions are defined in rx_ops_bridge.c (same directory).
 * Compile with -I<thirdparty/rx-ops/include>.
 *
 * MLIR lowering pass: lower-ada300hl-to-rxops
 *   ada300hl.pwnl {func=exp}  → rxops_bridge_exp_f32(out, in, n)
 *   ada300hl.pwnl {func=sqrt} → rxops_bridge_sqrt_f32(out, in, n)
 *   ada300hl.tensor_mma       → rxops_bridge_matmul_f32(C, A, B, M, N, K)
 */

#ifndef BUDDY_RUNTIME_ADA300_RX_OPS_BRIDGE_H_
#define BUDDY_RUNTIME_ADA300_RX_OPS_BRIDGE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Element-wise unary ops  (single-input / single-output, 1-D flat view)
 *
 *   out[i] = f(in[i])  for i in [0, n)
 *
 * `in` and `out` must point to non-overlapping float32 buffers of length n.
 * -------------------------------------------------------------------------*/

int rxops_bridge_exp_f32(float *out, const float *in, int64_t n);
int rxops_bridge_sqrt_f32(float *out, const float *in, int64_t n);
int rxops_bridge_log_f32(float *out, const float *in, int64_t n);
int rxops_bridge_rsqrt_f32(float *out, const float *in, int64_t n);

/* -------------------------------------------------------------------------
 * Matrix multiply  (row-major, no transpose)
 *
 *   C[M, N] = A[M, K] * B[K, N]
 *
 * All three buffers must be float32, row-major, contiguous.
 * -------------------------------------------------------------------------*/

int rxops_bridge_matmul_f32(float *C, const float *A, const float *B,
                            int64_t M, int64_t N, int64_t K);

/* -------------------------------------------------------------------------
 * Element-wise binary add
 *
 *   out[i] = in0[i] + in1[i]  for i in [0, n)
 * -------------------------------------------------------------------------*/

int rxops_bridge_add_f32(float *out, const float *in0, const float *in1,
                         int64_t n);

#ifdef __cplusplus
}
#endif

#endif /* BUDDY_RUNTIME_ADA300_RX_OPS_BRIDGE_H_ */
