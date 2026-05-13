/*
 * ada300-hetero-main.c - Ada300 SNPU persistent device firmware
 *                        for the heterogeneous demo.
 *
 * Boot assumptions (identical to ada300-baremetal-main.c):
 *   startup.S   : sets sp/gp, clears .bss, enables FP/V, calls bsp_main()
 *   bsp_main.c  : uart_init() + mr_heap_init(), then calls main()
 *   LPDDR heap  : mr_alloc(n, MR_REGION_LPDDR) gives 8 GiB-scale allocs
 *   mini_printf : printf() mapped to BSP UART (integer format only)
 *
 * Difference from ada300-baremetal-main.c:
 *   - NO call to _mlir_ciface_subgraph0 — this firmware does NOT run the
 *     full MLIR-compiled graph.
 *   - NO qemu_exit / csky_exit — firmware stays alive forever, serving
 *     MatMul commands from the host one at a time.
 *   - The main loop polls the ivshmem shared memory for commands,
 *     executes them using the Ada300 NPU backend (RXOPS_ADA300), and
 *     signals completion back to the host.
 *
 * MLIR runtime symbols provided here:
 *   malloc / free / calloc  : backed by LPDDR mr_heap (required by
 *                             rx_ops_bridge_ada300.c for fp16 scratch).
 *
 * Build:
 *   See CMakeLists.txt target ada300-hetero-device.
 *
 * Run:
 *   qemu-system-riscv64 -machine ada300s_evk -cpu ada300s          \
 *       -nographic -serial mon:stdio                               \
 *       -object memory-backend-file,id=shmem0,share=on,            \
 *              mem-path=/dev/shm/ada300_bar,size=64M               \
 *       -kernel build/ada300-hetero-device.elf
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "mini_printf.h"
#include "mr_heap.h"

/* Device-side ivshmem transport (hetero_shmem_device.c). */
#include "../include/hetero_shmem.h"
extern void hetero_device_poll_cmd(void);
extern void hetero_device_get_matmul(int32_t *M, int32_t *N, int32_t *K,
                                      float **A_out, float **B_out);
extern void hetero_device_get_sqrt(int32_t *n_out, const float **in_out);
extern void hetero_device_get_exp(int32_t *n_out, const float **in_out);
extern void hetero_device_get_add(int32_t *n_out,
                                   const float **in0_out,
                                   const float **in1_out);
extern void hetero_device_signal_done(const float *C, int32_t M, int32_t N,
                                       int rc);

/* Ada300 rx-ops bridge (rx_ops_bridge_ada300.c — uses RXOPS_ADA300). */
#include "../rx_ops_bridge.h"

/* =========================================================================
 * MLIR/rx-ops runtime memory allocators.
 *
 * rx_ops_bridge_ada300.c calls malloc/free for fp16 scratch buffers when
 * executing the matmul with the Ada300 hardware backend.  Back them with
 * the BSP LPDDR heap so large allocations work correctly on baremetal.
 * =========================================================================*/
void *malloc(size_t n)                { return mr_alloc(n, MR_REGION_LPDDR); }
void  free(void *p)                   { mr_free(p); }
void *calloc(size_t nmemb, size_t sz) { return mr_calloc(nmemb, sz, MR_REGION_LPDDR); }

/*
 * rxnn_mem_alloc / rxnn_mem_free — Ada300 backend heap override.
 * Declared weak in librx_ops; provide strong definitions here to route
 * all Ada300 backend scratch allocations through the LPDDR heap.
 */
void *rxnn_mem_alloc(int64_t size) { return mr_alloc((size_t)size, MR_REGION_LPDDR); }
void  rxnn_mem_free(void *ptr)     { mr_free(ptr); }

/* =========================================================================
 * RISC-V cycle / time CSR helper for simple profiling.
 * =========================================================================*/
static inline uint64_t rdtime(void)
{
    uint64_t t;
    __asm__ volatile ("rdtime %0" : "=r"(t));
    return t;
}

/* Print a float32 via mini_printf (no %f support). */
static void print_f32(float v)
{
    int neg = (v < 0.0f);
    if (neg) v = -v;
    int64_t whole = (int64_t)v;
    int64_t frac  = (int64_t)((v - (float)whole) * 10000.0f + 0.5f);
    if (frac >= 10000) { whole++; frac = 0; }
    if (neg) printf("-");
    printf("%lld.%04lld", (long long)whole, (long long)frac);
}

/* =========================================================================
 * main — persistent service loop
 * =========================================================================*/
int main(void)
{
    printf("[hetero-dev] Ada300 heterogeneous device firmware booted.\n");
    printf("[hetero-dev] Polling ivshmem at 0x%llx for commands...\n",
           (unsigned long long)ADA300_IVSHMEM_BASE);

    uint32_t cmd_count = 0;

    for (;;) {
        /* ----- Wait for a command from the host -------------------------- */
        hetero_device_poll_cmd();

        /* ----- Dispatch based on op_type --------------------------------- */
        volatile struct hetero_ctrl *ctrl = (volatile struct hetero_ctrl *)
            ((uintptr_t)ADA300_IVSHMEM_BASE + HETERO_CTRL_OFFSET);
        uint32_t op = ctrl->op_type;

        cmd_count++;

        if (op == HETERO_OP_MATMUL) {
            /* ----- Parse the matmul parameters --------------------------- */
            int32_t M, N, K;
            float *A = NULL, *B = NULL;
            hetero_device_get_matmul(&M, &N, &K, &A, &B);

            printf("[hetero-dev] cmd #%u: matmul [%d×%d] = [%d×%d] * [%d×%d]\n",
                   (unsigned)cmd_count, (int)M, (int)N,
                   (int)M, (int)K, (int)K, (int)N);

            float *C = (float *)mr_alloc((size_t)M * (size_t)N * sizeof(float),
                                         MR_REGION_LPDDR);
            if (!C) {
                printf("[hetero-dev] ERROR: alloc failed for C[%d×%d]\n",
                       (int)M, (int)N);
                hetero_device_signal_done(NULL, M, N, -1);
                continue;
            }

            uint64_t t0 = rdtime();
            int rc = rxops_bridge_ada300_matmul_f32(C, A, B, (int64_t)M, (int64_t)N, (int64_t)K);
            uint64_t t1 = rdtime();

            printf("[hetero-dev]   MatMul done in %llu ticks, rc=%d\n",
                   (unsigned long long)(t1 - t0), rc);

            if (rc == 0 && M > 0 && N > 0) {
                printf("[hetero-dev]   C[0][0] = ");
                print_f32(C[0]);
                printf("\n");
            }

            hetero_device_signal_done(rc == 0 ? C : NULL, M, N, rc);
            mr_free(C);

        } else if (op == HETERO_OP_SQRT) {
            /* ----- Parse the sqrt parameters ----------------------------- */
            int32_t n;
            const float *in = NULL;
            hetero_device_get_sqrt(&n, &in);

            printf("[hetero-dev] cmd #%u: sqrt [%d elements]\n",
                   (unsigned)cmd_count, (int)n);

            float *out_buf = (float *)mr_alloc((size_t)n * sizeof(float),
                                               MR_REGION_LPDDR);
            if (!out_buf) {
                printf("[hetero-dev] ERROR: alloc failed for sqrt out[%d]\n",
                       (int)n);
                hetero_device_signal_done(NULL, n, 1, -1);
                continue;
            }

            uint64_t t0 = rdtime();
            int rc = rxops_bridge_ada300_sqrt_f32(out_buf, in, (int64_t)n);
            uint64_t t1 = rdtime();

            printf("[hetero-dev]   Sqrt done in %llu ticks, rc=%d\n",
                   (unsigned long long)(t1 - t0), rc);

            if (rc == 0 && n > 0) {
                printf("[hetero-dev]   out[0] = ");
                print_f32(out_buf[0]);
                printf("\n");
            }

            /* M=n, N=1 reuses signal_done to write n floats. */
            hetero_device_signal_done(rc == 0 ? out_buf : NULL, n, 1, rc);
            mr_free(out_buf);

        } else if (op == HETERO_OP_EXP) {
            /* ----- Parse the exp parameters ------------------------------ */
            int32_t n;
            const float *in = NULL;
            hetero_device_get_exp(&n, &in);

            printf("[hetero-dev] cmd #%u: exp [%d elements]\n",
                   (unsigned)cmd_count, (int)n);

            float *out_buf = (float *)mr_alloc((size_t)n * sizeof(float),
                                               MR_REGION_LPDDR);
            if (!out_buf) {
                printf("[hetero-dev] ERROR: alloc failed for exp out[%d]\n",
                       (int)n);
                hetero_device_signal_done(NULL, n, 1, -1);
                continue;
            }

            uint64_t t0 = rdtime();
            int rc = rxops_bridge_ada300_exp_f32(out_buf, in, (int64_t)n);
            uint64_t t1 = rdtime();

            printf("[hetero-dev]   Exp done in %llu ticks, rc=%d\n",
                   (unsigned long long)(t1 - t0), rc);

            if (rc == 0 && n > 0) {
                printf("[hetero-dev]   out[0] = ");
                print_f32(out_buf[0]);
                printf("\n");
            }

            hetero_device_signal_done(rc == 0 ? out_buf : NULL, n, 1, rc);
            mr_free(out_buf);

        } else if (op == HETERO_OP_ADD) {
            /* ----- Parse the add parameters ------------------------------ */
            int32_t n;
            const float *in0 = NULL, *in1 = NULL;
            hetero_device_get_add(&n, &in0, &in1);

            printf("[hetero-dev] cmd #%u: add [%d elements]\n",
                   (unsigned)cmd_count, (int)n);

            float *out_buf = (float *)mr_alloc((size_t)n * sizeof(float),
                                               MR_REGION_LPDDR);
            if (!out_buf) {
                printf("[hetero-dev] ERROR: alloc failed for add out[%d]\n",
                       (int)n);
                hetero_device_signal_done(NULL, n, 1, -1);
                continue;
            }

            uint64_t t0 = rdtime();
            int rc = rxops_bridge_ada300_add_f32(out_buf, in0, in1, (int64_t)n);
            uint64_t t1 = rdtime();

            printf("[hetero-dev]   Add done in %llu ticks, rc=%d\n",
                   (unsigned long long)(t1 - t0), rc);

            if (rc == 0 && n > 0) {
                printf("[hetero-dev]   out[0] = ");
                print_f32(out_buf[0]);
                printf("\n");
            }

            /* M=n, N=1 reuses signal_done to write n floats. */
            hetero_device_signal_done(rc == 0 ? out_buf : NULL, n, 1, rc);
            mr_free(out_buf);

        } else {
            printf("[hetero-dev] cmd #%u: unknown op_type=%u, skipping\n",
                   (unsigned)cmd_count, (unsigned)op);
            hetero_device_signal_done(NULL, 0, 0, -1);
        }
    }

    /* unreachable */
    return 0;
}
