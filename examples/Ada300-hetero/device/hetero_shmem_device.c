/*
 * hetero_shmem_device.c - Device-side shared memory access for the Ada300
 * heterogeneous demo (RISC-V baremetal firmware).
 *
 * The device firmware accesses the ivshmem region via a raw MMIO pointer
 * cast to the physical address ADA300_IVSHMEM_BASE.  QEMU maps the 64 MiB
 * POSIX shared memory file into the guest address space at this base when the
 * machine is launched with:
 *
 *   -object memory-backend-file,id=shmem0,share=on,
 *          mem-path=/dev/shm/ada300_bar,size=64M
 *
 * The machine init code in ada300s_evk.c then calls
 * memory_region_add_subregion(..., ADA300_IVSHMEM_BASE, mr) to wire it in.
 *
 * Synchronisation:
 *   RISC-V fence instructions are used to order memory accesses:
 *     fence r,r   — ensures all reads of the cmd/blob are ordered after
 *                   the read of ctrl->cmd that detected CMD_RUN.
 *     fence w,w   — ensures all writes to the result buffer are visible
 *                   before ctrl->status = DONE is written.
 *
 * Public API:
 *
 *   void hetero_device_poll_cmd(void)
 *       Spin-wait (with wfi between iterations) until ctrl->cmd == RUN.
 *       Sets ctrl->status = RUNNING before returning.
 *
 *   void hetero_device_get_matmul(int32_t *M, int32_t *N, int32_t *K,
 *                                  float **A_out, float **B_out)
 *       Parse the matmul header from the command buffer.  A_out and B_out
 *       are set to point into the (non-volatile copy of the) command buffer.
 *       The caller must treat these pointers as read-only inputs to rx-ops.
 *
 *   void hetero_device_signal_done(const float *C, int32_t M, int32_t N,
 *                                   int rc)
 *       Write the result matrix C into the result buffer, set ctrl->rc and
 *       ctrl->result_size, then set ctrl->status = DONE and ctrl->cmd = IDLE.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/hetero_shmem.h"

#include <stdint.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Accessors for the three shared memory regions.
 * ADA300_IVSHMEM_BASE is defined in hetero_shmem.h.
 * -------------------------------------------------------------------------*/
static volatile struct hetero_ctrl *ctrl_regs(void)
{
    return (volatile struct hetero_ctrl *)
           ((uintptr_t)ADA300_IVSHMEM_BASE + HETERO_CTRL_OFFSET);
}

static volatile char *cmd_buf_base(void)
{
    return (volatile char *)
           ((uintptr_t)ADA300_IVSHMEM_BASE + HETERO_CMD_BUF_OFFSET);
}

static volatile char *result_buf_base(void)
{
    return (volatile char *)
           ((uintptr_t)ADA300_IVSHMEM_BASE + HETERO_RESULT_OFFSET);
}

/* -------------------------------------------------------------------------
 * hetero_device_poll_cmd
 *
 * Spin until the host writes HETERO_CMD_RUN into ctrl->cmd.  Between polls,
 * issue wfi to yield the hart to low-power state (the ivshmem interrupt can
 * wake it if wired, otherwise the timer interrupt provides a natural wake-up).
 * -------------------------------------------------------------------------*/
void hetero_device_poll_cmd(void)
{
    volatile struct hetero_ctrl *ctrl = ctrl_regs();

    while (ctrl->cmd != HETERO_CMD_RUN) {
        /* Busy-spin: the Ada300 QEMU machine has no timer interrupt wired
         * to ivshmem, so wfi would stall the hart indefinitely.  A plain
         * busy-spin is correct and efficient for a baremetal QEMU demo. */
        __asm__ volatile ("" ::: "memory");   /* prevent compiler hoisting */
    }

    /* Acquire barrier: all subsequent reads see the host's command writes. */
    __asm__ volatile ("fence r,r" ::: "memory");

    ctrl->status = HETERO_STATUS_RUNNING;
}

/* -------------------------------------------------------------------------
 * hetero_device_get_matmul
 *
 * Parse the matmul command blob from the command buffer.
 * The blob layout is: [ hetero_matmul_hdr | float A[M*K] | float B[K*N] ]
 *
 * We cast away volatile here because rx-ops read the data as ordinary
 * const float* inputs; the barrier in poll_cmd ensures ordering.
 * -------------------------------------------------------------------------*/
void hetero_device_get_matmul(int32_t *M, int32_t *N, int32_t *K,
                               float **A_out, float **B_out)
{
    volatile struct hetero_ctrl *ctrl = ctrl_regs();
    const char *blob = (const char *)cmd_buf_base() + ctrl->blob_offset;

    const struct hetero_matmul_hdr *hdr = (const struct hetero_matmul_hdr *)blob;
    *M = hdr->M;
    *N = hdr->N;
    *K = hdr->K;

    /* A and B immediately follow the header. */
    *A_out = (float *)(blob + sizeof(struct hetero_matmul_hdr));
    *B_out = (float *)(blob + sizeof(struct hetero_matmul_hdr)
                       + (size_t)(*M) * (size_t)(*K) * sizeof(float));
}

/* -------------------------------------------------------------------------
 * hetero_device_get_sqrt
 *
 * Parse the sqrt command blob from the command buffer.
 * The blob layout is: [ hetero_sqrt_hdr | float in[n] ]
 *
 * n_out is set to the number of elements.
 * in_out is set to point directly into the (non-volatile) command buffer.
 * The barrier in poll_cmd ensures ordering; caller treats in_out as read-only.
 * -------------------------------------------------------------------------*/
void hetero_device_get_sqrt(int32_t *n_out, const float **in_out)
{
    volatile struct hetero_ctrl *ctrl = ctrl_regs();
    const char *blob = (const char *)cmd_buf_base() + ctrl->blob_offset;

    const struct hetero_sqrt_hdr *hdr = (const struct hetero_sqrt_hdr *)blob;
    *n_out  = hdr->n;
    *in_out = (const float *)(blob + sizeof(struct hetero_sqrt_hdr));
}

/* -------------------------------------------------------------------------
 * hetero_device_get_add
 *
 * Parse the add command blob from the command buffer.
 * The blob layout is: [ hetero_add_hdr | float in0[n] | float in1[n] ]
 *
 * n_out  is set to the number of elements.
 * in0_out and in1_out point directly into the (non-volatile) command buffer.
 * -------------------------------------------------------------------------*/
void hetero_device_get_add(int32_t *n_out,
                            const float **in0_out, const float **in1_out)
{
    volatile struct hetero_ctrl *ctrl = ctrl_regs();
    const char *blob = (const char *)cmd_buf_base() + ctrl->blob_offset;

    const struct hetero_add_hdr *hdr = (const struct hetero_add_hdr *)blob;
    *n_out   = hdr->n;
    *in0_out = (const float *)(blob + sizeof(struct hetero_add_hdr));
    *in1_out = (const float *)(blob + sizeof(struct hetero_add_hdr)
                               + (size_t)hdr->n * sizeof(float));
}

/* -------------------------------------------------------------------------
 * hetero_device_get_exp
 *
 * Parse the exp command blob from the command buffer.
 * The blob layout is: [ hetero_exp_hdr | float in[n] ]
 * -------------------------------------------------------------------------*/
void hetero_device_get_exp(int32_t *n_out, const float **in_out)
{
    volatile struct hetero_ctrl *ctrl = ctrl_regs();
    const char *blob = (const char *)cmd_buf_base() + ctrl->blob_offset;

    const struct hetero_exp_hdr *hdr = (const struct hetero_exp_hdr *)blob;
    *n_out  = hdr->n;
    *in_out = (const float *)(blob + sizeof(struct hetero_exp_hdr));
}

/* -------------------------------------------------------------------------
 * hetero_device_signal_done
 *
 * Write C into the result buffer and signal completion to the host.
 *
 * Memory ordering:
 *   1. memcpy to result buffer.
 *   2. fence w,w — all result-buffer writes must be globally visible before
 *      ctrl->status = DONE is written.
 *   3. Write ctrl->rc, ctrl->result_size, ctrl->status.
 *   4. fence w,w — status write must be visible before clearing ctrl->cmd.
 *   5. Clear ctrl->cmd = IDLE so the host can issue the next command.
 * -------------------------------------------------------------------------*/
void hetero_device_signal_done(const float *C, int32_t M, int32_t N, int rc)
{
    volatile struct hetero_ctrl *ctrl = ctrl_regs();
    volatile char *rbuf = result_buf_base() + ctrl->result_offset;

    size_t c_bytes = (size_t)M * (size_t)N * sizeof(float);

    /* Write result data into the result buffer. */
    memcpy((void *)rbuf, C, c_bytes);

    /* Release barrier: result writes must precede status flag. */
    __asm__ volatile ("fence w,w" ::: "memory");

    ctrl->result_size = (uint32_t)c_bytes;
    ctrl->rc          = (int32_t)rc;
    ctrl->status      = HETERO_STATUS_DONE;

    /* Final barrier before resetting the command slot. */
    __asm__ volatile ("fence w,w" ::: "memory");

    /* Clear cmd so the host knows it can issue the next command. */
    ctrl->cmd = HETERO_CMD_IDLE;
}
