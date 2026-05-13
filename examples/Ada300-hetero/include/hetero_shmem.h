/*
 * hetero_shmem.h - Shared memory layout for the Ada300 heterogeneous demo.
 *
 * This header is included by both the host process (x86) and the RISC-V
 * baremetal device firmware.  It defines:
 *
 *   - The 64 MiB shared-memory region layout (offsets for control registers,
 *     command buffer, and result buffer).
 *   - struct hetero_ctrl  – the 8-word control-register block at offset 0.
 *   - struct hetero_matmul_hdr – the command blob header written by the host
 *     into the command buffer before triggering the device.
 *   - Constants for cmd / status field values.
 *
 * Transport summary:
 *
 *   Host (x86)                        Shared memory             Device (RISC-V)
 *   ──────────                        ─────────────             ───────────────
 *   hetero_shmem_open()               /dev/shm/ada300_bar       boot, poll ctrl->cmd
 *
 *   pack A,B → cmd_buffer
 *   ctrl->cmd = HETERO_CMD_RUN  ─────► ctrl->cmd == RUN ◄──────  wfi; wakes
 *                                                                 execute MatMul
 *                                                                 write C → result_buf
 *   ctrl->status == DONE ◄────────────  ctrl->status = DONE ◄─── signal_done()
 *   read C from result_buffer
 *   ctrl->cmd = HETERO_CMD_IDLE
 *
 * Shared memory layout (64 MiB):
 *
 *   Offset         Size     Purpose
 *   ─────────────  ───────  ─────────────────────────────────────────────
 *   0x0000_0000    4 KiB    Control registers  (struct hetero_ctrl)
 *   0x0000_1000    32 MiB   Command buffer     (hetero_matmul_hdr + A + B)
 *   0x0200_1000    32 MiB   Result buffer      (C matrix, float32)
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HETERO_SHMEM_H
#define HETERO_SHMEM_H

#include <stdint.h>

/* -------------------------------------------------------------------------
 * Region sizes and offsets within the 64 MiB shared window.
 * -------------------------------------------------------------------------*/
#define HETERO_SHM_SIZE          (64U * 1024U * 1024U)   /* 64 MiB total   */
#define HETERO_CTRL_OFFSET       0x00000000U             /* 4 KiB ctrl regs */
#define HETERO_CMD_BUF_OFFSET    0x00001000U             /* 32 MiB cmd buf  */
#define HETERO_RESULT_OFFSET     0x02001000U             /* 32 MiB result   */

/* -------------------------------------------------------------------------
 * ctrl->cmd values  (written by the host)
 * -------------------------------------------------------------------------*/
#define HETERO_CMD_IDLE          0U   /* channel idle, device polls         */
#define HETERO_CMD_RUN           1U   /* host has written a command blob    */

/* -------------------------------------------------------------------------
 * ctrl->op_type values  (written by the host alongside CMD_RUN)
 * -------------------------------------------------------------------------*/
#define HETERO_OP_MATMUL         0U   /* matmul: C[M,N] = A[M,K] * B[K,N]  */
#define HETERO_OP_SQRT           1U   /* element-wise sqrt: out[i]=sqrt(in[i]) */
#define HETERO_OP_ADD            2U   /* element-wise add:  out[i]=in0[i]+in1[i] */
#define HETERO_OP_EXP            3U   /* element-wise exp:  out[i]=exp(in[i])  */

/* -------------------------------------------------------------------------
 * ctrl->status values  (written by the device)
 * -------------------------------------------------------------------------*/
#define HETERO_STATUS_IDLE       0U   /* waiting for next command           */
#define HETERO_STATUS_RUNNING    1U   /* device is executing the op         */
#define HETERO_STATUS_DONE       2U   /* result is in result buffer         */

/* -------------------------------------------------------------------------
 * Control register block (at HETERO_CTRL_OFFSET in the shared window).
 *
 * Ownership protocol:
 *   - cmd, cmd_seq, blob_offset, blob_size, result_offset  → written by host
 *   - status, result_size, rc                              → written by device
 *
 * All fields are declared volatile so both sides see live values.
 * -------------------------------------------------------------------------*/
struct hetero_ctrl {
    volatile uint32_t cmd;            /* HETERO_CMD_*   — host triggers     */
    volatile uint32_t cmd_seq;        /* sequence number, host increments   */
    volatile uint32_t op_type;        /* HETERO_OP_*    — host sets w/ cmd  */
    volatile uint32_t blob_offset;    /* offset of input blob in cmd buf    */
    volatile uint32_t blob_size;      /* byte size of input blob            */
    volatile uint32_t result_offset;  /* offset in result buffer            */
    volatile uint32_t status;         /* HETERO_STATUS_* — device updates   */
    volatile uint32_t result_size;    /* output size in bytes, set by dev   */
    volatile int32_t  rc;             /* return code, set by device         */
};

/* -------------------------------------------------------------------------
 * Matmul command blob (at cmd_buffer + blob_offset).
 *
 * Layout:
 *   [ hetero_matmul_hdr ]        12 bytes
 *   [ float A[M * K]    ]        M*K * 4 bytes  (row-major)
 *   [ float B[K * N]    ]        K*N * 4 bytes  (row-major)
 *
 * Result buffer (at result_buffer + result_offset):
 *   [ float C[M * N]    ]        M*N * 4 bytes  (row-major)
 * -------------------------------------------------------------------------*/
struct hetero_matmul_hdr {
    int32_t M;    /* rows of A  /  rows of C  */
    int32_t N;    /* cols of B  /  cols of C  */
    int32_t K;    /* cols of A  /  rows of B  */
};

#define HETERO_MATMUL_HDR_SIZE  ((uint32_t)sizeof(struct hetero_matmul_hdr))

/* -------------------------------------------------------------------------
 * Sqrt command blob (at cmd_buffer + blob_offset).
 *
 * Layout:
 *   [ hetero_sqrt_hdr ]        4 bytes
 *   [ float in[n]     ]        n * 4 bytes  (input elements)
 *
 * Result buffer (at result_buffer + result_offset):
 *   [ float out[n]    ]        n * 4 bytes  (sqrt results)
 * -------------------------------------------------------------------------*/
struct hetero_sqrt_hdr {
    int32_t n;    /* number of elements */
};

#define HETERO_SQRT_HDR_SIZE    ((uint32_t)sizeof(struct hetero_sqrt_hdr))

/* -------------------------------------------------------------------------
 * Add command blob (at cmd_buffer + blob_offset).
 *
 * Layout:
 *   [ hetero_add_hdr  ]        4 bytes
 *   [ float in0[n]    ]        n * 4 bytes  (first operand)
 *   [ float in1[n]    ]        n * 4 bytes  (second operand)
 *
 * Result buffer (at result_buffer + result_offset):
 *   [ float out[n]    ]        n * 4 bytes  (add results)
 * -------------------------------------------------------------------------*/
struct hetero_add_hdr {
    int32_t n;    /* number of elements */
};

#define HETERO_ADD_HDR_SIZE     ((uint32_t)sizeof(struct hetero_add_hdr))

/* -------------------------------------------------------------------------
 * Exp command blob (at cmd_buffer + blob_offset).
 *
 * Layout:
 *   [ hetero_exp_hdr  ]        4 bytes
 *   [ float in[n]     ]        n * 4 bytes  (input elements)
 *
 * Result buffer (at result_buffer + result_offset):
 *   [ float out[n]    ]        n * 4 bytes  (exp results)
 * -------------------------------------------------------------------------*/
struct hetero_exp_hdr {
    int32_t n;    /* number of elements */
};

#define HETERO_EXP_HDR_SIZE     ((uint32_t)sizeof(struct hetero_exp_hdr))

/* -------------------------------------------------------------------------
 * Physical base address of the ivshmem region in the RISC-V guest.
 *
 * This constant is shared between:
 *   - ada300s_evk.h / ada300s_evk.c   (QEMU machine: maps hostmem here)
 *   - device/hetero_shmem_device.c    (firmware: raw MMIO pointer cast)
 *
 * Placement: gap between TCM end (0x41200000) and TEST (0x4C000000),
 *            fitting a 64 MiB window starting at 0x42000000.
 * -------------------------------------------------------------------------*/
#define ADA300_IVSHMEM_BASE    0x0042000000ULL
#define ADA300_IVSHMEM_SIZE    HETERO_SHM_SIZE

#endif /* HETERO_SHMEM_H */
