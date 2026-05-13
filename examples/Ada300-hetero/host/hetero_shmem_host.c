/*
 * hetero_shmem_host.c - Host-side shared memory transport for the Ada300
 * heterogeneous demo.
 *
 * Uses POSIX shm_open + mmap to create the 64 MiB shared window that QEMU
 * maps into the RISC-V guest at ADA300_IVSHMEM_BASE.  The host then
 * communicates with the baremetal device firmware via the hetero_ctrl
 * register block and the command / result buffers.
 *
 * Public API (declared in hetero_shmem_host.h or used via extern in callers):
 *
 *   void *hetero_shmem_open(const char *name, size_t size)
 *       Open (or create) a POSIX shared memory object, truncate it to size,
 *       and mmap it MAP_SHARED.  Returns the base pointer, or NULL on error.
 *       The control-register block at offset 0 is zeroed on open.
 *
 *   void hetero_shmem_close(void)
 *       Unmap the shared memory region.
 *
 *   int hetero_dispatch_matmul(void *base,
 *                              const float *A, const float *B,
 *                              float *C,
 *                              int32_t M, int32_t N, int32_t K)
 *       Pack a matmul command into the command buffer, signal the device,
 *       spin-poll until status == DONE, read the result, and return the
 *       device rc.  Writes to C are only performed when rc == 0.
 *
 * The global pointer g_hetero_base (set by hetero_shmem_open) is used by
 * rx_ops_bridge_hetero.c when resolving rxops_bridge_matmul_f32 calls.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/hetero_shmem.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>

/* Timeout for the spin-poll: 30 seconds. */
#define HETERO_TIMEOUT_NS  (30LL * 1000LL * 1000LL * 1000LL)

/* Shared memory base pointer — accessed by rx_ops_bridge_hetero.c. */
void *g_hetero_base = (void *)-1;   /* MAP_FAILED sentinel until open */

/* -------------------------------------------------------------------------
 * hetero_shmem_open
 * Open (or create) the POSIX shared memory object and mmap it.
 * Returns the mapped base address, or NULL on error.
 * -------------------------------------------------------------------------*/
void *hetero_shmem_open(const char *name, size_t size)
{
    int fd = shm_open(name, O_RDWR | O_CREAT, 0666);
    if (fd < 0) {
        fprintf(stderr, "[hetero] shm_open(%s): %s\n", name, strerror(errno));
        return NULL;
    }

    if (ftruncate(fd, (off_t)size) < 0) {
        fprintf(stderr, "[hetero] ftruncate: %s\n", strerror(errno));
        close(fd);
        return NULL;
    }

    void *addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);  /* fd can be closed after mmap */

    if (addr == MAP_FAILED) {
        fprintf(stderr, "[hetero] mmap: %s\n", strerror(errno));
        return NULL;
    }

    /* Zero the control-register block so the device sees idle state. */
    memset(addr, 0, sizeof(struct hetero_ctrl));

    g_hetero_base = addr;
    return addr;
}

/* -------------------------------------------------------------------------
 * hetero_shmem_close
 * -------------------------------------------------------------------------*/
void hetero_shmem_close(void)
{
    if (g_hetero_base != MAP_FAILED && g_hetero_base != NULL) {
        munmap(g_hetero_base, HETERO_SHM_SIZE);
        g_hetero_base = (void *)-1;
    }
}

/* -------------------------------------------------------------------------
 * hetero_dispatch_matmul
 *
 * Serialise a matmul command (M, N, K, A, B) into the command buffer,
 * trigger the device, wait for completion, then copy C from the result
 * buffer.
 *
 * Memory ordering:
 *   - __sync_synchronize() (full barrier) before setting ctrl->cmd = RUN
 *     ensures all command-buffer writes are visible before the trigger.
 *   - __sync_synchronize() after observing DONE ensures result-buffer
 *     reads happen after the device's writes.
 * -------------------------------------------------------------------------*/
int hetero_dispatch_matmul(void *base,
                            const float *A, const float *B,
                            float *C,
                            int32_t M, int32_t N, int32_t K)
{
    struct hetero_ctrl *ctrl = (struct hetero_ctrl *)
        ((char *)base + HETERO_CTRL_OFFSET);
    char *cmd_buf  = (char *)base + HETERO_CMD_BUF_OFFSET;
    char *res_buf  = (char *)base + HETERO_RESULT_OFFSET;

    size_t a_bytes   = (size_t)M * (size_t)K * sizeof(float);
    size_t b_bytes   = (size_t)K * (size_t)N * sizeof(float);
    size_t c_bytes   = (size_t)M * (size_t)N * sizeof(float);
    size_t blob_size = sizeof(struct hetero_matmul_hdr) + a_bytes + b_bytes;

    /* --- Write the command blob into the command buffer at offset 0 ------- */
    struct hetero_matmul_hdr hdr = { M, N, K };
    memcpy(cmd_buf,                                           &hdr,  sizeof(hdr));
    memcpy(cmd_buf + sizeof(hdr),                             A,     a_bytes);
    memcpy(cmd_buf + sizeof(hdr) + a_bytes,                   B,     b_bytes);

    /* --- Fill control registers ------------------------------------------- */
    ctrl->blob_offset   = 0U;
    ctrl->blob_size     = (uint32_t)blob_size;
    ctrl->result_offset = 0U;
    ctrl->op_type       = HETERO_OP_MATMUL;
    ctrl->status        = HETERO_STATUS_IDLE;
    ctrl->rc            = 0;

    /* Full barrier: all stores above must complete before the trigger. */
    __sync_synchronize();

    /* --- Trigger the device ----------------------------------------------- */
    ctrl->cmd_seq++;
    ctrl->cmd = HETERO_CMD_RUN;

    /* Full barrier: cmd write must be globally visible before we poll. */
    __sync_synchronize();

    /* --- Spin-poll until device signals DONE (with timeout) -------------- */
    {
        struct timespec ts_start, ts_now;
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
        while (ctrl->status != HETERO_STATUS_DONE) {
            clock_gettime(CLOCK_MONOTONIC, &ts_now);
            long long elapsed = (ts_now.tv_sec  - ts_start.tv_sec)  * 1000000000LL
                              + (ts_now.tv_nsec - ts_start.tv_nsec);
            if (elapsed > HETERO_TIMEOUT_NS) {
                fprintf(stderr,
                    "[hetero] timeout waiting for device (is QEMU running?)\n");
                ctrl->cmd = HETERO_CMD_IDLE;
                return -1;
            }
        }
    }

    /* Full barrier: device's result-buffer stores must be visible. */
    __sync_synchronize();

    int rc = (int)ctrl->rc;

    /* --- Copy result ------------------------------------------------------- */
    if (rc == 0) {
        memcpy(C, res_buf + ctrl->result_offset, c_bytes);
    }

    /* Reset command slot so the device knows it can accept the next cmd. */
    ctrl->cmd = HETERO_CMD_IDLE;

    return rc;
}

/* ------------------------------------------------------------------------- * hetero_dispatch_exp
 *
 * Serialise an exp command (n elements, input array in) into the command
 * buffer, trigger the device, wait for completion, then copy the result
 * into out.
 * -------------------------------------------------------------------------*/
int hetero_dispatch_exp(void *base,
                         const float *in, float *out, int32_t n)
{
    struct hetero_ctrl *ctrl = (struct hetero_ctrl *)
        ((char *)base + HETERO_CTRL_OFFSET);
    char *cmd_buf = (char *)base + HETERO_CMD_BUF_OFFSET;
    char *res_buf = (char *)base + HETERO_RESULT_OFFSET;

    size_t in_bytes  = (size_t)n * sizeof(float);
    size_t blob_size = sizeof(struct hetero_exp_hdr) + in_bytes;

    struct hetero_exp_hdr hdr = { n };
    memcpy(cmd_buf,               &hdr, sizeof(hdr));
    memcpy(cmd_buf + sizeof(hdr), in,   in_bytes);

    ctrl->blob_offset   = 0U;
    ctrl->blob_size     = (uint32_t)blob_size;
    ctrl->result_offset = 0U;
    ctrl->op_type       = HETERO_OP_EXP;
    ctrl->status        = HETERO_STATUS_IDLE;
    ctrl->rc            = 0;

    __sync_synchronize();
    ctrl->cmd_seq++;
    ctrl->cmd = HETERO_CMD_RUN;
    __sync_synchronize();

    {
        struct timespec ts_start, ts_now;
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
        while (ctrl->status != HETERO_STATUS_DONE) {
            clock_gettime(CLOCK_MONOTONIC, &ts_now);
            long long elapsed = (ts_now.tv_sec  - ts_start.tv_sec)  * 1000000000LL
                              + (ts_now.tv_nsec - ts_start.tv_nsec);
            if (elapsed > HETERO_TIMEOUT_NS) {
                fprintf(stderr,
                    "[hetero] timeout waiting for device (exp, is QEMU running?)\n");
                ctrl->cmd = HETERO_CMD_IDLE;
                return -1;
            }
        }
    }

    __sync_synchronize();
    int rc = (int)ctrl->rc;
    if (rc == 0)
        memcpy(out, res_buf + ctrl->result_offset, in_bytes);
    ctrl->cmd = HETERO_CMD_IDLE;
    return rc;
}

/* ------------------------------------------------------------------------- * hetero_dispatch_add
 *
 * Serialise an add command (n elements, in0 and in1 arrays) into the command
 * buffer, trigger the device, wait for completion, then copy the result
 * into out.
 * -------------------------------------------------------------------------*/
int hetero_dispatch_add(void *base,
                         const float *in0, const float *in1,
                         float *out, int32_t n)
{
    struct hetero_ctrl *ctrl = (struct hetero_ctrl *)
        ((char *)base + HETERO_CTRL_OFFSET);
    char *cmd_buf = (char *)base + HETERO_CMD_BUF_OFFSET;
    char *res_buf = (char *)base + HETERO_RESULT_OFFSET;

    size_t in_bytes  = (size_t)n * sizeof(float);
    size_t blob_size = sizeof(struct hetero_add_hdr) + 2 * in_bytes;

    /* --- Write the command blob into the command buffer at offset 0 ------- */
    struct hetero_add_hdr hdr = { n };
    memcpy(cmd_buf,                              &hdr,  sizeof(hdr));
    memcpy(cmd_buf + sizeof(hdr),                in0,   in_bytes);
    memcpy(cmd_buf + sizeof(hdr) + in_bytes,     in1,   in_bytes);

    /* --- Fill control registers ------------------------------------------- */
    ctrl->blob_offset   = 0U;
    ctrl->blob_size     = (uint32_t)blob_size;
    ctrl->result_offset = 0U;
    ctrl->op_type       = HETERO_OP_ADD;
    ctrl->status        = HETERO_STATUS_IDLE;
    ctrl->rc            = 0;

    /* Full barrier: all stores above must complete before the trigger. */
    __sync_synchronize();

    /* --- Trigger the device ----------------------------------------------- */
    ctrl->cmd_seq++;
    ctrl->cmd = HETERO_CMD_RUN;

    /* Full barrier: cmd write must be globally visible before we poll. */
    __sync_synchronize();

    /* --- Spin-poll until device signals DONE (with timeout) -------------- */
    {
        struct timespec ts_start, ts_now;
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
        while (ctrl->status != HETERO_STATUS_DONE) {
            clock_gettime(CLOCK_MONOTONIC, &ts_now);
            long long elapsed = (ts_now.tv_sec  - ts_start.tv_sec)  * 1000000000LL
                              + (ts_now.tv_nsec - ts_start.tv_nsec);
            if (elapsed > HETERO_TIMEOUT_NS) {
                fprintf(stderr,
                    "[hetero] timeout waiting for device (add, is QEMU running?)\n");
                ctrl->cmd = HETERO_CMD_IDLE;
                return -1;
            }
        }
    }

    /* Full barrier: device's result-buffer stores must be visible. */
    __sync_synchronize();

    int rc = (int)ctrl->rc;

    /* --- Copy result ------------------------------------------------------- */
    if (rc == 0) {
        memcpy(out, res_buf + ctrl->result_offset, in_bytes);
    }

    /* Reset command slot so the device knows it can accept the next cmd. */
    ctrl->cmd = HETERO_CMD_IDLE;

    return rc;
}

/* -------------------------------------------------------------------------
 * hetero_dispatch_sqrt
 *
 * Serialise a sqrt command (n elements, input array in) into the command
 * buffer, trigger the device, wait for completion, then copy the result
 * into out.
 * -------------------------------------------------------------------------*/
int hetero_dispatch_sqrt(void *base,
                          const float *in, float *out, int32_t n)
{
    struct hetero_ctrl *ctrl = (struct hetero_ctrl *)
        ((char *)base + HETERO_CTRL_OFFSET);
    char *cmd_buf = (char *)base + HETERO_CMD_BUF_OFFSET;
    char *res_buf = (char *)base + HETERO_RESULT_OFFSET;

    size_t in_bytes  = (size_t)n * sizeof(float);
    size_t blob_size = sizeof(struct hetero_sqrt_hdr) + in_bytes;

    /* --- Write the command blob into the command buffer at offset 0 ------- */
    struct hetero_sqrt_hdr hdr = { n };
    memcpy(cmd_buf,                       &hdr, sizeof(hdr));
    memcpy(cmd_buf + sizeof(hdr),         in,   in_bytes);

    /* --- Fill control registers ------------------------------------------- */
    ctrl->blob_offset   = 0U;
    ctrl->blob_size     = (uint32_t)blob_size;
    ctrl->result_offset = 0U;
    ctrl->op_type       = HETERO_OP_SQRT;
    ctrl->status        = HETERO_STATUS_IDLE;
    ctrl->rc            = 0;

    /* Full barrier: all stores above must complete before the trigger. */
    __sync_synchronize();

    /* --- Trigger the device ----------------------------------------------- */
    ctrl->cmd_seq++;
    ctrl->cmd = HETERO_CMD_RUN;

    /* Full barrier: cmd write must be globally visible before we poll. */
    __sync_synchronize();

    /* --- Spin-poll until device signals DONE (with timeout) -------------- */
    {
        struct timespec ts_start, ts_now;
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
        while (ctrl->status != HETERO_STATUS_DONE) {
            clock_gettime(CLOCK_MONOTONIC, &ts_now);
            long long elapsed = (ts_now.tv_sec  - ts_start.tv_sec)  * 1000000000LL
                              + (ts_now.tv_nsec - ts_start.tv_nsec);
            if (elapsed > HETERO_TIMEOUT_NS) {
                fprintf(stderr,
                    "[hetero] timeout waiting for device (sqrt, is QEMU running?)\n");
                ctrl->cmd = HETERO_CMD_IDLE;
                return -1;
            }
        }
    }

    /* Full barrier: device's result-buffer stores must be visible. */
    __sync_synchronize();

    int rc = (int)ctrl->rc;

    /* --- Copy result ------------------------------------------------------- */
    if (rc == 0) {
        memcpy(out, res_buf + ctrl->result_offset, in_bytes);
    }

    /* Reset command slot so the device knows it can accept the next cmd. */
    ctrl->cmd = HETERO_CMD_IDLE;

    return rc;
}
