# Ada300 Heterogeneous Execution: Host (x86) + QEMU (Ada300 SNPU)

> **Implementation status: complete and tested.**
> All source files described below exist; `ninja run-ada300-hetero` runs the
> full end-to-end demo.

## Overview

This example extends the `examples/Ada300` single-target demo into a
**heterogeneous execution model** where operators in the same inference graph
are dispatched to two different compute units:

| Compute Unit | Backend | Ops |
|---|---|---|
| Host x86 CPU | `RXOPS_C` (reference C) | `Exp`, `Sqrt` |
| Ada300 RISC-V SNPU (QEMU) | `RXOPS_ADA300` (`xadatmm`/`xadacv` ISA) | `MatMul`, `Add` |

The two units communicate via a **64 MiB POSIX shared memory region**
(`/dev/shm/ada300_bar`) that QEMU maps into the RISC-V guest at physical
address `0x42000000`.  This simulates a PCIe-attached accelerator board
where the host DMA-copies command tensors into the device's BAR and polls a
completion flag.  No QEMU ivshmem PCI device is needed — the region is
wired directly into the Ada300 address space via
`memory_region_add_subregion` in `ada300s_evk.c`.

---

## Model and Split Point

The Ada300 model is a two-layer feedforward network:

```
input [1×64]
  │
  ├─ MatMul(fc1_weight [64×128])  ◄─── DISPATCHED TO Ada300 SNPU (QEMU)
  ├─ Add(fc1_bias [1×128])        ◄─── DISPATCHED TO Ada300 SNPU (QEMU)
  │
  ├─ Exp [1×128]                  ◄─── RUNS ON HOST (RXOPS_C)
  ├─ Sqrt [1×128]                 ◄─── RUNS ON HOST (RXOPS_C)
  │
  ├─ MatMul(fc2_weight [128×64])  ◄─── DISPATCHED TO Ada300 SNPU (QEMU)
  ├─ Add(fc2_bias [1×64])         ◄─── DISPATCHED TO Ada300 SNPU (QEMU)
  │
output [1×64]
```

The model has two **synchronisation points**: after fc1 (host reads back
the `[1×128]` intermediate) and after fc2 (host reads back the `[1×64]`
final output).  Each synchronisation point is one `hetero_dispatch_matmul`
call which blocks until the Ada300 SNPU signals completion.

---

## Quick Start

```bash
# Build everything (from the repo root):
cd examples/ada300-hetero
mkdir -p build && cd build
cmake .. -G Ninja \
  -DMLIR_DIR=/usr/local/lib/cmake/mlir \
  -DHETERO_HOST=ON \
  -DHETERO_DEVICE=ON \
  -DADA300_LLVM_ROOT=/path/to/rx-llvm-main/build \
  -DADA300_QEMU=/path/to/rx-qemu-main/build/qemu-system-riscv64

# Build + run in one step:
ninja run-ada300-hetero
```

Expected output:

```
[run] Initialising shared memory at /dev/shm/ada300_bar...
[run] Starting QEMU ...
[run] Firmware ready (1 s).
[hetero] Inference time: ~10 ms
[hetero] Output (1×64):
  [1.7627, 1.7627, ...]
[hetero] Expected ≈ 1.7627 per element

=== Device UART log ===
[hetero-dev] Ada300 heterogeneous device firmware booted.
[hetero-dev] Polling ivshmem at 0x42000000 for commands...
[hetero-dev] cmd #1: matmul [1×128] = [1×64] * [64×128]
[hetero-dev]   MatMul done in ~300000 ticks, rc=0
[hetero-dev] cmd #2: add [128 elements]
[hetero-dev]   Add done in ~5000 ticks, rc=0
[hetero-dev] cmd #3: matmul [1×64] = [1×128] * [128×64]
[hetero-dev]   MatMul done in ~80000 ticks, rc=0
[hetero-dev] cmd #4: add [64 elements]
[hetero-dev]   Add done in ~2000 ticks, rc=0
```

> **Note:** The output is 1.7627 rather than the reference 1.7629 because the
> Ada300 SNPU has no fp32 matmul kernel.  The device firmware converts fp32 →
> fp16 before calling `rxops_matmul`, runs the Ada300 fp16 kernel, then
> converts fp16 → fp32.  The ~0.01% difference is normal fp16 rounding
> propagated through MatMul.  `Exp` and `Sqrt` run on the host CPU (RXOPS_C)
> with full fp32 precision.

---

## File Tree

```
examples/ada300-hetero/
├── CMakeLists.txt                    ← build targets (see §Build System)
├── cmake/
│   └── run-ada300-hetero.sh.in       ← script template for run-ada300-hetero target
├── include/
│   └── hetero_shmem.h                ← shared struct/offset definitions (host + device)
├── ada300-import.py               ← PyTorch → Top MLIR importer; edit DEVICE_* flags here
├── model.py                       ← Ada300Model definition
├── host/
│   ├── ada300-hetero-main.cpp     ← host orchestrator (opens shmem, calls subgraph0)
│   └── hetero_shmem_host.c       ← shm_open/mmap transport + dispatch helpers
├── device/
│   ├── ada300-hetero-main.c      ← persistent RISC-V firmware (polling loop)
│   └── hetero_shmem_device.c    ← MMIO-based shmem access on the RISC-V side
├── rx_ops_bridge.c               ← op bridge shared by host + device builds
│                                    #ifndef __riscv (x86): ada300_*→ivshmem dispatch
│                                    #ifdef  __riscv (device): ada300_*→RXOPS_ADA300 hw
│                                    plain ops always → RXOPS_C
└── rx_ops_bridge.h               ← bridge ABI declaration
```

External files modified for this example:

| File | Change |
|---|---|
| `rx-qemu-main/include/hw/riscv/ada300s_evk.h` | Added `ADA300S_EVK_IVSHMEM` memmap entry, `ADA300_IVSHMEM_BASE`, `ADA300_IVSHMEM_SIZE` |
| `rx-qemu-main/hw/riscv/ada300s_evk.c` | Added ivshmem memmap entry + `machine_init` mapping via `host_memory_backend_get_memory`; made HBF/LPDDR regions optional (fallback on OOM) |

---

## Transport Layer

### Shared Memory Layout (64 MiB)

```
Offset         Size     Purpose
─────────────  ───────  ─────────────────────────────────────────────────
0x0000_0000    4 KiB    Control registers  (struct hetero_ctrl)
0x0000_1000    32 MiB   Command buffer     host writes: matmul_hdr + A + B
0x0200_1000    32 MiB   Result buffer      device writes: C matrix (float32)
```

QEMU maps `/dev/shm/ada300_bar` into the RISC-V guest address space at
`ADA300_IVSHMEM_BASE = 0x0042000000` via `memory_region_add_subregion`.
The firmware accesses it as a raw MMIO pointer cast:

```c
volatile struct hetero_ctrl *ctrl =
    (volatile struct hetero_ctrl *)(uintptr_t)ADA300_IVSHMEM_BASE;
```

### Control Register Block (`struct hetero_ctrl`, offset 0)

```c
struct hetero_ctrl {
    volatile uint32_t cmd;           // host writes: IDLE=0 / RUN=1
    volatile uint32_t cmd_seq;       // host increments per command
    volatile uint32_t op_type;       // host writes: MATMUL=0 / SQRT=1 / ADD=2 / EXP=3
    volatile uint32_t blob_offset;   // byte offset of command blob in cmd buf
    volatile uint32_t blob_size;     // byte size of command blob
    volatile uint32_t result_offset; // byte offset in result buf
    volatile uint32_t status;        // device writes: IDLE=0 / RUNNING=1 / DONE=2
    volatile uint32_t result_size;   // device writes: byte size of result
    volatile  int32_t rc;            // device writes: 0=ok, <0=error
};
```

### Command Blob Layout (in command buffer)

Each MatMul command is packed as:

```
[ hetero_matmul_hdr (12 bytes) | A[M×K] float32 | B[K×N] float32 ]
```

```c
struct hetero_matmul_hdr { int32_t M, N, K; };
```

`A` and `B` immediately follow the header with no padding.

### Handshake Protocol

```
Host                             Shared Memory              Device Firmware
────                             ─────────────              ───────────────
memset(ctrl, 0)                                             boot; busy-poll ctrl->cmd

pack hdr+A+B → cmd_buffer
ctrl->blob_offset = 0
ctrl->blob_size   = …
ctrl->result_offset = 0
ctrl->status = IDLE
__sync_synchronize()
ctrl->cmd_seq++
ctrl->cmd = RUN          ──────► cmd == RUN ◄────────────  fence r,r
                                                            ctrl->status = RUNNING
                                                            parse hdr; rxops_matmul
                                                            memcpy C → result_buf
                                                            fence w,w
                                 ctrl->status = DONE ◄────  ctrl->status = DONE
                                                            fence w,w
                                                            ctrl->cmd = IDLE
poll(ctrl->status == DONE) ◄───
__sync_synchronize()
memcpy C from result_buf
ctrl->cmd = IDLE
```

Memory ordering uses `__sync_synchronize()` on the host (x86 TSO is strong
but the barrier prevents compiler reordering) and explicit RISC-V `fence`
instructions on the device (`fence r,r` after reading cmd, `fence w,w`
before writing status).

---

## Implementation Details

### Host Side

**`hetero_shmem_host.c`** — transport implementation

- `hetero_shmem_open(name, size)` — `shm_open` + `ftruncate` + `mmap(MAP_SHARED)`;
  zeros the control block on open.
- `hetero_dispatch_matmul(base, A, B, C, M, N, K)` — packs the command blob,
  triggers the device, spin-polls `ctrl->status` with a 30-second timeout,
  copies C from the result buffer.  Returns 0 on success, -1 on timeout or
  device error.

**`rx_ops_bridge.c`** — op bridge (host + device, auto-selects via `#ifndef __riscv`)

Provides two classes of symbols, resolved at link time:

| Symbol | Emitted when | Host dispatch | Device dispatch |
|---|---|---|---|
| `rxops_bridge_ada300_matmul_f32` | `device = "ada300"` on `top.MatMul` | `hetero_dispatch_matmul` → ivshmem → QEMU | `rxops_matmul(RXOPS_ADA300)` |
| `rxops_bridge_ada300_add_f32` | `device = "ada300"` on `top.Add` | `hetero_dispatch_add` → ivshmem → QEMU | `rxops_add(RXOPS_ADA300)` |
| `rxops_bridge_ada300_exp_f32` | `device = "ada300"` on `top.Exp` | `hetero_dispatch_exp` → ivshmem → QEMU | `rxops_exp(RXOPS_ADA300)` |
| `rxops_bridge_ada300_sqrt_f32` | `device = "ada300"` on `top.Sqrt` | `hetero_dispatch_sqrt` → ivshmem → QEMU | `rxops_sqrt(RXOPS_ADA300)` |
| `rxops_bridge_matmul_f32` | no `device` attr on `top.MatMul` | `rxops_matmul` via RXOPS_C | `rxops_matmul` via RXOPS_C |
| `rxops_bridge_exp_f32` | no `device` attr on `top.Exp` | `rxops_exp` via RXOPS_C | `rxops_exp` via RXOPS_C |
| `rxops_bridge_sqrt_f32` | no `device` attr on `top.Sqrt` | `rxops_sqrt` via RXOPS_C | `rxops_sqrt` via RXOPS_C |
| `rxops_bridge_add_f32` | no `device` attr on `top.Add` | `rxops_add` via RXOPS_C | `rxops_add` via RXOPS_C |
| `rxops_bridge_log_f32` | — | `rxops_log` via RXOPS_C | `rxops_log` via RXOPS_C |
| `rxops_bridge_rsqrt_f32` | — | `rxops_rsqrt` via RXOPS_C | `rxops_rsqrt` via RXOPS_C |

The dispatch decision is made by the **compiler** (`TopToRxOps` pass) at
lowering time, not at runtime.  `ada300-import.py` sets `DEVICE_MATMUL`,
`DEVICE_ADD`, `DEVICE_EXP`, and `DEVICE_SQRT` at the top of the file.  Set
a flag to `"ada300"` to emit the `ada300` variant (ivshmem on host,
hardware on device); set it to anything else to emit the plain variant
(RXOPS_C).  The linker resolves each symbol from this single file for both
host and device builds — no separate `*_hetero.c` file is needed.

**`ada300-hetero-main.cpp`** — host orchestrator

1. Opens `/dev/shm/ada300_bar` (or `$HETERO_SHM_NAME`) via `hetero_shmem_open`.
2. Allocates and fills model tensors (`fc1_weight`, `fc1_bias`, `fc2_weight`,
   `fc2_bias`, `input`).
3. Calls `_mlir_ciface_subgraph0` — the MLIR runtime calls back into the
   bridge functions above.
4. Prints the `[1×64]` output and inference time.

### Device Side (RISC-V Baremetal Firmware)

**`hetero_shmem_device.c`** — MMIO transport

- `hetero_device_poll_cmd()` — busy-spins on `ctrl->cmd` (no `wfi`; the
  Ada300 QEMU machine has no timer interrupt wired to ivshmem, so `wfi`
  would stall the hart).  Sets `ctrl->status = RUNNING` when cmd == RUN.
- `hetero_device_get_matmul(M, N, K, A, B)` — parses `hetero_matmul_hdr`
  from the command buffer; returns pointers into the shared window.
- `hetero_device_signal_done(C, M, N, rc)` — `memcpy` C into result buffer,
  `fence w,w`, writes `ctrl->status = DONE`, `fence w,w`, clears
  `ctrl->cmd = IDLE`.

**`rx_ops_bridge.c`** — same file, compiled for RISC-V (`#ifdef __riscv` path)

Provides all symbols declared in `rx_ops_bridge.h`:
- Plain ops forward to `RXOPS_C` directly (both host and device).
- `rxops_bridge_ada300_*` variants call `RXOPS_ADA300` hardware kernels.

On device, `rxnn_c_init()` is a `__attribute__((weak))` empty stub so lld
resolves it locally; on host, only `extern void rxnn_c_init(void)` is
declared (no definition), so the linker uses the real implementation from
`librx_ops.a` (placed with `--whole-archive`).  MatMul
requires a fp32 → fp16 → fp32 round-trip because the Ada300 hardware kernel
only supports fp16:

```
rxops_bridge_matmul_f32(C, A, B, M, N, K)
  1. malloc A16[M×K], B16[K×N], C16[M×N]   (LPDDR heap via mr_alloc)
  2. A → A16, B → B16  (fcvt.h.s, RISC-V Zfh)
  3. rxops_matmul(RXOPS_ADA300, fp16)
  4. C16 → C  (fcvt.s.h, RISC-V Zfh)
  5. free A16, B16, C16
  Returns 0 on success (normalised from RXOPS_TRUE=1)
```

**`ada300-hetero-main.c`** — persistent polling loop

```
main()
  printf "[hetero-dev] booted"
  for(;;):
    hetero_device_poll_cmd()          // busy-spin on ctrl->cmd
    op = ctrl->op_type
    if op == HETERO_OP_MATMUL:
      hetero_device_get_matmul(M,N,K,A,B)
      C = mr_alloc(M*N*4, LPDDR)
      rc = rxops_bridge_ada300_matmul_f32(C, A, B, M, N, K)
      hetero_device_signal_done(C, M, N, rc)
      mr_free(C)
    elif op == HETERO_OP_SQRT:
      hetero_device_get_sqrt(n, in)
      out = mr_alloc(n*4, LPDDR)
      rc = rxops_bridge_ada300_sqrt_f32(out, in, n)
      hetero_device_signal_done(out, n, 1, rc)
      mr_free(out)
    elif op == HETERO_OP_EXP:
      hetero_device_get_exp(n, in)
      out = mr_alloc(n*4, LPDDR)
      rc = rxops_bridge_ada300_exp_f32(out, in, n)
      hetero_device_signal_done(out, n, 1, rc)
      mr_free(out)
    elif op == HETERO_OP_ADD:
      hetero_device_get_add(n, in0, in1)
      out = mr_alloc(n*4, LPDDR)
      rc = rxops_bridge_ada300_add_f32(out, in0, in1, n)
      hetero_device_signal_done(out, n, 1, rc)
      mr_free(out)
    // loop — no qemu_exit(), stays alive for next command
```

`malloc`/`free` are backed by `mr_alloc(MR_REGION_LPDDR)` so the fp16
scratch buffers inside `rxops_bridge_matmul_f32` are allocated from the
8 GiB LPDDR heap.

### QEMU Machine Changes

**`ada300s_evk.h`**

```c
enum {
    ...
    ADA300S_EVK_IVSHMEM,   /* 64 MiB @ 0x42000000 */
    ...
};
#define ADA300_IVSHMEM_BASE  0x0042000000ULL
#define ADA300_IVSHMEM_SIZE  (64U * 1024U * 1024U)
```

**`ada300s_evk.c` — `ada300s_evk_board_init`**

The ivshmem memory region is mapped only when QEMU is launched with
`-object memory-backend-file,id=shmem0,...`:

```c
HostMemoryBackend *backend = object_find(…, "shmem0", …);
if (backend) {
    MemoryRegion *mr = host_memory_backend_get_memory(backend);
    memory_region_add_subregion(system_memory,
                                ada300s_evk_memmap[ADA300S_EVK_IVSHMEM].base,
                                mr);
    info_report("ada300s_evk: ivshmem mapped at 0x%llx (64 MiB)",
                ADA300_IVSHMEM_BASE);
}
```

If `shmem0` is absent the machine boots normally without ivshmem — the
hetero firmware will busy-poll forever but standalone firmware is unaffected.

LPDDR (8 GiB) and HBF (35 GiB) are made optional: if the host kernel's
overcommit limit (`vm.overcommit_memory=0`) prevents the allocation, LPDDR
falls back to 256 MiB and HBF is skipped silently.  The firmware entry
point is at `0x50000000` (LPDDR start), so 256 MiB is sufficient for the
hetero demo stack and heap.

---

## Build System

### CMake Options

| Option | Default | Description |
|---|---|---|
| `HETERO_HOST` | `ON` | Build `ada300-hetero-runner` (x86 host binary) |
| `HETERO_DEVICE` | `OFF` | Build `ada300-hetero-device.elf` (RISC-V firmware) |
| `ADA300_LLVM_ROOT` | `/workspace/rx-llvm-main/build` | Ada300 LLVM toolchain (provides `riscv64-unknown-elf-clang`) |
| `ADA300_SDK_ROOT` | auto-detected | Path to `ada300_snpu_sdk` (BSP + linker script) |
| `ADA300_QEMU` | `/opt/ada300-qemu/bin/qemu-system-riscv64` | QEMU binary with `ada300s_evk` machine |
| `ADA300_NEWLIB_SYSROOT` | `/usr/lib/picolibc/riscv64-unknown-elf` | Picolibc sysroot |

### Targets

| Target | Output | Requires |
|---|---|---|
| `ada300-hetero-host` | `ada300-hetero-runner` | `HETERO_HOST=ON` |
| `ada300-hetero-device` | `device/ada300-hetero-device.elf` | `HETERO_DEVICE=ON` |
| `run-ada300-hetero` | — | both ON |

### Host Build Pipeline (`HETERO_HOST=ON`)

The host runner is built from pre-compiled MLIR IR:

```
[Step 0 — ninja runs automatically when ada300-import.py or model.py change]
ada300-import.py
  DEVICE_MATMUL / DEVICE_ADD / DEVICE_EXP / DEVICE_SQRT flags at the top
  control which ops get device = "ada300" (→ ivshmem) vs. plain (→ RXOPS_C).
  current defaults: DEVICE_MATMUL="ada300", DEVICE_ADD="ada300",
                    DEVICE_EXP="host",    DEVICE_SQRT="host"
  └─ output/subgraph0_top.mlir          (Top dialect IR with device attrs)
       │
  [Step 1]
       ├─ tpuc-opt --convert-top-to-rxops
       │     device="ada300" ops  → llvm.call @rxops_bridge_ada300_<op>_f32
       │     plain ops            → llvm.call @rxops_bridge_<op>_f32
       │     → subgraph0_rxops.mlir
  [Steps 2–4]
       ├─ mlir-opt --lower-to-llvm         → subgraph0.ll  (LLVM 18 IR)
       ├─ llvm-as                          → subgraph0.bc  (LLVM 18 bitcode)
       └─ llc-14 -opaque-pointers          → subgraph0.o   (x86-64 object)
            │
  [Step 7]
            └─ clang++ link:
                  subgraph0.o
                  host/ada300-hetero-main.cpp
                  host/hetero_shmem_host.c
                  rx_ops_bridge.c              ← #ifndef __riscv: ada300_* → ivshmem dispatch
                                                 rxops_bridge_*  → RXOPS_C (unchanged)
                  third_party/rx-ops/build_host/librx_ops.a   ← x86 RXOPS_C
                  → ada300-hetero-runner
```

> `third_party/rx-ops/build_host/librx_ops.a` is the reference C backend
> built for x86 (`gcc`, no `CONFIG_BUILD_ADA300_QEMU`).  This is distinct
> from `build/librx_ops.a` which is the RISC-V Ada300 backend.

The system `llc` is found by probing `llc-18` … `llc-14` in
`/usr/lib/llvm-*/bin` (system LLVM with x86 target).  The Ada300 LLVM
toolchain (`rx-llvm-main/build/bin/llc`) is intentionally excluded because
it only targets RISC-V.

### Device Build Pipeline (`HETERO_DEVICE=ON`)

```
ada300_snpu_sdk/bsp/
  startup.S, uart.c, bsp_main.c, mr_heap.c, mini_printf.c
  └─ + device/ada300-hetero-main.c
  └─ + device/hetero_shmem_device.c
  └─ + rx_ops_bridge.c                  (#ifdef __riscv path: ada300_*→RXOPS_ADA300 hw,
  │                                       rxnn_c_init weak stub, no ivshmem headers)
  └─ + third_party/rx-ops/src/*.c          (RXOPS_ADA300 source)
  └─ + third_party/rx-ops/src/backend/ada300/**/*.S  (Ada300 ASM kernels)
       │
       └─ clang --target=riscv64-unknown-elf
              -march=rv64gcv_zfh_zvfh_zfbfmin0p8_zvfbfmin0p8_xadatmm_xadacv
              -mabi=lp64d -mcmodel=medany
              -ffreestanding -nostdlib -fuse-ld=lld
              -T ada300s_evk.ld
              → device/ada300-hetero-device.elf   (entry: 0x50000000)
```

---

## Execution Sequence Diagram

```
Host Process                          /dev/shm/ada300_bar           RISC-V Firmware
────────────                          ───────────────────           ───────────────
hetero_shmem_open("/ada300_bar")
  shm_open + mmap(MAP_SHARED)                                       boot → busy-poll ctrl->cmd

_mlir_ciface_subgraph0() ─────────────────────────────────────────►

  rxops_bridge_ada300_matmul_f32(fc1) ← device="ada300" → ivshmem → QEMU
    pack hdr+A+B → cmd_buffer
    ctrl->cmd = RUN             ─────────────────────────────────►  cmd==RUN detected
                                                                      status = RUNNING
                                                                      get_matmul(M,N,K,A,B)
                                                                      rxops_matmul(ADA300,fp16)
                                                                      memcpy C → result_buf
                                      status = DONE              ◄── signal_done()
    poll(status==DONE) ◄─────────────
    memcpy C ← result_buf
    return 0                                                          cmd = IDLE; loop

  rxops_bridge_ada300_add_f32(fc1_bias)  ← device="ada300" → ivshmem → QEMU
    pack hdr+in0+in1 → cmd_buffer
    ctrl->op_type = ADD
    ctrl->cmd = RUN          ─────────────────────────────────►  cmd==RUN detected
                                                                   op_type == ADD
                                                                   get_add(n, in0, in1)
                                                                   rxops_add(ADA300)
                                                                   memcpy out → result_buf
                               status = DONE              ◄──── signal_done()
    poll(status==DONE) ◄──────
    memcpy out ← result_buf
    return 0                                                       cmd = IDLE; loop

  rxops_bridge_exp_f32(hidden)         ← no device attr → local RXOPS_C, no IPC

  rxops_bridge_sqrt_f32(hidden)        ← no device attr → local RXOPS_C, no IPC

  rxops_bridge_ada300_matmul_f32(fc2) ← device="ada300" → ivshmem again
    …same handshake as above…                                         busy-poll → execute
                                      status = DONE              ◄──
    memcpy C ← result_buf

  rxops_bridge_ada300_add_f32(fc2_bias)  ← device="ada300" → ivshmem → QEMU
    …same handshake as above…

print output [1×64] ≈ 1.7627
hetero_shmem_close()
```

---

## Relationship to Real PCIe Hardware

This demo is a functional simulation of a PCIe-attached Ada300 board:

| Simulation (ivshmem) | Real Hardware (PCIe) |
|---|---|
| `shm_open` + `mmap(MAP_SHARED)` | `open(/dev/uio0)` or VFIO BAR mmap |
| `memory-backend-file` in QEMU | Physical LPDDR on Ada300 PCIe card |
| `ctrl->cmd = RUN` write | MMIO write to BAR0 doorbell register |
| Spin-poll `ctrl->status` | MSI/MSI-X interrupt or poll loop |
| Fixed offset `0x0200_1000` | DMA address negotiated at PCIe probe |
| `ADA300_IVSHMEM_BASE = 0x42000000` | Base address of PCIe BAR in RISC-V guest |

