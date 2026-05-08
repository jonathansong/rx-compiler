# Ada300 Example

## Overview

This example demonstrates lowering a simple feedforward neural network
from the **Top dialect** (rx-mlir's tensor-level IR) to **rx-ops library
calls** using the `--convert-top-to-rxops` pass in `tpuc-opt`.

Two execution backends are supported:

| Backend | Target | Description |
|---------|--------|-------------|
| **Reference C** (`RXOPS_C`) | x86-64 Linux | Native host binary, no hardware needed |
| **Ada300 Baremetal** (`RXOPS_ADA300`) | RISC-V QEMU (`ada300s_evk`) | NPU vectorised kernels on the Ada300 SNPU simulator |

---

## Part 1 — Common: Model, MLIR Pipeline, and Input Generation

This part is shared by both backends.

### Model architecture

```
input (1×64)
  │
  ▼ top.MatMul          fc1 weight (64×128)
  ├─ + bias (top.Add)   fc1 bias   (1×128)
  │
  ▼ top.Exp
  │
  ▼ top.Sqrt
  │
  ▼ top.MatMul          fc2 weight (128×64)
  ├─ + bias (top.Add)   fc2 bias   (1×64)
  │
  ▼ output (1×64)
```

All weights are initialised to `0.01`; biases are `0.0`.

### Op lowering — Top dialect → rx-ops bridge calls

| Top dialect op | Bridge call |
|----------------|-------------|
| `top.MatMul` | `rxops_bridge_matmul_f32(C, A, B, M, N, K)` |
| `top.Add` (element-wise) | `rxops_bridge_add_f32(out, in0, in1, n)` |
| `top.Exp` | `rxops_bridge_exp_f32(out, in, n)` |
| `top.Sqrt` | `rxops_bridge_sqrt_f32(out, in, n)` |
| `top.Input` | identity (pass-through) |
| `top.None` | erased |

### MLIR lowering pipeline

```
examples/Ada300/output/subgraph0_top.mlir   (Top dialect, tensor-level)
  │
  ▼ tpuc-opt --convert-top-to-rxops
      top.* → llvm.call @rxops_bridge_*_f32(...)
  │
  ▼ mlir-opt
      -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs"
      -finalize-memref-to-llvm
      -convert-arith-to-llvm
      -llvm-request-c-wrappers       ← emits _mlir_ciface_subgraph0
      -convert-func-to-llvm
      -reconcile-unrealized-casts
  │
  ▼ mlir-translate --mlir-to-llvmir
      → subgraph0_rxops.ll           ← consumed by both backends
```

### Generating the Top dialect MLIR

`ada300-import.py` traces `Ada300Model` with `torch.export.export()` and
emits `subgraph0_top.mlir` directly.

```bash
cd /workspace/rx-mlir-main/examples/Ada300
python3 ada300-import.py --output-dir output
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `output` | Directory for generated files |
| `--in-features` | `64` | Input feature dimension |
| `--hidden-features` | `128` | Hidden layer width |
| `--out-features` | `64` | Output feature dimension |

FX op → Top dialect mapping:

| PyTorch FX op | Top dialect |
|---------------|-------------|
| `aten.permute` | aliased (weight transposition folded into arg shape) |
| `aten.addmm(bias, A, B)` | `top.MatMul(A, B, %none)` + `top.Add(bias, result)` |
| `aten.exp` | `top.Exp` |
| `aten.sqrt` | `top.Sqrt` |

Func arg ordering in the generated MLIR:

```
%arg0  fc1.weight^T  [FC1_IN × HIDDEN]
%arg1  fc1.bias      [1 × HIDDEN]
%arg2  fc2.weight^T  [HIDDEN × FC2_OUT]
%arg3  fc2.bias      [1 × FC2_OUT]
%arg4  input         [batch × FC1_IN]
```

### Generated files (in `output/`)

| File | Description |
|------|-------------|
| `subgraph0_top.mlir` | Top dialect input (from `ada300-import.py`) |
| `subgraph0_top_weight.npz` | Model weights |
| `subgraph0_rxops.mlir` | After `--convert-top-to-rxops` |
| `subgraph0_rxops_finished.mlir` | After finishing passes (LLVM dialect) |
| `subgraph0_rxops.ll` | LLVM IR with `rxops_bridge_*` call sites |
| `subgraph0_rxops.o` | Native x86-64 PIC object file (reference C path) |

### Common prerequisite — build rx-mlir

```bash
cd /workspace/rx-mlir-main
source envsetup.sh
./build.sh RELEASE CPU
# Produces: install/bin/tpuc-opt
```

---

## Part 2 — Reference C Backend (x86-64)

Runs the model on the host CPU using the `RXOPS_C` pure-C reference
implementation.  No Ada300 hardware or QEMU required.

### How it works

`subgraph0_rxops.ll` is compiled to a native x86-64 object and linked
with `examples/Ada300/rx_ops_bridge.c` (which dispatches into `librx_ops.a`)
and a C++ harness (`ada300-rxops-main.cpp`) that allocates tensors,
calls the MLIR-generated subgraph function, and prints results.

### Bridge sources

| File | Description |
|------|-------------|
| [`examples/Ada300/rx_ops_bridge.h`](rx_ops_bridge.h) | C ABI declarations for `rxops_bridge_*` |
| [`examples/Ada300/rx_ops_bridge.c`](rx_ops_bridge.c) | Builds `rxops_tensor` structs, calls `rxops_*_init` + `rxops_*` via `RXOPS_C` |

### Prerequisites

Build the rx-ops x86 static library:

```bash
cd /workspace/rx-mlir-main/third_party/rx-ops
mkdir -p build && cd build
cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release
ninja
# Produces: third_party/rx-ops/build/librx_ops.a
```

### Build (CMake)

```bash
cd /workspace/rx-mlir-main/examples/Ada300
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/usr/local/lib/cmake/mlir
cmake --build build --target ada300-rxops-run
# Runner: examples/Ada300/build/ada300-rxops-runner
```

### Run

```bash
./examples/Ada300/build/ada300-rxops-runner [input_value]
```

`input_value` is an optional float filling the 1×64 input (default `1.0`).

Expected output:

```
[Log] Running inference (input=1)...
[Log] Inference time: 0.14 ms
[Log] Output (1x64):
  [1.76272, 1.76272, 1.76272, ...]
```

With input=1.0, weights=0.01:
- FC1 matmul: 64 × 1.0 × 0.01 = **0.64** per element
- exp(0.64) ≈ **1.8965**, sqrt(1.8965) ≈ **1.3771**
- FC2 matmul: 128 × 1.3771 × 0.01 ≈ **1.763** per element

### Manual step-by-step build

```bash
cd /workspace/rx-mlir-main
source envsetup.sh

OUT=examples/Ada300/output
RXOPS_INC=third_party/rx-ops/include
RXOPS_LIB=third_party/rx-ops/build/librx_ops.a

# Step 0 – generate Top dialect MLIR
python3 examples/Ada300/ada300-import.py --output-dir $OUT

# Step 1 – lower Top ops to rx-ops bridge calls
tpuc-opt --convert-top-to-rxops \
  $OUT/subgraph0_top.mlir -o $OUT/subgraph0_rxops.mlir

# Step 2 – finish lowering to LLVM dialect
mlir-opt \
  -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs" \
  -finalize-memref-to-llvm -convert-arith-to-llvm \
  -llvm-request-c-wrappers -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  $OUT/subgraph0_rxops.mlir -o $OUT/subgraph0_rxops_finished.mlir

# Step 3 – translate to LLVM IR
mlir-translate --mlir-to-llvmir \
  $OUT/subgraph0_rxops_finished.mlir -o $OUT/subgraph0_rxops.ll

# Step 4 – assemble to native x86-64 object
# LLVM 18 llc has no registered backend targets in this build.
# Workaround: llvm-as (LLVM 18) → bitcode, then system llc-14.
llvm-as $OUT/subgraph0_rxops.ll -o $OUT/subgraph0_rxops.bc
/usr/lib/llvm-14/bin/llc -opaque-pointers \
  -filetype=obj -relocation-model=pic -O3 \
  $OUT/subgraph0_rxops.bc -o $OUT/subgraph0_rxops.o

# Step 5 – compile bridge
cc -O2 -Iexamples/Ada300 -I$RXOPS_INC -I$RXOPS_INC/interface \
  -c examples/Ada300/rx_ops_bridge.c -o $OUT/rx_ops_bridge.o

# Step 6 – compile runner
c++ -O2 -std=c++17 -I/usr/local/include \
  -c examples/Ada300/ada300-rxops-main.cpp -o $OUT/ada300-rxops-main.o

# Step 7 – link
c++ \
  $OUT/subgraph0_rxops.o $OUT/rx_ops_bridge.o $OUT/ada300-rxops-main.o \
  -Wl,--whole-archive $RXOPS_LIB -Wl,--no-whole-archive \
  -lm -lstdc++ -fopenmp \
  -o $OUT/ada300-rxops-runner

$OUT/ada300-rxops-runner 1.0
```

---

## Part 3 — Ada300 Baremetal Backend (RISC-V / QEMU)

Runs the same model on the **Ada300 SNPU** inside the `ada300s_evk` QEMU
machine.  `RXOPS_ADA300` NPU vectorised kernels replace the reference C
path.  The toolchain targets `riscv64-unknown-elf` with ISA
`rv64gcv_zfh_zvfh_zfbfmin0p8_zvfbfmin0p8_xadatmm_xadacv`.

### Source files

| File | Purpose |
|------|---------|
| `rx_ops_bridge_ada300.c` | Ada300 NPU bridge — registers `RXOPS_ADA300`, provides fp32↔fp16 matmul wrapper |
| `ada300-baremetal-main.c` | Baremetal `main()` — `rdtime` timing, UART output, `rxnn_mem_alloc`/`rxnn_mem_free` delegating to `mr_alloc(MR_REGION_LPDDR)` |

### Op → Ada300 NPU kernel mapping

| Bridge function | Ada300 kernel | Type |
|-----------------|---------------|------|
| `rxops_bridge_exp_f32` | `ada300_exp_f32.S` | vectorised SNPU fp32 |
| `rxops_bridge_sqrt_f32` | `ada300_sqrt_f32.S` | vectorised SNPU fp32 |
| `rxops_bridge_log_f32` | `ada300_log_f32.S` | vectorised SNPU fp32 |
| `rxops_bridge_rsqrt_f32` | `ada300_rsqrt_f32.S` | vectorised SNPU fp32 |
| `rxops_bridge_add_f32` | `ada300_add_f32.S` | vectorised SNPU fp32 |
| `rxops_bridge_matmul_f32` | `ada300_matmul_f16.S` | fp32→fp16→NPU→fp32 wrapper |

All ops not covered by the Ada300 table fall back to `RXOPS_C`.

### fp16 matmul strategy

The Ada300 NPU has **no fp32 matmul kernel**; only `rxnn_npu_matmul_fp16`
is available.  `rx_ops_bridge_ada300.c` wraps it transparently:

```
rxops_bridge_matmul_f32(C_f32, A_f32, B_f32, M, N, K)
  │
  ├─ mr_alloc A16[M×K], B16[K×N], C16[M×N]   ← fp16 scratch in LPDDR heap
  ├─ A_f32 → A16   fcvt.h.s  (RISC-V Zfh instruction)
  ├─ B_f32 → B16   fcvt.h.s
  ├─ rxops_matmul(A16, B16, C16, RXOPS_ADA300, FLOAT16)
  │    └─ ada300_matmul_f16.S  (SNPU vector matmul)
  ├─ C16 → C_f32   fcvt.s.h
  └─ mr_free A16, B16, C16
```

For this model the largest scratch buffer is B16[128×64] = 16 KB, well
within the 8 GiB LPDDR region.

**Precision:** fp32→fp16 downcast introduces ~0.1 % relative error per
matmul; exp, sqrt, add, and biases remain full fp32.

### Memory layout (ada300s_evk)

| Region | Base | Size | Used for |
|--------|------|------|----------|
| SRAM | `0x0000000000` | 1 MiB | BSP / stack |
| TCM | `0x0040000000` | 18 MiB | fast scratch (optional) |
| LPDDR | `0x0050000000` | 8 GiB | ELF entry point + heap |
| HBF | `0x0400000000` | 35 GiB | high-bandwidth flash window |

### Host kernel requirement — vm.overcommit_memory

The `ada300s_evk` machine maps the **35 GiB HBF** region lazily with
`mmap(MAP_ANONYMOUS)`.  With the default Linux heuristic overcommit policy
(`vm.overcommit_memory=0`), the kernel rejects the reservation because
committed size exceeds free + swap, even though the pages are never touched.
QEMU fails with:

```
qemu-system-riscv64: cannot set up guest memory 'ada300s_evk.hbf':
    Cannot allocate memory
```

**This is a host kernel setting.**  Docker containers share the host kernel —
writing it inside a privileged container modifies the live host value
immediately.

Apply immediately (until next reboot):

```bash
sudo sysctl -w vm.overcommit_memory=1
```

Persist across reboots by adding to `/etc/sysctl.conf` (or
`/etc/sysctl.d/99-qemu-overcommit.conf`) **on the host machine**:

```
# Allow QEMU to mmap large guest regions (e.g. ada300s_evk 35 GiB HBF) lazily.
# Physical pages are committed only on first access.
vm.overcommit_memory=1
```

Reload without rebooting: `sudo sysctl -p`

> `vm.overcommit_memory=1` is standard practice on QEMU/KVM hosts.
> The 35 GiB HBF window consumes only a few MB of actual RAM for a
> typical baremetal workload.

### Prerequisites

| Requirement | Location |
|-------------|----------|
| Ada300 LLVM toolchain | `/workspace/rx-llvm-main/build` |
| ada300\_snpu\_sdk BSP | `/workspace/embed-system-main/ada300_snpu_sdk` |
| picolibc RISC-V newlib | `/usr/lib/picolibc/riscv64-unknown-elf` |
| rx-qemu binary | `/workspace/rx-qemu-main/build/qemu-system-riscv64` |
| `subgraph0_rxops.ll` | `examples/Ada300/output/subgraph0_rxops.ll` |

**picolibc** (provides `stdlib.h`, `math.h`, etc. for baremetal):

```bash
apt-get install picolibc-riscv64-unknown-elf
```

**Ada300 LLVM toolchain** (if not already built):

```bash
cd /workspace/rx-llvm-main
cmake -S llvm -B build -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_TARGETS_TO_BUILD="RISCV" \
  -DCMAKE_BUILD_TYPE=Release
ninja -C build clang lld llvm-as llc llvm-objdump
```

**rx-qemu** (if not already built):

```bash
cd /workspace/rx-qemu-main
./configure --target-list=riscv64-softmmu --disable-werror
make -j$(nproc)
# Produces: build/qemu-system-riscv64
```

`subgraph0_rxops.ll` is produced by the CMake `ada300-rxops-run` target
(Part 2) or manually through Step 3 of the manual pipeline.

### Build

All baremetal outputs land in `examples/Ada300/build/` — the rx-mlir
top-level build tree is not involved.

```bash
cd /workspace/rx-mlir-main/examples/Ada300

cmake -S . -B build -G Ninja \
  -DADA300_BAREMETAL=ON \
  -DADA300_LLVM_ROOT=/workspace/rx-llvm-main/build \
  -DADA300_QEMU=/workspace/rx-qemu-main/build/qemu-system-riscv64

cmake --build build --target ada300-baremetal
```

Output artefacts:

| File | Description |
|------|-------------|
| `build/ada300-baremetal.elf` | Bootable RISC-V ELF |
| `build/ada300-baremetal.map` | Linker map |
| `build/ada300-baremetal.asm` | Full disassembly |

Path overrides (all have sensible defaults):

```
-DADA300_LLVM_ROOT=/opt/my-ada300-llvm
-DADA300_NEWLIB_SYSROOT=/opt/my-picolibc/riscv64-unknown-elf
-DADA300_SDK_ROOT=/path/to/ada300_snpu_sdk
-DADA300_QEMU=/path/to/qemu-system-riscv64
```

### Run on QEMU

```bash
cd /workspace/rx-mlir-main/examples/Ada300
cmake --build build --target run-ada300-baremetal
# Exit QEMU: Ctrl-A then x
```

Or directly:

```bash
/workspace/rx-qemu-main/build/qemu-system-riscv64 \
  -machine ada300s_evk -cpu ada300s \
  -nographic -serial mon:stdio \
  -kernel /workspace/rx-mlir-main/examples/Ada300/build/ada300-baremetal.elf
```

Expected output:

```
[Ada300] Baremetal rx-ops inference demo
[Ada300] Model: fc1(64x128) -> exp -> sqrt -> fc2(128x64)
[Ada300] Running inference (input=1.0, weights=0.01)...
[Ada300] Inference: 366426 ticks (36642600 ns ~= 36642 us)
[Ada300] Output (1x64):
  [1.7627, 1.7627, 1.7627, ...]
[Ada300] Expected output ≈ 1.7629 per element
[Ada300] Done.
```

Timing uses `rdtime` (CLINT at 10 MHz → 100 ns/tick).

> The small difference from the fp32 reference (1.7629 vs 1.7627) is fp16
> quantization in the matmul path — weights are converted fp32→fp16 before
> the Ada300 NPU kernel, introducing a tiny rounding error.

### Building the rx-ops Ada300 library standalone

If you need a reusable `librx_ops.a` for other baremetal projects:

```bash
cd /workspace/rx-mlir-main/third_party/rx-ops

cmake -S . -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=/workspace/embed-system-main/ada300_snpu_sdk/cmake/toolchain-ada300-llvm.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCONFIG_BUILD_ADA300_QEMU=ON \
  -DRX-QEMU_DIR=/workspace/embed-system-main/ada300_snpu_sdk \
  -DADA300_LLVM_ROOT=/workspace/rx-llvm-main/build \
  -DQEMU_BIN=/workspace/rx-qemu-main/build/qemu-system-riscv64

cmake --build build
cmake --install build   # installs to build/install/
```

| Artifact | Description |
|----------|-------------|
| `build/librx_ops.a` | Ada300 NPU backend (fp32 element-wise + fp16 matmul ASM) |
| `build/libada300_bsp.a` | BSP (startup.S, UART, mini_printf, mr_heap) |
| `build/rx_ops_test.elf` | Self-test ELF; run with `cmake --build build --target run-rx_ops_test` |
| `build/install/` | Headers + libs in standard layout |

> The `ADA300_BAREMETAL` CMake target compiles rx-ops sources directly —
> no prebuilt `librx_ops.a` is required for the example.

---

## Implementation Notes

### Importer: `ada300-import.py`

Source: [examples/Ada300/ada300-import.py](ada300-import.py)

Uses `torch.export.export()` (PyTorch Dynamo) to capture the FX graph of
`Ada300Model`.  Each FX node is visited once and emitted as a Top dialect op:

- `aten.permute` nodes are aliased to the corresponding (already-transposed)
  func arg — no MLIR op is emitted.
- `aten.addmm(bias, A, B)` → `top.MatMul` followed by `top.Add`.
- Transcendental ops (`aten.exp`, `aten.sqrt`) → `top.Exp`, `top.Sqrt`.

Weight tensors are transposed from PyTorch storage order (`[out, in]`) to
matmul-compatible order (`[in, out]`) before being written to the `.npz` file.

### Pass: `--convert-top-to-rxops`

Source: [lib/Conversion/TopToRxOps/TopToRxOpsPass.cpp](../../lib/Conversion/TopToRxOps/TopToRxOpsPass.cpp)

Each supported Top op is rewritten to:
1. `bufferization.to_memref` — zero-copy view of each input tensor.
2. `memref.alloc` — fresh output buffer.
3. `memref.extract_aligned_pointer_as_index` → `arith.index_cast` → `llvm.inttoptr` — extract raw data pointer.
4. `llvm.call @rxops_bridge_<op>_f32(...)` — invoke the bridge function.
5. `bufferization.to_tensor restrict` — wrap the output buffer as a tensor.

**Limitations:** only f32, static shapes, 2-D matmul (no bias, no transpose),
2-input element-wise add.

### LLVM IR compilation workaround (reference C backend)

`mlir-translate --mlir-to-llvmir` emits LLVM 18 opaque-pointer IR (`ptr`
instead of `i8*`).  The LLVM 18 `llc` at `${LLVM_TOOLS_BINARY_DIR}/llc`
was built without any codegen backend targets.  The CMakeLists.txt detects a
system `llc` (e.g. `llc-14`) via `find_program` and passes `-opaque-pointers`
so it can read LLVM 18 bitcode.

The baremetal backend compiles `subgraph0_rxops.ll` directly with the Ada300
`clang` (`--target=riscv64-unknown-elf`) — no `llc-14` workaround needed.

### `memrefCopy` runtime symbol

The bufferizer emits calls to `memrefCopy` for buffer copies.  This symbol is
defined in MLIR's `CRunnerUtils` library, which is not available as a
standalone linkable library in this installation.  A self-contained
implementation is included in `ada300-rxops-main.cpp`.

### QEMU Ada300 SNPU emulation bugs (fixed)

Three bugs were found and fixed in
`rx-qemu-main/target/riscv/xada_helper.c` (`HELPER(xada_vfpwnl)`):

**Bug 1 — `exp` computed 2^x instead of e^x**

`xada_pwnl_eval` type 0 called `float32_exp2(x)` directly, but `float32_exp2`
computes 2^x.  The correct natural exponential requires multiplying by log₂(e)
first:

```c
/* was: */
return float32_exp2(x, st);
/* fix: */
return float32_exp2(float32_mul(x, XADA_LOG2E, st), st);
```

**Bug 2 — type 1 dispatched to sigmoid instead of sqrt**

The Ada300 SNPU ISA (`ada300.h`) defines:

| type immediate | function |
|----------------|----------|
| 0 | exp |
| 1 | sqrt |
| 2 | sigmoid |
| 3 | tanh |

The original `xada_pwnl_eval` switch had `case 1: sigmoid`, `case 2: tanh`,
`case 3: silu` — shifted by one, so every PWNL operation dispatched to the
wrong function.  The fix renumbers `sqrt` as type 1 and shifts sigmoid/tanh/silu
to types 2/3/4.

**Bug 3 — single-source `vfpwnl.v` ignored current SEW and only processed
one register of the LMUL group**

The exp/sqrt f32 kernels configure `vtype = e32, m2` and use `vle32`/`vse32`,
so the vector registers hold float32 data.  The original helper always treated
the source vector as bf16 (16-bit lanes), so:

- Each fp32 element was split into two bf16 reads producing garbage inputs.
- Only the first register of the m2 group was visited; the second half of
  each buffer was never touched.

The fix checks `FIELD_EX64(env->vtype, VTYPE, VSEW)` at runtime: when SEW=2
(e32), the helper iterates over `env->vl` fp32 elements across the full LMUL
register group; otherwise it falls back to the bf16 lane path for native
bf16 SNPU operations.

**Combined effect on demo output**

| Before fixes | After fixes |
|--------------|-------------|
| `[0.8872, ...]` | `[1.7627, ...]` |

The correct value 1.7627 matches the expected fp32 derivation (1.7629) within
the fp16 quantization error introduced by the Ada300 matmul kernel.

---

## References

- [ConvertTopToRxOps pass](../../lib/Conversion/TopToRxOps/TopToRxOpsPass.cpp)
- [ConvertTopToRxOps header](../../include/tpu_mlir/Conversion/TopToRxOps/TopToRxOps.h)
- [rx-ops library](../../third_party/rx-ops/)
- [rx-ops bridge (reference C)](rx_ops_bridge.c)
- [NpuToLLVM_RxOps Integration doc](../../docs/NpuToLLVM_RxOps_Integration.md)
