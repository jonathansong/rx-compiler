# Ada300 Example

## Overview

This example demonstrates how to use Buddy Compiler to trace a PyTorch model
with **TorchDynamo** (`DynamoCompiler`), capture its compute graph, lower it
to MLIR through `GraphDriver`, and then compile it all the way to native code
via four distinct lowering paths:

- **Path A** — generic target-independent pipeline (`linalg` → SCF loops → LLVM IR)
- **Path B** — Ada300 inline-asm pipeline (`Ada300HL` → `Ada300HW` → `llvm.inline_asm`)
- **Path C** — Ada300 ISA backend pipeline (`Ada300HL` → `Ada300HW` → LLVM intrinsics → RISC-V binary)
- **Path D** — rx-ops library-call pipeline (`Ada300HL` → `llvm.call` into `rxops_bridge_*` C wrappers → `librx_ops.a`)

Path C is the recommended production path for Ada300 hardware. Path D provides
a portable host-executable path (x86/Linux) via the rx-ops reference C backend,
useful for functional validation without Ada300 hardware.

### Op mapping

| PyTorch op | Buddy graph IR node | MLIR dialect op (after import) |
|------------|---------------------|-------------------------------|
| `nn.Linear` (addmm) | `AddMMOp` | `linalg.matmul` |
| `torch.exp` | `ExpOp` | `math.exp` |
| `torch.sqrt` | `SqrtOp` | `math.sqrt` |

### Model architecture

```
input
  │
  ▼
fc1  ──  nn.Linear(in=64, hidden=128)   →  linalg.matmul
  │
  ▼
exp  ──  torch.exp                       →  math.exp
  │
  ▼
sqrt ──  torch.sqrt                      →  math.sqrt
  │
  ▼
fc2  ──  nn.Linear(hidden=128, out=64)  →  linalg.matmul
  │
  ▼
output
```

### Generated files

| File | Description |
|------|-------------|
| `subgraph0.mlir` | Compute subgraph in tensor-level MLIR (`linalg` + `math` dialect) |
| `forward.mlir` | Top-level dispatch function that calls `subgraph0` |
| `arg0.data` | Flattened float32 model weights (for AOT execution) |
| `subgraph0_ada300.ll` | LLVM IR produced by Path B (Ada300 inline-asm) |
| `subgraph0_ada300.o` | Native RISC-V object file from Path B (requires stock RISC-V assembler) |
| `stage7_isa_finished.mlir` | Ada300HW-dialect MLIR after finishing passes — input to Path C |
| `subgraph0_isa.ll` | LLVM IR with `@llvm.riscv.ada300_*` intrinsic calls (Path C) |
| `subgraph0_isa.s` | RISC-V assembly listing with Ada300 mnemonics (Path C, `ada300-asm` target) |
| `subgraph0_isa.o` | RISC-V ELF64 relocatable object with encoded Ada300 binary (Path C, `ada300-isa` target) |
| `stage_rxops_calls.mlir` | Ada300HL lowered to `llvm.call` bridge stubs (Path D, step 1) |
| `stage_rxops_finished.mlir` | Fully lowered LLVM dialect MLIR (Path D, step 2) |
| `subgraph0_rxops.ll` | LLVM IR with `rxops_bridge_*` call sites (Path D) |
| `subgraph0_rxops.o` | Native x86 object file from Path D |

---

## Prerequisites

### 1. Build LLVM/MLIR with Python bindings

```bash
cd buddy-mlir
mkdir llvm/build && cd llvm/build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3)
ninja check-clang check-mlir omp
```

### 2. Build buddy-mlir with Python packages

```bash
cd buddy-mlir
mkdir build && cd build
cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE=$(which python3)
ninja
ninja check-buddy
```

The ISA backend path (Path C) is enabled by default
(`BUDDY_ADA300_ISA_BACKEND=ON`, `BUDDY_ADA300_EXAMPLES=ON`).
It requires only the standard `buddy-llc` produced by the build above — no
extra toolchain is needed.

Path D (rx-ops) is enabled when `BUDDY_ADA300_RXOPS_PATH=ON` and
requires the rx-ops static library to be pre-built at
`thirdparty/rx-ops/build/librx_ops.a`. See
[Building rx-ops](#building-rx-ops-path-d-only) below.

### 3. Install Python dependencies

```bash
conda activate <your-env>
cd buddy-mlir
pip install -r requirements.txt
```

### 4. Set `PYTHONPATH`

```bash
cd buddy-mlir/build
export BUDDY_MLIR_BUILD_DIR=$PWD
export PYTHONPATH=${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

---

## Running the example

### Step 1 — Import: trace the model and generate MLIR

The import step is shared by all paths. It runs `ada300-import.py` via
`DynamoCompiler` and `GraphDriver` to produce `subgraph0.mlir`,
`forward.mlir`, and `arg0.data`.

**Via CMake (recommended):**

```bash
cd buddy-mlir/build
cmake -G Ninja .. -DBUDDY_ADA300_EXAMPLES=ON
```

The import runs automatically as a dependency of any lowering target below.

**Standalone Python (no CMake):**

```bash
cd buddy-mlir/examples/Ada300
python ada300-import.py --output-dir ./out
```

Available arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | `./` | Directory to write generated files |
| `--in-features` | `64` | Number of model input features |
| `--hidden-features` | `128` | Width of the hidden layer |
| `--out-features` | `64` | Number of model output features |

---

### Step 2 — Lowering: Path A (generic)

Lowers `subgraph0.mlir` through standard MLIR passes to a generic native
object file. No Ada300-specific dialect is involved.

```bash
cd buddy-mlir/build
ninja ada300-import
```

**Output:** `build/examples/Ada300/subgraph0.o`, `forward.o`

**Pipeline:**
```
subgraph0.mlir
  ├─ one-shot-bufferize            tensor → memref
  ├─ convert-linalg-to-loops       linalg.matmul → scf loops
  ├─ convert-math-to-llvm          math.exp / math.sqrt → LLVM intrinsics
  ├─ convert-math-to-libm          fallback to libm
  └─ (SCF / CF / arith / memref → LLVM finishing passes)
         │
         ▼ mlir-translate + llc
subgraph0.o
```

---

### Step 3 — Lowering: Path B (Ada300HL → Ada300HW → inline-asm)

Lowers `subgraph0.mlir` through the Ada300-specific dialect stack, emitting
RISC-V inline-asm strings for the ADA300 vector/tensor units.

```bash
cd buddy-mlir/build
ninja ada300-lower
```

**Output:** `build/examples/Ada300/subgraph0_ada300.ll`, `forward.o`

**Pipeline:**

```
subgraph0.mlir
  │  Stages 1–5  (same as Path C up to Ada300HW dialect — see below)
  │
  │  Stage 6 – lower-ada300hw-to-llvm
  │    ada300hw.vfpwnl  →  llvm.inline_asm "vfpwnl.exp.16 $0, $1"
  │    ada300hw.gmma_mm →  llvm.inline_asm "gmma.mm $0, $1, $2"
  │    ada300hw.tcsync  →  llvm.inline_asm "tcsync"
  │
  │  Stage 7 – finishing passes → LLVM dialect
  │
  ▼  mlir-translate -mlir-to-llvmir
subgraph0_ada300.ll   (LLVM IR with Ada300 instructions as inline asm strings)
```

---

### Step 4 — Lowering: Path C (Ada300HW → LLVM intrinsics → RISC-V binary) ✓ recommended

Lowers `subgraph0.mlir` through the Ada300 dialect stack but **skips
`lower-ada300hw-to-llvm`**, keeping `ada300hw.*` ops alive until
`buddy-translate --buddy-to-llvmir` maps them to
`@llvm.riscv.ada300_*` intrinsics. `buddy-llc` then encodes those intrinsics
as exact R-format machine binary using the TableGen patterns in
`RISCVInstrInfoAda300HW.td`.

**Produce the RISC-V object file:**

```bash
cd buddy-mlir/build
ninja ada300-isa
```

**Output:** `examples/Ada300/output/subgraph0_isa.o`
— ELF64 RISC-V relocatable, hard-float ABI, 13 Ada300 custom instructions.

**Produce a human-readable assembly listing:**

```bash
cd buddy-mlir/build
ninja ada300-asm
```

**Output:** `examples/Ada300/output/subgraph0_isa.s`

Sample from the assembly listing:
```asm
subgraph0:
        ada300.set_gmm_cfg   a0, zero
        ada300.set_gmm_type  zero, zero
        ada300.set_gmm_iter  s2, s2
        ada300.gmma_mm       s11, s10, s4
        ada300.tcsync
        ada300.vfpwnl        a3, zero, a5
        ...
```

**Full pipeline:**

```
subgraph0.mlir  (linalg.matmul + math.exp + math.sqrt, tensor level)
  │
  │  Stage 1 – tosa-to-linalg + convert-elementwise-to-linalg
  │
  │  Stage 2 – one-shot-bufferize
  │    All tensors → memref
  │
  │  Stage 3 – canonicalize + cse + normalize-memrefs
  │
  │  Stage 4a – math-to-ada300hl
  │    math.exp  linalg.generic  →  ada300hl.pwnl {func=exp,  segments=16}
  │    math.sqrt linalg.generic  →  ada300hl.pwnl {func=sqrt, segments=16}
  │
  │  Stage 4b – linalg-to-ada300hl
  │    linalg.matmul  →  ada300hl.copy_to_sram + ada300hl.tensor_mma
  │                      + ada300hl.tensor_sync
  │
  │  Stage 5 – lower-ada300hl-to-ada300hw
  │    ada300hl.pwnl        →  ada300hw.vfpwnl
  │    ada300hl.tensor_mma  →  ada300hw.set_gmm_cfg / set_gmm_type /
  │                             set_gmm_iter / gmma_mm
  │    ada300hl.tensor_sync →  ada300hw.tcsync
  │
  │  Stage 7 (ISA) – finishing passes WITHOUT lower-ada300hw-to-llvm
  │    convert-linalg-to-loops, expand-strided-metadata, lower-affine,
  │    convert-vector-to-llvm, convert-math-to-llvm, convert-math-to-libm,
  │    convert-scf-to-cf, convert-cf-to-llvm, convert-arith-to-llvm,
  │    finalize-memref-to-llvm, convert-func-to-llvm,
  │    reconcile-unrealized-casts
  │    (ada300hw.* ops survive into the output MLIR)
  │
  ▼  buddy-translate --buddy-to-llvmir
subgraph0_isa.ll   (LLVM IR with @llvm.riscv.ada300_* intrinsic calls)
  │
  ▼  buddy-llc -filetype=obj -mtriple=riscv64 -mattr=+buddyext,+v -float-abi=hard -O3
subgraph0_isa.o    (ELF64 RISC-V relocatable, Ada300 instructions binary-encoded)
  │
  ▼  buddy-llc -filetype=asm  (same flags)
subgraph0_isa.s    (human-readable RISC-V assembly with ada300.* mnemonics)
```

**Ada300 instruction encoding** (CUSTOM_3 opcode `0x7b`, func3=`0b011`):

| Instruction | func7 | Example encoding |
|-------------|-------|-----------------|
| `ada300.set_gmm_cfg` | 64 (`0b1000000`) | `8005307b` |
| `ada300.set_gmm_type` | 65 (`0b1000001`) | `8200307b` |
| `ada300.set_gmm_iter` | 66 (`0b1000010`) | `8529307b` |
| `ada300.gmma_mm` | 68 (`0b1000100`) | `894d3dfb` |
| `ada300.tcsync` | 73 (`0b1001001`) | `9200307b` |
| `ada300.vfpwnl` | 74 (`0b1001010`) | `94f036fb` |

---

### Step 5 — Lowering: Path D (Ada300HL → rx-ops library calls) ✓ host-portable

Path D lowers Ada300HL ops **directly** to `llvm.call` instructions targeting
flat C wrapper functions (`rxops_bridge_*`) that delegate to the rx-ops
operator library. It bypasses the Ada300HW dialect entirely, making the result
runnable on any x86 host using the reference C backend.

#### Building rx-ops (Path D only)

```bash
cd buddy-mlir/thirdparty/rx-ops
mkdir -p build && cd build
cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release
ninja
# Produces: thirdparty/rx-ops/build/librx_ops.a
```

#### Building and running Path D

```bash
cd buddy-mlir/build
cmake -G Ninja .. -DBUDDY_ADA300_RXOPS_PATH=ON
ninja ada300-rxops-run
```

This builds `build/examples/Ada300/ada300-rxops-runner`.

**Run inference:**

```bash
LD_LIBRARY_PATH=/path/to/buddy-mlir/llvm/build/lib:$LD_LIBRARY_PATH \
  ./examples/Ada300/ada300-rxops-runner \
  /path/to/buddy-mlir/examples/Ada300/output/arg0.data \
  1.0
```

Arguments:

| Argument | Description |
|----------|-------------|
| `arg0.data` | Path to the binary float32 params file (generated by `ada300-import.py`) |
| `input_value` | Optional float to fill the 1×64 input tensor (default `1.0`) |

**Example output:**

```
[Log] Loaded 16576 params from .../arg0.data
[Log] Running inference (input=1)...
[Log] Inference time: 0.19 ms
[Log] Output (1x64):
  [0.20243, 0.0211748, -0.556863, ...]
```

**Pipeline:**

```
subgraph0.mlir  (linalg.matmul + math.exp + math.sqrt, tensor level)
  │
  │  Stages 1–4b  (shared with Paths B/C — see above)
  │
  │  Path D Step 1 – lower-ada300hl-to-rxops
  │    ada300hl.pwnl {func=exp}   →  llvm.call @rxops_bridge_exp_f32(out, in, n)
  │    ada300hl.pwnl {func=sqrt}  →  llvm.call @rxops_bridge_sqrt_f32(out, in, n)
  │    ada300hl.tensor_mma        →  llvm.call @rxops_bridge_matmul_f32(C, A, B, M, N, K)
  │    ada300hl.tensor_sync       →  erased (sync handled inside the library)
  │    ada300hl.copy_to_sram      →  memref.copy (passthrough)
  │    ada300hl.copy_from_sram    →  memref.copy (passthrough)
  │
  │  Path D Step 2 – finishing passes
  │    convert-linalg-to-loops, convert-vector-to-llvm,
  │    finalize-memref-to-llvm, -llvm-request-c-wrappers,
  │    convert-func-to-llvm, reconcile-unrealized-casts
  │
  ▼  mlir-translate --mlir-to-llvmir
subgraph0_rxops.ll   (LLVM IR with rxops_bridge_* call sites)
  │
  ▼  llc -filetype=obj -relocation-model=pic -O3
subgraph0_rxops.o    (native x86-64 PIC object)
  │
  ▼  c++ link: subgraph0_rxops.o + rx_ops_bridge.o + forward.o
               + librx_ops.a + libmlir_c_runner_utils + -fopenmp
ada300-rxops-runner  (standalone x86-64 binary)
```

**Bridge sources** (in `runtime/Ada300/`):

| File | Description |
|------|-------------|
| `rx_ops_bridge.h` | Flat C ABI declarations for `rxops_bridge_*` wrappers |
| `rx_ops_bridge.c` | Implementations: fill `rxops_tensor` structs, call `rxops_*_init` + `rxops_*` |

The bridge uses `RXOPS_C` (reference C backend) for host execution. Swap to
`RXOPS_ADA300` and call `rxnn_ada300_init()` when targeting real Ada300
hardware.

**Key design note — `llvm.func` vs `func.func` for bridge declarations:**
The pass uses `LLVM::LLVMFuncOp` (i.e., `llvm.func`) rather than
`func::FuncOp` for bridge declarations. This ensures `-llvm-request-c-wrappers`
does **not** redirect calls to `_mlir_ciface_rxops_bridge_*` names — those
wrappers don't exist and would cause linker errors.

---

## Inspecting intermediate IR

After running either path you can inspect the generated files in
`examples/Ada300/output/` (or `build/examples/Ada300/` for Path A/B):

```bash
# Verify the three ops are present in the imported MLIR
grep -E "linalg.matmul|math.exp|math.sqrt" examples/Ada300/output/subgraph0.mlir

# Inspect the Ada300HW MLIR that feeds the ISA backend
cat examples/Ada300/output/stage7_isa_finished.mlir

# Inspect the LLVM IR with Ada300 intrinsic calls (Path C)
cat examples/Ada300/output/subgraph0_isa.ll

# View the assembly listing (Path C)
cat examples/Ada300/output/subgraph0_isa.s

# Inspect the Ada300 inline-asm LLVM IR (Path B)
grep -E "vfpwnl|gmma.mm|tcsync" examples/Ada300/output/subgraph0_ada300.ll
```

To stop the pipeline at a specific stage and inspect the intermediate MLIR,
run `buddy-opt` manually with only the passes up to that stage:

```bash
# Example: inspect IR after Stage 5 (Ada300HW dialect, before finishing passes)
buddy-opt examples/Ada300/output/subgraph0.mlir \
  -pass-pipeline="builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith),convert-elementwise-to-linalg)" \
  -one-shot-bufferize="bufferize-function-boundaries" \
  -ownership-based-buffer-deallocation \
  -buffer-deallocation-simplification \
  -bufferization-lower-deallocations \
  -canonicalize -cse -normalize-memrefs \
  -math-to-ada300hl \
  -linalg-to-ada300hl \
  -lower-ada300hl-to-ada300hw \
  -o stage5_ada300hw.mlir
```

---

## Implementation notes

### unrealized_conversion_cast handling

After `finalize-memref-to-llvm`, Ada300HW execute ops (`gmma_mm`, etc.) hold
`memref`-typed operands that are connected to their translated `!llvm.struct`
values via `builtin.unrealized_conversion_cast` bridge ops.
`reconcile-unrealized-casts` cannot remove these because the Ada300HW ops are
genuine memref users.

The fix lives entirely inside buddy-mlir — the `llvm/` submodule is unmodified:

- `BuddyBuiltinLLVMIRTranslationInterface` (in
  [midend/lib/Target/LLVMIR/Dialect/Ada300HW/Ada300HWToLLVMIRTranslation.cpp](../../midend/lib/Target/LLVMIR/Dialect/Ada300HW/Ada300HWToLLVMIRTranslation.cpp))
  overrides the standard builtin translation and forwards the already-translated
  `!llvm.struct` value to the cast result, so `lookupValue()` succeeds for the
  memref-typed operand.
- `registerAda300HWDialectTranslation()` is called **before**
  `registerAllToLLVMIRTranslations()` in
  [midend/lib/Target/LLVMIR/ConvertBuddyToLLVMIR.cpp](../../midend/lib/Target/LLVMIR/ConvertBuddyToLLVMIR.cpp),
  so our interface wins the `Dialect::addInterface` `try_emplace` race for the
  `BuiltinDialect` slot.

---

## References

- [ADA300 MLIR Dialect Design](../../docs/ADA300_MLIR_Dialect_Design.md)
- [Ada300HW LLVM ISA Backend](../../docs/Ada300HW_LLVM_ISA_Backend.md)
- [Buddy Compiler Python Environment](../../docs/PythonEnvironment.md)
- [rx-ops library](../../thirdparty/rx-ops/)
- [rx-ops bridge header](../../runtime/Ada300/rx_ops_bridge.h)
- [rx-ops bridge implementation](../../runtime/Ada300/rx_ops_bridge.c)
- [LowerAda300HLToRxOps pass](../../midend/lib/Conversion/LowerAda300HLToRxOps/LowerAda300HLToRxOpsPass.cpp)

