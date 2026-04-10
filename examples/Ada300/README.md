# Ada300 Example

## Overview

This example demonstrates how to use Buddy Compiler to trace a PyTorch model
with **TorchDynamo** (`DynamoCompiler`), capture its compute graph, lower it
to MLIR through `GraphDriver`, and then compile it all the way to native code
via two distinct lowering paths:

- **Path A** — generic target-independent pipeline (`linalg` → SCF loops → LLVM IR)
- **Path B** — Ada300-specific pipeline through the `Ada300HL` and `Ada300HW` dialects

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
| `subgraph0_ada300.ll` | LLVM IR produced by the Ada300HL/Ada300HW pipeline (Path B) |
| `subgraph0_ada300.o` | Native RISC-V object file from Path B (requires Ada300-enabled toolchain — see `ada300-assemble`) |

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

The import step is shared by both paths. It runs `ada300-import.py` via
`DynamoCompiler` and `GraphDriver` to produce `subgraph0.mlir`,
`forward.mlir`, and `arg0.data`.

**Via CMake (recommended):**

```bash
cd buddy-mlir/build
cmake -G Ninja .. -DBUDDY_ADA300_EXAMPLES=ON
```

The import runs automatically as a dependency of either lowering target below.

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

### Step 3 — Lowering: Path B (Ada300HL → Ada300HW → LLVM IR)

Lowers `subgraph0.mlir` through the Ada300-specific dialect stack, emitting
RISC-V inline-asm instructions for the ADA300 vector/tensor units.

```bash
cd buddy-mlir/build
ninja ada300-lower
```

**Output:** `build/examples/Ada300/subgraph0_ada300.ll`, `forward.o`

> **Note — assembling to `.o`:** The custom Ada300 instruction mnemonics
> (`gmm.cfg`, `gmma.mm`, `tcsync`, `vfpwnl`) are not yet registered in the
> BuddyExt RISC-V LLVM backend (`RISCVInstrInfoBuddyExt.td`), so a stock
> RISC-V assembler will reject them.  Once those TableGen entries are added
> and `buddy-llc` is rebuilt, use the separate target:
> ```bash
> ninja ada300-assemble   # produces subgraph0_ada300.o
> ```

**Pipeline:**

```
subgraph0.mlir  (linalg.matmul + math.exp + math.sqrt, tensor level)
  │
  │  Stage 1 – tosa-to-linalg + convert-elementwise-to-linalg
  │    tosa.transpose → linalg.transpose
  │    math.exp  %tensor  →  linalg.generic (scalar body: math.exp)
  │    math.sqrt %tensor  →  linalg.generic (scalar body: math.sqrt)
  │
  │  Stage 2 – one-shot-bufferize
  │    All tensors → memref
  │    (+ ownership-based-buffer-deallocation / deallocation-simplification /
  │       bufferization-lower-deallocations)
  │
  │  Stage 3 – canonicalize + cse + normalize-memrefs
  │    Folds constant strides; rewrites non-identity affine-mapped memrefs
  │    to plain memrefs + explicit index arithmetic so Ada300HL passes see
  │    clean identity-layout operands.
  │
  │  Stage 4a – math-to-ada300hl
  │    math.exp  (scalar)  →  ada300hl.pwnl {func=exp,  segments=16}
  │    math.sqrt (scalar)  →  ada300hl.pwnl {func=sqrt, segments=16}
  │
  │  Stage 4b – linalg-to-ada300hl
  │    linalg.matmul →  ada300hl.copy_to_sram  %A, %A_sram
  │                     ada300hl.copy_to_sram  %B, %B_sram
  │                     ada300hl.tensor_mma    %C, %A_sram, %B_sram
  │                     ada300hl.tensor_sync
  │
  │  Stage 5 – lower-ada300hl-to-ada300hw
  │    ada300hl.pwnl       →  ada300hw.vfpwnl
  │    ada300hl.tensor_mma →  ada300hw.set_gmm_cfg / set_gmm_type /
  │                            set_gmm_iter / gmma_mm
  │    ada300hl.tensor_sync→  ada300hw.tcsync
  │
  │  Stage 6 – lower-ada300hw-to-llvm
  │    ada300hw.vfpwnl     →  llvm.inline_asm "vfpwnl.exp.16 $0, $1"
  │    ada300hw.gmma_mm    →  llvm.inline_asm "gmma.mm $0, $1, $2"
  │    ada300hw.tcsync     →  llvm.inline_asm "tcsync"
  │
  │  Stage 7 – finishing passes
  │    convert-linalg-to-loops    remaining linalg.generic / linalg.transpose
  │                                → scf loops
  │    expand-strided-metadata
  │    lower-affine
  │    convert-vector-to-llvm
  │    convert-math-to-llvm        scalar math → LLVM intrinsics
  │    convert-math-to-libm        fallback libm calls
  │    convert-scf-to-cf
  │    convert-cf-to-llvm
  │    convert-arith-to-llvm
  │    finalize-memref-to-llvm
  │    convert-func-to-llvm
  │    reconcile-unrealized-casts
  │
  ▼ mlir-translate -mlir-to-llvmir
subgraph0_ada300.ll   (LLVM IR with Ada300 instructions as inline asm)
  │
  ▼ (ada300-assemble target — requires Ada300-enabled buddy-llc)
  llvm-as | buddy-llc -march=riscv64 -mattr=+v,+buddyext
subgraph0_ada300.o    (RISC-V native object file)
```

---

## Inspecting intermediate IR

After running either path you can inspect the generated files in
`build/examples/Ada300/`:

```bash
# Verify the three ops are present in the imported MLIR
grep -E "linalg.matmul|math.exp|math.sqrt" build/examples/Ada300/subgraph0.mlir

# Inspect the Ada300-lowered LLVM IR (inline asm visible here)
cat build/examples/Ada300/subgraph0_ada300.ll

# Check for Ada300 inline asm instructions
grep -E "vfpwnl|gmma.mm|tcsync" build/examples/Ada300/subgraph0_ada300.ll
```

To stop the pipeline at a specific stage and inspect the intermediate MLIR,
run `buddy-opt` manually with only the passes up to that stage:

```bash
# Example: inspect IR after Stage 4b (Ada300HL dialect)
buddy-opt build/examples/Ada300/subgraph0.mlir \
  -pass-pipeline="builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith),convert-elementwise-to-linalg)" \
  -one-shot-bufferize="bufferize-function-boundaries" \
  -ownership-based-buffer-deallocation \
  -buffer-deallocation-simplification \
  -bufferization-lower-deallocations \
  -canonicalize -cse -normalize-memrefs \
  -math-to-ada300hl \
  -linalg-to-ada300hl \
  -o stage4b_ada300hl.mlir
```

---

## References

- [ADA300 MLIR Dialect Design](../../docs/ADA300_MLIR_Dialect_Design.md)
- [Buddy Compiler Python Environment](../../docs/PythonEnvironment.md)
