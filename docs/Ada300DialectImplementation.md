# ADA300 Dialect Implementation Guide

This document describes the implementation of the two ADA300 MLIR dialects
(`Ada300HL` and `Ada300HW`) and the four conversion passes that connect them
to the broader MLIR/LLVM pipeline.

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Ada300HL Dialect](#ada300hl-dialect)
   - [Attributes](#attributes)
   - [Types](#types)
   - [Memory Constraints](#memory-constraints)
   - [Operations](#ada300hl-operations)
4. [Ada300HW Dialect](#ada300hw-dialect)
   - [Operations](#ada300hw-operations)
5. [Conversion Passes](#conversion-passes)
   - [MathToAda300HL](#1-mathtoadabl300hl)
   - [LinalgToAda300HL](#2-linalgtoadabl300hl)
   - [LowerAda300HLToAda300HW](#3-lowerada300hltoadabl300hw)
   - [LowerAda300HWToLLVM](#4-lowerada300hwtollvm)
6. [Full Pipeline](#full-pipeline)
7. [CMake Integration](#cmake-integration)

---

## Overview

The ADA300 ISA is a RISC-V 64IF + RVV v1.0 based processor with two key
hardware subsystems:

- **Vector unit** – RVV-style register-to-register ops including a
  piecewise-linear nonlinear approximation instruction (`vfpwnl`) and
  mixed-precision arithmetic.
- **Tensor Core** – A long-latency SRAM-based matrix accelerator driven by
  hardware configuration registers (`gmm_cfg`, `gmm_type`, `gmm_iter`) and
  execute instructions (`gmma_mm`, `gmma_mt`, etc.).

To support optimisation and code generation for both subsystems, two MLIR
dialects are layered as follows:

```
math / arith / vector / linalg
        │
        ▼  MathToAda300HL / LinalgToAda300HL
   ada300hl  ← high-level hardware-aware ops (optimisable)
        │
        ▼  LowerAda300HLToAda300HW
   ada300hw  ← ISA-level ops (direct hardware protocol)
        │
        ▼  LowerAda300HWToLLVM
   LLVM dialect (llvm.inline_asm)
        │
        ▼
   RISC-V / ADA300 backend
```

---

## Directory Structure

```
midend/
  include/
    Dialect/
      Ada300HL/
        Ada300HLDialect.td     # Dialect registration + base Op class
        Ada300HLAttrs.td       # 7 enum attrs + LayoutAttr
        Ada300HLTypes.td       # Custom Fp8Type (E4M3)
        Ada300HLMemory.td      # Memory-space type constraints
        Ada300HLOps.td         # Root tablegen file; includes all sub-files
        Ada300HLDialect.h      # C++ dialect header
        Ada300HLOps.h          # C++ op declarations
        Transforms.h           # Public API for lowering passes
      Ada300HW/
        Ada300HWDialect.td     # Dialect registration + base Op class
        Ada300HWAttrs.td       # Re-exports Ada300HL attrs (shared namespace)
        Ada300HWOps.td         # Root tablegen file; 14 ISA-level ops
        Ada300HWDialect.h      # C++ dialect header
        Ada300HWOps.h          # C++ op declarations
        Transforms.h           # Public API for Ada300HW → LLVM pass
  lib/
    Dialect/
      Ada300HL/
        Ada300HLDialect.cpp    # Dialect registration (ops + types + attrs)
        Ada300HLAttrs.cpp      # Enum + AttrDef implementations
        Ada300HLTypes.cpp      # Fp8Type implementation
        Ada300HLOps.cpp        # Op verifiers
      Ada300HW/
        Ada300HWDialect.cpp    # Dialect registration
        Ada300HWOps.cpp        # Op verifiers
    Conversion/
      MathToAda300HL/
        MathToAda300HLPass.cpp
        CMakeLists.txt
      LinalgToAda300HL/
        LinalgToAda300HLPass.cpp
        CMakeLists.txt
      LowerAda300HLToAda300HW/
        LowerAda300HLToAda300HWPass.cpp
        CMakeLists.txt
      LowerAda300HWToLLVM/
        LowerAda300HWToLLVMPass.cpp
        CMakeLists.txt
```

---

## Ada300HL Dialect

**Namespace:** `::buddy::ada300hl`  
**Mnemonic prefix:** `ada300hl`  
**Root tablegen file:** `Ada300HLOps.td`

The high-level dialect captures hardware-related semantics while deferring
ISA-specific encoding details.  Its ops are amenable to pattern rewriting,
fusion, and cost-model-based selection before being lowered to `Ada300HW`.

### Attributes

All attributes are defined in `Ada300HLAttrs.td` and share the
`::buddy::ada300hl` C++ namespace.  Ada300HW ops reuse them directly (the
`Ada300HWAttrs.td` file is a thin re-export).

| Attribute | Mnemonic | Values | Purpose |
|---|---|---|---|
| `NonlinearFuncAttr` | `#ada300hl.nlfunc<...>` | `exp`, `log`, `sqrt`, `rsqrt`, `div` | Selects the PWL function kind |
| `SegmentCountAttr` | `#ada300hl.segments<...>` | `16`, `32` | Number of PWL approximation segments |
| `TensorModeAttr` | `#ada300hl.tensor_mode<...>` | `inner`, `outer` | Tensor Core accumulation mode |
| `TensorDataTypeAttr` | `#ada300hl.dtype<...>` | `fp16`, `bf16`, `int16`, `fp8`, `int8`, `int4` | Operand data type for gmm/gmv ops |
| `MemorySpaceAttr` | `#ada300hl.memory_space<...>` | `global`, `sram`, `vr` | Memory domain annotation |
| `PartAttr` | `#ada300hl.part<...>` | `low`, `high` | Sub-vector half selection |
| `MajorAttr` | `#ada300hl.major<...>` | `row_major`, `col_major`, `blocked` | Matrix storage order |
| `LayoutAttr` | `#ada300hl.layout<...>` | `{block_m, block_n, block_k, rhs_transposed, major}` | Tile layout for pack/unpack |

### Types

Defined in `Ada300HLTypes.td`:

| Type | Mnemonic | Description |
|---|---|---|
| `Fp8Type` | `!ada300hl.fp8<e4m3>` | 8-bit float, E4M3 format, for mixed-precision paths |

Standard MLIR types (`f16`, `bf16`, `f32`, `i8`, `i16`, `vector<...>`,
`memref<...>`) are reused without wrapping.

### Memory Constraints

`Ada300HLMemory.td` defines tablegen type constraint aliases used in op
signatures to document which memory domain a memref operand must come from:

| Constraint | Meaning |
|---|---|
| `Ada300HL_GlobalMemRef` | `memref<...>` in global DRAM |
| `Ada300HL_SRAMMemRef` | `memref<..., #ada300hl.memory_space<sram>>` |
| `Ada300HL_VRMemRef` | `memref<..., #ada300hl.memory_space<vr>>` |
| `Ada300HL_AnyMemRef` | Any of the above |

### Ada300HL Operations

#### Vector Nonlinear / PWL Ops

| Op | Traits | Description | Example |
|---|---|---|---|
| `ada300hl.exp` | `Pure` | Element-wise `exp(x)` via PWL unit | `%y = ada300hl.exp %x {segments = #ada300hl.segments<16>} : vector<64xf16> -> vector<64xf16>` |
| `ada300hl.pwnl` | `Pure` | Generalised PWL (exp/log/sqrt/rsqrt/div) | `%y = ada300hl.pwnl %x {func = #ada300hl.nlfunc<sqrt>, segments = #ada300hl.segments<32>} : ...` |
| `ada300hl.cvt` | `Pure` | Mixed-precision type conversion | `%y = ada300hl.cvt %x {dst_type = #ada300hl.dtype<fp8>, part = #ada300hl.part<low>} : ...` |
| `ada300hl.vmul_mixed` | `Pure` | Mixed-precision vector multiply | `%z = ada300hl.vmul_mixed %a, %b {acc_type = #ada300hl.dtype<fp16>, part = #ada300hl.part<low>} : ...` |

#### Tensor Core Ops

| Op | Traits | Description |
|---|---|---|
| `ada300hl.tensor_mma` | `MemRead`, `MemWrite` | Full-semantic Tensor Core matmul. Carries mode, dimensions, data types, and block counts so that later passes can make informed lowering decisions. |
| `ada300hl.tensor_sync` | `MemRead`, `MemWrite` | Abstract barrier waiting for Tensor Core completion. |

`tensor_mma` attributes:

| Attribute | Type | Meaning |
|---|---|---|
| `mode` | `TensorModeAttr` | `inner` or `outer` product |
| `rhs_transposed` | `bool` | Whether to use `gmma_mt` (transposed weights) |
| `row_size`, `col_size`, `acc_size` | `i32` | Matrix tile dimensions |
| `out_type`, `act_type`, `wht_type` | `TensorDataTypeAttr` | Hardware data types |
| `blk_cnt_a`, `blk_cnt_w` | `i32` | SRAM block iteration counts |

#### Layout / Data-Movement Ops

| Op | Description |
|---|---|
| `ada300hl.pack` | Reformats a flat matrix buffer into the blocked layout the Tensor Core requires. |
| `ada300hl.unpack` | Inverse of `pack`; converts blocked layout back to flat. |
| `ada300hl.layout_cast` | View-like reinterpretation of layout; no data movement. |
| `ada300hl.copy_to_sram` | Copies a buffer from global/VR memory into on-chip SRAM. |
| `ada300hl.copy_from_sram` | Copies a buffer from SRAM back to global/VR memory. |
| `ada300hl.copy_to_vr` | Copies into vector-register-backed storage. |
| `ada300hl.copy_from_vr` | Copies out of vector-register-backed storage. |

---

## Ada300HW Dialect

**Namespace:** `::buddy::ada300hw`  
**Mnemonic prefix:** `ada300hw`  
**Root tablegen file:** `Ada300HWOps.td`

The low-level ISA dialect models the ADA300 hardware protocol directly.  Its
ops have a 1:1 or near-1:1 correspondence to machine instructions and must be
preserved by the code generator.

Attributes are shared with Ada300HL — `Ada300HWAttrs.td` simply includes
`Ada300HLAttrs.td`, so the same `#ada300hl.*` attribute syntax is used in both
dialects.

### Ada300HW Operations

#### Vector Ops

| Op | Traits | ISA Instruction | Description |
|---|---|---|---|
| `ada300hw.vfpwnl` | `Pure` | `vfpwnl.<func>.<segs>` | PWL nonlinear function. `func` and `segments` select the approximation table. |
| `ada300hw.vfcvt` | `Pure` | `vfcvt.<part>.*` | Floating-point precision conversion (low/high half). |
| `ada300hw.vfmul_low` | `Pure` | `vfmul.low.*` | Mixed-precision multiply into low sub-vector. |
| `ada300hw.vfmul_high` | `Pure` | `vfmul.high.*` | Mixed-precision multiply into high sub-vector. |

#### Tensor Core Configuration Ops

These ops write hardware configuration registers and must not be reordered or
eliminated.  They always precede the corresponding execute op.

| Op | Register | Key Attributes |
|---|---|---|
| `ada300hw.set_gmm_cfg` | `gmm_cfg` | `row_size`, `col_size`, `acc_size`, `mode` |
| `ada300hw.set_gmm_type` | `gmm_type` | `out_type`, `act_type`, `wht_type` |
| `ada300hw.set_gmm_iter` | `gmm_iter` | `blk_cnt_a`, `blk_cnt_w` |

#### Tensor Core Execute Ops

All execute ops carry `MemRead` + `MemWrite` side effects.

| Op | ISA Instruction | Description |
|---|---|---|
| `ada300hw.gmm_mm` | `gmm.mm` | Non-accumulating matrix multiply (clears dst). |
| `ada300hw.gmma_mm` | `gmma.mm` | Accumulating matrix multiply (adds into dst). |
| `ada300hw.gmma_mt` | `gmma.mt` | Accumulating matmul, weight matrix transposed. |
| `ada300hw.gmv_mm` | `gmv.mm` | Non-accumulating matrix-vector multiply. |
| `ada300hw.gmva_mm` | `gmva.mm` | Accumulating matrix-vector multiply. |
| `ada300hw.gmva_mt` | `gmva.mt` | Accumulating mat-vec multiply, matrix transposed. |

#### Synchronisation

| Op | ISA Instruction | Description |
|---|---|---|
| `ada300hw.tcsync` | `tcsync` | Polls the Tensor Core completion flag. Blocks until the hardware signals done. |

---

## Conversion Passes

### 1. MathToAda300HL

**Pass argument:** `--math-to-ada300hl`  
**Source file:** `midend/lib/Conversion/MathToAda300HL/MathToAda300HLPass.cpp`  
**Pass class:** `MathToAda300HLPass` (operates on `func::FuncOp`)

Replaces standard `math` dialect ops that have direct ADA300 vector unit
support with the corresponding high-level `ada300hl` ops.  Only vector-typed
operands are lowered; scalar ops are left for the standard arith pipeline.

| Input | Output | Notes |
|---|---|---|
| `math.exp %v` | `ada300hl.exp %v {segments = 16}` | Default 16-segment table |
| `math.log %v` | `ada300hl.pwnl %v {func=log, segments=16}` | |
| `math.sqrt %v` | `ada300hl.pwnl %v {func=sqrt, segments=16}` | |
| `math.rsqrt %v` | `ada300hl.pwnl %v {func=rsqrt, segments=16}` | |

The default segment count (16) is conservative.  A subsequent tuning pass can
increase it to 32 for higher-accuracy applications.

---

### 2. LinalgToAda300HL

**Pass argument:** `--linalg-to-ada300hl`  
**Source file:** `midend/lib/Conversion/LinalgToAda300HL/LinalgToAda300HLPass.cpp`  
**Pass class:** `LinalgToAda300HLPass` (operates on `func::FuncOp`)

Converts bufferized `linalg.matmul` to the Ada300HL Tensor Core sequence.
The IR must be fully bufferized (memref operands) before this pass runs.

**Expansion pattern** for `linalg.matmul ins(%A, %B) outs(%C)`:

```mlir
%A_sram = memref.alloc(...)                    // SRAM staging buffer
%B_sram = memref.alloc(...)
ada300hl.copy_to_sram %A, %A_sram              // Stage A into SRAM
ada300hl.copy_to_sram %B, %B_sram              // Stage B into SRAM
ada300hl.tensor_mma %C, %A_sram, %B_sram {
    mode         = inner,
    rhs_transposed = false,
    row_size     = <M (static) or 16 (dynamic)>,
    col_size     = <N (static) or 16 (dynamic)>,
    acc_size     = <K (static) or 16 (dynamic)>,
    out_type     = <inferred from C element type>,
    act_type     = <inferred from A element type>,
    wht_type     = <inferred from B element type>,
    blk_cnt_a    = 1,
    blk_cnt_w    = 1
}
ada300hl.tensor_sync
memref.dealloc %A_sram
memref.dealloc %B_sram
```

Matrix dimensions are encoded directly when statically known; otherwise the
conservative default (16) is used.  Block counts start at 1 and can be
increased by a tiling pass.

SRAM buffer addresses are resolved by a downstream memory-planning pass; at
this stage they are plain `memref.alloc`s.

---

### 3. LowerAda300HLToAda300HW

**Pass argument:** `--lower-ada300hl-to-ada300hw`  
**Source file:** `midend/lib/Conversion/LowerAda300HLToAda300HW/LowerAda300HLToAda300HWPass.cpp`  
**Pass class:** `LowerAda300HLToAda300HWPass` (operates on `func::FuncOp`)

Lowers every `ada300hl` op to the equivalent `ada300hw` ISA-level op or op
sequence.  After this pass the function body contains no `ada300hl` ops.

#### Lowering Table

| Ada300HL op | Ada300HW output | Notes |
|---|---|---|
| `ada300hl.exp` | `ada300hw.vfpwnl {func=exp, masked=false}` | `func` is fixed to `exp` |
| `ada300hl.pwnl` | `ada300hw.vfpwnl {func=<forwarded>, masked=false}` | `func` attr forwarded |
| `ada300hl.cvt` | `ada300hw.vfcvt` | `dst_type` and `part` forwarded |
| `ada300hl.vmul_mixed` (`part=low`) | `ada300hw.vfmul_low` | `src_a/b_type` inferred from MLIR element types |
| `ada300hl.vmul_mixed` (`part=high`) | `ada300hw.vfmul_high` | |
| `ada300hl.tensor_mma` | `ada300hw.set_gmm_cfg` + `set_gmm_type` + `set_gmm_iter` + `gmma_mm` or `gmma_mt` | 1 → 4 op expansion; `gmma_mt` selected when `rhs_transposed = true` |
| `ada300hl.tensor_sync` | `ada300hw.tcsync` | Direct replacement |
| `ada300hl.pack` | `memref.copy` (placeholder) | Backend DMA codegen owns data movement |
| `ada300hl.unpack` | `memref.copy` (placeholder) | |
| `ada300hl.layout_cast` | erased (source value forwarded) | Layout info consumed during lowering |
| `ada300hl.copy_to_sram` | `memref.copy` (placeholder) | |
| `ada300hl.copy_from_sram` | `memref.copy` (placeholder) | |
| `ada300hl.copy_to_vr` | `memref.copy` (placeholder) | |
| `ada300hl.copy_from_vr` | `memref.copy` (placeholder) | |

Uses `applyPatternsAndFoldGreedily` for pattern application.

---

### 4. LowerAda300HWToLLVM

**Pass argument:** `--lower-ada300hw-to-llvm`  
**Source file:** `midend/lib/Conversion/LowerAda300HWToLLVM/LowerAda300HWToLLVMPass.cpp`  
**Pass class:** `LowerAda300HWToLLVMPass` (operates on `func::FuncOp`)

Lowers Ada300HW ISA-level ops to `LLVM::InlineAsmOp` operations.  This allows
the LLVM backend to emit the custom ADA300 instruction byte sequences via the
existing RISC-V inline-asm emission path, without requiring full LLVM target
support for the ADA300 custom extension.

Uses `applyPartialConversion` with `LLVMConversionTarget`; all `ada300hw` ops
are marked illegal, `llvm` ops are legal.  MemRef → LLVM struct type
conversion patterns are also populated so that pointer operands for tensor
execute ops can be extracted.

#### Instruction Encoding Convention

| Ada300HW op | Inline asm mnemonic | Constraints |
|---|---|---|
| `vfpwnl` | `vfpwnl.<func>.<segments>  vd, vs1` | `=vr,vr` |
| `vfcvt` | `vfcvt.<part>  vd, vs1` | `=vr,vr` |
| `vfmul_low` | `vfmul.low  vd, vs1, vs2` | `=vr,vr,vr` |
| `vfmul_high` | `vfmul.high  vd, vs1, vs2` | `=vr,vr,vr` |
| `set_gmm_cfg` | `gmm.cfg <row>, <col>, <acc>, <mode>` | side-effect, no SSA operands |
| `set_gmm_type` | `gmm.type <out_dt>, <act_dt>, <wht_dt>` | side-effect, no SSA operands |
| `set_gmm_iter` | `gmm.iter <blk_a>, <blk_w>` | side-effect, no SSA operands |
| `gmm_mm` | `gmm.mm  $0, $1, $2` | `r,r,r` (GPR pointers) |
| `gmma_mm` | `gmma.mm  $0, $1, $2` | `r,r,r` |
| `gmma_mt` | `gmma.mt  $0, $1, $2` | `r,r,r` |
| `gmv_mm` | `gmv.mm  $0, $1, $2` | `r,r,r` |
| `gmva_mm` | `gmva.mm  $0, $1, $2` | `r,r,r` |
| `gmva_mt` | `gmva.mt  $0, $1, $2` | `r,r,r` |
| `tcsync` | `tcsync` | side-effect, no operands |

Config register values (`row_size`, `col_size`, `mode`, data-type codes, etc.)
are compile-time constants derived from op attributes; they are embedded
directly into the asm string at conversion time.

For tensor execute ops, the aligned pointer (field index 1 of the LLVM memref
descriptor struct) is extracted via `llvm.extractvalue` and passed as a GPR
(`"r"`) operand.

---

## Full Pipeline

A complete compilation pipeline for a function containing `linalg.matmul`
and `math.exp` on vectors:

```
1.  linalg.matmul + math.exp   (standard MLIR)
         │
         │  (bufferization: one-shot-bufferize or similar)
         │
2.  linalg.matmul + math.exp   (bufferized, memref form)
         │
         │  --math-to-ada300hl
         │
3.  linalg.matmul              (linalg)
    ada300hl.exp               (high-level)
         │
         │  --linalg-to-ada300hl
         │
4.  ada300hl.copy_to_sram
    ada300hl.tensor_mma
    ada300hl.tensor_sync
    ada300hl.exp               (all ada300hl)
         │
         │  --lower-ada300hl-to-ada300hw
         │
5.  ada300hw.set_gmm_cfg
    ada300hw.set_gmm_type
    ada300hw.set_gmm_iter
    ada300hw.gmma_mm
    ada300hw.tcsync
    ada300hw.vfpwnl            (all ada300hw)
    memref.copy                (data-movement placeholders)
         │
         │  --lower-ada300hw-to-llvm
         │  (+ finalize-memref-to-llvm)
         │
6.  llvm.inline_asm            (LLVM dialect, ready for backend)
```

---

## CMake Integration

The two dialect libraries and four pass libraries are wired into the midend
build system as follows:

### Dialect Libraries

```cmake
# midend/include/Dialect/Ada300HL/CMakeLists.txt
add_mlir_dialect(Ada300HLOps ada300hl)          # generates ops/types/dialect inc files
mlir_tablegen(Ada300HLOpsAttrs.h.inc  ...)      # enum + attr declarations
mlir_tablegen(Ada300HLOpsEnums.cpp.inc ...)     # enum definitions
add_public_tablegen_target(BuddyAda300HLAttrsIncGen)

# midend/lib/Dialect/Ada300HL/CMakeLists.txt
add_mlir_dialect_library(BuddyAda300HL
  Ada300HLDialect.cpp Ada300HLAttrs.cpp Ada300HLTypes.cpp Ada300HLOps.cpp
  DEPENDS MLIRAda300HLOpsIncGen BuddyAda300HLAttrsIncGen
  ...)

# midend/lib/Dialect/Ada300HW/CMakeLists.txt
add_mlir_dialect_library(BuddyAda300HW
  Ada300HWDialect.cpp Ada300HWOps.cpp
  DEPENDS MLIRAda300HWOpsIncGen BuddyAda300HLAttrsIncGen
  LINK_LIBS PUBLIC BuddyAda300HL ...)
```

### Pass Libraries

```cmake
# midend/lib/Conversion/CMakeLists.txt
add_subdirectory(MathToAda300HL)
add_subdirectory(LinalgToAda300HL)
add_subdirectory(LowerAda300HLToAda300HW)
add_subdirectory(LowerAda300HWToLLVM)
```

Each pass directory contains a standalone `add_mlir_library(...)` target that
links the relevant dialect libraries and standard MLIR infrastructure
(`MLIRPass`, `MLIRTransformUtils`, `MLIRFuncDialect`, etc.).

### Key Generated Files

| Target | Generated from | Used in |
|---|---|---|
| `MLIRAda300HLOpsIncGen` | `Ada300HLOps.td` | `Ada300HLOps.h`, `Ada300HLDialect.cpp`, `Ada300HLOps.cpp` |
| `BuddyAda300HLAttrsIncGen` | `Ada300HLOps.td` | `Ada300HLAttrs.h`, `Ada300HLDialect.h`, `Ada300HLAttrs.cpp` |
| `MLIRAda300HWOpsIncGen` | `Ada300HWOps.td` | `Ada300HWOps.h`, `Ada300HWDialect.cpp` |
