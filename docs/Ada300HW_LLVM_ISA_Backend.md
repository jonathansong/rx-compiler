# Ada300HW LLVM ISA Backend

This document describes the design and implementation of the LLVM target backend
for the `ada300hw` dialect, which produces true machine binary for ADA300 custom
RISC-V instructions via the buddy patched LLVM backend.

## Overview

The `ada300hw` dialect has two independent code-generation paths that coexist:

| Path | Trigger | Output |
|---|---|---|
| **Inline-asm** (existing) | `buddy-opt --lower-ada300hw-to-llvm` | `llvm.inline_asm` ops; assembler parses mnemonic strings at emit time |
| **LLVM ISA backend** (new) | `mlir-translate --buddy-to-llvmir` | LLVM IR intrinsic calls; backend encodes exact binary via tablegen |

The ISA backend path enables full compiler optimisation (register allocation,
instruction scheduling, CSE) across Ada300HW operations and produces
disassemble-able binary without any external assembler dependency.

---

## Compilation Pipeline

```
Ada300HL dialect ops
    │
    │  LowerAda300HLToAda300HWPass (buddy-opt)
    ▼
Ada300HW dialect ops  ◄──── this document covers everything below this line
    │
    │  Ada300HWToLLVMIRTranslation  (mlir-translate --buddy-to-llvmir)
    │  (LLVMTranslationDialectInterface)
    ▼
LLVM IR intrinsic calls
    │  call void @llvm.riscv.ada300_gmm_mm(i64 %dst, i64 %act, i64 %wht)
    │
    │  buddy RISC-V LLVM backend (requires patched LLVM from backend/)
    │  DAG ISel matches via RVInstR tablegen patterns
    ▼
Machine binary
    │  R-format:  [func7(7)][rs2(5)][rs1(5)][func3(3)][rd(5)][opcode(7)]
    │  opcode = OPC_CUSTOM_3 (0b1111011)
    │  func3  = 0b011
    │  func7  = per-instruction tag (see encoding table)
    ▼
ADA300 hardware execution
```

---

## File Map

### Backend / LLVM layer

These files live inside the patched LLVM build (`backend/`) that the buddy project
maintains to support all custom extensions.

```
backend/
├── include/llvm/IR/
│   ├── IntrinsicsBuddyExt.td           ← top-level include router (adds +1 line)
│   ├── IntrinsicsRISCVBuddyExt.td      ← Gemmini / RVV / AME / IME intrinsics
│   └── IntrinsicsRISCVAda300HW.td      ← NEW: Ada300HW intrinsic declarations
└── llvm/lib/Target/RISCV/
    ├── RISCVBuddyExt.td                ← subtarget feature + register classes
    ├── RISCVInstrInfoBuddyExt.td       ← Gemmini / AME encodings (adds +1 line)
    └── RISCVInstrInfoAda300HW.td       ← NEW: Ada300HW binary encodings
```

### MLIR / midend layer

```
midend/
├── include/Target/LLVMIR/Dialect/
│   └── Ada300HW/
│       └── Ada300HWToLLVMIRTranslation.h   ← NEW: registration declarations
├── lib/Target/LLVMIR/
│   ├── ConvertBuddyToLLVMIR.cpp            ← patched: +include, +register call
│   ├── CMakeLists.txt                       ← patched: +BuddyAda300HWToLLVMIRTranslation
│   └── Dialect/
│       ├── CMakeLists.txt                   ← patched: +add_subdirectory(Ada300HW)
│       └── Ada300HW/
│           ├── Ada300HWToLLVMIRTranslation.cpp   ← NEW: op → intrinsic translation
│           └── CMakeLists.txt                     ← NEW: build target
```

---

## Layer 1: LLVM Intrinsic Declarations

**File:** [`backend/include/llvm/IR/IntrinsicsRISCVAda300HW.td`](../backend/include/llvm/IR/IntrinsicsRISCVAda300HW.td)

Each ADA300 instruction is declared as an LLVM target intrinsic with
`TargetPrefix = "riscv"`.  LLVM tablegen generates `llvm::Intrinsic::riscv_ada300_*`
enum values from these declarations, which the translation layer uses to look up
functions in the LLVM module.

### Operand conventions

**Config ops** — two `i64` arguments carrying compile-time constants packed from
MLIR attributes:

| Intrinsic | `rs1` | `rs2` |
|---|---|---|
| `int_riscv_ada300_set_gmm_cfg` | `row_size \| (col_size<<8) \| (acc_size<<16)` | `mode` |
| `int_riscv_ada300_set_gmm_type` | `out_type \| (act_type<<8)` | `wht_type` |
| `int_riscv_ada300_set_gmm_iter` | `blk_cnt_a` | `blk_cnt_w` |

**Execute ops** — three `i64` arguments carrying SRAM buffer aligned pointers
(cast from memref descriptors via `ptrtoint`):

| Intrinsic | `rs1` | `rs2` | `rs3` |
|---|---|---|---|
| `int_riscv_ada300_gmm_mm` | `dst ptr` | `act ptr` | `wht ptr` |
| `int_riscv_ada300_gmma_mm` | `dst ptr` | `act ptr` | `wht ptr` |
| `int_riscv_ada300_gmma_mt` | `dst ptr` | `act ptr` | `wht ptr` |
| `int_riscv_ada300_gmv_mm` | `dst ptr` | `mat ptr` | `vec ptr` |
| `int_riscv_ada300_gmva_mm` | `dst ptr` | `mat ptr` | `vec ptr` |
| `int_riscv_ada300_gmva_mt` | `dst ptr` | `mat ptr` | `vec ptr` |

**Vector ops** — `i64` base-address convention (placeholder; see TODOs):

| Intrinsic | `rs1` | `rs2` | `rs3` | result |
|---|---|---|---|---|
| `int_riscv_ada300_vfpwnl` | `input addr` | `func enum` | `segments` | `output addr` |
| `int_riscv_ada300_vfcvt` | `input addr` | `dst_type` | `part` | `output addr` |
| `int_riscv_ada300_vfmul_low` | `lhs addr` | `rhs addr` | — | `result addr` |
| `int_riscv_ada300_vfmul_high` | `lhs addr` | `rhs addr` | — | `result addr` |

**Sync op:**

| Intrinsic | Operands | Result |
|---|---|---|
| `int_riscv_ada300_tcsync` | none | none |

### Include chain

```
IntrinsicsBuddyExt.td
  ├── Intrinsics.td                    (upstream LLVM)
  ├── IntrinsicsRISCVBuddyExt.td       (Gemmini / RVV / IME / AME)
  └── IntrinsicsRISCVAda300HW.td       ← Ada300HW additions
```

---

## Layer 2: Machine Instruction Binary Encodings

**File:** [`backend/llvm/lib/Target/RISCV/RISCVInstrInfoAda300HW.td`](../backend/llvm/lib/Target/RISCV/RISCVInstrInfoAda300HW.td)

All Ada300HW instructions use the RISC-V **R-format** with:
- `opcode = OPC_CUSTOM_3` (`0b1111011`) — the fourth RISC-V custom opcode slot
- `func3  = 0b011` — consistent with the Gemmini buddy extension convention
- `func7` — unique per instruction (Ada300HW occupies the `0b1xxxxxxx` range)

### func7 encoding table

| `func7` (binary) | Decimal | Instruction |
|---|---|---|
| `0b1000000` | 64 | `set_gmm_cfg` |
| `0b1000001` | 65 | `set_gmm_type` |
| `0b1000010` | 66 | `set_gmm_iter` |
| `0b1000011` | 67 | `gmm_mm` |
| `0b1000100` | 68 | `gmma_mm` |
| `0b1000101` | 69 | `gmma_mt` |
| `0b1000110` | 70 | `gmv_mm` |
| `0b1000111` | 71 | `gmva_mm` |
| `0b1001000` | 72 | `gmva_mt` |
| `0b1001001` | 73 | `tcsync` |
| `0b1001010` | 74 | `vfpwnl` |
| `0b1001011` | 75 | `vfcvt` |
| `0b1001100` | 76 | `vfmul_low` |
| `0b1001101` | 77 | `vfmul_high` |

> **TODO:** Replace these placeholder assignments with the actual func7 values
> specified in the ADA300 ISA manual once the chip specification is available.

### R-format bit layout

```
 31      25 24   20 19   15 14  12 11    7 6       0
┌──────────┬───────┬───────┬──────┬───────┬─────────┐
│  func7   │  rs2  │  rs1  │func3 │  rd   │ opcode  │
│  7 bits  │ 5 bits│ 5 bits│3 bits│ 5 bits│  7 bits │
└──────────┴───────┴───────┴──────┴───────┴─────────┘
```

### Execute op register usage

The R-format only has `rd`, `rs1`, `rs2` (three register fields).  Execute ops
need three pointer operands.  Ada300HW encodes them as:

```
rd  = wht / vec buffer pointer
rs1 = dst buffer pointer
rs2 = act / mat buffer pointer
```

### Include chain

```
RISCVBuddyExt.td
  ├── RISCV.td                         (upstream LLVM RISC-V)
  ├── ...register classes...
  └── RISCVInstrInfoBuddyExt.td
        ├── ...Gemmini encodings...
        └── RISCVInstrInfoAda300HW.td  ← Ada300HW additions
```

---

## Layer 3: MLIR → LLVM IR Translation

**File:** [`midend/lib/Target/LLVMIR/Dialect/Ada300HW/Ada300HWToLLVMIRTranslation.cpp`](../midend/lib/Target/LLVMIR/Dialect/Ada300HW/Ada300HWToLLVMIRTranslation.cpp)

Implements `LLVMTranslationDialectInterface::convertOperation()`.  For each
Ada300HW op, it builds LLVM IR instructions directly using `IRBuilder` —
no intermediate lowering pass is needed because Ada300HW ops already model
ISA-level instructions one-to-one.

### Helper functions

| Helper | Purpose |
|---|---|
| `i64Ty(builder)` | Returns `llvm::Type::getInt64Ty` |
| `i64Const(builder, val)` | Builds a constant `i64` IR value |
| `getIntrinsic(module, id)` | Looks up / inserts an intrinsic declaration |
| `alignedPtrFromMemref(llvmMemref, builder)` | Extracts field 1 (aligned ptr) from a memref descriptor struct, or returns the value directly for bare-pointer convention |
| `ptrToI64(ptr, builder)` | Emits `ptrtoint ptr to i64` |
| `memrefToI64(llvmMemref, builder)` | Combines aligned-ptr extraction + ptrtoint |

### Translation logic per op group

#### Config ops

MLIR attributes (integers, enums) are converted to `i64` IR constants.
Multiple fields are bit-packed into a single register operand.

```
ada300hw.set_gmm_cfg {row_size=16, col_size=16, acc_size=16,  ──►  call void @llvm.riscv.ada300_set_gmm_cfg(
    mode = #ada300hl.tensor_mode<inner>}                               i64 16|(16<<8)|(16<<16),    ; rs1 = 0x101010
                                                                        i64 0)                     ; rs2 = inner=0
```

#### Execute ops

The memref operands are looked up from the `ModuleTranslation` value map,
the aligned pointer is extracted, and cast to `i64`:

```
ada300hw.gmm_mm %dst, %act, %wht  ──►  %d = ptrtoint ptr %dst.aligned to i64
                                         %a = ptrtoint ptr %act.aligned to i64
                                         %w = ptrtoint ptr %wht.aligned to i64
                                         call void @llvm.riscv.ada300_gmm_mm(i64 %d, i64 %a, i64 %w)
```

#### Vector ops

Vector operands are passed as pointer-to-i64 casts (placeholder convention).
Enum attributes become `i64` constants.  The intrinsic result (an `i64` address)
is mapped back to the MLIR result SSA value:

```
%y = ada300hw.vfpwnl %x {func=exp, segments=16}  ──►  %xi = ptrtoint ptr %x to i64
                                                         %yi = call i64 @llvm.riscv.ada300_vfpwnl(
                                                                   i64 %xi, i64 0, i64 16)
                                                         ; mapValue(%y, %yi)
```

#### Sync op

```
ada300hw.tcsync  ──►  call void @llvm.riscv.ada300_tcsync()
```

---

## Layer 4: Build System

### New CMakeLists

**`midend/lib/Target/LLVMIR/Dialect/Ada300HW/CMakeLists.txt`:**

```cmake
add_mlir_translation_library(BuddyAda300HWToLLVMIRTranslation
  Ada300HWToLLVMIRTranslation.cpp

  DEPENDS
  buddy_intrinsics_gen          # ensures IntrinsicsRISCV.h is generated first

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  BuddyAda300HW
)
```

`buddy_intrinsics_gen` is the tablegen target defined in
`backend/include/llvm/IR/CMakeLists.txt` that generates `IntrinsicsRISCV.h`
(containing the `llvm::Intrinsic::riscv_ada300_*` enum values) from
`IntrinsicsBuddyExt.td`.

### Patched existing files

| File | Change |
|---|---|
| `backend/include/llvm/IR/IntrinsicsBuddyExt.td` | `include "IntrinsicsRISCVAda300HW.td"` |
| `backend/llvm/lib/Target/RISCV/RISCVInstrInfoBuddyExt.td` | `include "RISCVInstrInfoAda300HW.td"` |
| `midend/lib/Target/LLVMIR/Dialect/CMakeLists.txt` | `add_subdirectory(Ada300HW)` |
| `midend/lib/Target/LLVMIR/ConvertBuddyToLLVMIR.cpp` | include header + `registerAda300HWDialectTranslation(registry)` |
| `midend/lib/Target/LLVMIR/CMakeLists.txt` | `BuddyAda300HWToLLVMIRTranslation` in `LINK_LIBS` |

---

## Coexistence with the Inline-Asm Path

Both paths are active simultaneously.  The user selects the path via the tool:

```bash
# Inline-asm path (stock RISC-V toolchain, no buddy LLVM required):
buddy-opt input.mlir --lower-ada300hw-to-llvm | \
  mlir-translate --mlir-to-llvmir -o out.ll

# ISA backend path (requires buddy patched LLVM):
buddy-opt input.mlir |
  mlir-translate --buddy-to-llvmir -o out.ll
# Then compile out.ll with the buddy LLVM clang/llc targeting riscv64 +buddyext
```

The two paths share the same Ada300HW dialect ops and attributes — there is
no duplication at the MLIR level.

---

## Attribute Enum Values

Enum values used for attribute packing in the translation layer.
Defined in `midend/include/Dialect/Ada300HL/Ada300HLAttrs.td`.

### NonlinearFunc (for `vfpwnl` `func` attribute)

| Name | Value |
|---|---|
| `exp` | 0 |
| `log` | 1 |
| `sqrt` | 2 |
| `rsqrt` | 3 |
| `div` | 4 |
| `sin` | 5 |
| `cos` | 6 |

### TensorDataType (for `set_gmm_type`, `vfcvt` `dst_type`)

| Name | Value |
|---|---|
| `fp16` | 0 |
| `bf16` | 1 |
| `int16` | 2 |
| `fp8` | 3 |
| `int8` | 4 |
| `int4` | 5 |

### TensorMode (for `set_gmm_cfg` `mode`)

| Name | Value |
|---|---|
| `inner` | 0 |
| `outer` | 1 |

### Part (for `vfcvt` `part`)

| Name | Value |
|---|---|
| `low` | 0 |
| `high` | 1 |

---

## Known Limitations and TODOs

### func7 values are placeholders

The `func7` bit assignments in `RISCVInstrInfoAda300HW.td` use the range
`0b1000000–0b1001101` (64–77) to avoid collision with the Gemmini assignments
(`0–35`).  These **must be updated** with the actual values from the ADA300 ISA
specification before targeting real hardware.

**File to update:** `backend/llvm/lib/Target/RISCV/RISCVInstrInfoAda300HW.td`

### Vector ops use GPR / i64 addressing (not RVV registers)

`vfpwnl`, `vfcvt`, `vfmul_low`, and `vfmul_high` currently encode the vector
register as an `i64` integer base address.  The correct encoding should use
proper RVV-style vector register operand classes.

**When:** Once the ADA300 V-extension instruction format is known and the buddy
LLVM backend is extended with the appropriate register classes, update:
- `backend/include/llvm/IR/IntrinsicsRISCVAda300HW.td` — change `llvm_i64_ty` to an overloaded vector type
- `backend/llvm/lib/Target/RISCV/RISCVInstrInfoAda300HW.td` — replace `GPR` operand class with a vector register class
- `midend/lib/Target/LLVMIR/Dialect/Ada300HW/Ada300HWToLLVMIRTranslation.cpp` — remove `ptrToI64` for vector operands; pass vector values directly

### No DAG ISel patterns yet

The current tablegen defines instruction records (for assembler/disassembler
generation) but does not yet add `Pat<>` DAG selection patterns.  LLVM will
use the intrinsic→instruction lowering path, which requires the intrinsic to
be marked as a builtin in the instruction record.  This is sufficient for
codegen via `llc` with `-O0` but may need explicit `Pat<>` entries for
optimised compilation.
