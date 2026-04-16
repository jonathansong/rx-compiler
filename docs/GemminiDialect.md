# Gemmini Dialect

The `gemmini` dialect targets the [Gemmini accelerator](https://github.com/ucb-bar/gemmini), a systolic array-based accelerator for RISC-V developed at UC Berkeley.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Compilation Pipeline](#compilation-pipeline)
- [Operation Reference](#operation-reference)
  - [Configuration Ops](#configuration-ops)
  - [Data Movement Ops](#data-movement-ops)
  - [Low-Level Compute Ops](#low-level-compute-ops)
  - [High-Level Ops](#high-level-ops)
  - [Intrinsic Ops](#intrinsic-ops)
- [Hardware Constants](#hardware-constants)
- [Pass 1: convert-linalg-to-gemmini](#pass-1-convert-linalg-to-gemmini)
  - [MatmulLowering](#matmullowering)
  - [Conv2DNhwcFhwcLowering](#conv2dnhwcfhwclowering)
  - [Conv2DNchwFchwLowering](#conv2dnchwfchwlowering)
  - [Conv2DNhwcHwcfLowering](#conv2dnhwchwcflowering)
  - [BatchMatMulOpLowering](#batchmatmulopplowering)
- [Pass 2: lower-gemmini](#pass-2-lower-gemmini)
  - [Pass Composition](#pass-composition)
  - [Config Op Lowering](#config-op-lowering)
  - [DMA Op Lowering](#dma-op-lowering)
  - [GemminiTileMatMulLowering](#gemminitilematmullowering)
  - [GemminiTileConvLowering](#gemminitileconvlowering)
- [Lowering Pattern Summary](#lowering-pattern-summary)
- [RISC-V ISA Encoding](#risc-v-isa-encoding)
- [Examples](#examples)

---

## Architecture Overview

Gemmini is a generator for systolic array-based deep learning accelerators. The buddy-mlir compiler provides a complete stack from high-level Linalg operations down to custom RISC-V machine instructions:

```
linalg.matmul / linalg.conv_*
        │
        ▼  --convert-linalg-to-gemmini
gemmini.tile_matmul / gemmini.tile_conv
        │
        ▼  --lower-gemmini
gemmini.intr.loop_ws* / gemmini.intr.loop_conv_ws* / gemmini.intr.mvin / ...
        │
        ▼  LLVM backend (buddy RISC-V fork)
custom RISC-V instructions (OPC_CUSTOM_3)
```

---

## Compilation Pipeline

### Source files

| Component | Path |
|-----------|------|
| Dialect definition (TableGen) | `midend/include/Dialect/Gemmini/Gemmini.td` |
| Dialect header | `midend/include/Dialect/Gemmini/GemminiDialect.h` |
| Ops header | `midend/include/Dialect/Gemmini/GemminiOps.h` |
| Hardware constants | `midend/include/Dialect/Gemmini/Transform.h` |
| Linalg → Gemmini pass | `midend/lib/Conversion/LowerLinalgToGemmini/LowerLinalgToGemmini.cpp` |
| Gemmini → LLVM pass | `midend/lib/Conversion/LowerGemmini/LowerGemminiPass.cpp` |
| Legalization patterns | `midend/lib/Dialect/Gemmini/Transforms/LegalizeForLLVMExport.cpp` |
| LLVM IR translation | `midend/lib/Target/LLVMIR/Dialect/Gemmini/GemminiToLLVMIRTranslation.cpp` |
| RISC-V intrinsics | `backend/include/llvm/IR/IntrinsicsRISCVBuddyExt.td` |
| RISC-V instruction encoding | `backend/llvm/lib/Target/RISCV/RISCVInstrInfoBuddyExt.td` |
| Examples | `examples/GemminiDialect/` |

---

## Operation Reference

### Configuration Ops

#### `gemmini.flush %skip : i64`

Flushes the TLB and drains the Gemmini pipeline.

- `skip=1`: skip the current TLB request
- `skip=0`: retry the current TLB request

#### `gemmini.config_ld %stride : i64`

Configures the DMA load (mvin) pipeline.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `scale` | f32 | 1.0 | Input scale factor |
| `shrunk` | bool | false | Use shrunk format |
| `id` | i64 | 0 | Which load pipeline (0=A, 1=B, 2=D) |
| `block_mvin_stride` | i64 | -1 (→ dim) | Block stride for grouped loads |
| `pixel_repeats` | i64 | 1 | Pixel repeat count |

#### `gemmini.config_st %stride : i64`

Configures the DMA store (mvout) pipeline.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `activation` | i64 | 0 | 0=none, 1=RELU |
| `scale` | f32 | 1.0 | Output scale factor |

#### `gemmini.config_ex`

Configures the systolic array execute pipeline.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataflow` | i64 | 0 | 0=OUTPUT_STATIONARY, 1=WEIGHT_STATIONARY |
| `sysAct` | i64 | 0 | 0=none, 1=RELU |
| `sysShift` | i64 | 0 | Right-shift bits on systolic output |
| `sysAccScale` | f32 | 1.0 | Scale when reading from accumulator |
| `aStride` | i64 | 1 | Scratchpad row stride for A |
| `cStride` | i64 | 1 | Scratchpad row stride for C |
| `aTranspose` | bool | false | Transpose A before multiply |
| `bTranspose` | bool | false | Transpose B before multiply |
| `setOnlyStrides` | bool | false | Update strides only |

#### `gemmini.config_norm`

Configures the normalization pipeline (BERT, IGELU, SOFTMAX).

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `qConst` | i64 | 0 | Quantization constant |
| `qConstType` | i64 | 0 | Type of qConst |
| `setStatsIdOnly` | i64 | 0 | Update StatsId only flag |
| `actMsb` | i64 | 0 | Activation MSB flag |
| `StatsId` | i64 | 0 | Statistics identifier |
| `igeluQb` | i64 | 0 | IGELU quantization `b` parameter |
| `igeluQc` | i64 | 0 | IGELU quantization `c` parameter |

---

### Data Movement Ops

#### `gemmini.mvin %memref %spadAddr : memref<RxCxT> i64`
#### `gemmini.mvin2 %memref %spadAddr : memref<RxCxT> i64`
#### `gemmini.mvin3 %memref %spadAddr : memref<RxCxT> i64`

Load a 2D tile from DRAM → scratchpad. The three variants correspond to the three load pipelines (A, B, D) for double-buffering and pipelining.

- `memref`: 2D source in main memory (rows × cols)
- `spadAddr`: base address in scratchpad (i64)

#### `gemmini.mvout %memref %spadAddr : memref<RxCxT> i64`

Store a 2D tile from scratchpad → L2/DRAM.

- `memref`: 2D destination in main memory
- `spadAddr`: base address in scratchpad

---

### Low-Level Compute Ops

These expose the raw systolic array interface for manual tiling.

#### `gemmini.preload_zeros %addr %rows %cols : i64 i64 i64`

Preload zeros into scratchpad/accumulator at `addr`. Used to initialize the C/accumulator tile before a new matrix multiply tile.

#### `gemmini.preload %bdAddr %cAddr %bdRows %bdCols %cRows %cCols`

Preloads a matrix tile into the systolic array:

- `bdAddr`: scratchpad address of B (weight-stationary) or D (output-stationary)
- `cAddr`: scratchpad address where C output will be written; set to `GARBAGE_ADDR (0xFFFFFFFF)` to suppress write-back
- `bdRows/bdCols`: dimensions of B/D tile
- `cRows/cCols`: dimensions of C tile

#### `gemmini.compute_preloaded %aAddr %bdAddr %aRows %aCols %bdRows %bdCols`

Execute `A × B` (or `A × D`) using a freshly preloaded tile. Use on the **first** K-loop iteration.

#### `gemmini.compute_accumulated %aAddr %bdAddr %aRows %aCols %bdRows %bdCols`

Same as `compute_preloaded` but **accumulates** into the existing C tile. Use on all subsequent K-loop iterations.

**Typical manual tiling pattern:**

```mlir
gemmini.config_st %stride : i64
gemmini.config_ld %stride : i64
gemmini.mvin %A %addrA : memref<4x4xi8> i64
gemmini.mvin %B %addrB : memref<4x4xi8> i64
gemmini.config_ex
gemmini.preload_zeros %addrC %rows %cols : i64 i64 i64
// k=0: fresh accumulator
gemmini.compute_preloaded %addrA %addrB %r %c %r %c : i64 i64 i64 i64 i64 i64
gemmini.mvout %out %addrC : memref<4x4xi8> i64
// k=1: accumulate
gemmini.compute_accumulated %addrA %addrB %r %c %r %c : i64 i64 i64 i64 i64 i64
gemmini.mvout %out %addrC : memref<4x4xi8> i64
```

---

### High-Level Ops

These perform automatic tiling and emit all necessary config/DMA/compute instructions.

#### `gemmini.tile_matmul %A %B %C %D`

Computes `C = A × B + D` with full hardware tiling.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataflow` | i64 | 1 | 0=OUTPUT_STATIONARY, 1=WEIGHT_STATIONARY |
| `act` | i64 | 0 | 0=none, 1=RELU, 2=LAYERNORM, 3=IGELU, 4=SOFTMAX |
| `aScaleFactor` | f32 | 1.0 | A matrix input scale |
| `bScaleFactor` | f32 | 1.0 | B matrix input scale |
| `dScaleFactor` | f32 | 1.0 | D (bias) matrix input scale |
| `accScale` | f32 | 1.0 | Accumulator output scale |
| `bertScale` | f32 | 0.0 | BERT-style normalization scale |
| `aTranspose` | bool | false | Transpose A before multiply |
| `bTranspose` | bool | false | Transpose B before multiply |
| `repeatingBias` | bool | false | Broadcast D as a bias row |
| `fullC` | bool | false | Output full i32 accumulator instead of downcasting to i8 |
| `lowD` | bool | false | D is in elem_t format (not acc_t) |
| `weightA` | i64 | 0 | Weight bank selector |

#### `gemmini.tile_conv %input %weights %bias %output %outRowDim %outColDim %kernelDim`

Perform convolution. Weights must be pre-reshaped to 2D `(H×W×C_in, C_out)` and output to `(N×H'×W', C_out)`.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `scale` | f32 | 1.0 | Output scale |
| `stride` | i64 | 1 | Convolution stride |
| `inputDilation` | i64 | 1 | Input dilation |
| `kernelDilation` | i64 | 1 | Kernel dilation |
| `padding` | i64 | 0 | Zero padding |
| `wrot180` | bool | false | Rotate weights 180° (for backprop) |
| `transOutput1203` | bool | false | Transpose output layout 1203 |
| `transInput3120` | bool | false | Transpose input layout 3120 |
| `transWeight1203` | bool | false | Transpose weight layout 1203 |
| `transWeight0132` | bool | false | Transpose weight layout 0132 |
| `act` | i64 | 0 | Activation function |
| `poolSize` | i64 | 0 | Max pool window size (0=disabled) |
| `poolStride` | i64 | 0 | Max pool stride |
| `poolPadding` | i64 | 0 | Max pool padding |

#### `gemmini.print %memref`

Print memref values for debugging. Supports i8, i32, f32, f64 element types. Lowered to `printf` calls.

---

### Intrinsic Ops

All `gemmini.intr.*` ops map 1:1 to LLVM intrinsics (`llvm.riscv.*`) which are encoded as RISC-V custom instructions.

| Intrinsic | Arguments | Description |
|-----------|-----------|-------------|
| `gemmini.intr.mvin` | ptr, spad | DMA load (pipeline 0/A) |
| `gemmini.intr.mvin2` | ptr, spad | DMA load (pipeline 1/B) |
| `gemmini.intr.mvin3` | ptr, spad | DMA load (pipeline 2/D) |
| `gemmini.intr.mvout` | ptr, spad | DMA store |
| `gemmini.intr.config` | rs1, rs2 | Pipeline config (LD/ST/EX/BERT) |
| `gemmini.intr.flush` | rs1, rs2 | TLB flush / pipeline drain |
| `gemmini.intr.preload` | rs1, rs2 | Preload B/D tile |
| `gemmini.intr.compute_preloaded` | rs1, rs2 | Systolic compute (fresh) |
| `gemmini.intr.compute_accumulated` | rs1, rs2 | Systolic compute (accumulate) |
| `gemmini.intr.loop_ws_config_bounds` | rs1, rs2 | WS loop: tile bounds |
| `gemmini.intr.loop_ws_config_addrs_ab` | a, b | WS loop: A/B DRAM addresses |
| `gemmini.intr.loop_ws_config_addrs_dc` | d, c | WS loop: D/C DRAM addresses |
| `gemmini.intr.loop_ws_config_strides_ab` | rs1, rs2 | WS loop: A/B row strides |
| `gemmini.intr.loop_ws_config_strides_dc` | rs1, rs2 | WS loop: D/C row strides |
| `gemmini.intr.loop_ws` | rs1, rs2 | Fire weight-stationary hardware loop |
| `gemmini.intr.loop_conv_ws_config1`–`6` | rs1, rs2 | Conv WS loop config (6 instructions) |
| `gemmini.intr.loop_conv_ws` | rs1, rs2 | Fire weight-stationary conv hardware loop |

---

## Hardware Constants

Defined in `midend/include/Dialect/Gemmini/Transform.h`:

```c
// Hardware layout
BANK_NUM  = 4           // scratchpad banks
MAX_BYTES = 64          // max DMA bytes per transaction (one systolic row)

// Sentinel address
GARBAGE_ADDR = 0xFFFFFFFF   // do not write output

// Dataflow modes
OUTPUT_STATIONARY = 0
WEIGHT_STATIONARY = 1

// Activation functions
NO_ACTIVATION = 0
RELU          = 1
LAYERNORM     = 2
IGELU         = 3
SOFTMAX       = 4

// Config command IDs (packed into rs1[1:0])
CONFIG_EX   = 0
CONFIG_LD   = 1
CONFIG_ST   = 2
CONFIG_BERT = 3

// Scale identity values
MVIN_SCALE_IDENTITY = 1.0
ACC_SCALE_IDENTITY  = 1.0
```

**`--lower-gemmini` pass parameters** (must match physical chip):

| Flag | Default | Description |
|------|---------|-------------|
| `--dim` | 16 | Systolic array dimension (DIM × DIM) |
| `--addr_len` | 32 | Scratchpad address bit width |
| `--acc_rows` | 1024 | Accumulator row count |
| `--bank_rows` | 4096 | Scratchpad rows per bank |
| `--elem_t` | `i8` | Input element type (`i8` or `f32`) |
| `--acc_t` | `i32` | Accumulator type (`i32` or `f32`) |

---

## Pass 1: `--convert-linalg-to-gemmini`

**File:** `midend/lib/Conversion/LowerLinalgToGemmini/LowerLinalgToGemmini.cpp`

A `DialectConversion` pass that rewrites standard Linalg ops to Gemmini high-level ops.

### MatmulLowering

`linalg.matmul` → `gemmini.tile_matmul`

1. Allocates a zero-filled `i32` (or `f32`) bias buffer via `linalg.fill`
2. Collapses `1×M×K` tensors to `M×K` if batch dimension is 1 (via `memref.collapse_shape`)
3. Replaces with `gemmini.tile_matmul(A, B, C, bias)`
4. Inserts `memref.dealloc` for the temporary bias

### Conv2DNhwcFhwcLowering

`linalg.conv_2d_nhwc_fhwc` → `gemmini.tile_conv`

Input layout: `(N,H,W,C)`, Weights layout: `(F,H,W,C)`

Weight reshape `(F,H,W,C)` → `(H×W×C, F)` via explicit SCF loops:
```
row = h*(W*C) + w*C + c
col = f
```

Output `(N,H',W',F)` → `(N×H'×W', F)` for tile_conv, then copied back via SCF loop.

Constraint: input must be square (`inputShape[1] == inputShape[2]`)

### Conv2DNchwFchwLowering

`linalg.conv_2d_nchw_fchw` → `gemmini.tile_conv`

Input layout: `(N,C,H,W)`, Weights layout: `(F,C,kH,kW)`

1. Transposes input NCHW → NHWC via SCF loop: `(n,c,h,w) → (n,h,w,c)`
2. Reshapes weights `(F,C,kH,kW)` → `(kH×kW×C, F)`:
   ```
   row = krow*(kernelDim*inC) + kcol*inC + inchannel
   col = f
   ```
3. Calls `gemmini.tile_conv`
4. Copies 2D output back to NCHW format

### Conv2DNhwcHwcfLowering

`linalg.conv_2d_nhwc_hwcf` → `gemmini.tile_conv`

Input layout: `(N,H,W,C)`, Weights layout: `(H,W,C,F)`

Reshapes weights `(H,W,C,F)` → `(H×W×C, F)` and output `(N,H',W',F)` → `(N×H'×W', F)`.

Constraint: input and kernel must be square.

### BatchMatMulOpLowering

`linalg.batch_matmul` → loop of `gemmini.tile_matmul`

Emits one `gemmini.tile_matmul` per batch element via a SCF loop over the batch dimension. Also handles `batch_matmul_transpose_b`.

---

## Pass 2: `--lower-gemmini`

**Files:**
- `midend/lib/Conversion/LowerGemmini/LowerGemminiPass.cpp` — pass registration
- `midend/lib/Dialect/Gemmini/Transforms/LegalizeForLLVMExport.cpp` — lowering patterns

### Pass Composition

All dialects are lowered to LLVM IR in a single `applyPartialConversion` call by chaining multiple pattern sets:

```
populateGemminiLegalizeForLLVMExportPatterns   // Gemmini → LLVM intrinsics
populateAffineToStdConversionPatterns
populateSCFToControlFlowConversionPatterns
populateArithToLLVMConversionPatterns
populateFinalizeMemRefToLLVMConversionPatterns
populateControlFlowToLLVMConversionPatterns
populateFuncToLLVMConversionPatterns
PrintOpLowering                                // gemmini.print → printf
```

### Config Op Lowering

Each config op packs its attributes into two 64-bit register values (rs1, rs2) matching the Gemmini ISA encoding, then emits one `gemmini.intr.config`.

**`config_ex` encoding:**
```
rs1 = sysAccScale[63:32] | aStride[23:16] | bTranspose[9] | aTranspose[8]
    | setOnlyStrides[7]  | sysAct[5:3]    | dataflow[3:2] | CONFIG_EX(0)
rs2 = cStride[63:48] | sysShift[15:0]
```

**`config_ld` encoding:**
```
rs1 = scale[63:32] | blockMvinStride[31:16] | pixelRepeats[15:8]
    | id[5:3] | shrunk[2] | CONFIG_LD(1)
rs2 = stride value
```

**`config_st` encoding:**
```
rs1 = activation[3:2] | CONFIG_ST(2)
rs2 = scale[63:32] | stride[31:0]
```

**`config_norm` encoding (BERT/IGELU/SOFTMAX):**
```
rs1 = qConst[63:32] | qConstType[18] | setStatsIdOnly[17] | actMsb[16]
    | StatsId[15:8] | CONFIG_BERT(3)
rs2 = igeluQc[63:32] | igeluQb[31:0]
```

### DMA Op Lowering

All mvin/mvout ops extract the raw pointer from the memref and pack the spad address with shape information:

```cpp
// Extract pointer
ptr = memref.extract_aligned_pointer_as_index(memref)
ptr_i64 = arith.index_cast(ptr, i64)

// Pack spad address encoding
// rows[63:addrLen+16] | cols[addrLen+15:addrLen] | baseAddr[addrLen-1:0]
spadInt = rows << (addrLen+16) | cols << addrLen | baseAddr

emit Mvin_IntrOp(ptr_i64, spadInt)
```

Handles strided memref layouts by adding byte offsets for sub-tile access in the OS tiling loop.

### GemminiTileMatMulLowering

The most complex lowering — performs static compile-time tiling.

#### Step 1: Tile size selection

```
dbPartitionRows = (BANK_NUM * bankRows / 2) / 2
dbMatsInAcc     = (accRows / 2) / dim
dbMaxTileIJ     = sqrt(dbMatsInAcc)       // initial tile size limit
dbMaxTileK      = dbPartitionRows / dbMaxTileIJ

tileI, tileJ, tileK initialized to min(dimPadded/dim, dbMax*)

// Special case: LAYERNORM/SOFTMAX require full-row tiles
if act == LAYERNORM || SOFTMAX:
    tileI=1, tileJ=dimJ/dim, tileK=1
else:
    // Greedy expansion: grow tiles while fitting in scratchpad + accumulator
    while (tiledMatmulTotalSpadRows(tileI, tileJ+1, tileK) <= maxSpadRows
           && tiledMatmulTotalAccRows(tileI, tileJ+1) <= maxAccRows):
        tileJ++
    // similarly for tileI, tileK
```

#### Step 2: Emit config instructions

```mlir
gemmini.config_ex {dataflow, act & 3, shift=0, ACC_SCALE_IDENTITY}
gemmini.config_st %strideC * sizeofC {act & 3, scale}
gemmini.config_ld %strideA * sizeOfElemT {aScale, id=0}   // A pipeline
gemmini.config_ld %strideB * sizeOfElemT {bScale, id=1}   // B pipeline
gemmini.config_ld %strideD * sizeofD     {dScale, id=2}   // D pipeline

// IGELU: derive qb, qc from bertScale
//   qb = -1.769 / (bertScale / sqrt(2))
//   qc = 1.0 / (-0.2888 * (bertScale^2 / 2))
// SOFTMAX: derive qln2, qln2_inv, qb, qc from bertScale (2 config_norm ops)
```

#### Step 3: Outer tile loop (I0 × J0 × K0)

For each tile `(i0, j0, k0)`:

- Compute DRAM byte offset: `(tileRow * stride + tileCol) * dim * elemSize`
- Dispatch to dataflow-specific tiling function:

**Weight-Stationary (WS) dataflow** — uses Gemmini's hardware loop engine:
```mlir
gemmini.intr.loop_ws_config_bounds  rs1=(padK<<32|padJ<<16|padI), rs2=(K<<32|J<<16|I)
gemmini.intr.loop_ws_config_addrs_ab  a_ptr, b_ptr
gemmini.intr.loop_ws_config_addrs_dc  d_ptr, c_ptr
gemmini.intr.loop_ws_config_strides_ab  strideA, strideB
gemmini.intr.loop_ws_config_strides_dc  strideD (or 0 if repeatingBias), strideC
gemmini.intr.loop_ws  rs1=(aSpadId<<18|bSpadId<<16|act<<8|lowD<<2|fullC<<1|exAccum),
                      rs2=(isResadd<<2|bTranspose<<1|aTranspose)
```

**Output-Stationary (OS) dataflow** — emits explicit mvin/compute/mvout:
```
// Scratchpad address layout:
aSpAddrStart = 0
bSpAddrStart = BANK_NUM * bankRows - k*j*dim
dSpAddrStart = 1 << (addrLen-1)            // accumulator region
cSpAddrStart = (3 << (addrLen-2)) | (fullC << (addrLen-3))

// Load D tiles (bias) → accumulator
ConfigLd(strideD), mvin3 for each (i0,j0) sub-tile

// Load B tiles → scratchpad
ConfigLd(strideB), mvin for each (k0,j0) sub-tile

// Load A tiles → scratchpad
ConfigLd(strideA), mvin for each (i0,k0) sub-tile

// Inner compute loop (i0,j0,k0):
//   k0==0: preload + compute_preloaded  (fresh accumulator)
//   k0>0:  preload(GARBAGE_ADDR) + compute_accumulated (accumulate)

// Store C tiles → DRAM
mvout for each (i0,j0) sub-tile
```

#### Step 4: Memory fence and flush

```mlir
LLVM::FenceOp(seq_cst)   // seq_cst memory fence for CPU-NPU ordering
gemmini.intr.flush(0, 0)  // drain Gemmini pipeline
```

### GemminiTileConvLowering

`gemmini.tile_conv` → `loop_conv_ws_config1`–`6` + `loop_conv_ws` + `flush`

Uses Gemmini's hardware convolution loop engine. The 6 config instructions encode: batch/channel/spatial dimensions, strides, padding, dilation, pooling parameters, activation, and the DRAM addresses for input/weight/bias/output.

```mlir
gemmini.intr.loop_conv_ws_config1 (batch, in_channels, out_channels, in_dim)
gemmini.intr.loop_conv_ws_config2 (kernel_dim, pool_size, pool_stride, pool_pad)
gemmini.intr.loop_conv_ws_config3 (out_dim, stride, padding, dilation)
gemmini.intr.loop_conv_ws_config4 (input_ptr, weights_ptr)
gemmini.intr.loop_conv_ws_config5 (output_ptr, bias_ptr)
gemmini.intr.loop_conv_ws_config6 (act, scale, pixel_repeats, ...)
gemmini.intr.loop_conv_ws  (control flags)
gemmini.intr.flush(0, 0)
```

---

## Lowering Pattern Summary

| Pattern Class | Input | Output |
|---|---|---|
| `MatmulLowering` | `linalg.matmul` | `gemmini.tile_matmul` |
| `Conv2DNhwcFhwcLowering` | `linalg.conv_2d_nhwc_fhwc` | `gemmini.tile_conv` |
| `Conv2DNchwFchwLowering` | `linalg.conv_2d_nchw_fchw` | `gemmini.tile_conv` |
| `Conv2DNhwcHwcfLowering` | `linalg.conv_2d_nhwc_hwcf` | `gemmini.tile_conv` |
| `BatchMatMulOpLowering` | `linalg.batch_matmul` | loop of `gemmini.tile_matmul` |
| `BatchMatMulTransposeBLowering` | `linalg.batch_matmul_transpose_b` | loop of `gemmini.tile_matmul` |
| `GemminiFlushLowering` | `gemmini.flush` | `gemmini.intr.flush` |
| `GemminiConfigStLowering` | `gemmini.config_st` | `gemmini.intr.config` |
| `GemminiConfigLdLowering` | `gemmini.config_ld` | `gemmini.intr.config` |
| `GemminiConfigExLowering` | `gemmini.config_ex` | `gemmini.intr.config` |
| `GemminiConfigNormLowering` | `gemmini.config_norm` | `gemmini.intr.config` |
| `GemminiMvinLowering` | `gemmini.mvin` | `gemmini.intr.mvin` |
| `GemminiMvin2Lowering` | `gemmini.mvin2` | `gemmini.intr.mvin2` |
| `GemminiMvin3Lowering` | `gemmini.mvin3` | `gemmini.intr.mvin3` |
| `GemminiMvoutLowering` | `gemmini.mvout` | `gemmini.intr.mvout` |
| `GemminiPreloadZerosLowering` | `gemmini.preload_zeros` | `gemmini.intr.preload` |
| `GemminiPreloadLowering` | `gemmini.preload` | `gemmini.intr.preload` |
| `GemminiComputePreloadedLowering` | `gemmini.compute_preloaded` | `gemmini.intr.compute_preloaded` |
| `GemminiComputeAccumulatedLowering` | `gemmini.compute_accumulated` | `gemmini.intr.compute_accumulated` |
| `GemminiTileMatMulLowering` | `gemmini.tile_matmul` | WS: `loop_ws_config*` + `loop_ws`; OS: mvin/compute/mvout |
| `GemminiTileConvLowering` | `gemmini.tile_conv` | `loop_conv_ws_config1-6` + `loop_conv_ws` |
| `PrintOpLowering` | `gemmini.print` | `printf` calls |

---

## RISC-V ISA Encoding

All Gemmini intrinsics are encoded as R-type RISC-V custom instructions using buddy-mlir's LLVM fork:

```
[func7(7)][rs2(5)][rs1(5)][func3(3)][rd(5)][opcode(7)]
                                            └── 0b1111011 (OPC_CUSTOM_3)
                               └─────────────── 0b011 (Gemmini convention)
└──────────────────────────────────────────────── per-instruction identifier
```

- **`opcode`**: `0b1111011` (`OPC_CUSTOM_3`)
- **`func3`**: `0b011` (Gemmini convention)
- **`func7`**: distinguishes each Gemmini command

Intrinsic declarations: `backend/include/llvm/IR/IntrinsicsRISCVBuddyExt.td`
Instruction encoding: `backend/llvm/lib/Target/RISCV/RISCVInstrInfoBuddyExt.td`

---

## Examples

All examples are in `examples/GemminiDialect/`.

### Linalg → Gemmini (`matmul.mlir`)

```bash
buddy-opt matmul.mlir --convert-linalg-to-gemmini
```

```mlir
// Input
linalg.matmul
  ins(%A, %B : memref<8x8xi8>, memref<8x8xi8>)
  outs(%C : memref<8x8xi8>)

// Output
gemmini.tile_matmul %alloc %alloc_0 %alloc_1 %alloc_2 :
  memref<8x8xi8> memref<8x8xi8> memref<8x8xi8> memref<8x8xi32>
```

### Gemmini → LLVM intrinsics (`tile-matmul.mlir`)

```bash
buddy-opt tile-matmul.mlir --lower-gemmini
```

```mlir
// Input
gemmini.tile_matmul %aArray %bArray %cArray %dArray :
  memref<64x64xi8> memref<64x64xi8> memref<64x64xi8> memref<64x64xi32>

// Output (WS dataflow)
"gemmini.intr.config"(...)           // config_ex
"gemmini.intr.config"(...)           // config_st
"gemmini.intr.config"(...)           // config_ld A
"gemmini.intr.config"(...)           // config_ld B
"gemmini.intr.config"(...)           // config_ld D
"gemmini.intr.loop_ws_config_bounds"(...)
"gemmini.intr.loop_ws_config_addrs_ab"(...)
"gemmini.intr.loop_ws_config_addrs_dc"(...)
"gemmini.intr.loop_ws_config_strides_ab"(...)
"gemmini.intr.loop_ws_config_strides_dc"(...)
"gemmini.intr.loop_ws"(...)
"gemmini.intr.flush"(...)
```

### Convolution (`tile-conv.mlir`)

```bash
buddy-opt tile-conv.mlir --lower-gemmini
```

```mlir
// Input: batchSize=1, inputDim=5, inChannels=1, outChannels=2, kernelDim=3
gemmini.tile_conv %input %weight %bias %output %3 %3 %3 {stride = 1} :
  memref<1x5x5x1xi8> memref<9x2xi8> memref<2xi32> memref<9x2xi8> i64 i64 i64

// Output
"gemmini.intr.loop_conv_ws_config1"(...)
"gemmini.intr.loop_conv_ws_config2"(...)
"gemmini.intr.loop_conv_ws_config3"(...)
"gemmini.intr.loop_conv_ws_config4"(...)
"gemmini.intr.loop_conv_ws_config5"(...)
"gemmini.intr.loop_conv_ws_config6"(...)
"gemmini.intr.loop_conv_ws"(...)
"gemmini.intr.flush"(...)
```

### Weight-Stationary with RELU (`tile-matmul-ws-relu.mlir`)

```mlir
gemmini.tile_matmul %A %B %C %D {dataflow=1, act=1} :
  memref<5x5xi8> memref<5x5xi8> memref<5x5xi8> memref<5x5xi32>
```

### Manual DMA (`mvin-mvout.mlir`)

```mlir
gemmini.config_st %stride16 : i64   // configure store stride
gemmini.config_ld %stride16 : i64   // configure load stride
gemmini.mvin  %arrayA %spadAddr : memref<2x16xi8> i64
gemmini.mvout %arrayB %spadAddr : memref<3x16xi8> i64
gemmini.config_st %stride8 : i64    // reconfigure store stride
gemmini.mvout %arrayC %spadAddr : memref<2x8xi8> i64
```
