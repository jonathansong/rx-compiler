# Tiling Optimization in buddy-mlir

This document describes how loop tiling is implemented in buddy-mlir as a set of custom MLIR conversion passes, targeting matrix multiplication, convolution, digital signal processing, and transpose operations.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Common Pattern](#2-common-pattern)
3. [Passes](#3-passes)
   - [BatchMatMul Tile Optimize](#31-batchmatmul-tile-optimize)
   - [Conv2D NHWC-FHWC Tile Optimize](#32-conv2d-nhwc-fhwc-tile-optimize)
   - [CB Convolution Vectorization](#33-cb-convolution-vectorization)
   - [DAP Tiled Vectorization](#34-dap-tiled-vectorization)
   - [AMX Matmul Tiling](#35-amx-matmul-tiling)
   - [Transpose Tiling](#36-transpose-tiling)
   - [Gemmini Hardware Tiling](#37-gemmini-hardware-tiling)
4. [Tail Handling](#4-tail-handling)
5. [CLI Options](#5-cli-options)
6. [Examples](#6-examples)

---

## 1. Overview

Loop tiling (also called strip mining or blocking) improves performance by restructuring iteration spaces to increase data locality and expose SIMD parallelism. In buddy-mlir, tiling is applied as **custom MLIR rewrite passes** that lower high-level `linalg` operations into tiled loop nests using `affine`, `scf`, and `vector` dialect operations.

The general lowering chain is:

```
linalg.conv / linalg.matmul / linalg.batch_matmul
    ↓  (tiling pass)
affine.parallel / scf.forall  (outer tiles)
    ↓
affine.for / scf.for           (inner loops)
    ↓
vector.fma / vector.load / vector.store  (SIMD)
```

---

## 2. Common Pattern

Every tiling pass is a `ConversionPattern` (or `OpRewritePattern`) with parameterized tile sizes passed as `PassOptions`. The structure is:

1. **Extract dimensions** from the input/output memrefs using `memref.dim`.
2. **Compute tile step sizes** — e.g. `stepOH = ceil(OH / tilingOH)`.
3. **Create the outer tiled loop nest** — `affine.parallel`, `scf.forall`, or `affine.for`.
4. **Build the inner vectorized loop** — broadcast scalars, load tiles, apply `vector.fma`.
5. **Handle boundary tiles** — `affine.if` guards and `vector.create_mask` for remainder elements.
6. **Erase the original operation** with `rewriter.eraseOp(op)`.

---

## 3. Passes

### 3.1 BatchMatMul Tile Optimize

**Pass flag:** `--batchmatmul-tile-optimize`  
**Source:** `midend/lib/Conversion/MatMulOptimization/BatchMatMulTileOptimize.cpp`  
**Target op:** `linalg.batch_matmul`

Tiles the batch, M, and N dimensions. An `affine.parallel` loop parallelizes across the batch dimension (with a prefetch hint for tensor A), followed by a 2D tiled loop over N and M using `affine.buildAffineLoopNest`. The inner K loop performs scalar loads from A, vector loads from B, and accumulates with `vector.fma` (float) or `arith.muli`/`arith.addi` (integer).

**Tile shape:** `kernelM × (kernelN × vecSize)` per thread.

```
affine.parallel %b in [0, batch) {
  affine.for %j in [0, N) step (kernelN * vecSize) {
    affine.for %i in [0, M) step kernelM {
      affine.for %k in [0, K) step 1 {
        // vector.fma over kernelM × kernelN vectors
      }
    }
  }
}
```

### 3.2 Conv2D NHWC-FHWC Tile Optimize

**Pass flag:** `--conv-nhwc-fhwc-tile-optimize`  
**Source:** `midend/lib/Conversion/ConvOptimization/ConvNhwcFhwcTileOptimize.cpp`  
**Target op:** `linalg.conv_2d_nhwc_fhwc`

Tiles the output spatial dimensions (OH, OW) and output channels (OC) using two nested `scf.forall` loops. The inner computation iterates over the input channels (IC) with a `scf.for`, loading input and filter slices as vectors and accumulating with `vector.fma`. A `vector.reduction` collapses the channel dimension before writing back to the output.

```
scf.forall (n, oh_tile, ow_tile, oc_tile) in (N, stepOH, stepOW, stepOC) {
  scf.forall (oh, ow, oc) in tile bounds {
    scf.for %ic in [0, IC) step vecSize {
      scf.for %fh ...
        scf.for %fw ...
          vector.fma(input_vec, filter_vec, acc_vec)
      vector.reduction ADD -> scalar
    }
  }
}
```

### 3.3 CB Convolution Vectorization

**Pass flag:** `--cb-conv-vectorize`  
**Source:** `midend/lib/Conversion/ConvVectorization/CBConvVectorization.cpp`  
**Target op:** `linalg.conv_2d`

Implements the Coefficient Broadcasting (CB) algorithm. The outer loop iterates over kernel elements; each kernel scalar is broadcast to a 2D vector. The corresponding input tile and output tile are loaded with `vector.transfer_read`, accumulated with `vector.fma`, and written back with `vector.transfer_write`.

```
affine.for %krow, %kcol in kernel_dims {
  kernel_vec = broadcast(kernel[krow, kcol])     // scalar → 2D vector
  input_vec  = transfer_read(input[krow, kcol])  // tile read
  output_vec = transfer_read(output[0, 0])
  result_vec = vector.fma(input_vec, kernel_vec, output_vec)
  transfer_write(result_vec, output[0, 0])
}
```

### 3.4 DAP Tiled Vectorization

**Source:** `midend/lib/Conversion/DAPVectorization/DAPVectorization.cpp`  
**Target:** FIR/IIR filter operations

Uses a fixed two-level tiling strategy for 1D signal processing:
- **Outer tile:** `tileStep = 2048` input samples for cache locality.
- **Inner vector loop:** `vlStep = 16` elements processed as `vector<16xf32>` with `vector.splat` + `vector.fma`.

### 3.5 AMX Matmul Tiling

**Source:** `midend/lib/Conversion/MatMulOptimization/MatmulAMX.cpp`  
**Target:** matmul on Intel AMX hardware

Tiles M and N in fixed **16×16** blocks matching AMX tile register dimensions, with K iterating in chunks of **32** BF16 elements. The B matrix is pre-packed for AMX-friendly memory layout, then each tile is computed with `amx.tile_load`, `amx.tile_mul_f`, and `amx.tile_store`.

```
scf.for %m in [0, M) step 16 {
  scf.for %n in [0, N) step 16 {
    acc = amx.tile_zero
    scf.for %k in [0, K) step 32 {
      tA  = amx.tile_load(A[m, k])      // 16x32 BF16
      tB  = amx.tile_load(Bpack[k, n])  // 16x32 BF16
      acc = amx.tile_mul_f(tA, tB, acc) // 16x16 F32
    }
    amx.tile_store(C[m, n], acc)
  }
}
```

### 3.6 Transpose Tiling

**Source:** `midend/lib/Conversion/TransposeOptimization/BuiltinTransposeVectorization.cpp`  
**Target:** rank-2 tensor transpose

Tiles the matrix into `affineVectorSize × affineVectorSize` blocks processed in an `affine.parallel` loop. Each block is loaded with a `vector.transfer_read` using `permutation_map = (d0,d1) -> (d0,d1)` and written back transposed with `permutation_map = (d0,d1) -> (d1,d0)`. Separate `affine.if` branches handle unaligned row and column tails with `vector.create_mask`.

### 3.7 Gemmini Hardware Tiling

**Source:** `midend/lib/Conversion/LowerLinalgToGemmini/LowerLinalgToGemmini.cpp`  
**Target ops:** `linalg.conv_2d_nhwc_fhwc`, `linalg.conv_2d_nchw_fchw`, `linalg.batch_matmul`, `linalg.batch_matmul_transpose_b`

Lowers to Gemmini's systolic-array tile instructions. Convolution kernels are reshaped via im2col (kernel `[F,H,W,C] → [H×W×C, F]`, output `[N,H,W,C] → [N×H×W, C]`) and then dispatched to `gemmini.tile_conv`. Matmul operations are dispatched to `gemmini.tile_matmul`. The hardware handles all internal tiling to match the systolic array dimensions.

---

## 4. Tail Handling

When tensor dimensions are not exact multiples of the tile or vector size, the passes use two complementary mechanisms:

**`affine.if` guards** — branch on whether the current index is within bounds:
```mlir
affine.if #set(%i)[%M] {  // check: i < M
  // full vector computation
}
```

**`vector.create_mask` + `vector.maskedstore`** — compute the number of remaining elements and generate a predicate mask:
```cpp
Value tailLength = builder.create<affine::AffineApplyOp>(loc,
    AffineMap::get(2, 0, -d0 + d1), ValueRange{fixedJV, N});
Value maskVector = builder.create<vector::CreateMaskOp>(loc, i1VecTy, tailLength);
builder.create<MaskedStoreOp>(loc, cptr, indices, maskVector, vecC);
```

---

## 5. CLI Options

Pass options are supplied to `buddy-opt` as `--pass-name{option=value}`.

### `--batchmatmul-tile-optimize`

| Option | Default | Description |
|---|---|---|
| `vec-size` | 16 | SIMD vector width (number of elements per vector register) |
| `kernel-m` | 4 | Number of rows per M-dimension tile |
| `kernel-n` | 2 | Number of N-dimension tiles per step (total N tile = `kernel-n × vec-size`) |

### `--conv-nhwc-fhwc-tile-optimize`

| Option | Default | Description |
|---|---|---|
| `vec-size` | 16 | SIMD vector width for the IC reduction loop |
| `tiling-height` | 1 | Number of output-height tiles |
| `tiling-width` | 1 | Number of output-width tiles |
| `tiling-channel` | 1 | Number of output-channel tiles |

### `--cb-conv-vectorize`

| Option | Default | Description |
|---|---|---|
| `stride` | 1 | Convolution stride |
| `tile` | (empty) | 2D tile size `[rows, cols]`; if empty, uses splitting mode |

---

## 6. Examples

### BatchMatMul with tiling and vectorization

```bash
buddy-opt input.mlir \
  --batchmatmul-tile-optimize="vec-size=16 kernel-m=4 kernel-n=2" \
  --lower-affine \
  --convert-vector-to-llvm \
  --convert-memref-to-llvm \
  --convert-func-to-llvm \
  -o output.mlir
```

### Conv2D NHWC-FHWC with spatial tiling

```bash
buddy-opt input.mlir \
  --conv-nhwc-fhwc-tile-optimize="vec-size=16 tiling-height=4 tiling-width=4 tiling-channel=2" \
  --lower-affine \
  --convert-vector-to-llvm \
  -o output.mlir
```

### Im2col + BatchMatMul tiling (SIMD experiment)

See `examples/SIMDExperiment/conv2d-nchw-fchw-im2col-tiling.mlir` for a hand-written example of a `[4, 32, 8]` tile schedule applied to a `64×576×3136` batch matmul produced by im2col lowering of `conv2d_nchw_fchw`.

### Gemmini tile_matmul

See `examples/GemminiDialect/tile-matmul.mlir`:

```bash
buddy-opt examples/GemminiDialect/tile-matmul.mlir --lower-gemmini
```

This lowers `gemmini.tile_matmul` to the low-level Gemmini intrinsic sequence:
`loop_ws_config_bounds` → `loop_ws_config_addrs_ab` → `loop_ws_config_addrs_dc` → `loop_ws_config_strides_ab` → `loop_ws_config_strides_dc` → `loop_ws`.
