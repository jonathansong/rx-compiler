//===- Ada300HLOps.td - Ada300 High-Level Dialect Ops ----------*- tablegen -*-===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Root tablegen file for the Ada300HL dialect.  This is the single file
// passed to mlir-tablegen / add_mlir_dialect; it includes all sub-files so
// that one tablegen invocation generates the complete set of .inc files.
//
// Include hierarchy:
//   Ada300HLOps.td
//     Ada300HLDialect.td   – dialect registration + base Op class
//     Ada300HLAttrs.td     – enum attrs (NonlinearFunc, SegmentCount,
//                            TensorMode, TensorDataType, MemorySpace,
//                            Part, Major) + LayoutAttr
//     Ada300HLTypes.td     – custom Fp8 type
//     Ada300HLMemory.td    – memory-space type constraints
//     mlir/Interfaces/...  – InferType + SideEffect interfaces
//
// Op groups:
//   1. Vector nonlinear / PWL ops  – pwnl, cvt, vmul_mixed
//   2. Tensor Core ops             – tensor_mma, tensor_sync
//   3. Layout / data-movement ops  – pack, unpack, layout_cast,
//                                    copy_to/from_sram, copy_to/from_vr
//
//===----------------------------------------------------------------------===//

#ifndef ADA300HL_ADA300HLOPS_TD
#define ADA300HL_ADA300HLOPS_TD

include "Ada300HL/Ada300HLDialect.td"
include "Ada300HL/Ada300HLAttrs.td"
include "Ada300HL/Ada300HLTypes.td"
include "Ada300HL/Ada300HLMemory.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// 1. Vector Nonlinear / PWL Ops
//===----------------------------------------------------------------------===//

// ada300hl.pwnl
//   Piecewise-linear nonlinear op covering all five functions
//   (exp, log, sqrt, rsqrt, div).  Use func = #ada300hl.nlfunc<exp> for
//   the exponential case.
//
//   %y = ada300hl.pwnl %x {func = #ada300hl.nlfunc<sqrt>, segments = #ada300hl.segments<32>}
//            : vector<64xf32> -> vector<64xf32>
def Ada300HL_PwnlOp : Ada300HL_Op<"pwnl", [Pure, SameOperandsAndResultType]> {
  let summary = "Apply a PWL nonlinear function (exp, log, sqrt, rsqrt, div) to a vector";
  let description = [{
    Applies a hardware piecewise-linear (PWL) approximation of the selected
    nonlinear function to each element of the input vector.  The `func`
    attribute selects the function (exp, log, sqrt, rsqrt, div); `segments`
    selects the approximation precision (16 or 32 segments).  Lowers to
    `ada300hw.vfpwnl`.
  }];

  let arguments = (ins
    AnyVectorOfAnyRank:$input,
    Ada300HL_NonlinearFuncAttr:$func,
    Ada300HL_SegmentCountAttr:$segments
  );
  let results = (outs AnyVectorOfAnyRank:$result);

  let assemblyFormat =
    "$input attr-dict `:` type($input) `->` type($result)";

  let hasVerifier = 1;
}

// ada300hl.cvt
//   Mixed-precision type conversion (e.g. f16 -> fp8).  The `dst_type`
//   attribute encodes the destination element type; `part` selects which
//   half of a paired src register is converted.
//
//   %y = ada300hl.cvt %x {dst_type = #ada300hl.dtype<fp8>, part = #ada300hl.part<low>}
//            : vector<64xf16> -> vector<64xf16>
def Ada300HL_CvtOp : Ada300HL_Op<"cvt", [Pure]> {
  let summary = "Mixed-precision element-wise type conversion";
  let description = [{
    Mirrors the `vfcvt.low/high.*` instruction family.  The `dst_type`
    attribute records the intended hardware destination format (this may differ
    from the MLIR result element type when packing is deferred).  `part`
    selects the low or high half of the source register pair.

    Lowers to `ada300hw.vfcvt`.
  }];

  let arguments = (ins
    AnyVectorOfAnyRank:$input,
    Ada300HL_TensorDataTypeAttr:$dst_type,
    Ada300HL_PartAttr:$part
  );
  let results = (outs AnyVectorOfAnyRank:$result);

  let assemblyFormat =
    "$input attr-dict `:` type($input) `->` type($result)";
}

// ada300hl.vmul_mixed
//   Mixed-precision vector multiply.  Models the `vfmul.low/high.*`
//   instruction pair at the high level without fixing the exact encoding.
//
//   %z = ada300hl.vmul_mixed %a, %b
//            {acc_type = #ada300hl.dtype<fp32>, part = #ada300hl.part<low>}
//            : vector<64xbf16>, vector<64xbf16> -> vector<64xf32>
def Ada300HL_VMulMixedOp : Ada300HL_Op<"vmul_mixed", [Pure]> {
  let summary = "Mixed-precision element-wise vector multiply";
  let description = [{
    Multiplies two vectors whose element type may be lower-precision (e.g.
    bf16) and accumulates into a higher-precision result (e.g. f32).  The
    `acc_type` attribute records the hardware accumulator element type;
    `part` selects the low or high sub-vector.

    Lowers to `ada300hw.vfmul_low` or `ada300hw.vfmul_high` depending on
    `part`.
  }];

  let arguments = (ins
    AnyVectorOfAnyRank:$lhs,
    AnyVectorOfAnyRank:$rhs,
    Ada300HL_TensorDataTypeAttr:$acc_type,
    Ada300HL_PartAttr:$part
  );
  let results = (outs AnyVectorOfAnyRank:$result);

  let assemblyFormat =
    "$lhs `,` $rhs attr-dict"
    "`:` type($lhs) `,` type($rhs) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// 2. Tensor Core Ops
//===----------------------------------------------------------------------===//

// ada300hl.tensor_mma
//   High-level Tensor Core matrix-multiply-accumulate.  Holds all the sizing,
//   data-type, and layout information so that later passes can make informed
//   decisions before expanding to the ISA-level config+execute sequence.
//
//   ada300hl.tensor_mma %dst, %act, %wht
//       {mode = #ada300hl.tensor_mode<inner>, rhs_transposed = false,
//        row_size = 16, col_size = 16, acc_size = 16,
//        out_type = #ada300hl.dtype<bf16>, act_type = #ada300hl.dtype<bf16>,
//        wht_type = #ada300hl.dtype<bf16>, blk_cnt_a = 4, blk_cnt_w = 4}
//       : memref<...>, memref<...>, memref<...>
def Ada300HL_TensorMmaOp : Ada300HL_Op<"tensor_mma",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "ADA300 Tensor Core matrix-multiply-accumulate";
  let description = [{
    Captures the complete high-level semantics of a Tensor Core matrix
    operation: output accumulation mode (inner / outer product), matrix
    dimensions, operand data types, and the block-iteration counts used to
    tile over SRAM buffers.

    All three operands are memrefs.  `dst` is both read and written (in-place
    accumulation); `act` and `wht` are read-only inputs.  The verifier checks
    that sizes are positive and block counts are legal.

    Lowers to the sequence:
      ada300hw.set_gmm_cfg → ada300hw.set_gmm_type → ada300hw.set_gmm_iter
      → ada300hw.gmma_mm or gmma_mt → ada300hw.tcsync
  }];

  let arguments = (ins
    AnyMemRef:$dst,
    AnyMemRef:$act,
    AnyMemRef:$wht,
    Ada300HL_TensorModeAttr:$mode,
    BoolAttr:$rhs_transposed,
    I32Attr:$row_size,
    I32Attr:$col_size,
    I32Attr:$acc_size,
    Ada300HL_TensorDataTypeAttr:$out_type,
    Ada300HL_TensorDataTypeAttr:$act_type,
    Ada300HL_TensorDataTypeAttr:$wht_type,
    I32Attr:$blk_cnt_a,
    I32Attr:$blk_cnt_w
  );

  let assemblyFormat =
    "$dst `,` $act `,` $wht attr-dict"
    "`:` type($dst) `,` type($act) `,` type($wht)";

  let hasVerifier = 1;
}

// ada300hl.tensor_sync
//   Waits until the Tensor Core has completed all pending operations.
//   Must appear after every tensor_mma before its result can be consumed.
//   Lowers to ada300hw.tcsync.
def Ada300HL_TensorSyncOp : Ada300HL_Op<"tensor_sync",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Wait for Tensor Core execution to complete";
  let description = [{
    An abstract synchronisation barrier.  MLIR passes must not move or remove
    this op across Tensor Core reads or writes.  Lowers to `ada300hw.tcsync`.
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// 3. Layout / Data-Movement Ops
//===----------------------------------------------------------------------===//

// ada300hl.pack
//   Reformat a row-major or column-major memref into the blocked layout that
//   the Tensor Core requires, placing the result in an SRAM buffer.
//
//   %packed = ada300hl.pack %src {layout = #ada300hl.layout<...>}
//               : memref<64x64xf16> -> memref<4x4x16x16xf16>
def Ada300HL_PackOp : Ada300HL_Op<"pack",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Pack a matrix buffer into the Tensor Core blocked layout";
  let description = [{
    Reads `src` and writes `dst` in the tiled/blocked layout described by the
    `layout` attribute.  Both `src` and `dst` are supplied by the caller; this
    op models the reformat in-place so that subsequent lowering can emit the
    appropriate data-movement or permutation code.
  }];

  let arguments = (ins
    AnyMemRef:$src,
    AnyMemRef:$dst,
    Ada300HL_LayoutAttr:$layout
  );

  let assemblyFormat =
    "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}

// ada300hl.unpack
//   Inverse of ada300hl.pack: converts from blocked layout back to the
//   requested storage order.
def Ada300HL_UnpackOp : Ada300HL_Op<"unpack",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Unpack a Tensor Core blocked-layout buffer into a flat matrix";
  let description = [{
    Inverse of `ada300hl.pack`.  Reads from a blocked-layout `src` buffer and
    writes linearised data to `dst`.
  }];

  let arguments = (ins
    AnyMemRef:$src,
    AnyMemRef:$dst,
    Ada300HL_LayoutAttr:$layout
  );

  let assemblyFormat =
    "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}

// ada300hl.layout_cast
//   Reinterprets the layout attribute of a memref without moving data.
//   Used to annotate a buffer as blocked (or otherwise) in preparation for
//   pack/pack-free Tensor Core lowering paths.
def Ada300HL_LayoutCastOp : Ada300HL_Op<"layout_cast",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Reinterpret buffer layout without moving data";
  let description = [{
    A view-like op that attaches a new `layout` descriptor to an existing
    memref.  No data movement occurs; this op exists to make layout information
    explicit in the IR so that downstream passes can reason about it without
    relying on implicit type attributes.
  }];

  let arguments = (ins
    AnyMemRef:$src,
    Ada300HL_LayoutAttr:$layout
  );
  let results = (outs AnyMemRef:$result);

  let assemblyFormat =
    "$src attr-dict `:` type($src) `->` type($result)";
}

// ada300hl.copy_to_sram
//   Copies data from global memory (or a VR-backed buffer) into SRAM so that
//   the Tensor Core can access it.
def Ada300HL_CopyToSramOp : Ada300HL_Op<"copy_to_sram",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Copy a buffer from global/VR memory into on-chip SRAM";
  let description = [{
    Models a DMA or explicit copy from higher-latency memory into the Tensor
    Core's local SRAM.  Both `src` and `dst` must be supplied.  The verifier
    may check that `dst` has `sram` memory-space annotation.
  }];

  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst);
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}

// ada300hl.copy_from_sram
//   Copies data out of SRAM back to global memory or a VR-backed buffer.
def Ada300HL_CopyFromSramOp : Ada300HL_Op<"copy_from_sram",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Copy a buffer from on-chip SRAM to global/VR memory";

  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst);
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}

// ada300hl.copy_to_vr
//   Copies data from SRAM or global memory into a VR-backed buffer.
def Ada300HL_CopyToVrOp : Ada300HL_Op<"copy_to_vr",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Copy a buffer into vector-register-backed storage";

  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst);
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}

// ada300hl.copy_from_vr
//   Copies data out of a VR-backed buffer to SRAM or global memory.
def Ada300HL_CopyFromVrOp : Ada300HL_Op<"copy_from_vr",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Copy a vector-register-backed buffer to global/SRAM memory";

  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst);
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}

//===----------------------------------------------------------------------===//
// 4. Element-wise Activation / Arithmetic Ops
//===----------------------------------------------------------------------===//

// ada300hl.sin
//   Element-wise sin(x) via the hardware PWL unit.
//   Lowers to ada300hw.vfpwnl {func=sin}.
def Ada300HL_SinOp : Ada300HL_Op<"sin", [Pure, SameOperandsAndResultType]> {
  let summary = "Compute element-wise sin(x) via the ADA300 PWL unit";
  let description = [{
    Applies the sine function to each element of the input vector using the
    hardware piecewise-linear (PWL) approximation unit.  Lowers to
    `ada300hw.vfpwnl` with `func = sin`.
  }];
  let arguments = (ins
    AnyVectorOfAnyRank:$input,
    Ada300HL_SegmentCountAttr:$segments
  );
  let results = (outs AnyVectorOfAnyRank:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

// ada300hl.cos
//   Element-wise cos(x) via the hardware PWL unit.
//   Lowers to ada300hw.vfpwnl {func=cos}.
def Ada300HL_CosOp : Ada300HL_Op<"cos", [Pure, SameOperandsAndResultType]> {
  let summary = "Compute element-wise cos(x) via the ADA300 PWL unit";
  let description = [{
    Applies the cosine function to each element of the input vector using the
    hardware piecewise-linear (PWL) approximation unit.  Lowers to
    `ada300hw.vfpwnl` with `func = cos`.
  }];
  let arguments = (ins
    AnyVectorOfAnyRank:$input,
    Ada300HL_SegmentCountAttr:$segments
  );
  let results = (outs AnyVectorOfAnyRank:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

// ada300hl.sigmoid
//   Element-wise sigmoid: f(x) = 1 / (1 + exp(-x)).
//   Decomposes to negation + exp + scalar add + div; all PWL work uses
//   the `segments` table.  Lowers to a short sequence of ada300hw ops.
def Ada300HL_SigmoidOp : Ada300HL_Op<"sigmoid", [Pure, SameOperandsAndResultType]> {
  let summary = "Compute element-wise sigmoid(x) = 1 / (1 + exp(-x))";
  let description = [{
    High-level sigmoid whose lowering expands to:
      vfpwnl(neg(x), exp, segments) → scalar_add(1) → vfpwnl(reciprocal, div)
    The `segments` attribute controls approximation precision for all internal
    PWL calls.
  }];
  let arguments = (ins
    AnyVectorOfAnyRank:$input,
    Ada300HL_SegmentCountAttr:$segments
  );
  let results = (outs AnyVectorOfAnyRank:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

// ada300hl.silu
//   Element-wise SiLU (Sigmoid Linear Unit): f(x) = x * sigmoid(x).
//   Requires exp via the PWL unit.
def Ada300HL_SiluOp : Ada300HL_Op<"silu", [Pure, SameOperandsAndResultType]> {
  let summary = "Compute element-wise SiLU(x) = x * sigmoid(x)";
  let description = [{
    SiLU activation function.  Lowers to `ada300hl.sigmoid` followed by an
    element-wise vector multiply.  The `segments` attribute is forwarded to the
    internal sigmoid expansion.
  }];
  let arguments = (ins
    AnyVectorOfAnyRank:$input,
    Ada300HL_SegmentCountAttr:$segments
  );
  let results = (outs AnyVectorOfAnyRank:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

// ada300hl.softplus
//   Element-wise softplus: f(x) = log(1 + exp(x)).
//   Uses both exp and log PWL calls.
def Ada300HL_SoftplusOp : Ada300HL_Op<"softplus", [Pure, SameOperandsAndResultType]> {
  let summary = "Compute element-wise softplus(x) = log(1 + exp(x))";
  let description = [{
    Soft version of ReLU.  Lowers to:
      vfpwnl(x, exp, segments) → scalar_add(1) → vfpwnl(log, segments)
    The same `segments` table is used for both PWL calls.
  }];
  let arguments = (ins
    AnyVectorOfAnyRank:$input,
    Ada300HL_SegmentCountAttr:$segments
  );
  let results = (outs AnyVectorOfAnyRank:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

// ada300hl.x2
//   Element-wise square: f(x) = x * x.
//   Pure arithmetic; no PWL unit required.
//   Lowers to a single vector-multiply instruction.
def Ada300HL_X2Op : Ada300HL_Op<"x2", [Pure, SameOperandsAndResultType]> {
  let summary = "Compute element-wise x^2 (element-wise square)";
  let description = [{
    Multiplies each element of the input vector by itself.  No PWL hardware is
    needed; lowers to an element-wise vector multiply in the ada300hw dialect.
  }];
  let arguments = (ins AnyVectorOfAnyRank:$input);
  let results = (outs AnyVectorOfAnyRank:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// 5. Reduction Ops
//===----------------------------------------------------------------------===//

// ada300hl.redmax
//   Reduce a vector to its maximum element along the given axis.
//   Result type is typically the element type or a rank-(n-1) vector.
def Ada300HL_RedmaxOp : Ada300HL_Op<"redmax", [Pure]> {
  let summary = "Reduce a vector to its maximum element";
  let description = [{
    Computes the maximum value across all elements along `axis`.  The result
    has the same element type as the input but with the reduced dimension
    removed.  Lowers to a tree-reduction sequence of compare+select or to a
    dedicated vector-reduce instruction when available.
  }];
  let arguments = (ins
    AnyVectorOfAnyRank:$input,
    I32Attr:$axis
  );
  let results = (outs AnyType:$result);
  let assemblyFormat =
    "$input attr-dict `:` type($input) `->` type($result)";
}

// ada300hl.redsum
//   Reduce a vector to the sum of its elements along the given axis.
//   Also known as ReduceSum.
def Ada300HL_RedsumOp : Ada300HL_Op<"redsum", [Pure]> {
  let summary = "Reduce a vector to the sum of its elements (ReduceSum)";
  let description = [{
    Computes the sum of all elements along `axis`.  The result has the same
    element type as the input with the reduced dimension removed.  Lowers to a
    tree-reduction or horizontal-add sequence.
  }];
  let arguments = (ins
    AnyVectorOfAnyRank:$input,
    I32Attr:$axis
  );
  let results = (outs AnyType:$result);
  let assemblyFormat =
    "$input attr-dict `:` type($input) `->` type($result)";
}

// ada300hl.rms
//   Compute the Root-Mean-Square of a vector: rms(x) = sqrt(mean(x^2)).
//   Uses the PWL sqrt unit; `segments` selects approximation precision.
def Ada300HL_RmsOp : Ada300HL_Op<"rms", [Pure]> {
  let summary = "Compute root-mean-square of a vector: sqrt(mean(x^2))";
  let description = [{
    Computes rms(x) = sqrt( (1/n) * sum_i(x_i^2) ) over the whole input
    vector.  Lowers to x2 → redsum → scalar-div-by-n → ada300hl.pwnl(sqrt).
    The result is a scalar value with the element type of the input.
  }];
  let arguments = (ins
    AnyVectorOfAnyRank:$input,
    Ada300HL_SegmentCountAttr:$segments
  );
  let results = (outs AnyType:$result);
  let assemblyFormat =
    "$input attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// 6. Compound Normalization Ops
//===----------------------------------------------------------------------===//

// ada300hl.softmax
//   Numerically-stable softmax along `axis`:
//     softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
//   Input and output have the same vector type.
def Ada300HL_SoftmaxOp : Ada300HL_Op<"softmax", [Pure, SameOperandsAndResultType]> {
  let summary = "Compute numerically-stable softmax along a vector axis";
  let description = [{
    Applies the softmax function to the input vector along `axis`.  The
    numerically-stable form subtracts the max before exponentiation.  Lowers
    to:
      redmax → broadcast-subtract → pwnl(exp) → redsum → broadcast-divide
    All PWL calls use the `segments` approximation table.
  }];
  let arguments = (ins
    AnyVectorOfAnyRank:$input,
    Ada300HL_SegmentCountAttr:$segments,
    I32Attr:$axis
  );
  let results = (outs AnyVectorOfAnyRank:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

// ada300hl.rms_norm
//   RMS Normalization: y = x / rms(x) * scale.
//   Used in LLM transformer blocks (e.g. LLaMA RMSNorm).
//   `scale` is a learned per-channel weight vector with the same type as `input`.
def Ada300HL_RmsNormOp : Ada300HL_Op<"rms_norm", [Pure]> {
  let summary = "Apply RMS normalization: y = x / rms(x) * scale";
  let description = [{
    Normalises `input` by its root-mean-square and then scales element-wise by
    the `scale` vector.  Lowers to:
      ada300hl.rms(input) → broadcast-div → element-wise-mul(scale)
    The `segments` attribute is forwarded to the internal `rms` expansion.

    Input and scale must share the same vector type.  The result has the same
    type as `input`.
  }];
  let arguments = (ins
    AnyVectorOfAnyRank:$input,
    AnyVectorOfAnyRank:$scale,
    Ada300HL_SegmentCountAttr:$segments
  );
  let results = (outs AnyVectorOfAnyRank:$result);
  let assemblyFormat =
    "$input `,` $scale attr-dict `:` type($input) `,` type($scale) `->` type($result)";
  let hasVerifier = 1;
}

#endif // ADA300HL_ADA300HLOPS_TD
