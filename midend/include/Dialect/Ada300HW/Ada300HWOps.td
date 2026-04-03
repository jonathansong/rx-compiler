//===- Ada300HWOps.td - Ada300 Hardware/ISA Dialect Ops --------*- tablegen -*-===//
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
// Root tablegen file for the Ada300HW dialect.  Include hierarchy:
//   Ada300HWOps.td
//     Ada300HWDialect.td  – dialect registration + base Op class
//     Ada300HWAttrs.td    – re-exports Ada300HL attrs (shared namespace)
//     mlir/Interfaces/SideEffectInterfaces.td
//
// Operation groups:
//   1. Vector ops   – vfpwnl, vfcvt, vfmul_low, vfmul_high
//   2. Config ops   – set_gmm_cfg, set_gmm_type, set_gmm_iter
//   3. Execute ops  – gmm_mm, gmma_mm, gmma_mt, gmv_mm, gmva_mm, gmva_mt
//   4. Sync op      – tcsync
//
//===----------------------------------------------------------------------===//

#ifndef ADA300HW_ADA300HWOPS_TD
#define ADA300HW_ADA300HWOPS_TD

include "Ada300HW/Ada300HWDialect.td"
include "Ada300HW/Ada300HWAttrs.td"
include "mlir/IR/OpDefinition.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// 1. Vector Ops
//===----------------------------------------------------------------------===//

// ada300hw.vfpwnl
//   Direct counterpart to the ADA300 `vfpwnl` instruction.
//   Computes a piecewise-linear approximation of `func` applied element-wise.
//   The `masked` flag indicates whether the operation uses a vector mask.
//
//   %y = ada300hw.vfpwnl %x {func = #ada300hl.nlfunc<exp>,
//                             segments = #ada300hl.segments<16>,
//                             masked = false}
//            : vector<64xf16> -> vector<64xf16>
def Ada300HW_VfpwnlOp : Ada300HW_Op<"vfpwnl",
    [Pure, SameOperandsAndResultType]> {
  let summary = "ADA300 vfpwnl: piecewise-linear nonlinear function on a vector";
  let description = [{
    Maps directly to the `vfpwnl` machine instruction.  `func` selects the
    mathematical function; `segments` selects the 16- or 32-segment table;
    `masked` indicates whether a vector mask register is in effect (mask
    register not yet modelled as an explicit operand).

    This op is `Pure` when `masked = false`.  Masked execution may be modelled
    as side-effectful in a future extension.
  }];

  let arguments = (ins
    AnyVectorOfAnyRank:$input,
    Ada300HL_NonlinearFuncAttr:$func,
    Ada300HL_SegmentCountAttr:$segments,
    BoolAttr:$masked
  );
  let results = (outs AnyVectorOfAnyRank:$result);

  let assemblyFormat =
    "$input attr-dict `:` type($input) `->` type($result)";

  let hasVerifier = 1;
}

// ada300hw.vfcvt
//   Maps to the `vfcvt.low.*` / `vfcvt.high.*` instruction family.
//   Converts between floating-point precisions with explicit half selection.
//
//   %y = ada300hw.vfcvt %x {dst_type = #ada300hl.dtype<fp8>,
//                            part = #ada300hl.part<high>}
//            : vector<64xf16> -> vector<64xf16>
def Ada300HW_VfcvtOp : Ada300HW_Op<"vfcvt", [Pure]> {
  let summary = "ADA300 vfcvt: floating-point precision conversion";
  let description = [{
    Corresponds to the `vfcvt.low/high.*` instruction family.  The `dst_type`
    attribute records the hardware destination element format; `part` selects
    which register half is converted.  Verifier enforces that the src/dst type
    combination is ISA-legal.
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

// ada300hw.vfmul_low
//   Maps to the `vfmul.low.*` instruction: multiplies two lower-precision
//   vectors and accumulates into a higher-precision result (low sub-vector).
//
//   %z = ada300hw.vfmul_low %a, %b
//            {src_a_type = #ada300hl.dtype<bf16>,
//             src_b_type = #ada300hl.dtype<bf16>,
//             dst_type   = #ada300hl.dtype<fp16>}
//            : vector<64xbf16>, vector<64xbf16> -> vector<64xf32>
def Ada300HW_VfmulLowOp : Ada300HW_Op<"vfmul_low", [Pure]> {
  let summary = "ADA300 vfmul.low: mixed-precision multiply (low sub-vector)";

  let arguments = (ins
    AnyVectorOfAnyRank:$lhs,
    AnyVectorOfAnyRank:$rhs,
    Ada300HL_TensorDataTypeAttr:$src_a_type,
    Ada300HL_TensorDataTypeAttr:$src_b_type,
    Ada300HL_TensorDataTypeAttr:$dst_type
  );
  let results = (outs AnyVectorOfAnyRank:$result);

  let assemblyFormat =
    "$lhs `,` $rhs attr-dict"
    "`:` type($lhs) `,` type($rhs) `->` type($result)";
}

// ada300hw.vfmul_high
//   Same as vfmul_low but targets the high sub-vector.
def Ada300HW_VfmulHighOp : Ada300HW_Op<"vfmul_high", [Pure]> {
  let summary = "ADA300 vfmul.high: mixed-precision multiply (high sub-vector)";

  let arguments = (ins
    AnyVectorOfAnyRank:$lhs,
    AnyVectorOfAnyRank:$rhs,
    Ada300HL_TensorDataTypeAttr:$src_a_type,
    Ada300HL_TensorDataTypeAttr:$src_b_type,
    Ada300HL_TensorDataTypeAttr:$dst_type
  );
  let results = (outs AnyVectorOfAnyRank:$result);

  let assemblyFormat =
    "$lhs `,` $rhs attr-dict"
    "`:` type($lhs) `,` type($rhs) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// 2. Tensor Core Configuration Ops
//===----------------------------------------------------------------------===//
// All three config ops have side effects because they write hardware
// configuration registers.  They must not be DCE-eliminated or reordered
// past the corresponding gmm/gmva execute op.

// ada300hw.set_gmm_cfg
//   Writes the gmm_cfg hardware register: matrix dimensions and accumulation mode.
//
//   ada300hw.set_gmm_cfg {row_size = 16, col_size = 16, acc_size = 16,
//                          mode = #ada300hl.tensor_mode<inner>}
def Ada300HW_SetGmmCfgOp : Ada300HW_Op<"set_gmm_cfg",
    [MemoryEffects<[MemWrite]>]> {
  let summary = "Write gmm_cfg configuration register";
  let description = [{
    Configures the Tensor Core matrix dimensions (`row_size`, `col_size`,
    `acc_size`) and the accumulation mode (inner or outer product).  Must
    precede any `gmm_mm`, `gmma_mm`, or related execute op in the same block
    that relies on these parameters.
  }];

  let arguments = (ins
    I32Attr:$row_size,
    I32Attr:$col_size,
    I32Attr:$acc_size,
    Ada300HL_TensorModeAttr:$mode
  );

  let assemblyFormat = "attr-dict";

  let hasVerifier = 1;
}

// ada300hw.set_gmm_type
//   Writes the gmm_type hardware register: data types for output, activation,
//   and weight operands.
//
//   ada300hw.set_gmm_type {out_type = #ada300hl.dtype<bf16>,
//                           act_type = #ada300hl.dtype<bf16>,
//                           wht_type = #ada300hl.dtype<bf16>}
def Ada300HW_SetGmmTypeOp : Ada300HW_Op<"set_gmm_type",
    [MemoryEffects<[MemWrite]>]> {
  let summary = "Write gmm_type configuration register";
  let description = [{
    Sets the element data types for the Tensor Core output (`out_type`),
    activation (`act_type`), and weight (`wht_type`) operands.  Must be
    written before the corresponding execute op.
  }];

  let arguments = (ins
    Ada300HL_TensorDataTypeAttr:$out_type,
    Ada300HL_TensorDataTypeAttr:$act_type,
    Ada300HL_TensorDataTypeAttr:$wht_type
  );

  let assemblyFormat = "attr-dict";
}

// ada300hw.set_gmm_iter
//   Writes the gmm_iter hardware register: iteration block counts.
//
//   ada300hw.set_gmm_iter {blk_cnt_a = 4, blk_cnt_w = 4}
def Ada300HW_SetGmmIterOp : Ada300HW_Op<"set_gmm_iter",
    [MemoryEffects<[MemWrite]>]> {
  let summary = "Write gmm_iter configuration register";
  let description = [{
    Configures the number of SRAM blocks that the Tensor Core will iterate
    over for the activation (`blk_cnt_a`) and weight (`blk_cnt_w`) matrices.
  }];

  let arguments = (ins
    I32Attr:$blk_cnt_a,
    I32Attr:$blk_cnt_w
  );

  let assemblyFormat = "attr-dict";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 3. Tensor Core Execute Ops
//===----------------------------------------------------------------------===//
// All execute ops have both memory-read and memory-write side effects:
// they read from activation/weight SRAM buffers and write to the destination.
// They must not be moved relative to the preceding config ops or tcsync.

// ada300hw.gmm_mm
//   Non-accumulating matrix-multiply (clears dst before writing).
def Ada300HW_GmmMmOp : Ada300HW_Op<"gmm_mm",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "ADA300 gmm_mm: Tensor Core non-accumulating matmul";
  let description = [{
    Executes a Tensor Core matrix multiply that clears the destination
    buffer before writing results.  All three buffers must reside in SRAM.
  }];

  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$act, AnyMemRef:$wht);
  let assemblyFormat =
    "$dst `,` $act `,` $wht attr-dict"
    "`:` type($dst) `,` type($act) `,` type($wht)";
}

// ada300hw.gmma_mm
//   Accumulating matrix-multiply (adds into dst).
def Ada300HW_GmmaMmOp : Ada300HW_Op<"gmma_mm",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "ADA300 gmma_mm: Tensor Core accumulating matmul";
  let description = [{
    Executes a Tensor Core matrix multiply-accumulate.  The result is added
    into the existing contents of `dst`.  All three buffers must reside in
    SRAM.
  }];

  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$act, AnyMemRef:$wht);
  let assemblyFormat =
    "$dst `,` $act `,` $wht attr-dict"
    "`:` type($dst) `,` type($act) `,` type($wht)";
}

// ada300hw.gmma_mt
//   Accumulating matrix-multiply with RHS (weight) transposition.
def Ada300HW_GmmaMtOp : Ada300HW_Op<"gmma_mt",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "ADA300 gmma_mt: Tensor Core accumulating matmul, weight transposed";

  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$act, AnyMemRef:$wht);
  let assemblyFormat =
    "$dst `,` $act `,` $wht attr-dict"
    "`:` type($dst) `,` type($act) `,` type($wht)";
}

// ada300hw.gmv_mm
//   Non-accumulating matrix-vector multiply.
def Ada300HW_GmvMmOp : Ada300HW_Op<"gmv_mm",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "ADA300 gmv_mm: Tensor Core non-accumulating matrix-vector multiply";

  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$mat, AnyMemRef:$vec);
  let assemblyFormat =
    "$dst `,` $mat `,` $vec attr-dict"
    "`:` type($dst) `,` type($mat) `,` type($vec)";
}

// ada300hw.gmva_mm
//   Accumulating matrix-vector multiply.
def Ada300HW_GmvaMmOp : Ada300HW_Op<"gmva_mm",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "ADA300 gmva_mm: Tensor Core accumulating matrix-vector multiply";

  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$mat, AnyMemRef:$vec);
  let assemblyFormat =
    "$dst `,` $mat `,` $vec attr-dict"
    "`:` type($dst) `,` type($mat) `,` type($vec)";
}

// ada300hw.gmva_mt
//   Accumulating matrix-vector multiply with matrix transposition.
def Ada300HW_GmvaMtOp : Ada300HW_Op<"gmva_mt",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "ADA300 gmva_mt: Tensor Core accumulating matrix-vector multiply, matrix transposed";

  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$mat, AnyMemRef:$vec);
  let assemblyFormat =
    "$dst `,` $mat `,` $vec attr-dict"
    "`:` type($dst) `,` type($mat) `,` type($vec)";
}

//===----------------------------------------------------------------------===//
// 4. Synchronisation
//===----------------------------------------------------------------------===//

// ada300hw.tcsync
//   Maps to the `tcsync` instruction.  Blocks the processor until the Tensor
//   Core's completion flag is set.  Must appear after all gmm/gmva execute ops
//   whose results will be consumed.
//
//   ada300hw.tcsync
def Ada300HW_TcsyncOp : Ada300HW_Op<"tcsync",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "ADA300 tcsync: poll Tensor Core completion flag";
  let description = [{
    Blocks execution until the Tensor Core signals that all pending operations
    have completed.  This op must not be moved before any preceding gmm/gmva
    execute op in the same block, nor after the point where the destination
    buffer is first consumed.

    Models both read and write side effects to prevent DCE and motion past
    adjacent memory operations.
  }];

  let assemblyFormat = "attr-dict";
}

#endif // ADA300HW_ADA300HWOPS_TD
