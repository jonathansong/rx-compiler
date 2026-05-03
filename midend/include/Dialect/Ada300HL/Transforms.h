//===- Transforms.h - Ada300HL Transformation/Lowering Entrypoints -*-C++-*===//
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
// Declares the public API for lowering passes that operate on the Ada300HL
// dialect.
//
// Currently provided lowering:
//   Ada300HL → Ada300HW  (Ada300HLToAda300HW)
//     Maps every ada300hl op to the equivalent ada300hw ISA-level sequence.
//     After this pass the IR contains no ada300hl ops.
//
// Typical pipeline:
//   math / vector / linalg
//     ↓  (upstream conversion passes)
//   ada300hl
//     ↓  populateAda300HLToAda300HWPatterns (this pass)
//   ada300hw
//     ↓  (future Ada300HWToLLVM pass)
//   LLVM dialect / custom intrinsics
//
//===----------------------------------------------------------------------===//

#ifndef ADA300HL_TRANSFORMS_H
#define ADA300HL_TRANSFORMS_H

namespace mlir {

class RewritePatternSet;

namespace buddy {

/// Populate `patterns` with rewrite patterns that lower every Ada300HL op to
/// the corresponding Ada300HW op sequence.
///
/// Op mappings:
///   ada300hl.exp         → ada300hw.vfpwnl {func=exp, masked=false}
///   ada300hl.pwnl        → ada300hw.vfpwnl {func=<attr>, masked=false}
///   ada300hl.cvt         → ada300hw.vfcvt
///   ada300hl.vmul_mixed  → ada300hw.vfmul_low  (part=low)
///                          ada300hw.vfmul_high (part=high)
///   ada300hl.tensor_mma  → ada300hw.set_gmm_cfg
///                          ada300hw.set_gmm_type
///                          ada300hw.set_gmm_iter
///                          ada300hw.gmma_mm / gmma_mt
///   ada300hl.tensor_sync → ada300hw.tcsync
///   ada300hl.pack        → ada300hw (erased; data movement is target-specific)
///   ada300hl.unpack      → ada300hw (erased; data movement is target-specific)
///   ada300hl.layout_cast → (erased, layout info has been consumed by lowering)
///   ada300hl.copy_*      → (lowered to memref.copy as a placeholder; a real
///                           target backend would replace these with DMA calls)
void populateAda300HLToAda300HWPatterns(RewritePatternSet &patterns);

/// Register the Ada300HL → Ada300HW lowering pass with the pass manager.
void registerLowerAda300HLToAda300HWPass();

/// Register the Math → Ada300HL lowering pass with the pass manager.
/// Lowers math.exp / math.log / math.sqrt / math.rsqrt on vector types to
/// the corresponding ada300hl.exp / ada300hl.pwnl ops.
void registerMathToAda300HLPass();

/// Register the Linalg → Ada300HL lowering pass with the pass manager.
/// Lowers bufferized linalg.matmul to ada300hl.tensor_mma + tensor_sync.
void registerLinalgToAda300HLPass();

/// Register the TOSA → Ada300HL lowering pass with the pass manager.
///
/// Directly lowers selected TOSA ops to Ada300HL ops for static-shape tensors:
///   tosa.exp          → ada300hl.pwnl {func=exp,   segments=16}
///   tosa.log          → ada300hl.pwnl {func=log,   segments=16}
///   tosa.rsqrt        → ada300hl.pwnl {func=rsqrt, segments=16}
///   tosa.sigmoid      → ada300hl.sigmoid {segments=16}
///   tosa.matmul       → linalg.batch_matmul (tensor form, for downstream
///                        LinalgToAda300HL after bufferization)
///
/// Ops with dynamic dimensions are left unchanged for the standard TOSA
/// lowering pipeline.
void registerTosaToAda300HLPass();

/// Register the Ada300HL → rx-ops library-call lowering pass.
/// Replaces every Ada300HL op with a call to the corresponding
/// rxops_bridge_* wrapper (rx_ops_bridge.h / rx_ops_bridge.c).
void registerLowerAda300HLToRxOpsPass();

} // namespace buddy
} // namespace mlir

#endif // ADA300HL_TRANSFORMS_H
