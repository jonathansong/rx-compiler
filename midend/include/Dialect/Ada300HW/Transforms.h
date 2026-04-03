//===- Transforms.h - Ada300HW Lowering Pass Entrypoints -----------C++-*-===//
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
// Declares the public API for passes that lower Ada300HW ISA-level ops to
// LLVM dialect (RISC-V inline asm / intrinsics).
//
// Provided lowering:
//   Ada300HW → LLVM  (LowerAda300HWToLLVM)
//
//     Vector ops:
//       ada300hw.vfpwnl    → LLVM inline asm: vfpwnl.<func>.<segs>
//       ada300hw.vfcvt     → LLVM inline asm: vfcvt.<part>.*
//       ada300hw.vfmul_low → LLVM inline asm: vfmul.low.*
//       ada300hw.vfmul_high→ LLVM inline asm: vfmul.high.*
//
//     Tensor Core config ops → LLVM inline asm (side-effect CSR/config writes):
//       ada300hw.set_gmm_cfg
//       ada300hw.set_gmm_type
//       ada300hw.set_gmm_iter
//
//     Tensor Core execute ops → LLVM inline asm (memory pointer args):
//       ada300hw.gmm_mm    → gmm.mm
//       ada300hw.gmma_mm   → gmma.mm
//       ada300hw.gmma_mt   → gmma.mt
//       ada300hw.gmv_mm    → gmv.mm
//       ada300hw.gmva_mm   → gmva.mm
//       ada300hw.gmva_mt   → gmva.mt
//
//     Sync op:
//       ada300hw.tcsync    → LLVM inline asm: tcsync
//
//===----------------------------------------------------------------------===//

#ifndef ADA300HW_TRANSFORMS_H
#define ADA300HW_TRANSFORMS_H

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;

namespace buddy {

/// Populate `patterns` with rewrite patterns that lower all Ada300HW ops to
/// LLVM dialect operations (inline asm or intrinsic calls).
///
/// `converter` must be an LLVMTypeConverter configured for the target triple
/// (typically RISC-V 64-bit with the ADA300 custom extension).
void populateAda300HWToLLVMPatterns(LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns);

/// Register the Ada300HW → LLVM lowering pass.
void registerLowerAda300HWToLLVMPass();

} // namespace buddy
} // namespace mlir

#endif // ADA300HW_TRANSFORMS_H
