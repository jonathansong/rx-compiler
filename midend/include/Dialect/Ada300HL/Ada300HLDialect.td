//===- Ada300HLDialect.td - Ada300 High-Level Dialect Base ------*- tablegen -*-===//
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
// Defines only the Ada300HL dialect registration record and the base Op class.
// Attribute, type, and memory definitions live in their own sub-files:
//   Ada300HLAttrs.td  – enum attrs + LayoutAttr
//   Ada300HLTypes.td  – custom Fp8 type
//   Ada300HLMemory.td – memory-space type constraints
//   Ada300HLOps.td    – all op definitions (root tablegen file)
//
//===----------------------------------------------------------------------===//

#ifndef ADA300HL_ADA300HLDIALECT_TD
#define ADA300HL_ADA300HLDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def Ada300HL_Dialect : Dialect {
  let name = "ada300hl";
  let cppNamespace = "::buddy::ada300hl";
  let summary = "ADA300 high-level hardware-semantic dialect";
  let description = [{
    The `ada300hl` dialect expresses hardware-relevant semantics for the ADA300
    Vector/Tensor ISA without exposing low-level instruction encoding.  It sits
    between standard MLIR dialects (math, vector, linalg) and the low-level
    `ada300hw` dialect and provides a stable optimisation surface for fusion,
    legalization, and cost-model-guided transformations.

    Lowering pipeline position:
      math / vector / linalg → ada300hl → ada300hw → LLVM / backend

    Key design choices:
    - Pure vector ops (exp, pwnl, cvt, vmul_mixed) carry semantic attributes
      (nonlinear-function kind, segment count, precision part).
    - Tensor Core ops (tensor_mma, pack, unpack, copy_*) model SRAM-dependent
      side-effectful operations with explicit layout information.
    - Memory spaces (global / sram / vr) are first-class attributes that
      lowering passes can validate for data-movement legality.
    - Custom types: Fp8 (e4m3) for mixed-precision paths.
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base Op Class
//===----------------------------------------------------------------------===//

class Ada300HL_Op<string mnemonic, list<Trait> traits = []> :
    Op<Ada300HL_Dialect, mnemonic, traits>;

#endif // ADA300HL_ADA300HLDIALECT_TD
