//===- Ada300HLTypes.td - Ada300HL Custom Type Definitions ------*- tablegen -*-===//
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
// Defines custom MLIR types for the Ada300HL dialect.
//
// ADA300 ISA introduces FP8 (E4M3 format) as a first-class data type used in
// mixed-precision Tensor Core paths.  Since upstream MLIR's Float8E4M3FNType
// may or may not be available depending on the LLVM version in use, we define
// a dialect-local Fp8Type that carries the ISA-specific semantics.
//
// Future types (VregType for register-backed vectors, etc.) can be added here.
//
//===----------------------------------------------------------------------===//

#ifndef ADA300HL_ADA300HLTYPES_TD
#define ADA300HL_ADA300HLTYPES_TD

include "Ada300HL/Ada300HLDialect.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Fp8Type – ADA300 FP8 floating-point type (E4M3 format)
//
// The ADA300 ISA supports FP8 operands in vfcvt and mixed-precision Tensor
// Core paths.  The hardware uses E4M3 (4 exponent bits, 3 mantissa bits)
// as the sole FP8 encoding.
//
// IR spelling:  !ada300hl.fp8<e4m3>
// Example use:  vector<64x!ada300hl.fp8<e4m3>>
//===----------------------------------------------------------------------===//

def Ada300HL_Fp8Type : TypeDef<Ada300HL_Dialect, "Fp8", []> {
  let mnemonic = "fp8";
  let summary = "ADA300 8-bit floating-point type (E4M3 format)";
  let description = [{
    Represents the FP8 E4M3 data format used in ADA300 mixed-precision
    vfcvt and Tensor Core operations.  This type exists because:
    - The ADA300 ISA encodes the precision selection in instruction fields.
    - Higher-level ops (cvt, tensor_mma) must retain the element type so
      that verifiers and cost models can reason about precision.
    - The assembly format `<e4m3>` documents the encoding explicitly and
      leaves room for future variants (E5M2, etc.).

    Lowering note: ops producing `!ada300hl.fp8<e4m3>` elements must
    eventually be lowered to the appropriate `vfcvt` instruction or a
    custom intrinsic before reaching LLVM IR.
  }];

  // The only supported format is E4M3; the parameter is purely documentary
  // and allows round-tripping through the textual IR.
  let parameters = (ins StringRefParameter<"fp8 format identifier", "\"e4m3\"">:$format);

  let assemblyFormat = "`<` $format `>`";

  let builders = [
    // Convenience builder that always produces E4M3.
    TypeBuilder<(ins), [{
      return $_get($_ctxt, "e4m3");
    }]>
  ];

  let extraClassDeclaration = [{
    // Returns true when this type carries the E4M3 format.
    bool isE4M3() const { return getFormat() == "e4m3"; }
  }];
}

#endif // ADA300HL_ADA300HLTYPES_TD
