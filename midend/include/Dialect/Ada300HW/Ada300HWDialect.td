//===- Ada300HWDialect.td - Ada300 Hardware Dialect Base --------*- tablegen -*-===//
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
// Defines only the Ada300HW dialect registration record and the base Op class.
// Attribute and op definitions live in their own sub-files:
//   Ada300HWAttrs.td – re-exports all Ada300HL attribute types
//   Ada300HWOps.td   – all ISA-level op definitions (root tablegen file)
//
// Ada300HW ops intentionally reference attribute types from the Ada300HL
// namespace (e.g. #ada300hl.nlfunc<exp>) so that the lowering pass can
// forward attr values directly without translation.
//
//===----------------------------------------------------------------------===//

#ifndef ADA300HW_ADA300HWDIALECT_TD
#define ADA300HW_ADA300HWDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def Ada300HW_Dialect : Dialect {
  let name = "ada300hw";
  let cppNamespace = "::buddy::ada300hw";
  let summary = "ADA300 low-level ISA / hardware-protocol dialect";
  let description = [{
    The `ada300hw` dialect models the ADA300 ISA as closely as possible,
    covering:

    - Vector nonlinear / conversion ops: vfpwnl, vfcvt, vfmul_low/high
    - Tensor Core configuration registers: set_gmm_cfg, set_gmm_type,
      set_gmm_iter
    - Tensor Core execution: gmm_mm, gmma_mm, gmma_mt, gmv_mm, gmva_mm,
      gmva_mt
    - Tensor Core synchronisation: tcsync

    These ops map one-to-one (or nearly so) to ADA300 machine instructions and
    are the immediate source for final code-generation to LLVM intrinsics or
    inline assembly.

    Attribute types (NonlinearFuncAttr, TensorDataTypeAttr, etc.) are shared
    with the `ada300hl` dialect to avoid translation overhead during lowering.

    Lowering pipeline position:
      ada300hl → ada300hw → LLVM dialect / custom intrinsic / inline asm
  }];

  // Ada300HW has no custom attribute or type printer/parser;
  // all used attribute types are owned by ada300hl.
  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
}

//===----------------------------------------------------------------------===//
// Base Op Class
//===----------------------------------------------------------------------===//

class Ada300HW_Op<string mnemonic, list<Trait> traits = []> :
    Op<Ada300HW_Dialect, mnemonic, traits>;

#endif // ADA300HW_ADA300HWDIALECT_TD
