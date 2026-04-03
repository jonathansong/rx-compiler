//===- Ada300HLDialect.cpp - Ada300 High-Level Dialect Registration -------===//
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
// Registers the Ada300HL dialect with MLIR by implementing initialize().
// Implementation details are split across three files:
//   Ada300HLAttrs.cpp  – enum utilities + attribute class bodies
//   Ada300HLTypes.cpp  – custom type class bodies (Fp8Type)
//   Ada300HLOps.cpp    – op class bodies + hand-written verifiers
//
//===----------------------------------------------------------------------===//

#include "Ada300HL/Ada300HLDialect.h"
#include "Ada300HL/Ada300HLOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace buddy::ada300hl;

//===----------------------------------------------------------------------===//
// Generated dialect boilerplate (print/parse dispatch, etc.)
//===----------------------------------------------------------------------===//

#include "Ada300HL/Ada300HLOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Ada300HLDialect::initialize
//===----------------------------------------------------------------------===//

void Ada300HLDialect::initialize() {
  // Register all ops from Ada300HLOps.td.
  addOperations<
#define GET_OP_LIST
#include "Ada300HL/Ada300HLOps.cpp.inc"
  >();

  // Register custom types (Fp8Type).
  addTypes<
#define GET_TYPEDEF_LIST
#include "Ada300HL/Ada300HLOpsTypes.cpp.inc"
  >();

  // Register attribute classes (enum attrs + LayoutAttr).
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Ada300HL/Ada300HLOpsAttrs.cpp.inc"
  >();
}
