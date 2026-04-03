//===- Ada300HLTypes.cpp - Ada300HL Custom Type Implementations -----------===//
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
// Provides the implementation bodies for Ada300HL custom type classes.
// Currently this covers:
//   - Fp8Type (E4M3 format FP8 used in mixed-precision Tensor Core paths)
//
// The registration of these type classes with the dialect (addTypes) happens
// in Ada300HLDialect.cpp::initialize().
//
//===----------------------------------------------------------------------===//

#include "Ada300HL/Ada300HLDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace buddy::ada300hl;

//===----------------------------------------------------------------------===//
// TypeDef class implementations – parser, printer, and storage bodies for
// all TypeDef definitions in Ada300HLTypes.td (currently only Fp8Type).
// Generated via the -gen-typedef-defs tablegen backend.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Ada300HL/Ada300HLOpsTypes.cpp.inc"
