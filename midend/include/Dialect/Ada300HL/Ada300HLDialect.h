//===- Ada300HLDialect.h - Ada300 High-Level Dialect Declaration -*-C++-*-===//
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
// Master C++ header for the Ada300HL dialect.  Including this file makes
// available:
//   - All enum C++ types (NonlinearFunc, SegmentCount, TensorMode,
//     TensorDataType, MemorySpace, Part, Major)
//   - All attribute class declarations (EnumAttr specialisations + LayoutAttr)
//   - All custom type class declarations (Fp8Type)
//   - The Ada300HLDialect class
//
//===----------------------------------------------------------------------===//

#ifndef ADA300HL_ADA300HLDIALECT_H
#define ADA300HL_ADA300HLDIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// --- Enum declarations (NonlinearFunc, SegmentCount, TensorMode,
//     TensorDataType, MemorySpace, Part, Major) -------------------------
// Generated from the I32EnumAttr definitions in Ada300HLAttrs.td via
// the -gen-enum-decls tablegen backend.
#include "Ada300HL/Ada300HLOpsEnums.h.inc"

// --- Custom type class declarations (Fp8Type) -------------------------
// Generated from Ada300HLTypes.td via the -gen-typedef-decls backend.
#define GET_TYPEDEF_CLASSES
#include "Ada300HL/Ada300HLOpsTypes.h.inc"

// --- Attribute class declarations (EnumAttr specialisations +
//     LayoutAttr) --------------------------------------------------------
// Generated from Ada300HLAttrs.td via the -gen-attrdef-decls backend.
#define GET_ATTRDEF_CLASSES
#include "Ada300HL/Ada300HLOpsAttrs.h.inc"

// --- Dialect class declaration -----------------------------------------
// Generated from Ada300HLDialect.td via the -gen-dialect-decls backend.
#include "Ada300HL/Ada300HLOpsDialect.h.inc"

#endif // ADA300HL_ADA300HLDIALECT_H
