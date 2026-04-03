//===- Ada300HLAttrs.cpp - Ada300HL Attribute Implementations -------------===//
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
// Provides the implementation bodies for Ada300HL attribute classes.
//
// Specifically this file pulls in:
//   - Enum stringification / parsing utilities (Ada300HLOpsEnums.cpp.inc)
//   - AttrDef class method bodies for all EnumAttr specialisations and for
//     LayoutAttr (Ada300HLOpsAttrs.cpp.inc)
//
// The registration of these attribute classes with the dialect (addAttributes)
// happens in Ada300HLDialect.cpp::initialize().
//
//===----------------------------------------------------------------------===//

#include "Ada300HL/Ada300HLDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace buddy::ada300hl;

//===----------------------------------------------------------------------===//
// Enum utilities – stringification, parsing, and operator<< overloads.
// Generated from the I32EnumAttr definitions in Ada300HLAttrs.td.
//===----------------------------------------------------------------------===//

#include "Ada300HL/Ada300HLOpsEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// AttrDef class implementations – parser, printer, and storage bodies for
// all EnumAttr specialisations (NonlinearFuncAttr, SegmentCountAttr, …)
// and for the structured LayoutAttr.
// Generated from the AttrDef / EnumAttr definitions in Ada300HLAttrs.td via
// the -gen-attrdef-defs tablegen backend.
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "Ada300HL/Ada300HLOpsAttrs.cpp.inc"
