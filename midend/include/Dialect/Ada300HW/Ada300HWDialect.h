//===- Ada300HWDialect.h - Ada300 Hardware Dialect Declaration ---*-C++-*-===//
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

#ifndef ADA300HW_ADA300HWDIALECT_H
#define ADA300HW_ADA300HWDIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Ada300HW ops reference attribute types owned by the Ada300HL dialect.
// Include the HL dialect headers so those types are available.
#include "Ada300HL/Ada300HLDialect.h"

// Dialect class declaration for Ada300HW — generated from Ada300HWOps.td.
#include "Ada300HW/Ada300HWOpsDialect.h.inc"

#endif // ADA300HW_ADA300HWDIALECT_H
