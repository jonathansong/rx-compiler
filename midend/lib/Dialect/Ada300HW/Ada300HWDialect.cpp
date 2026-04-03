//===- Ada300HWDialect.cpp - Ada300 Hardware Dialect Definition -----------===//
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
// Registers the Ada300HW dialect and its operations.  No custom attribute
// classes are defined in this dialect; attribute types are owned by Ada300HL.
//
//===----------------------------------------------------------------------===//

#include "Ada300HW/Ada300HWDialect.h"
#include "Ada300HW/Ada300HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace buddy::ada300hw;

// Generated dialect boilerplate.
#include "Ada300HW/Ada300HWOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Ada300HWDialect::initialize
//===----------------------------------------------------------------------===//

void Ada300HWDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ada300HW/Ada300HWOps.cpp.inc"
  >();
}
