//===- Ada300HWOps.cpp - Ada300 Hardware Dialect Op Definitions -----------===//
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
// Hand-written verifier logic for Ada300HW ops.  Tablegen-generated op bodies
// are pulled in via the GET_OP_CLASSES include below.
//
//===----------------------------------------------------------------------===//

#include "Ada300HW/Ada300HWOps.h"
#include "Ada300HW/Ada300HWDialect.h"
#include "Ada300HL/Ada300HLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace buddy::ada300hw;
using namespace buddy::ada300hl;

//===----------------------------------------------------------------------===//
// Tablegen-generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Ada300HW/Ada300HWOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Ada300HW_VfpwnlOp
//===----------------------------------------------------------------------===//

LogicalResult Ada300HW_VfpwnlOp::verify() {
  // Segment count must be one of the two ISA-defined values.
  SegmentCount seg = getSegments().getValue();
  if (seg != SegmentCount::seg16 && seg != SegmentCount::seg32)
    return emitOpError("'segments' must be 16 or 32");
  return success();
}

//===----------------------------------------------------------------------===//
// Ada300HW_SetGmmCfgOp
//===----------------------------------------------------------------------===//

LogicalResult Ada300HW_SetGmmCfgOp::verify() {
  if (getRowSize() <= 0)
    return emitOpError("'row_size' must be a positive integer");
  if (getColSize() <= 0)
    return emitOpError("'col_size' must be a positive integer");
  if (getAccSize() <= 0)
    return emitOpError("'acc_size' must be a positive integer");
  return success();
}

//===----------------------------------------------------------------------===//
// Ada300HW_SetGmmIterOp
//===----------------------------------------------------------------------===//

LogicalResult Ada300HW_SetGmmIterOp::verify() {
  if (getBlkCntA() <= 0)
    return emitOpError("'blk_cnt_a' must be a positive integer");
  if (getBlkCntW() <= 0)
    return emitOpError("'blk_cnt_w' must be a positive integer");
  return success();
}
