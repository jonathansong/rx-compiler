//===- Ada300HLAttrs.td - Ada300HL Attribute Definitions -------*- tablegen -*-===//
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
// Defines all custom attributes for the Ada300HL dialect:
//   - Six I32 enum attributes covering nonlinear function kind, PWL segment
//     count, tensor accumulation mode, tensor data type, memory space,
//     sub-vector part selection, and buffer storage order.
//   - LayoutAttr: a structured attribute encoding block tile sizes and
//     transposition flags for Tensor Core pack/unpack operations.
//
// This file is included by Ada300HLOps.td (and transitively by Ada300HWAttrs.td)
// so that Ada300HW ops can reference the same attribute types without
// re-declaring them.
//
//===----------------------------------------------------------------------===//

#ifndef ADA300HL_ADA300HLATTRS_TD
#define ADA300HL_ADA300HLATTRS_TD

include "Ada300HL/Ada300HLDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// NonlinearFuncAttr
// Identifies the mathematical function targeted by the PWL hardware unit.
// Values: exp | log | sqrt | rsqrt | div
//===----------------------------------------------------------------------===//

def Ada300HL_NLFunc_Exp   : I32EnumAttrCase<"exp",   0, "exp">;
def Ada300HL_NLFunc_Log   : I32EnumAttrCase<"log",   1, "log">;
def Ada300HL_NLFunc_Sqrt  : I32EnumAttrCase<"sqrt",  2, "sqrt">;
def Ada300HL_NLFunc_Rsqrt : I32EnumAttrCase<"rsqrt", 3, "rsqrt">;
def Ada300HL_NLFunc_Div   : I32EnumAttrCase<"div",   4, "div">;
def Ada300HL_NLFunc_Sin   : I32EnumAttrCase<"sin",   5, "sin">;
def Ada300HL_NLFunc_Cos   : I32EnumAttrCase<"cos",   6, "cos">;

def Ada300HL_NonlinearFuncEnum : I32EnumAttr<"NonlinearFunc",
    "Nonlinear function kind for PWL approximation",
    [Ada300HL_NLFunc_Exp, Ada300HL_NLFunc_Log, Ada300HL_NLFunc_Sqrt,
     Ada300HL_NLFunc_Rsqrt, Ada300HL_NLFunc_Div,
     Ada300HL_NLFunc_Sin, Ada300HL_NLFunc_Cos]> {
  let cppNamespace = "::buddy::ada300hl";
  let genSpecializedAttr = 0;
}

def Ada300HL_NonlinearFuncAttr :
    EnumAttr<Ada300HL_Dialect, Ada300HL_NonlinearFuncEnum, "nlfunc"> {
  let summary = "Nonlinear function kind for PWL approximation (exp/log/sqrt/rsqrt/div/sin/cos)";
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// SegmentCountAttr
// Selects the 16- or 32-segment PWL approximation table.
// Only these two values are legal per the ADA300 ISA specification.
//===----------------------------------------------------------------------===//

def Ada300HL_Seg_16 : I32EnumAttrCase<"seg16", 16, "16">;
def Ada300HL_Seg_32 : I32EnumAttrCase<"seg32", 32, "32">;

def Ada300HL_SegmentCountEnum : I32EnumAttr<"SegmentCount",
    "Number of PWL approximation segments (must be 16 or 32)",
    [Ada300HL_Seg_16, Ada300HL_Seg_32]> {
  let cppNamespace = "::buddy::ada300hl";
  let genSpecializedAttr = 0;
}

def Ada300HL_SegmentCountAttr :
    EnumAttr<Ada300HL_Dialect, Ada300HL_SegmentCountEnum, "segments"> {
  let summary = "Number of PWL segments (16 or 32)";
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// TensorModeAttr
// Selects inner-product or outer-product accumulation mode for the Tensor Core.
//===----------------------------------------------------------------------===//

def Ada300HL_TMode_Inner : I32EnumAttrCase<"inner", 0, "inner">;
def Ada300HL_TMode_Outer : I32EnumAttrCase<"outer", 1, "outer">;

def Ada300HL_TensorModeEnum : I32EnumAttr<"TensorMode",
    "Tensor Core output accumulation mode",
    [Ada300HL_TMode_Inner, Ada300HL_TMode_Outer]> {
  let cppNamespace = "::buddy::ada300hl";
  let genSpecializedAttr = 0;
}

def Ada300HL_TensorModeAttr :
    EnumAttr<Ada300HL_Dialect, Ada300HL_TensorModeEnum, "tensor_mode"> {
  let summary = "Tensor Core output accumulation mode (inner or outer product)";
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// TensorDataTypeAttr
// Corresponds to the data-type fields encoded in the gmm_type / gmv_type ISA
// configuration registers.
// Values: fp16 | bf16 | int16 | fp8 | int8 | int4
//===----------------------------------------------------------------------===//

def Ada300HL_DType_FP16  : I32EnumAttrCase<"fp16",  0, "fp16">;
def Ada300HL_DType_BF16  : I32EnumAttrCase<"bf16",  1, "bf16">;
def Ada300HL_DType_INT16 : I32EnumAttrCase<"int16", 2, "int16">;
def Ada300HL_DType_FP8   : I32EnumAttrCase<"fp8",   3, "fp8">;
def Ada300HL_DType_INT8  : I32EnumAttrCase<"int8",  4, "int8">;
def Ada300HL_DType_INT4  : I32EnumAttrCase<"int4",  5, "int4">;

def Ada300HL_TensorDataTypeEnum : I32EnumAttr<"TensorDataType",
    "Tensor Core operand or result data type",
    [Ada300HL_DType_FP16, Ada300HL_DType_BF16, Ada300HL_DType_INT16,
     Ada300HL_DType_FP8,  Ada300HL_DType_INT8,  Ada300HL_DType_INT4]> {
  let cppNamespace = "::buddy::ada300hl";
  let genSpecializedAttr = 0;
}

def Ada300HL_TensorDataTypeAttr :
    EnumAttr<Ada300HL_Dialect, Ada300HL_TensorDataTypeEnum, "dtype"> {
  let summary = "Tensor Core operand data type (fp16/bf16/int16/fp8/int8/int4)";
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// MemorySpaceAttr
// Distinguishes the three storage domains visible to ADA300 ops:
//   global – ordinary DRAM / host-visible memory
//   sram   – on-chip SRAM accessed by the Tensor Core
//   vr     – storage backed by vector registers
//
// Usage (as a memref memory-space attribute):
//   memref<128x128xf16, #ada300hl.memory_space<sram>>
//===----------------------------------------------------------------------===//

def Ada300HL_MSpace_Global : I32EnumAttrCase<"global", 0, "global">;
def Ada300HL_MSpace_SRAM   : I32EnumAttrCase<"sram",   1, "sram">;
def Ada300HL_MSpace_VR     : I32EnumAttrCase<"vr",     2, "vr">;

def Ada300HL_MemorySpaceEnum : I32EnumAttr<"MemorySpace",
    "ADA300 memory space (global DRAM, on-chip SRAM, or VR-backed)",
    [Ada300HL_MSpace_Global, Ada300HL_MSpace_SRAM, Ada300HL_MSpace_VR]> {
  let cppNamespace = "::buddy::ada300hl";
  let genSpecializedAttr = 0;
}

def Ada300HL_MemorySpaceAttr :
    EnumAttr<Ada300HL_Dialect, Ada300HL_MemorySpaceEnum, "memory_space"> {
  let summary = "ADA300 memory space used as memref memory-space annotation";
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// PartAttr
// Selects which half of a sub-vector or register pair is operated on.
// Used by vfcvt and vfmul_low/vfmul_high style instructions.
//===----------------------------------------------------------------------===//

def Ada300HL_Part_Low  : I32EnumAttrCase<"low",  0, "low">;
def Ada300HL_Part_High : I32EnumAttrCase<"high", 1, "high">;

def Ada300HL_PartEnum : I32EnumAttr<"Part",
    "Low or high half selection for mixed-precision operations",
    [Ada300HL_Part_Low, Ada300HL_Part_High]> {
  let cppNamespace = "::buddy::ada300hl";
  let genSpecializedAttr = 0;
}

def Ada300HL_PartAttr :
    EnumAttr<Ada300HL_Dialect, Ada300HL_PartEnum, "part"> {
  let summary = "Half selection (low or high) for sub-vector operations";
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// MajorAttr
// Describes the logical storage order (major axis) of a matrix buffer.
//===----------------------------------------------------------------------===//

def Ada300HL_Major_RowMajor : I32EnumAttrCase<"row_major", 0, "row_major">;
def Ada300HL_Major_ColMajor : I32EnumAttrCase<"col_major", 1, "col_major">;
def Ada300HL_Major_Blocked  : I32EnumAttrCase<"blocked",   2, "blocked">;

def Ada300HL_MajorEnum : I32EnumAttr<"Major",
    "Matrix buffer storage order",
    [Ada300HL_Major_RowMajor, Ada300HL_Major_ColMajor,
     Ada300HL_Major_Blocked]> {
  let cppNamespace = "::buddy::ada300hl";
  let genSpecializedAttr = 0;
}

def Ada300HL_MajorAttr :
    EnumAttr<Ada300HL_Dialect, Ada300HL_MajorEnum, "major"> {
  let summary = "Matrix buffer storage order (row-major, col-major, or blocked)";
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// LayoutAttr (structured AttrDef)
// Encodes the complete tiling and transposition descriptor for Tensor Core
// pack/unpack and data-movement operations.
//
// Example:
//   #ada300hl.layout<block_m = 16, block_n = 16, block_k = 16,
//                    rhs_transposed = false, major = 2>
//===----------------------------------------------------------------------===//

def Ada300HL_LayoutAttr : AttrDef<Ada300HL_Dialect, "Layout", []> {
  let mnemonic = "layout";
  let summary = "Block-tiling and transposition descriptor for Tensor Core data";
  let description = [{
    Groups the tile dimensions (block_m × block_n × block_k), an optional RHS
    transposition flag, and the major storage order so that pack/unpack ops and
    the Tensor Core lowering can reason about data layout without embedding it
    in the memref type system.

    The `major` field uses the integer encoding of `Major`:
      0 = row_major, 1 = col_major, 2 = blocked.
  }];

  let parameters = (ins
    "int32_t":$block_m,
    "int32_t":$block_n,
    "int32_t":$block_k,
    "bool":$rhs_transposed,
    "int32_t":$matrix_major
  );

  let assemblyFormat = [{
    `<` `block_m` `=` $block_m `,`
        `block_n` `=` $block_n `,`
        `block_k` `=` $block_k `,`
        `rhs_transposed` `=` $rhs_transposed `,`
        `major` `=` $matrix_major `>`
  }];
}

#endif // ADA300HL_ADA300HLATTRS_TD
