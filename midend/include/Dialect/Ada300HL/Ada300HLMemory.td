//===- Ada300HLMemory.td - Ada300HL Memory Space Modelling ------*- tablegen -*-===//
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
// Provides type constraints and documentation for ADA300 memory space modelling.
//
// ADA300 has three distinct storage domains:
//   global  – ordinary DRAM / host-visible memory (default memref space)
//   sram    – on-chip SRAM exclusively used by the Tensor Core
//   vr      – storage backed by vector registers
//
// Memory spaces are carried by Ada300HL_MemorySpaceAttr (defined in
// Ada300HLAttrs.td) and annotated on memref types as the memory-space
// attribute:
//
//   memref<128x128xf16, #ada300hl.memory_space<sram>>
//   memref<64x64x!ada300hl.fp8<e4m3>, #ada300hl.memory_space<global>>
//   memref<1xvector<64xf16>, #ada300hl.memory_space<vr>>
//
// This file defines:
//   1. Type-constraint aliases for use in op argument/result definitions that
//      demand a memref without committing to a fixed element type.
//   2. Documentation of the expected memref memory-space usage patterns.
//
// Verifier logic that cross-checks the declared memory space against
// actual tensor-core addressing requirements lives in Ada300HLOps.td and
// its generated C++ verifier bodies.
//
//===----------------------------------------------------------------------===//

#ifndef ADA300HL_ADA300HLMEMORY_TD
#define ADA300HL_ADA300HLMEMORY_TD

include "Ada300HL/Ada300HLDialect.td"
include "Ada300HL/Ada300HLAttrs.td"
include "mlir/IR/BuiltinTypes.td"

//===----------------------------------------------------------------------===//
// Memory-space type constraints
//
// These constraints do not encode the memory-space value in the type
// predicate (MLIR type predicates operate on the MLIR type, not on an
// attribute carried alongside it).  They serve as *documentation aliases*:
// op argument types annotated with these defs communicate intent to readers
// and allow future predicate refinements without changing op signatures.
//
// Actual memory-space validation is performed by op verifiers.
//===----------------------------------------------------------------------===//

// A buffer that should reside in global (DRAM) memory.
// Annotation: #ada300hl.memory_space<global>
def Ada300HL_GlobalMemRef : Type<IsMemRefTypePred,
    "memref expected in global (DRAM) memory space">;

// A buffer that must reside in on-chip SRAM for Tensor Core access.
// Annotation: #ada300hl.memory_space<sram>
def Ada300HL_SRAMMemRef : Type<IsMemRefTypePred,
    "memref expected in on-chip SRAM memory space">;

// A buffer backed by vector registers.
// Annotation: #ada300hl.memory_space<vr>
def Ada300HL_VRMemRef : Type<IsMemRefTypePred,
    "memref expected in vector-register backing storage">;

//===----------------------------------------------------------------------===//
// Constraint helper: any memref (no memory-space restriction from tablegen).
// Prefer the typed aliases above when the expected space is known.
//===----------------------------------------------------------------------===//

def Ada300HL_AnyMemRef : Type<IsMemRefTypePred,
    "any memref (memory space checked by verifier)">;

#endif // ADA300HL_ADA300HLMEMORY_TD
