//===- Ada300HWAttrs.td - Ada300HW Attribute Re-exports --------*- tablegen -*-===//
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
// Ada300HW intentionally reuses all attribute types defined in Ada300HL.
// This file is the single include point for attribute types used in Ada300HW
// op definitions.  Including Ada300HW/Ada300HWAttrs.td brings in:
//
//   - Ada300HL_NonlinearFuncAttr  (nlfunc:  exp/log/sqrt/rsqrt/div)
//   - Ada300HL_SegmentCountAttr   (segments: 16 / 32)
//   - Ada300HL_TensorModeAttr     (tensor_mode: inner / outer)
//   - Ada300HL_TensorDataTypeAttr (dtype: fp16/bf16/int16/fp8/int8/int4)
//   - Ada300HL_MemorySpaceAttr    (memory_space: global / sram / vr)
//   - Ada300HL_PartAttr           (part: low / high)
//   - Ada300HL_MajorAttr          (major: row_major / col_major / blocked)
//   - Ada300HL_LayoutAttr         (layout: block_m/n/k, rhs_transposed, major)
//
// Rationale: sharing the Ada300HL attribute namespace means that the lowering
// pass (Ada300HLToAda300HW) can forward attr values directly without any
// translation step, and the IR remains consistent across both dialect levels.
//
// ISA-level attributes specific to Ada300HW (e.g. instruction encoding
// immediates) should be added to this file when they arise.
//
//===----------------------------------------------------------------------===//

#ifndef ADA300HW_ADA300HWATTRS_TD
#define ADA300HW_ADA300HWATTRS_TD

// Bring in all Ada300HL attribute definitions.
include "Ada300HL/Ada300HLAttrs.td"

// *** Placeholder for future Ada300HW-specific attributes ***
// Example pattern for a future ISA-specific attribute:
//
//   def Ada300HW_SomeISAAttr : AttrDef<Ada300HW_Dialect, "SomeISA", []> {
//     let mnemonic = "isa_attr";
//     ...
//   }
//
// Until such attributes exist, Ada300HW ops reference only Ada300HL attrs.

#endif // ADA300HW_ADA300HWATTRS_TD
