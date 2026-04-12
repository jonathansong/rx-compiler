//===- RISCVInstrInfoAda300HW.td - Ada300HW Instruction Encodings ---*- tablegen -*-===//
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
// RISC-V instruction binary encodings for the ADA300 custom ISA extension.
//
// All instructions use:
//   opcode = OPC_CUSTOM_3  (0b1111011)
//   func3  = 0b011          (same convention as the Gemmini buddy extension)
//   func7  = per-instruction mnemonic tag (see table below)
//
// func7 allocation (Ada300HW uses 0b1xxxxxxx range to avoid Gemmini collision):
//
//   0b1000000 = 64  set_gmm_cfg
//   0b1000001 = 65  set_gmm_type
//   0b1000010 = 66  set_gmm_iter
//   0b1000011 = 67  gmm_mm
//   0b1000100 = 68  gmma_mm
//   0b1000101 = 69  gmma_mt
//   0b1000110 = 70  gmv_mm
//   0b1000111 = 71  gmva_mm
//   0b1001000 = 72  gmva_mt
//   0b1001001 = 73  tcsync
//   0b1001010 = 74  vfpwnl
//   0b1001011 = 75  vfcvt
//   0b1001100 = 76  vfmul_low
//   0b1001101 = 77  vfmul_high
//
// TODO: Replace the func7 values above with the actual ADA300 ISA encoding
//       once the chip specification is available.
//
// TODO: Vector ops (vfpwnl, vfcvt, vfmul_low, vfmul_high) operate on vector
//       register files.  The R-format GPR encoding below treats the vector
//       register address as an integer pointer value.  When proper RVV-style
//       vector-register operand classes are added to the buddy LLVM backend,
//       these definitions should be updated to use them instead.
//
//===----------------------------------------------------------------------===//

// ── 2. Config ops ───────────────────────────────────────────────────────────

// ada300.set_gmm_cfg
//   Writes gmm_cfg: rs1 = packed dims, rs2 = mode.
let hasSideEffects = 1, mayLoad = 0, mayStore = 1, Predicates = [HasBuddyExt] in
def ADA300_SET_GMM_CFG : RVInstR<0b1000000, 0b011, OPC_CUSTOM_3, (outs),
    (ins GPR:$rs1, GPR:$rs2), "ada300.set_gmm_cfg", "$rs1, $rs2"> {
  let rd = 0;
}

// ada300.set_gmm_type
//   Writes gmm_type: rs1 = out_type | (act_type<<8), rs2 = wht_type.
let hasSideEffects = 1, mayLoad = 0, mayStore = 1, Predicates = [HasBuddyExt] in
def ADA300_SET_GMM_TYPE : RVInstR<0b1000001, 0b011, OPC_CUSTOM_3, (outs),
    (ins GPR:$rs1, GPR:$rs2), "ada300.set_gmm_type", "$rs1, $rs2"> {
  let rd = 0;
}

// ada300.set_gmm_iter
//   Writes gmm_iter: rs1 = blk_cnt_a, rs2 = blk_cnt_w.
let hasSideEffects = 1, mayLoad = 0, mayStore = 1, Predicates = [HasBuddyExt] in
def ADA300_SET_GMM_ITER : RVInstR<0b1000010, 0b011, OPC_CUSTOM_3, (outs),
    (ins GPR:$rs1, GPR:$rs2), "ada300.set_gmm_iter", "$rs1, $rs2"> {
  let rd = 0;
}

// ── 3. Execute ops ──────────────────────────────────────────────────────────

// ada300.gmm_mm
//   Non-accumulating matmul: rs1=dst ptr, rs2=act ptr, rd=wht ptr.
//   NOTE: Three-register variant — rd encodes the third operand (wht).
let hasSideEffects = 1, mayLoad = 1, mayStore = 1, Predicates = [HasBuddyExt] in
def ADA300_GMM_MM : RVInstR<0b1000011, 0b011, OPC_CUSTOM_3, (outs),
    (ins GPR:$rd, GPR:$rs1, GPR:$rs2), "ada300.gmm_mm", "$rd, $rs1, $rs2"> {
}

// ada300.gmma_mm
//   Accumulating matmul: rs1=dst ptr, rs2=act ptr, rd=wht ptr.
let hasSideEffects = 1, mayLoad = 1, mayStore = 1, Predicates = [HasBuddyExt] in
def ADA300_GMMA_MM : RVInstR<0b1000100, 0b011, OPC_CUSTOM_3, (outs),
    (ins GPR:$rd, GPR:$rs1, GPR:$rs2), "ada300.gmma_mm", "$rd, $rs1, $rs2"> {
}

// ada300.gmma_mt
//   Accumulating matmul, weight transposed.
let hasSideEffects = 1, mayLoad = 1, mayStore = 1, Predicates = [HasBuddyExt] in
def ADA300_GMMA_MT : RVInstR<0b1000101, 0b011, OPC_CUSTOM_3, (outs),
    (ins GPR:$rd, GPR:$rs1, GPR:$rs2), "ada300.gmma_mt", "$rd, $rs1, $rs2"> {
}

// ada300.gmv_mm
//   Non-accumulating matrix-vector multiply.
let hasSideEffects = 1, mayLoad = 1, mayStore = 1, Predicates = [HasBuddyExt] in
def ADA300_GMV_MM : RVInstR<0b1000110, 0b011, OPC_CUSTOM_3, (outs),
    (ins GPR:$rd, GPR:$rs1, GPR:$rs2), "ada300.gmv_mm", "$rd, $rs1, $rs2"> {
}

// ada300.gmva_mm
//   Accumulating matrix-vector multiply.
let hasSideEffects = 1, mayLoad = 1, mayStore = 1, Predicates = [HasBuddyExt] in
def ADA300_GMVA_MM : RVInstR<0b1000111, 0b011, OPC_CUSTOM_3, (outs),
    (ins GPR:$rd, GPR:$rs1, GPR:$rs2), "ada300.gmva_mm", "$rd, $rs1, $rs2"> {
}

// ada300.gmva_mt
//   Accumulating matrix-vector multiply, matrix transposed.
let hasSideEffects = 1, mayLoad = 1, mayStore = 1, Predicates = [HasBuddyExt] in
def ADA300_GMVA_MT : RVInstR<0b1001000, 0b011, OPC_CUSTOM_3, (outs),
    (ins GPR:$rd, GPR:$rs1, GPR:$rs2), "ada300.gmva_mt", "$rd, $rs1, $rs2"> {
}

// ── 4. Sync op ──────────────────────────────────────────────────────────────

// ada300.tcsync
//   Poll Tensor Core completion flag; no operands.
let hasSideEffects = 1, mayLoad = 1, mayStore = 1, Predicates = [HasBuddyExt] in
def ADA300_TCSYNC : RVInstR<0b1001001, 0b011, OPC_CUSTOM_3, (outs),
    (ins), "ada300.tcsync", ""> {
  let rd  = 0;
  let rs1 = 0;
  let rs2 = 0;
}

// ── 1. Vector ops ───────────────────────────────────────────────────────────
// TODO: These use GPR addressing as a placeholder.  Replace with proper
//       RVV-style vector register operands once the ADA300 V-extension
//       instruction format is integrated into the buddy LLVM backend.

// ada300.vfpwnl
//   rs1 = input vector address, rs2 = func enum, rd = segments.
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, Predicates = [HasBuddyExt] in
def ADA300_VFPWNL : RVInstR<0b1001010, 0b011, OPC_CUSTOM_3, (outs GPR:$rd_out),
    (ins GPR:$rd, GPR:$rs1, GPR:$rs2), "ada300.vfpwnl", "$rd, $rs1, $rs2"> {
}

// ada300.vfcvt
//   rs1 = input vector address, rs2 = dst_type, rd = part.
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, Predicates = [HasBuddyExt] in
def ADA300_VFCVT : RVInstR<0b1001011, 0b011, OPC_CUSTOM_3, (outs GPR:$rd_out),
    (ins GPR:$rd, GPR:$rs1, GPR:$rs2), "ada300.vfcvt", "$rd, $rs1, $rs2"> {
}

// ada300.vfmul_low
//   rs1 = lhs vector address, rs2 = rhs vector address.
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, Predicates = [HasBuddyExt] in
def ADA300_VFMUL_LOW : RVInstR<0b1001100, 0b011, OPC_CUSTOM_3, (outs GPR:$rd),
    (ins GPR:$rs1, GPR:$rs2), "ada300.vfmul_low", "$rd, $rs1, $rs2"> {
}

// ada300.vfmul_high
//   rs1 = lhs vector address, rs2 = rhs vector address.
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, Predicates = [HasBuddyExt] in
def ADA300_VFMUL_HIGH : RVInstR<0b1001101, 0b011, OPC_CUSTOM_3, (outs GPR:$rd),
    (ins GPR:$rs1, GPR:$rs2), "ada300.vfmul_high", "$rd, $rs1, $rs2"> {
}
