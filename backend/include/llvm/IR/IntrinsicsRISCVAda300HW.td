//===- IntrinsicsRISCVAda300HW.td - Ada300HW LLVM Intrinsics ---*- tablegen -*-===//
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
// LLVM intrinsic declarations for the ADA300 custom RISC-V ISA extension.
//
// These intrinsics bridge the MLIR Ada300HW dialect and the LLVM RISC-V
// backend.  The backend emits each intrinsic as its corresponding machine
// instruction binary (defined in RISCVInstrInfoAda300HW.td).
//
// Instruction groups:
//
//   1. Vector ops   – vfpwnl, vfcvt, vfmul_low, vfmul_high
//        Vector operands are passed as pointer-sized integers (i64) holding
//        the base address of the vector register file entry; exact RVV-style
//        encoding is refined by the target backend.
//
//   2. Config ops   – set_gmm_cfg, set_gmm_type, set_gmm_iter
//        All compile-time configuration attributes are packed into two i64
//        register operands, matching the RVInstR two-source convention.
//        Packing layout:
//          set_gmm_cfg  | rs1 = row_size | (col_size<<8) | (acc_size<<16)
//                       | rs2 = mode
//          set_gmm_type | rs1 = out_type | (act_type<<8)
//                       | rs2 = wht_type
//          set_gmm_iter | rs1 = blk_cnt_a
//                       | rs2 = blk_cnt_w
//
//   3. Execute ops  – gmm_mm, gmma_mm, gmma_mt, gmv_mm, gmva_mm, gmva_mt
//        Three i64 register operands carry the aligned base addresses of the
//        dst, act/mat, and wht/vec SRAM buffers.
//
//   4. Sync op      – tcsync
//        No operands; side-effect only.
//
//===----------------------------------------------------------------------===//

// ── 1. Vector ops ───────────────────────────────────────────────────────────

// vfpwnl: piecewise-linear nonlinear function approximation.
//   rs1 = input vector register address
//   rs2 = func enum  (0=exp,1=log,2=sqrt,3=rsqrt,4=div,5=sin,6=cos)
//   rs3 = segments   (16 or 32)
//   result = output vector register address
let TargetPrefix = "riscv" in
def int_riscv_ada300_vfpwnl
    : Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty], []>;

// vfcvt: floating-point precision conversion (low/high half).
//   rs1 = input vector register address
//   rs2 = dst_type enum  (fp16=0,bf16=1,int16=2,fp8=3,int8=4,int4=5)
//   rs3 = part           (0=low, 1=high)
let TargetPrefix = "riscv" in
def int_riscv_ada300_vfcvt
    : Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty], []>;

// vfmul_low: mixed-precision vector multiply (low sub-vector).
//   rs1 = lhs vector register address
//   rs2 = rhs vector register address
let TargetPrefix = "riscv" in
def int_riscv_ada300_vfmul_low
    : Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty], []>;

// vfmul_high: mixed-precision vector multiply (high sub-vector).
let TargetPrefix = "riscv" in
def int_riscv_ada300_vfmul_high
    : Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty], []>;

// ── 2. Config ops ───────────────────────────────────────────────────────────

// set_gmm_cfg: write gmm_cfg hardware register.
//   rs1 = row_size | (col_size<<8) | (acc_size<<16)
//   rs2 = mode  (0=inner, 1=outer)
let TargetPrefix = "riscv" in
def int_riscv_ada300_set_gmm_cfg
    : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

// set_gmm_type: write gmm_type hardware register.
//   rs1 = out_type | (act_type<<8)
//   rs2 = wht_type
let TargetPrefix = "riscv" in
def int_riscv_ada300_set_gmm_type
    : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

// set_gmm_iter: write gmm_iter hardware register.
//   rs1 = blk_cnt_a
//   rs2 = blk_cnt_w
let TargetPrefix = "riscv" in
def int_riscv_ada300_set_gmm_iter
    : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

// ── 3. Execute ops ──────────────────────────────────────────────────────────

// gmm_mm: non-accumulating Tensor Core matrix-multiply.
//   rs1 = dst  SRAM buffer aligned pointer
//   rs2 = act  SRAM buffer aligned pointer
//   rs3 = wht  SRAM buffer aligned pointer
let TargetPrefix = "riscv" in
def int_riscv_ada300_gmm_mm
    : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty], []>;

// gmma_mm: accumulating Tensor Core matrix-multiply.
let TargetPrefix = "riscv" in
def int_riscv_ada300_gmma_mm
    : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty], []>;

// gmma_mt: accumulating Tensor Core matmul, weight transposed.
let TargetPrefix = "riscv" in
def int_riscv_ada300_gmma_mt
    : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty], []>;

// gmv_mm: non-accumulating Tensor Core matrix-vector multiply.
//   rs1 = dst  SRAM buffer aligned pointer
//   rs2 = mat  SRAM buffer aligned pointer
//   rs3 = vec  SRAM buffer aligned pointer
let TargetPrefix = "riscv" in
def int_riscv_ada300_gmv_mm
    : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty], []>;

// gmva_mm: accumulating Tensor Core matrix-vector multiply.
let TargetPrefix = "riscv" in
def int_riscv_ada300_gmva_mm
    : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty], []>;

// gmva_mt: accumulating Tensor Core matrix-vector multiply, matrix transposed.
let TargetPrefix = "riscv" in
def int_riscv_ada300_gmva_mt
    : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty], []>;

// ── 4. Sync op ──────────────────────────────────────────────────────────────

// tcsync: wait for Tensor Core completion.
let TargetPrefix = "riscv" in
def int_riscv_ada300_tcsync
    : Intrinsic<[], [], []>;
