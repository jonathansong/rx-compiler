# ===- CMakeLists.txt ----------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# CMake build rules for the Ada300 example.
#
# There are two lowering paths for subgraph0.mlir:
#
#   Path A – Generic (target-independent):
#     subgraph0.mlir → one-shot-bufferize → linalg/SCF loops → LLVM IR → .o
#     Target: ada300-import
#
#   Path B – Ada300-specific (Ada300HL → Ada300HW → LLVM IR):
#     subgraph0.mlir
#       ↓ convert-elementwise-to-linalg
#       ↓ one-shot-bufferize                   tensor → memref
#       ↓ func.func(linalg-vectorize)           linalg.generic → vector<> math ops
#       ↓ func.func(math-to-ada300hl)           math.exp/sqrt → ada300hl.pwnl
#       ↓ linalg-to-ada300hl                    linalg.matmul → ada300hl.tensor_mma
#       ↓ lower-ada300hl-to-ada300hw            ada300hl → ada300hw ISA ops
#       ↓ lower-ada300hw-to-llvm                ada300hw → LLVM inline asm
#       ↓ (standard finishing passes)
#       ↓ mlir-translate -mlir-to-llvmir        → subgraph0_ada300.ll  (LLVM IR)
#       ↓ llc                                   → subgraph0_ada300.o
#     Target: ada300-lower
#
# Ops in subgraph0.mlir after import:
#   - linalg.matmul  (from nn.Linear / AddMMOp)
#   - math.exp       (from torch.exp / ExpOp)
#   - math.sqrt      (from torch.sqrt / SqrtOp)
#
# Ada300HL / Ada300HW pass names in buddy-opt:
#   math-to-ada300hl          – vector math.* → ada300hl.pwnl
#   linalg-to-ada300hl        – bufferized linalg.matmul → ada300hl.tensor_mma
#   lower-ada300hl-to-ada300hw – ada300hl → ada300hw ISA ops
#   lower-ada300hw-to-llvm    – ada300hw → LLVM inline_asm
#
# ===---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step 1: Import – run the Python importer to generate MLIR and parameters
# ---------------------------------------------------------------------------
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
         ${CMAKE_CURRENT_BINARY_DIR}/arg0.data
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ada300-import.py
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Ada300: tracing model and generating subgraph0.mlir, forward.mlir, arg0.data"
  VERBATIM
)

# ---------------------------------------------------------------------------
# Step 2 (Path A): Compile subgraph0.mlir → subgraph0.o  (generic path)
#
# Lowering pipeline:
#   • one-shot-bufferize       – tensor → memref
#   • convert-linalg-to-loops  – linalg.matmul / linalg.generic → scf loops
#   • convert-math-to-llvm     – math.exp / math.sqrt → LLVM intrinsics
#   • convert-math-to-libm     – fallback to libm for any remaining math ops
#   • Standard LLVM dialect lowering passes
# ---------------------------------------------------------------------------
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -expand-strided-metadata
            -lower-affine
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
          buddy-opt
  COMMENT "Ada300: lowering subgraph0.mlir → subgraph0.o (generic path)"
  VERBATIM
)

# ---------------------------------------------------------------------------
# Step 3 (Path B): subgraph0.mlir → subgraph0_ada300.ll / subgraph0_ada300.o
#          via Ada300HL and Ada300HW dialects
#
# Full lowering pipeline:
#
#   Stage 1 – prepare elementwise math ops for vectorization
#     convert-elementwise-to-linalg
#       math.exp %tensor  →  linalg.generic (body: scalar math.exp)
#       math.sqrt %tensor →  linalg.generic (body: scalar math.sqrt)
#
#   Stage 2 – bufferize
#     one-shot-bufferize="bufferize-function-boundaries"
#       All tensors (linalg.matmul, linalg.generic) become memref-based.
#
#   Stage 3 – vectorize linalg.generic (exp / sqrt)
#     func.func(linalg-vectorize)
#       linalg.generic (memref, scalar body)
#         → vector.transfer_read + math.exp<vector<Nxf32>> + vector.transfer_write
#       After this step math.exp / math.sqrt operate on MLIR VectorType,
#       which is the form required by math-to-ada300hl.
#
#   Stage 4 – Ada300HL conversion
#     func.func(math-to-ada300hl)
#       math.exp  <vector<Nxf32>>  →  ada300hl.pwnl {func=exp,  segments=16}
#       math.sqrt <vector<Nxf32>>  →  ada300hl.pwnl {func=sqrt, segments=16}
#
#     linalg-to-ada300hl
#       linalg.matmul (memref M×K, K×N → M×N)
#         →  ada300hl.copy_to_sram  %A, %A_sram
#            ada300hl.copy_to_sram  %B, %B_sram
#            ada300hl.tensor_mma    %C, %A_sram, %B_sram
#              {mode=inner, row=M, col=N, acc=K, blk_cnt_a=1, blk_cnt_w=1, ...}
#            ada300hl.tensor_sync
#
#   Stage 5 – Ada300HW ISA-level expansion
#     lower-ada300hl-to-ada300hw
#       ada300hl.pwnl       →  ada300hw.vfpwnl {func=exp/sqrt, masked=false}
#       ada300hl.tensor_mma →  ada300hw.set_gmm_cfg
#                              ada300hw.set_gmm_type
#                              ada300hw.set_gmm_iter
#                              ada300hw.gmma_mm
#       ada300hl.tensor_sync→  ada300hw.tcsync
#       ada300hl.copy_to_sram   →  memref.copy (placeholder)
#
#   Stage 6 – LLVM inline asm emission
#     lower-ada300hw-to-llvm
#       ada300hw.vfpwnl     →  llvm.inline_asm "vfpwnl.exp.16 $0, $1"
#       ada300hw.set_gmm_cfg→  llvm.inline_asm "gmm.cfg M, N, K, 0"
#       ada300hw.set_gmm_type → llvm.inline_asm "gmm.type ..."
#       ada300hw.set_gmm_iter → llvm.inline_asm "gmm.iter 1, 1"
#       ada300hw.gmma_mm    →  llvm.inline_asm "gmma.mm $0, $1, $2"  [r,r,r]
#       ada300hw.tcsync     →  llvm.inline_asm "tcsync"
#
#   Stage 7 – Standard finishing
#     expand-strided-metadata, lower-affine, convert-vector-to-llvm,
#     convert-scf-to-cf, convert-cf-to-llvm, convert-arith-to-llvm,
#     finalize-memref-to-llvm, convert-func-to-llvm,
#     reconcile-unrealized-casts
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Path B staged dumps – one add_custom_command per stage so that each
# intermediate MLIR file is written to disk and can be inspected.
#
#   subgraph0.mlir               (import output – linalg + math, tensor level)
#     → stage1_elementwise.mlir  (TOSA → linalg; math.exp/sqrt → linalg.generic)
#     → stage2_bufferized.mlir   (tensors → memrefs)
#     → stage3_normalized.mlir   (canonicalize + normalize-memrefs: flatten strided layouts)
#     → stage4a_ada300hl_math.mlir    (math.* → ada300hl.pwnl)
#     → stage4b_ada300hl_matmul.mlir  (linalg.matmul → ada300hl.tensor_mma)
#     → stage5_ada300hw.mlir          (ada300hl → ada300hw ISA ops)
#     → stage6_llvm_dialect.mlir      (ada300hw → llvm.inline_asm)
#     → stage7_finished.mlir          (all standard finishing passes applied)
#     → subgraph0_ada300.ll           (mlir-translate → LLVM IR text)
# ---------------------------------------------------------------------------

# Stage 1: lower TOSA ops → linalg/arith/tensor, then convert elementwise
#          math ops on tensors → linalg.generic
#   tosa.transpose / tosa.* → linalg.generic / arith.*
#   math.exp(%tensor)       → linalg.generic { math.exp(scalar) }
#   math.sqrt(%tensor)      → linalg.generic { math.sqrt(scalar) }
set(ADA300_TOSA_PIPELINE
    "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-tensor,tosa-to-arith),convert-elementwise-to-linalg)")
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stage1_elementwise.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
            -pass-pipeline=${ADA300_TOSA_PIPELINE}
            -o ${CMAKE_CURRENT_BINARY_DIR}/stage1_elementwise.mlir
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.mlir
          buddy-opt
  COMMENT "Ada300 Stage 1: tosa-to-linalg + convert-elementwise-to-linalg → stage1_elementwise.mlir"
  VERBATIM
)

# Stage 2: bufferize – all tensors become memrefs
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stage2_bufferized.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/stage1_elementwise.mlir
            -one-shot-bufferize="bufferize-function-boundaries"
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -o ${CMAKE_CURRENT_BINARY_DIR}/stage2_bufferized.mlir
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/stage1_elementwise.mlir
          buddy-opt
  COMMENT "Ada300 Stage 2: one-shot-bufferize → stage2_bufferized.mlir"
  VERBATIM
)

# Stage 3: normalize memref layouts
#   After bufferization, subview-based memrefs carry dynamic strided layouts
#   (e.g. strided<[?, ?], offset: ?>) that confuse linalg-to-ada300hl.
#   - canonicalize: folds constant strides/offsets into static values where
#                   possible, eliminating the dynamic-symbol operand mismatch.
#   - normalize-memrefs: rewrites memrefs with non-identity affine layout maps
#                        into plain memrefs + explicit index arithmetic, so all
#                        memrefs reaching the Ada300HL passes have identity layout.
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stage3_normalized.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/stage2_bufferized.mlir
            -canonicalize
            -cse
            -normalize-memrefs
            -o ${CMAKE_CURRENT_BINARY_DIR}/stage3_normalized.mlir
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/stage2_bufferized.mlir
          buddy-opt
  COMMENT "Ada300 Stage 3: canonicalize + normalize-memrefs → stage3_normalized.mlir"
  VERBATIM
)

# Stage 4a: math.exp / math.sqrt → ada300hl.pwnl
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stage4a_ada300hl_math.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/stage3_normalized.mlir
            -math-to-ada300hl
            -o ${CMAKE_CURRENT_BINARY_DIR}/stage4a_ada300hl_math.mlir
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/stage3_normalized.mlir
          buddy-opt
  COMMENT "Ada300 Stage 4a: math-to-ada300hl → stage4a_ada300hl_math.mlir"
  VERBATIM
)

# Stage 4b: bufferized linalg.matmul → ada300hl.tensor_mma
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stage4b_ada300hl_matmul.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/stage4a_ada300hl_math.mlir
            -linalg-to-ada300hl
            -o ${CMAKE_CURRENT_BINARY_DIR}/stage4b_ada300hl_matmul.mlir
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/stage4a_ada300hl_math.mlir
          buddy-opt
  COMMENT "Ada300 Stage 4b: linalg-to-ada300hl → stage4b_ada300hl_matmul.mlir"
  VERBATIM
)

# Stage 5: ada300hl ops → ada300hw ISA ops
#   ada300hl.pwnl       → ada300hw.vfpwnl
#   ada300hl.tensor_mma → ada300hw.set_gmm_cfg / gmma_mm / tcsync
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stage5_ada300hw.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/stage4b_ada300hl_matmul.mlir
            -lower-ada300hl-to-ada300hw
            -o ${CMAKE_CURRENT_BINARY_DIR}/stage5_ada300hw.mlir
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/stage4b_ada300hl_matmul.mlir
          buddy-opt
  COMMENT "Ada300 Stage 5: lower-ada300hl-to-ada300hw → stage5_ada300hw.mlir"
  VERBATIM
)

# Stage 6: ada300hw ops → llvm.inline_asm nodes
#   ada300hw.vfpwnl  → llvm.inline_asm "vfpwnl.exp.16 $0, $1"
#   ada300hw.gmma_mm → llvm.inline_asm "gmma.mm $0, $1, $2"
#   ada300hw.tcsync  → llvm.inline_asm "tcsync"
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stage6_llvm_dialect.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/stage5_ada300hw.mlir
            -lower-ada300hw-to-llvm
            -o ${CMAKE_CURRENT_BINARY_DIR}/stage6_llvm_dialect.mlir
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/stage5_ada300hw.mlir
          buddy-opt
  COMMENT "Ada300 Stage 6: lower-ada300hw-to-llvm → stage6_llvm_dialect.mlir"
  VERBATIM
)

# Stage 7: standard finishing passes – produce fully lowered LLVM dialect MLIR
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stage7_finished.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/stage6_llvm_dialect.mlir
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -convert-vector-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts
            -o ${CMAKE_CURRENT_BINARY_DIR}/stage7_finished.mlir
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/stage6_llvm_dialect.mlir
          buddy-opt
  COMMENT "Ada300 Stage 7: finishing passes → stage7_finished.mlir"
  VERBATIM
)

# Translate the finished LLVM dialect MLIR → LLVM IR text (.ll)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_ada300.ll
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-translate
            ${CMAKE_CURRENT_BINARY_DIR}/stage7_finished.mlir
            -mlir-to-llvmir
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_ada300.ll
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/stage7_finished.mlir
  COMMENT "Ada300: mlir-translate → subgraph0_ada300.ll (LLVM IR)"
  VERBATIM
)

# Assemble the Ada300-lowered LLVM IR into an object file
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_ada300.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/llvm-as
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_ada300.ll
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_ada300.bc &&
          ${LLVM_TOOLS_BINARY_DIR}/llc
            -filetype=obj -relocation-model=pic -O3
            -march=riscv64 -mattr=+v
            ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_ada300.bc
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_ada300.o
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_ada300.ll
  COMMENT "Ada300: assembling subgraph0_ada300.ll → subgraph0_ada300.o (RISC-V + BuddyExt)"
  VERBATIM
)

# ---------------------------------------------------------------------------
# Step 4: Compile forward.mlir → forward.o
#
# The forward graph is a thin dispatcher (func calls + memref manipulation),
# so only the standard LLVM lowering pipeline is needed.
# ---------------------------------------------------------------------------
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
            -pass-pipeline
            "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg,
             tosa-to-tensor, tosa-to-arith), empty-tensor-to-alloc-tensor,
             convert-elementwise-to-linalg)" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline
            "builtin.module(func.func(buffer-deallocation-simplification,
             convert-linalg-to-loops), eliminate-empty-tensors,
             func.func(llvm-request-c-wrappers),
             convert-math-to-llvm, convert-math-to-libm,
             convert-scf-to-cf, convert-cf-to-llvm,
             convert-arith-to-llvm, expand-strided-metadata,
             finalize-memref-to-llvm, convert-func-to-llvm,
             reconcile-unrealized-casts)" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/forward.o
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/forward.mlir
  COMMENT "Ada300: lowering forward.mlir → forward.o"
  VERBATIM
)

# ---------------------------------------------------------------------------
# Convenience target: build both object files (generic path, Path A)
# ---------------------------------------------------------------------------
add_custom_target(ada300-import
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o
    ${CMAKE_CURRENT_BINARY_DIR}/forward.o
  COMMENT "Ada300: model import and compilation complete (generic path)"
)

# ---------------------------------------------------------------------------
# Convenience target: Ada300-specific lowering (Path B)
#
# The final deliverable of the Ada300 lowering pipeline is subgraph0_ada300.ll
# (LLVM IR with Ada300 custom instructions as inline assembly).  Compiling to
# a native object file (subgraph0_ada300.o) requires an Ada300-enabled RISC-V
# toolchain whose instruction mnemonics (gmm.cfg / gmma.mm / tcsync / vfpwnl)
# are not yet registered in the BuddyExt LLVM backend.  Once those TableGen
# entries are added and the backend is rebuilt, update this target to also
# depend on subgraph0_ada300.o.
# ---------------------------------------------------------------------------
add_custom_target(ada300-lower
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_ada300.ll
    ${CMAKE_CURRENT_BINARY_DIR}/forward.o
  COMMENT "Ada300: Ada300HL/Ada300HW lowering complete (LLVM IR stage)"
)

# ---------------------------------------------------------------------------
# Optional target: assemble Ada300 LLVM IR → object file.
# Requires buddy-llc built with Ada300 ISA support (-mattr=+buddyext,+v and
# Ada300 instruction TableGen entries in RISCVInstrInfoBuddyExt.td).
# ---------------------------------------------------------------------------
add_custom_target(ada300-assemble
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_ada300.o
    ${CMAKE_CURRENT_BINARY_DIR}/forward.o
  COMMENT "Ada300: assembling LLVM IR → object files (requires Ada300-enabled toolchain)"
)
