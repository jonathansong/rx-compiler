#!/usr/bin/env bash

# Build Directories
MLIR_BUILD_DIR        := ../../llvm/build/
BUDDY_MLIR_BUILD_DIR  := ../../build/
CROSS_LLVM_BUILD_DIR  := ../../llvm/build-cross-clang-rv/

# Tools
MLIR_OPT       := ${MLIR_BUILD_DIR}/bin/mlir-opt
MLIR_TRANSLATE := ${MLIR_BUILD_DIR}/bin/mlir-translate
LLC            := ${MLIR_BUILD_DIR}/bin/llc
LOCAL_CLANG    := ${MLIR_BUILD_DIR}/bin/clang
BUDDY_OPT      := ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
BUDDY_TRANSLATE := ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-translate

# RISC-V toolchain (built as part of buddy-mlir)
RISCV_GNU_TOOLCHAIN      := ${BUDDY_MLIR_BUILD_DIR}/thirdparty/riscv-gnu-toolchain
RISCV_GNU_TOOLCHAIN_SYSROOT := ${RISCV_GNU_TOOLCHAIN}/sysroot
QEMU := ${RISCV_GNU_TOOLCHAIN}/bin/qemu-riscv64

# RVV target attributes
RV_TRIPLE  := riscv64-unknown-linux-gnu
RV_MATTR   := +m,+f,+d,+v

# Tile shape: m_r=4, n_r=4 (VLMAX for VLEN=128 FP32 LMUL=1), k_r=1
BLOCK_FACTORS := 4,4,1

# Standard lowering pipeline (tensor → LLVM dialect)
LOWER_PIPELINE := \
	--one-shot-bufferize="bufferize-function-boundaries \
	                      allow-return-allocs-from-loops allow-unknown-ops" \
	--linalg-generalize-named-ops \
	--convert-linalg-to-loops \
	--lower-affine \
	--convert-scf-to-cf \
	--convert-vector-to-llvm \
	--convert-arith-to-llvm \
	--convert-func-to-llvm \
	--finalize-memref-to-llvm \
	--reconcile-unrealized-casts

# ─── How to get rvv-matmul-static.mlir ───────────────────────────────────────
#
#  WAY 1 — generate it automatically (RECOMMENDED for inspection):
#    Run the --linalg-block-pack-matmul pass on the simple input file.
#    The pass rewrites linalg.matmul into pack/generic/unpack form.
#    Use 'make derive-packed' to see the output.
#
#  WAY 2 — write it by hand:
#    Once you know the tile shapes (m_r, n_r, k_r), write the
#    linalg.pack / linalg.generic / linalg.unpack IR directly.
#    See rvv-matmul-static.mlir for the hand-annotated version.
#
#  NOTE: linalg.pack does not yet have a complete lowering path in the
#  current buddy-mlir build.  rvv-matmul-static.mlir is the TARGET form
#  (documentation).  rvv-matmul-kernel.mlir is the runnable demo using
#  the rvv dialect directly.
#
# ─────────────────────────────────────────────────────────────────────────────

# Way 1: derive the packed MLIR from the input and print it (for inspection)
derive-packed:
	@echo "=== Packed form derived by --linalg-block-pack-matmul ==="
	@${MLIR_OPT} ./rvv-matmul-input.mlir \
		--linalg-block-pack-matmul="block-factors=${BLOCK_FACTORS}"

# Way 1: save the derived packed form to a file
derive-packed-save:
	@${MLIR_OPT} ./rvv-matmul-input.mlir \
		--linalg-block-pack-matmul="block-factors=${BLOCK_FACTORS}" \
		-o ./rvv-matmul-derived.mlir
	@echo "Saved to rvv-matmul-derived.mlir"

# Lower the hand-written (or derived) static MLIR to LLVM dialect
lower-static:
	@${MLIR_OPT} ./rvv-matmul-static.mlir \
		${LOWER_PIPELINE} \
		-o ./log.mlir

# Lower to LLVM IR (.ll)
lower-llvmir:
	@${MLIR_OPT} ./rvv-matmul-static.mlir \
		${LOWER_PIPELINE} \
	| ${MLIR_TRANSLATE} --mlir-to-llvmir \
		-o ./rvv-matmul.ll
	@echo "LLVM IR written to rvv-matmul.ll"

# Compile to RISC-V assembly (host llc — cross-compilation)
compile-riscv-asm:
	@${MLIR_OPT} ./rvv-matmul-static.mlir \
		${LOWER_PIPELINE} \
	| ${MLIR_TRANSLATE} --mlir-to-llvmir \
	| ${LLC} -mtriple=${RV_TRIPLE} -mattr=${RV_MATTR} -O2 \
		-o ./rvv-matmul-static.s
	@echo "RVV assembly written to rvv-matmul-static.s"
	@echo ""
	@echo "=== Checking for vfmacc / vsetvli ==="
	@grep -E "vfmacc|vsetvli|vle32|vse32" ./rvv-matmul-static.s | head -20

# Compile to object file for RISC-V
compile-riscv-obj:
	@${MLIR_OPT} ./rvv-matmul-static.mlir \
		${LOWER_PIPELINE} \
	| ${MLIR_TRANSLATE} --mlir-to-llvmir \
	| ${LLC} -mtriple=${RV_TRIPLE} -mattr=${RV_MATTR} -O2 \
		-filetype=obj -o ./rvv-matmul-static.o
	@echo "Object file: rvv-matmul-static.o"

# RVV dialect lowering pipeline (requires buddy-opt / buddy-translate)
KERNEL_LOWER_PIPELINE := \
	-lower-rvv \
	-convert-vector-to-llvm \
	-convert-scf-to-cf \
	-convert-cf-to-llvm \
	-convert-math-to-llvm \
	-convert-arith-to-llvm \
	-convert-func-to-llvm \
	-finalize-memref-to-llvm \
	-reconcile-unrealized-casts

# Lower the rvv-dialect microkernel to LLVM IR
kernel-lower:
	@${BUDDY_OPT} ./rvv-matmul-kernel.mlir \
		${KERNEL_LOWER_PIPELINE} \
	| ${BUDDY_TRANSLATE} --buddy-to-llvmir \
		-o ./rvv-matmul-kernel.ll
	@echo "LLVM IR written to rvv-matmul-kernel.ll"

# Compile the microkernel to RISC-V assembly
kernel-asm:
	@${BUDDY_OPT} ./rvv-matmul-kernel.mlir \
		${KERNEL_LOWER_PIPELINE} \
	| ${BUDDY_TRANSLATE} --buddy-to-llvmir \
	| ${LLC} -mtriple=${RV_TRIPLE} -mattr=${RV_MATTR} -O2 \
		-o ./rvv-matmul-kernel.s
	@echo "RVV assembly written to rvv-matmul-kernel.s"
	@echo ""
	@echo "=== Key RVV instructions ==="
	@grep -E "vsetvli|vle32|vse32|vfmacc|vfmadd" ./rvv-matmul-kernel.s | head -30

# Compile the microkernel to RISC-V object file
kernel-obj:
	@${BUDDY_OPT} ./rvv-matmul-kernel.mlir \
		${KERNEL_LOWER_PIPELINE} \
	| ${BUDDY_TRANSLATE} --buddy-to-llvmir \
	| ${LLC} -mtriple=${RV_TRIPLE} -mattr=${RV_MATTR} -O2 \
		-filetype=obj -o ./rvv-matmul-kernel.o
	@echo "Object file: rvv-matmul-kernel.o"

.PHONY: derive-packed derive-packed-save lower-static lower-llvmir \
        compile-riscv-asm compile-riscv-obj \
        kernel-lower kernel-asm kernel-obj
