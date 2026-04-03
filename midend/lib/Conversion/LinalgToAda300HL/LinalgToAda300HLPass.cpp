//===- LinalgToAda300HLPass.cpp - Linalg → Ada300HL Lowering Pass ---------===//
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
// Implements the Linalg → Ada300HL lowering pass.
//
// This pass converts structured linear-algebra ops from the `linalg` dialect
// into the equivalent high-level Ada300HL ops, preserving enough semantic
// information for the subsequent Ada300HL → Ada300HW lowering to emit
// efficient Tensor Core code.
//
// At this stage of the pipeline the IR is expected to be fully bufferized
// (memref operands, no tensor types).
//
// Lowering summary:
//
//   linalg.matmul ins(%A, %B) outs(%C)
//     →
//       ada300hl.copy_to_sram %A, %A_sram   // stage A into on-chip SRAM
//       ada300hl.copy_to_sram %B, %B_sram   // stage B into on-chip SRAM
//       ada300hl.tensor_mma  %C, %A_sram, %B_sram {
//           mode      = inner,
//           rhs_transposed = false,
//           row_size  = M (or 16 if dynamic),
//           col_size  = N (or 16 if dynamic),
//           acc_size  = K (or 16 if dynamic),
//           out_type  = <inferred from C element type>,
//           act_type  = <inferred from A element type>,
//           wht_type  = <inferred from B element type>,
//           blk_cnt_a = 1,
//           blk_cnt_w = 1
//       }
//       ada300hl.tensor_sync
//       ada300hl.copy_from_sram %C_sram, %C  // write result back
//
// Notes:
// - SRAM buffers are allocated via `memref.alloc`; a later pass (e.g. memory
//   planning or DMA insertion) is expected to map these to the actual on-chip
//   address space.
// - When the matrix dimensions are statically known they are encoded directly
//   in the op attributes; otherwise the conservative default (16) is used and
//   a verifier note will guide the user to tile/pad their workload.
// - Block counts (blk_cnt_a, blk_cnt_w) are set to 1 at this stage; a
//   tiling pass can increase them.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Ada300HL/Ada300HLDialect.h"
#include "Ada300HL/Ada300HLOps.h"

using namespace mlir;
using namespace buddy::ada300hl;

//===----------------------------------------------------------------------===//
// Helper – map MLIR element type to Ada300HL TensorDataType enum
//===----------------------------------------------------------------------===//

namespace {

static TensorDataType mlirTypeToTensorDataType(Type elemTy) {
  if (elemTy.isF16())   return TensorDataType::fp16;
  if (elemTy.isBF16())  return TensorDataType::bf16;
  if (elemTy.isInteger(16)) return TensorDataType::int16;
  if (elemTy.isInteger(8))  return TensorDataType::int8;
  if (elemTy.isInteger(4))  return TensorDataType::int4;
  if (isa<Fp8Type>(elemTy)) return TensorDataType::fp8;
  // Default: fp16-like for unknown / f32 (fp16 accumulator is widened).
  return TensorDataType::fp16;
}

/// Return the static dim at `idx` from a MemRefType, or `defaultVal` if the
/// dimension is dynamic.
static int32_t staticDimOr(MemRefType mrt, unsigned idx, int32_t defaultVal) {
  if (idx >= (unsigned)mrt.getRank())
    return defaultVal;
  int64_t d = mrt.getDimSize(idx);
  return ShapedType::isDynamic(d) ? defaultVal : static_cast<int32_t>(d);
}

} // namespace

//===----------------------------------------------------------------------===//
// Pattern: linalg.matmul → ada300hl.tensor_mma + tensor_sync
//===----------------------------------------------------------------------===//

namespace {
struct LinalgMatmulToAda300HLTensorMma
    : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    // Only handle bufferized (memref) form.
    if (op.hasPureTensorSemantics())
      return rewriter.notifyMatchFailure(op,
          "linalgToAda300HL requires bufferized (memref) matmul");

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    Value A = op.getInputs()[0]; // activation  (M x K)
    Value B = op.getInputs()[1]; // weights      (K x N)
    Value C = op.getOutputs()[0]; // output/acc   (M x N)

    auto aTy = cast<MemRefType>(A.getType());
    auto bTy = cast<MemRefType>(B.getType());
    auto cTy = cast<MemRefType>(C.getType());

    // Derive matrix dimension sizes from static shapes when available.
    // A is M×K, B is K×N, C is M×N.
    const int32_t kDefault = 16;
    int32_t rowSize = staticDimOr(aTy, 0, kDefault); // M
    int32_t colSize = staticDimOr(bTy, 1, kDefault); // N
    int32_t accSize = staticDimOr(aTy, 1, kDefault); // K

    // Infer hardware data types from element types.
    TensorDataType actDT = mlirTypeToTensorDataType(aTy.getElementType());
    TensorDataType whtDT = mlirTypeToTensorDataType(bTy.getElementType());
    TensorDataType outDT = mlirTypeToTensorDataType(cTy.getElementType());

    // Build attribute helpers.
    auto mkDType = [&](TensorDataType d) {
      return TensorDataTypeAttr::get(ctx, d);
    };
    auto mkI32 = [&](int32_t v) { return rewriter.getI32IntegerAttr(v); };

    // --- Allocate SRAM staging buffers -----------------------------------
    // Use the same element type and shape as the inputs; the address space
    // will be reconciled in a later memory-planning pass.
    Value aSram = rewriter.create<memref::AllocOp>(loc, aTy);
    Value bSram = rewriter.create<memref::AllocOp>(loc, bTy);

    // --- Copy A and B into SRAM ------------------------------------------
    rewriter.create<Ada300HL_CopyToSramOp>(loc, A, aSram);
    rewriter.create<Ada300HL_CopyToSramOp>(loc, B, bSram);

    // --- Emit the Tensor Core MMA ----------------------------------------
    rewriter.create<Ada300HL_TensorMmaOp>(
        loc,
        /*dst=*/C,
        /*act=*/aSram,
        /*wht=*/bSram,
        /*mode=*/TensorModeAttr::get(ctx, TensorMode::inner),
        /*rhs_transposed=*/rewriter.getBoolAttr(false),
        /*row_size=*/mkI32(rowSize),
        /*col_size=*/mkI32(colSize),
        /*acc_size=*/mkI32(accSize),
        /*out_type=*/mkDType(outDT),
        /*act_type=*/mkDType(actDT),
        /*wht_type=*/mkDType(whtDT),
        /*blk_cnt_a=*/mkI32(1),
        /*blk_cnt_w=*/mkI32(1));

    rewriter.create<Ada300HL_TensorSyncOp>(loc);

    // --- Free the staging buffers ----------------------------------------
    rewriter.create<memref::DeallocOp>(loc, aSram);
    rewriter.create<memref::DeallocOp>(loc, bSram);

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace {

struct LinalgToAda300HLPass
    : public PassWrapper<LinalgToAda300HLPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToAda300HLPass)

  StringRef getArgument() const final { return "linalg-to-ada300hl"; }

  StringRef getDescription() const final {
    return "Lower bufferized Linalg ops to Ada300HL high-level ops.";
  }

  LinalgToAda300HLPass() = default;
  LinalgToAda300HLPass(const LinalgToAda300HLPass &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Ada300HLDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<LinalgMatmulToAda300HLTensorMma>(ctx);

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace mlir::buddy {

void registerLinalgToAda300HLPass() {
  PassRegistration<LinalgToAda300HLPass>();
}

} // namespace mlir::buddy
