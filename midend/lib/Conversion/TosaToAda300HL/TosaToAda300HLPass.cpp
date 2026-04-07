//===- TosaToAda300HLPass.cpp - TOSA → Ada300HL Lowering Pass -------------===//
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
// Implements the TOSA → Ada300HL lowering pass.
//
// This pass lowers selected TOSA element-wise and matrix ops directly to
// Ada300HL high-level hardware-semantic ops, bypassing the generic
// TOSA → linalg → vector → Ada300HL chain for the most performance-critical
// ops.
//
// The pass uses the `vector.transfer_read / vector.transfer_write` idiom to
// bridge TOSA's tensor-typed operands to Ada300HL's vector-typed ops.  Only
// ops on **fully-static-shape** tensors are handled; ops with dynamic
// dimensions are left unchanged for the standard TOSA lowering pipeline.
//
// Lowering summary:
//
//   Element-wise PWL ops (static shapes only):
//     tosa.exp   %t → ada300hl.pwnl %v {func=exp,   segments=16} → tensor
//     tosa.log   %t → ada300hl.pwnl %v {func=log,   segments=16} → tensor
//     tosa.rsqrt %t → ada300hl.pwnl %v {func=rsqrt, segments=16} → tensor
//
//   Compound activation (static shapes only):
//     tosa.sigmoid %t → ada300hl.sigmoid %v {segments=16} → tensor
//
//   Matrix multiply (static shapes only):
//     tosa.matmul %a[b,m,k] %b[b,k,n]
//       → linalg.batch_matmul (tensor form, zero-initialised accumulator)
//     The resulting linalg.batch_matmul is picked up by the LinalgToAda300HL
//     pass after one-shot bufferization.
//
// Pipeline placement:
//   tosa (frontend IR)
//     ↓  tosa-to-ada300hl  (this pass)
//   ada300hl / linalg (mixed: elementwise already in ada300hl,
//                              matmul as linalg.batch_matmul)
//     ↓  one-shot-bufferization
//     ↓  linalg-to-ada300hl
//     ↓  lower-ada300hl-to-ada300hw
//   ada300hw
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Ada300HL/Ada300HLDialect.h"
#include "Ada300HL/Ada300HLOps.h"

using namespace mlir;
using namespace ::buddy::ada300hl;

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

namespace {

/// Returns the default SegmentCountAttr (16 segments, conservative).
static SegmentCountAttr defaultSegments(MLIRContext *ctx) {
  return SegmentCountAttr::get(ctx, SegmentCount::seg16);
}

/// Return the VectorType matching the shape and element type of a
/// fully-static RankedTensorType.
static VectorType tensorToVecType(RankedTensorType ty) {
  return VectorType::get(ty.getShape(), ty.getElementType());
}

/// Emit `vector.transfer_read %src[0…0], %pad` that reads the entire tensor
/// into a vector of matching shape.  The in-bounds array is set to all-true
/// because tile sizes are always equal to the tensor dimensions.
static Value readIntoVector(PatternRewriter &rewriter, Location loc,
                             Value src, VectorType vecTy) {
  auto srcTy = cast<RankedTensorType>(src.getType());
  int64_t rank = srcTy.getRank();

  SmallVector<Value> zeros(
      rank, rewriter.create<arith::ConstantIndexOp>(loc, 0));

  Type elemTy = srcTy.getElementType();
  Value pad;
  if (isa<FloatType>(elemTy))
    pad = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemTy, 0.0));
  else
    pad = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(elemTy, 0));

  SmallVector<bool> inBoundsVals(rank, true);
  return rewriter.create<vector::TransferReadOp>(loc, vecTy, src, zeros, pad,
                                                  inBoundsVals);
}

/// Emit `vector.transfer_write %vec, %empty[0…0]` back to a freshly-created
/// empty tensor with the same shape/element-type as `resultTy`.
/// Returns the resulting tensor value.
static Value writeFromVector(PatternRewriter &rewriter, Location loc,
                              Value vec, RankedTensorType resultTy) {
  Value empty = rewriter.create<tensor::EmptyOp>(
      loc, resultTy.getShape(), resultTy.getElementType());

  int64_t rank = resultTy.getRank();
  SmallVector<Value> zeros(
      rank, rewriter.create<arith::ConstantIndexOp>(loc, 0));

  SmallVector<bool> inBoundsVals(rank, true);
  return rewriter
      .create<vector::TransferWriteOp>(loc, vec, empty, zeros, inBoundsVals)
      ->getResult(0);
}

} // namespace

//===----------------------------------------------------------------------===//
// Pattern: tosa.exp → ada300hl.pwnl {func=exp, segments=16}
//===----------------------------------------------------------------------===//

namespace {
struct TosaExpToAda300HL : public OpRewritePattern<tosa::ExpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ExpOp op,
                                PatternRewriter &rewriter) const override {
    auto resultTy = cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic shapes not supported");

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    VectorType vecTy = tensorToVecType(resultTy);

    Value vec = readIntoVector(rewriter, loc, op->getOperand(0), vecTy);
    Value result = rewriter.create<PwnlOp>(
        loc, vecTy, vec, NonlinearFuncAttr::get(ctx, NonlinearFunc::exp),
        defaultSegments(ctx));
    rewriter.replaceOp(op, writeFromVector(rewriter, loc, result, resultTy));
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern: tosa.log → ada300hl.pwnl {func=log, segments=16}
//===----------------------------------------------------------------------===//

namespace {
struct TosaLogToAda300HL : public OpRewritePattern<tosa::LogOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::LogOp op,
                                PatternRewriter &rewriter) const override {
    auto resultTy = cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic shapes not supported");

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    VectorType vecTy = tensorToVecType(resultTy);

    Value vec = readIntoVector(rewriter, loc, op->getOperand(0), vecTy);
    Value result = rewriter.create<PwnlOp>(
        loc, vecTy, vec, NonlinearFuncAttr::get(ctx, NonlinearFunc::log),
        defaultSegments(ctx));
    rewriter.replaceOp(op, writeFromVector(rewriter, loc, result, resultTy));
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern: tosa.rsqrt → ada300hl.pwnl {func=rsqrt, segments=16}
//===----------------------------------------------------------------------===//

namespace {
struct TosaRsqrtToAda300HL : public OpRewritePattern<tosa::RsqrtOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::RsqrtOp op,
                                PatternRewriter &rewriter) const override {
    auto resultTy = cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic shapes not supported");

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    VectorType vecTy = tensorToVecType(resultTy);

    Value vec = readIntoVector(rewriter, loc, op->getOperand(0), vecTy);
    Value result = rewriter.create<PwnlOp>(
        loc, vecTy, vec, NonlinearFuncAttr::get(ctx, NonlinearFunc::rsqrt),
        defaultSegments(ctx));
    rewriter.replaceOp(op, writeFromVector(rewriter, loc, result, resultTy));
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern: tosa.sigmoid → ada300hl.sigmoid {segments=16}
//
// tosa.sigmoid computes 1 / (1 + exp(-x)).  The Ada300HL sigmoid op carries
// the same compound semantics and lowers to an efficient PWL-based sequence
// via the Ada300HW dialect.
//===----------------------------------------------------------------------===//

namespace {
struct TosaSigmoidToAda300HL : public OpRewritePattern<tosa::SigmoidOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::SigmoidOp op,
                                PatternRewriter &rewriter) const override {
    auto resultTy = cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic shapes not supported");

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    VectorType vecTy = tensorToVecType(resultTy);

    // tosa.sigmoid uses operand named "input" (cf. SiLUFusion.cpp).
    Value vec = readIntoVector(rewriter, loc, op.getInput(), vecTy);
    Value result =
        rewriter.create<SigmoidOp>(loc, vecTy, vec, defaultSegments(ctx));
    rewriter.replaceOp(op, writeFromVector(rewriter, loc, result, resultTy));
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern: tosa.matmul [b,m,k] × [b,k,n] → linalg.batch_matmul (tensor form)
//
// TOSA matmul operates on 3D batched tensors [batch, M, K] × [batch, K, N]
// producing [batch, M, N].  This pattern lowers it to `linalg.batch_matmul`
// (tensor-semantics) with a zero-initialised accumulator injected as the
// output operand.  After one-shot bufferization the resulting linalg op is
// handled by the LinalgToAda300HL pass.
//
// Zero-point operands (a_zp, b_zp) are intentionally ignored because this
// codebase always emits zero values for them (floating-point inference path).
// Quantised models with non-zero zero-points should use the standard TOSA
// lowering instead.
//
// Only static-shape operands are handled; ops with dynamic dimensions fall
// through to the standard TOSA lowering pipeline.
//===----------------------------------------------------------------------===//

namespace {
struct TosaMatmulToAda300HL : public OpRewritePattern<tosa::MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    Value a = op.getA();
    Value b = op.getB();
    auto aTy = cast<RankedTensorType>(a.getType());
    auto resultTy = cast<RankedTensorType>(op->getResult(0).getType());

    if (!aTy.hasStaticShape() || !resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "dynamic shapes not supported for tosa.matmul lowering");

    Location loc = op.getLoc();
    Type elemTy = resultTy.getElementType();

    // Allocate a zero-initialised output accumulator.
    Value emptyOut = rewriter.create<tensor::EmptyOp>(
        loc, resultTy.getShape(), elemTy);

    Value zero;
    if (isa<FloatType>(elemTy))
      zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(elemTy, 0.0));
    else
      zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(elemTy, 0));

    Value filledOut =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyOut})
            .getResult(0);

    // Lower to linalg.batch_matmul:  C[b,m,n] += A[b,m,k] * B[b,k,n]
    auto batchMm = rewriter.create<linalg::BatchMatmulOp>(
        loc, TypeRange{resultTy}, ValueRange{a, b}, ValueRange{filledOut});
    rewriter.replaceOp(op, batchMm.getResult(0));
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace {
struct TosaToAda300HLPass
    : public PassWrapper<TosaToAda300HLPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TosaToAda300HLPass)

  StringRef getArgument() const final { return "tosa-to-ada300hl"; }

  StringRef getDescription() const final {
    return "Lower TOSA element-wise and matmul ops to Ada300HL high-level ops "
           "(static shapes only).";
  }

  TosaToAda300HLPass() = default;
  TosaToAda300HLPass(const TosaToAda300HLPass &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Ada300HLDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<
        TosaExpToAda300HL,
        TosaLogToAda300HL,
        TosaRsqrtToAda300HL,
        TosaSigmoidToAda300HL,
        TosaMatmulToAda300HL
    >(ctx);

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace mlir::buddy {

void registerTosaToAda300HLPass() {
  PassRegistration<TosaToAda300HLPass>();
}

} // namespace mlir::buddy
