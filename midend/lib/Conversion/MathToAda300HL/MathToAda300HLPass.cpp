//===- MathToAda300HLPass.cpp - Math → Ada300HL Lowering Pass -------------===//
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
// Implements the Math → Ada300HL lowering pass.
//
// This pass replaces standard MLIR `math` dialect ops that have direct
// hardware support on the ADA300 vector unit with the corresponding high-level
// `ada300hl` ops.  The `ada300hl` ops preserve enough semantic information for
// subsequent optimisation passes (e.g. segment-count tuning, fusion) before
// they are eventually lowered to the Ada300HW ISA-level dialect.
//
// Lowering summary:
//
//   math.exp  %v  →  ada300hl.pwnl %v  {func = #ada300hl.nlfunc<exp>,
//                                        segments = #ada300hl.segments<16>}
//   math.log  %v  →  ada300hl.pwnl %v  {func = #ada300hl.nlfunc<log>,
//                                        segments = #ada300hl.segments<16>}
//   math.sqrt %v  →  ada300hl.pwnl %v  {func = #ada300hl.nlfunc<sqrt>,
//                                        segments = #ada300hl.segments<16>}
//   math.rsqrt %v →  ada300hl.pwnl %v  {func = #ada300hl.nlfunc<rsqrt>,
//                                        segments = #ada300hl.segments<16>}
//
// Only vector-typed operands are lowered; scalar math ops are left unchanged
// so that they can be handled by the standard arith lowering pipeline.
//
// The default segment count (16) is conservative and can be changed by a
// subsequent pass that has target-quality data.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Ada300HL/Ada300HLDialect.h"
#include "Ada300HL/Ada300HLOps.h"

using namespace mlir;
using namespace ::buddy::ada300hl;

//===----------------------------------------------------------------------===//
// Helper – default segment-count attribute
//===----------------------------------------------------------------------===//

namespace {

/// Returns the default SegmentCountAttr (16 segments).
static SegmentCountAttr defaultSegments(MLIRContext *ctx) {
  return SegmentCountAttr::get(ctx, SegmentCount::seg16);
}

/// Returns true when the type is a vector type (the only form the ADA300
/// vector unit can process directly).
static bool isVectorTy(Type t) { return isa<VectorType>(t); }

} // namespace

//===----------------------------------------------------------------------===//
// Pattern: math.exp → ada300hl.pwnl {func=exp}
//===----------------------------------------------------------------------===//

namespace {
struct MathExpToAda300HLPwnl : public OpRewritePattern<math::ExpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::ExpOp op,
                                PatternRewriter &rewriter) const override {
    if (!isVectorTy(op.getOperand().getType()))
      return failure();

    auto funcAttr =
        NonlinearFuncAttr::get(rewriter.getContext(), NonlinearFunc::exp);
    rewriter.replaceOpWithNewOp<PwnlOp>(
        op, op.getResult().getType(), op.getOperand(), funcAttr,
        defaultSegments(rewriter.getContext()));
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern: math.log → ada300hl.pwnl {func=log}
//===----------------------------------------------------------------------===//

namespace {
struct MathLogToAda300HLPwnl : public OpRewritePattern<math::LogOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::LogOp op,
                                PatternRewriter &rewriter) const override {
    if (!isVectorTy(op.getOperand().getType()))
      return failure();

    auto funcAttr =
        NonlinearFuncAttr::get(rewriter.getContext(), NonlinearFunc::log);
    rewriter.replaceOpWithNewOp<PwnlOp>(
        op, op.getResult().getType(), op.getOperand(), funcAttr,
        defaultSegments(rewriter.getContext()));
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern: math.sqrt → ada300hl.pwnl {func=sqrt}
//===----------------------------------------------------------------------===//

namespace {
struct MathSqrtToAda300HLPwnl : public OpRewritePattern<math::SqrtOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::SqrtOp op,
                                PatternRewriter &rewriter) const override {
    if (!isVectorTy(op.getOperand().getType()))
      return failure();

    auto funcAttr =
        NonlinearFuncAttr::get(rewriter.getContext(), NonlinearFunc::sqrt);
    rewriter.replaceOpWithNewOp<PwnlOp>(
        op, op.getResult().getType(), op.getOperand(), funcAttr,
        defaultSegments(rewriter.getContext()));
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern: math.rsqrt → ada300hl.pwnl {func=rsqrt}
//===----------------------------------------------------------------------===//

namespace {
struct MathRsqrtToAda300HLPwnl : public OpRewritePattern<math::RsqrtOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::RsqrtOp op,
                                PatternRewriter &rewriter) const override {
    if (!isVectorTy(op.getOperand().getType()))
      return failure();

    auto funcAttr =
        NonlinearFuncAttr::get(rewriter.getContext(), NonlinearFunc::rsqrt);
    rewriter.replaceOpWithNewOp<PwnlOp>(
        op, op.getResult().getType(), op.getOperand(), funcAttr,
        defaultSegments(rewriter.getContext()));
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace {

struct MathToAda300HLPass
    : public PassWrapper<MathToAda300HLPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MathToAda300HLPass)

  StringRef getArgument() const final { return "math-to-ada300hl"; }

  StringRef getDescription() const final {
    return "Lower Math dialect ops to Ada300HL high-level ops "
           "(vector operands only).";
  }

  MathToAda300HLPass() = default;
  MathToAda300HLPass(const MathToAda300HLPass &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Ada300HLDialect>();
    registry.insert<math::MathDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<
        MathExpToAda300HLPwnl,
        MathLogToAda300HLPwnl,
        MathSqrtToAda300HLPwnl,
        MathRsqrtToAda300HLPwnl
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

void registerMathToAda300HLPass() {
  PassRegistration<MathToAda300HLPass>();
}

} // namespace mlir::buddy
