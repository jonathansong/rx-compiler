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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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
// Pattern: linalg.generic { scalar MathOp } → transfer_read + pwnl + write
//
// Matches elementwise linalg.generic ops whose body contains exactly one
// scalar math op (exp, log, sqrt, rsqrt) and rewrites the whole generic into
// a flat vector read + ada300hl.pwnl + vector write, which is the form the
// Ada300 vector unit expects.
//
// Requirements on the matched linalg.generic:
//   - All iterator types are parallel.
//   - Exactly 1 DPS input and 1 DPS init (in-place: input == output buffer).
//   - The input/output memref has a static shape.
//   - Region body: ^bb0(%in, %out): %r = MathOp %in; linalg.yield %r
//===----------------------------------------------------------------------===//

namespace {

/// Helper: try to cast Value to MemRefType with fully static shape.
static MemRefType getStaticMemRefType(Value v) {
  auto ty = dyn_cast<MemRefType>(v.getType());
  if (!ty || !ty.hasStaticShape())
    return {};
  return ty;
}

template <typename MathOp>
struct LinalgGenericMathToAda300HLPwnl
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  NonlinearFunc func;
  explicit LinalgGenericMathToAda300HLPwnl(MLIRContext *ctx, NonlinearFunc f)
      : OpRewritePattern<linalg::GenericOp>(ctx), func(f) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // All iterators must be parallel.
    if (!llvm::all_of(op.getIteratorTypesArray(), [](utils::IteratorType t) {
          return t == utils::IteratorType::parallel;
        }))
      return failure();

    // Exactly 1 input and 1 in-place output.
    if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
      return failure();

    Value input = op.getDpsInputOperand(0)->get();
    Value output = op.getDpsInitOperand(0)->get();

    MemRefType inputTy = getStaticMemRefType(input);
    MemRefType outputTy = getStaticMemRefType(output);
    if (!inputTy || !outputTy)
      return failure();

    // Input and output element types must match and be floating-point.
    if (inputTy.getElementType() != outputTy.getElementType())
      return failure();
    if (!isa<FloatType>(inputTy.getElementType()))
      return failure();

    // Body: ^bb0(%in, %out): %r = MathOp %in; linalg.yield %r
    Block &body = op.getRegion().front();
    if (body.getOperations().size() != 2)
      return failure();
    auto mathOp = dyn_cast<MathOp>(body.front());
    if (!mathOp || mathOp.getOperand() != body.getArgument(0))
      return failure();
    auto yieldOp = cast<linalg::YieldOp>(body.back());
    if (yieldOp.getValues().size() != 1 ||
        yieldOp.getValues()[0] != mathOp.getResult())
      return failure();

    // Build flat vector type: vector<NxelemTy> where N = total element count.
    int64_t numElems = 1;
    for (int64_t d : inputTy.getShape())
      numElems *= d;
    auto vecTy =
        VectorType::get({numElems}, inputTy.getElementType());

    Location loc = op.getLoc();

    // Zero indices for transfer_read/write.
    SmallVector<Value> zeros(
        inputTy.getRank(),
        rewriter.create<arith::ConstantIndexOp>(loc, 0));

    // Padding value (zero) required by transfer_read.
    Value pad = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(inputTy.getElementType()));

    // vector.transfer_read from input.
    Value vec =
        rewriter.create<vector::TransferReadOp>(loc, vecTy, input, zeros, pad);

    // ada300hl.pwnl on the vector.
    auto funcAttr = NonlinearFuncAttr::get(rewriter.getContext(), func);
    Value result = rewriter.create<PwnlOp>(loc, vecTy, vec, funcAttr,
                                           defaultSegments(rewriter.getContext()));

    // vector.transfer_write to output.
    rewriter.create<vector::TransferWriteOp>(loc, result, output, zeros);

    rewriter.eraseOp(op);
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
    registry.insert<arith::ArithDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<math::MathDialect>();
    registry.insert<vector::VectorDialect>();
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
    patterns.add<LinalgGenericMathToAda300HLPwnl<math::ExpOp>>(
        ctx, NonlinearFunc::exp);
    patterns.add<LinalgGenericMathToAda300HLPwnl<math::LogOp>>(
        ctx, NonlinearFunc::log);
    patterns.add<LinalgGenericMathToAda300HLPwnl<math::SqrtOp>>(
        ctx, NonlinearFunc::sqrt);
    patterns.add<LinalgGenericMathToAda300HLPwnl<math::RsqrtOp>>(
        ctx, NonlinearFunc::rsqrt);

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
