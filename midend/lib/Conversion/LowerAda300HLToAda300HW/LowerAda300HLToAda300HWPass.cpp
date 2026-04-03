//===- LowerAda300HLToAda300HWPass.cpp - Ada300HL to Ada300HW Lowering ---===//
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
// Implements the Ada300HL → Ada300HW lowering pass.
//
// This pass lowers each high-level ada300hl op to the corresponding sequence
// of ada300hw ISA-level ops.  After the pass, the function body contains no
// more ada300hl ops.
//
// Lowering summary:
//
//   Vector / PWL path:
//     ada300hl.pwnl        → ada300hw.vfpwnl {func=<attr>, masked=false}
//     ada300hl.cvt         → ada300hw.vfcvt
//     ada300hl.vmul_mixed  → ada300hw.vfmul_low  (when part=low)
//                            ada300hw.vfmul_high (when part=high)
//
//   Tensor Core path:
//     ada300hl.tensor_mma  → ada300hw.set_gmm_cfg
//                            ada300hw.set_gmm_type
//                            ada300hw.set_gmm_iter
//                            ada300hw.gmma_mm or gmma_mt
//     ada300hl.tensor_sync → ada300hw.tcsync
//
//   Data-movement / layout ops:
//     ada300hl.pack        → erased (backend DMA codegen owns this)
//     ada300hl.unpack      → erased
//     ada300hl.layout_cast → erased (layout consumed during lowering)
//     ada300hl.copy_to_sram    → memref.copy (placeholder)
//     ada300hl.copy_from_sram  → memref.copy (placeholder)
//     ada300hl.copy_to_vr      → memref.copy (placeholder)
//     ada300hl.copy_from_vr    → memref.copy (placeholder)
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Ada300HL/Ada300HLDialect.h"
#include "Ada300HL/Ada300HLOps.h"
#include "Ada300HL/Transforms.h"
#include "Ada300HW/Ada300HWDialect.h"
#include "Ada300HW/Ada300HWOps.h"

using namespace mlir;
using namespace buddy::ada300hl;
using namespace buddy::ada300hw;

//===----------------------------------------------------------------------===//
// Helper – infer Ada300HL TensorDataType from an MLIR element type
//===----------------------------------------------------------------------===//

namespace {

/// Map an MLIR scalar/element type to the Ada300HW ISA data-type encoding.
/// Returns TensorDataType::fp16 as a safe fallback for unrecognised types;
/// callers that need strict checking should validate before calling.
static TensorDataTypeAttr inferDataTypeAttr(MLIRContext *ctx, Type elemTy) {
  TensorDataType kind = TensorDataType::fp16; // safe default
  if (elemTy.isF16())
    kind = TensorDataType::fp16;
  else if (elemTy.isBF16())
    kind = TensorDataType::bf16;
  else if (elemTy.isInteger(16))
    kind = TensorDataType::int16;
  else if (elemTy.isInteger(8))
    kind = TensorDataType::int8;
  else if (elemTy.isInteger(4))
    kind = TensorDataType::int4;
  else if (isa<Fp8Type>(elemTy))
    kind = TensorDataType::fp8;
  return TensorDataTypeAttr::get(ctx, kind);
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pattern: ada300hl.pwnl → ada300hw.vfpwnl
//===----------------------------------------------------------------------===//

namespace {
struct LowerPwnlOp : public OpRewritePattern<Ada300HL_PwnlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Ada300HL_PwnlOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<Ada300HW_VfpwnlOp>(
        op, op.getResult().getType(), op.getInput(),
        op.getFuncAttr(), op.getSegmentsAttr(), rewriter.getBoolAttr(false));
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern: ada300hl.cvt → ada300hw.vfcvt
//===----------------------------------------------------------------------===//

namespace {
struct LowerCvtOp : public OpRewritePattern<Ada300HL_CvtOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Ada300HL_CvtOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<Ada300HW_VfcvtOp>(
        op, op.getResult().getType(), op.getInput(),
        op.getDstTypeAttr(), op.getPartAttr());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern: ada300hl.vmul_mixed → ada300hw.vfmul_low / vfmul_high
//===----------------------------------------------------------------------===//

namespace {
struct LowerVMulMixedOp : public OpRewritePattern<Ada300HL_VMulMixedOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Ada300HL_VMulMixedOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = rewriter.getContext();

    // Derive src type attrs from the MLIR input element types.
    Type lhsElem = cast<VectorType>(op.getLhs().getType()).getElementType();
    Type rhsElem = cast<VectorType>(op.getRhs().getType()).getElementType();
    auto srcAAttr = inferDataTypeAttr(ctx, lhsElem);
    auto srcBAttr = inferDataTypeAttr(ctx, rhsElem);
    Type resultTy = op.getResult().getType();

    if (op.getPartAttr().getValue() == Part::low) {
      rewriter.replaceOpWithNewOp<Ada300HW_VfmulLowOp>(
          op, resultTy, op.getLhs(), op.getRhs(),
          srcAAttr, srcBAttr, op.getAccTypeAttr());
    } else {
      rewriter.replaceOpWithNewOp<Ada300HW_VfmulHighOp>(
          op, resultTy, op.getLhs(), op.getRhs(),
          srcAAttr, srcBAttr, op.getAccTypeAttr());
    }
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern: ada300hl.tensor_mma →
//   ada300hw.set_gmm_cfg
//   ada300hw.set_gmm_type
//   ada300hw.set_gmm_iter
//   ada300hw.gmma_mm  or  ada300hw.gmma_mt
//
// Note: tcsync is NOT emitted here; it is emitted by LowerTensorSyncOp
//       from the corresponding ada300hl.tensor_sync op.
//===----------------------------------------------------------------------===//

namespace {
struct LowerTensorMmaOp : public OpRewritePattern<Ada300HL_TensorMmaOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Ada300HL_TensorMmaOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // 1. Write gmm_cfg configuration register.
    rewriter.create<Ada300HW_SetGmmCfgOp>(
        loc, op.getRowSizeAttr(), op.getColSizeAttr(),
        op.getAccSizeAttr(), op.getModeAttr());

    // 2. Write gmm_type configuration register.
    rewriter.create<Ada300HW_SetGmmTypeOp>(
        loc, op.getOutTypeAttr(), op.getActTypeAttr(), op.getWhtTypeAttr());

    // 3. Write gmm_iter configuration register.
    rewriter.create<Ada300HW_SetGmmIterOp>(
        loc, op.getBlkCntAAttr(), op.getBlkCntWAttr());

    // 4. Emit the execute instruction.
    //    Use gmma_mt (transposed weights) when rhs_transposed = true,
    //    otherwise use gmma_mm.
    if (op.getRhsTransposed())
      rewriter.create<Ada300HW_GmmaMtOp>(
          loc, op.getDst(), op.getAct(), op.getWht());
    else
      rewriter.create<Ada300HW_GmmaMmOp>(
          loc, op.getDst(), op.getAct(), op.getWht());

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern: ada300hl.tensor_sync → ada300hw.tcsync
//===----------------------------------------------------------------------===//

namespace {
struct LowerTensorSyncOp : public OpRewritePattern<Ada300HL_TensorSyncOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Ada300HL_TensorSyncOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<Ada300HW_TcsyncOp>(op);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Patterns for layout / data-movement ops
//
// pack, unpack, layout_cast:  These ops have no direct ISA counterpart at
//   the Ada300HW level; the actual data-movement is either handled by
//   software loops (lowered by separate passes) or by DMA units modelled
//   outside this dialect.  We erase them here so that the pass produces clean
//   Ada300HW-only IR.
//
// copy_to/from_sram, copy_to/from_vr:  Lowered to memref.copy as a portable
//   placeholder.  A real ADA300 backend pass would replace these with
//   target-specific DMA or bulk-load intrinsics.
//===----------------------------------------------------------------------===//

namespace {

struct LowerPackOp : public OpRewritePattern<Ada300HL_PackOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Ada300HL_PackOp op,
                                PatternRewriter &rewriter) const override {
    // Emit a memref.copy as a semantics-preserving placeholder.
    rewriter.replaceOpWithNewOp<memref::CopyOp>(op, op.getSrc(), op.getDst());
    return success();
  }
};

struct LowerUnpackOp : public OpRewritePattern<Ada300HL_UnpackOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Ada300HL_UnpackOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::CopyOp>(op, op.getSrc(), op.getDst());
    return success();
  }
};

struct LowerLayoutCastOp : public OpRewritePattern<Ada300HL_LayoutCastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Ada300HL_LayoutCastOp op,
                                PatternRewriter &rewriter) const override {
    // layout_cast is view-like: erase it and replace uses with the source.
    rewriter.replaceOp(op, op.getSrc());
    return success();
  }
};

struct LowerCopyToSramOp : public OpRewritePattern<Ada300HL_CopyToSramOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Ada300HL_CopyToSramOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::CopyOp>(op, op.getSrc(), op.getDst());
    return success();
  }
};

struct LowerCopyFromSramOp : public OpRewritePattern<Ada300HL_CopyFromSramOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Ada300HL_CopyFromSramOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::CopyOp>(op, op.getSrc(), op.getDst());
    return success();
  }
};

struct LowerCopyToVrOp : public OpRewritePattern<Ada300HL_CopyToVrOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Ada300HL_CopyToVrOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::CopyOp>(op, op.getSrc(), op.getDst());
    return success();
  }
};

struct LowerCopyFromVrOp : public OpRewritePattern<Ada300HL_CopyFromVrOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Ada300HL_CopyFromVrOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::CopyOp>(op, op.getSrc(), op.getDst());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population function (public API)
//===----------------------------------------------------------------------===//

void mlir::buddy::populateAda300HLToAda300HWPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  // clang-format off
  patterns.add<
      LowerPwnlOp,
      LowerCvtOp,
      LowerVMulMixedOp,
      LowerTensorMmaOp,
      LowerTensorSyncOp,
      LowerPackOp,
      LowerUnpackOp,
      LowerLayoutCastOp,
      LowerCopyToSramOp,
      LowerCopyFromSramOp,
      LowerCopyToVrOp,
      LowerCopyFromVrOp
  >(ctx);
  // clang-format on
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace {

struct LowerAda300HLToAda300HWPass
    : public PassWrapper<LowerAda300HLToAda300HWPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerAda300HLToAda300HWPass)

  StringRef getArgument() const final {
    return "lower-ada300hl-to-ada300hw";
  }

  StringRef getDescription() const final {
    return "Lower Ada300HL high-level ops to Ada300HW ISA-level ops.";
  }

  LowerAda300HLToAda300HWPass() = default;
  LowerAda300HLToAda300HWPass(const LowerAda300HLToAda300HWPass &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Ada300HLDialect>();
    registry.insert<Ada300HWDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    mlir::buddy::populateAda300HLToAda300HWPatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

void mlir::buddy::registerLowerAda300HLToAda300HWPass() {
  PassRegistration<LowerAda300HLToAda300HWPass>();
}
