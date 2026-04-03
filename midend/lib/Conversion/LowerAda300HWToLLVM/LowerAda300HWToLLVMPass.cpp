//===- LowerAda300HWToLLVMPass.cpp - Ada300HW → LLVM Lowering Pass --------===//
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
// Implements the Ada300HW → LLVM dialect lowering pass.
//
// Each Ada300HW op is lowered to an `llvm.inline_asm` operation that encodes
// the corresponding ADA300 custom RISC-V instruction.  This allows the LLVM
// backend to emit the instruction byte sequence via the existing RISC-V
// inline-asm emission path, without requiring full LLVM target support for the
// ADA300 custom extension.
//
// Instruction string conventions:
//
//   Vector ops (RVV-style register-to-register):
//     vfpwnl.<func>.<segments> vd, vs1
//     vfcvt.<part>             vd, vs1
//     vfmul.low                vd, vs1, vs2
//     vfmul.high               vd, vs1, vs2
//
//   Config register writes (encoded as immediates in asm string):
//     gmm.cfg  <row>, <col>, <acc>, <mode>
//     gmm.type <out>, <act>, <wht>
//     gmm.iter <blk_a>, <blk_w>
//
//   Tensor Core execute (pointer operands):
//     gmm.mm   rd, ra, rw
//     gmma.mm  rd, ra, rw
//     gmma.mt  rd, ra, rw
//     gmv.mm   rd, rm, rv
//     gmva.mm  rd, rm, rv
//     gmva.mt  rd, rm, rv
//
//   Synchronisation:
//     tcsync
//
// Attributes are embedded in the asm string at conversion time (they are
// compile-time constants derived from op attributes).
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FormatVariadic.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "Ada300HW/Ada300HWDialect.h"
#include "Ada300HW/Ada300HWOps.h"
#include "Ada300HW/Transforms.h"

using namespace mlir;
using namespace buddy::ada300hw;
using namespace buddy::ada300hl; // for shared attrs

//===----------------------------------------------------------------------===//
// Attribute → string helpers
//===----------------------------------------------------------------------===//

namespace {

static StringRef nlFuncStr(NonlinearFunc f) {
  switch (f) {
  case NonlinearFunc::exp:   return "exp";
  case NonlinearFunc::log:   return "log";
  case NonlinearFunc::sqrt:  return "sqrt";
  case NonlinearFunc::rsqrt: return "rsqrt";
  case NonlinearFunc::div:   return "div";
  }
  llvm_unreachable("unknown NonlinearFunc");
}

static StringRef partStr(Part p) {
  return p == Part::low ? "low" : "high";
}

static int dtypeInt(TensorDataType d) {
  return static_cast<int>(d);
}

static int tensorModeInt(TensorMode m) {
  return static_cast<int>(m);
}

/// Helper: build a zero-result LLVM inline asm op with side effects and no
/// SSA operands.  Used for pure-attribute config / sync ops.
static void emitSideEffectAsm(Location loc, StringRef asmStr,
                               PatternRewriter &rewriter) {
  MLIRContext *ctx = rewriter.getContext();
  auto asmDialect =
      LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT);
  rewriter.create<LLVM::InlineAsmOp>(
      loc,
      /*res=*/TypeRange{},
      /*operands=*/ValueRange{},
      /*asm_string=*/rewriter.getStringAttr(asmStr),
      /*constraints=*/rewriter.getStringAttr(""),
      /*has_side_effects=*/true,
      /*is_align_stack=*/false,
      /*asm_dialect=*/asmDialect,
      /*operand_attrs=*/ArrayAttr{});
}

/// Helper: build a single-result LLVM inline asm op for vector register ops.
/// Constraint: "=vr,vr"  (output vr, one input vr).
static Value emitVectorUnaryAsm(Location loc, Type resultLLVMTy, Value input,
                                 StringRef asmStr,
                                 ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = rewriter.getContext();
  auto asmDialect =
      LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT);
  auto asmOp = rewriter.create<LLVM::InlineAsmOp>(
      loc,
      /*res=*/resultLLVMTy,
      /*operands=*/ValueRange{input},
      /*asm_string=*/rewriter.getStringAttr(asmStr),
      /*constraints=*/rewriter.getStringAttr("=vr,vr"),
      /*has_side_effects=*/false,
      /*is_align_stack=*/false,
      /*asm_dialect=*/asmDialect,
      /*operand_attrs=*/ArrayAttr{});
  return asmOp.getRes();
}

/// Helper: build a single-result LLVM inline asm op for binary vector ops.
/// Constraint: "=vr,vr,vr"
static Value emitVectorBinaryAsm(Location loc, Type resultLLVMTy, Value lhs,
                                  Value rhs, StringRef asmStr,
                                  ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = rewriter.getContext();
  auto asmDialect =
      LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT);
  auto asmOp = rewriter.create<LLVM::InlineAsmOp>(
      loc,
      /*res=*/resultLLVMTy,
      /*operands=*/ValueRange{lhs, rhs},
      /*asm_string=*/rewriter.getStringAttr(asmStr),
      /*constraints=*/rewriter.getStringAttr("=vr,vr,vr"),
      /*has_side_effects=*/false,
      /*is_align_stack=*/false,
      /*asm_dialect=*/asmDialect,
      /*operand_attrs=*/ArrayAttr{});
  return asmOp.getRes();
}

/// Helper: emits a three-pointer side-effect inline asm (for tensor execute
/// ops like gmma.mm).  Pointer arguments are "r" constrained (GPR).
static void emitThreePtrAsm(Location loc, Value pDst, Value pAct, Value pWht,
                              StringRef mnemonic,
                              ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = rewriter.getContext();
  auto asmDialect =
      LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT);
  std::string asmStr = (mnemonic + " $0, $1, $2").str();
  rewriter.create<LLVM::InlineAsmOp>(
      loc,
      /*res=*/TypeRange{},
      /*operands=*/ValueRange{pDst, pAct, pWht},
      /*asm_string=*/rewriter.getStringAttr(asmStr),
      /*constraints=*/rewriter.getStringAttr("r,r,r"),
      /*has_side_effects=*/true,
      /*is_align_stack=*/false,
      /*asm_dialect=*/asmDialect,
      /*operand_attrs=*/ArrayAttr{});
}

/// Extract a raw i8* (opaque ptr) to the first element of a converted memref.
/// The adaptor value for a memref is an LLVM struct; `extractvalue 1` gives
/// the aligned pointer.
static Value extractAlignedPtr(Location loc, Value llvmMemref,
                                LLVM::LLVMStructType structTy,
                                ConversionPatternRewriter &rewriter) {
  // struct { ptr, ptr, i64, [...], [...] }
  // index 1 = aligned pointer
  return rewriter.create<LLVM::ExtractValueOp>(loc, llvmMemref, 1);
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Vector Op Patterns
//===----------------------------------------------------------------------===//

namespace {

// ada300hw.vfpwnl → vfpwnl.<func>.<segments>  vd, vs1
struct LowerVfpwnlOp : public ConvertOpToLLVMPattern<Ada300HW_VfpwnlOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Ada300HW_VfpwnlOp op,
                  Ada300HW_VfpwnlOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultLLVMTy =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultLLVMTy)
      return failure();

    NonlinearFunc func = op.getFuncAttr().getValue();
    SegmentCount segs = op.getSegmentsAttr().getValue();
    int segInt = (segs == SegmentCount::seg16) ? 16 : 32;
    std::string asmStr =
        llvm::formatv("vfpwnl.{0}.{1} $0, $1", nlFuncStr(func), segInt).str();

    Value result = emitVectorUnaryAsm(op.getLoc(), resultLLVMTy,
                                      adaptor.getInput(), asmStr, rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// ada300hw.vfcvt → vfcvt.<part>  vd, vs1
struct LowerVfcvtOp : public ConvertOpToLLVMPattern<Ada300HW_VfcvtOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Ada300HW_VfcvtOp op,
                  Ada300HW_VfcvtOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultLLVMTy =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultLLVMTy)
      return failure();

    std::string asmStr =
        llvm::formatv("vfcvt.{0} $0, $1",
                      partStr(op.getPartAttr().getValue()))
            .str();
    Value result = emitVectorUnaryAsm(op.getLoc(), resultLLVMTy,
                                      adaptor.getInput(), asmStr, rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// ada300hw.vfmul_low → vfmul.low  vd, vs1, vs2
struct LowerVfmulLowOp : public ConvertOpToLLVMPattern<Ada300HW_VfmulLowOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Ada300HW_VfmulLowOp op,
                  Ada300HW_VfmulLowOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultLLVMTy =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultLLVMTy)
      return failure();

    Value result =
        emitVectorBinaryAsm(op.getLoc(), resultLLVMTy, adaptor.getLhs(),
                             adaptor.getRhs(), "vfmul.low $0, $1, $2",
                             rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// ada300hw.vfmul_high → vfmul.high  vd, vs1, vs2
struct LowerVfmulHighOp
    : public ConvertOpToLLVMPattern<Ada300HW_VfmulHighOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Ada300HW_VfmulHighOp op,
                  Ada300HW_VfmulHighOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultLLVMTy =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultLLVMTy)
      return failure();

    Value result =
        emitVectorBinaryAsm(op.getLoc(), resultLLVMTy, adaptor.getLhs(),
                             adaptor.getRhs(), "vfmul.high $0, $1, $2",
                             rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Tensor Core Config Op Patterns
//===----------------------------------------------------------------------===//

// ada300hw.set_gmm_cfg → gmm.cfg <row>, <col>, <acc>, <mode>
struct LowerSetGmmCfgOp
    : public ConvertOpToLLVMPattern<Ada300HW_SetGmmCfgOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Ada300HW_SetGmmCfgOp op,
                  Ada300HW_SetGmmCfgOp::Adaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    std::string asmStr =
        llvm::formatv("gmm.cfg {0}, {1}, {2}, {3}",
                      op.getRowSize(), op.getColSize(), op.getAccSize(),
                      tensorModeInt(op.getModeAttr().getValue()))
            .str();
    emitSideEffectAsm(op.getLoc(), asmStr, rewriter);
    rewriter.eraseOp(op);
    return success();
  }
};

// ada300hw.set_gmm_type → gmm.type <out_dt>, <act_dt>, <wht_dt>
struct LowerSetGmmTypeOp
    : public ConvertOpToLLVMPattern<Ada300HW_SetGmmTypeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Ada300HW_SetGmmTypeOp op,
                  Ada300HW_SetGmmTypeOp::Adaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    std::string asmStr =
        llvm::formatv("gmm.type {0}, {1}, {2}",
                      dtypeInt(op.getOutTypeAttr().getValue()),
                      dtypeInt(op.getActTypeAttr().getValue()),
                      dtypeInt(op.getWhtTypeAttr().getValue()))
            .str();
    emitSideEffectAsm(op.getLoc(), asmStr, rewriter);
    rewriter.eraseOp(op);
    return success();
  }
};

// ada300hw.set_gmm_iter → gmm.iter <blk_cnt_a>, <blk_cnt_w>
struct LowerSetGmmIterOp
    : public ConvertOpToLLVMPattern<Ada300HW_SetGmmIterOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Ada300HW_SetGmmIterOp op,
                  Ada300HW_SetGmmIterOp::Adaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    std::string asmStr =
        llvm::formatv("gmm.iter {0}, {1}",
                      op.getBlkCntA(), op.getBlkCntW())
            .str();
    emitSideEffectAsm(op.getLoc(), asmStr, rewriter);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Tensor Core Execute Op Patterns
//
// The memref operands are lowered to LLVM structs by MemRefToLLVM patterns.
// We extract the aligned pointer (field index 1 of the descriptor struct) and
// pass it as a GPR ("r") operand to the inline asm.
//===----------------------------------------------------------------------===//

/// Utility base for three-memref-argument tensor ops.
template <typename OpTy>
struct ThreeMemrefOpLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

  /// Subclasses provide the asm mnemonic.
  virtual StringRef mnemonic() const = 0;

  LogicalResult
  matchAndRewrite(OpTy op,
                  typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto extractPtr = [&](Value llvmStruct) -> Value {
      auto structTy =
          dyn_cast<LLVM::LLVMStructType>(llvmStruct.getType());
      if (!structTy)
        return llvmStruct; // already a pointer
      return rewriter.create<LLVM::ExtractValueOp>(loc, llvmStruct, 1);
    };

    Value pDst = extractPtr(adaptor.getDst());
    Value pAct = extractPtr(adaptor.getAct());
    Value pWht = extractPtr(adaptor.getWht());

    emitThreePtrAsm(loc, pDst, pAct, pWht, mnemonic(), rewriter);
    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerGmmMmOp : public ThreeMemrefOpLowering<Ada300HW_GmmMmOp> {
  using ThreeMemrefOpLowering::ThreeMemrefOpLowering;
  StringRef mnemonic() const override { return "gmm.mm"; }
};

struct LowerGmmaMmOp : public ThreeMemrefOpLowering<Ada300HW_GmmaMmOp> {
  using ThreeMemrefOpLowering::ThreeMemrefOpLowering;
  StringRef mnemonic() const override { return "gmma.mm"; }
};

struct LowerGmmaMtOp : public ThreeMemrefOpLowering<Ada300HW_GmmaMtOp> {
  using ThreeMemrefOpLowering::ThreeMemrefOpLowering;
  StringRef mnemonic() const override { return "gmma.mt"; }
};

/// Utility base for matrix-vector ops (dst, mat, vec) -- same three-memref
/// structure but different naming convention in the op.
template <typename OpTy>
struct ThreeMemrefMVOpLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

  virtual StringRef mnemonic() const = 0;

  LogicalResult
  matchAndRewrite(OpTy op,
                  typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto extractPtr = [&](Value llvmStruct) -> Value {
      auto structTy =
          dyn_cast<LLVM::LLVMStructType>(llvmStruct.getType());
      if (!structTy)
        return llvmStruct;
      return rewriter.create<LLVM::ExtractValueOp>(loc, llvmStruct, 1);
    };

    Value pDst = extractPtr(adaptor.getDst());
    Value pMat = extractPtr(adaptor.getMat());
    Value pVec = extractPtr(adaptor.getVec());

    emitThreePtrAsm(loc, pDst, pMat, pVec, mnemonic(), rewriter);
    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerGmvMmOp : public ThreeMemrefMVOpLowering<Ada300HW_GmvMmOp> {
  using ThreeMemrefMVOpLowering::ThreeMemrefMVOpLowering;
  StringRef mnemonic() const override { return "gmv.mm"; }
};

struct LowerGmvaMmOp : public ThreeMemrefMVOpLowering<Ada300HW_GmvaMmOp> {
  using ThreeMemrefMVOpLowering::ThreeMemrefMVOpLowering;
  StringRef mnemonic() const override { return "gmva.mm"; }
};

struct LowerGmvaMtOp : public ThreeMemrefMVOpLowering<Ada300HW_GmvaMtOp> {
  using ThreeMemrefMVOpLowering::ThreeMemrefMVOpLowering;
  StringRef mnemonic() const override { return "gmva.mt"; }
};

//===----------------------------------------------------------------------===//
// Synchronisation Op
//===----------------------------------------------------------------------===//

struct LowerTcsyncOp : public ConvertOpToLLVMPattern<Ada300HW_TcsyncOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Ada300HW_TcsyncOp op,
                  Ada300HW_TcsyncOp::Adaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    emitSideEffectAsm(op.getLoc(), "tcsync", rewriter);
    rewriter.eraseOp(op);
    return success();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pattern population function (public API)
//===----------------------------------------------------------------------===//

void mlir::buddy::populateAda300HWToLLVMPatterns(LLVMTypeConverter &converter,
                                                  RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
      LowerVfpwnlOp,
      LowerVfcvtOp,
      LowerVfmulLowOp,
      LowerVfmulHighOp,
      LowerSetGmmCfgOp,
      LowerSetGmmTypeOp,
      LowerSetGmmIterOp,
      LowerGmmMmOp,
      LowerGmmaMmOp,
      LowerGmmaMtOp,
      LowerGmvMmOp,
      LowerGmvaMmOp,
      LowerGmvaMtOp,
      LowerTcsyncOp
  >(converter);
  // clang-format on
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace {

struct LowerAda300HWToLLVMPass
    : public PassWrapper<LowerAda300HWToLLVMPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerAda300HWToLLVMPass)

  StringRef getArgument() const final { return "lower-ada300hw-to-llvm"; }

  StringRef getDescription() const final {
    return "Lower Ada300HW ISA-level ops to LLVM dialect (inline asm).";
  }

  LowerAda300HWToLLVMPass() = default;
  LowerAda300HWToLLVMPass(const LowerAda300HWToLLVMPass &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Ada300HWDialect>();
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    LLVMTypeConverter converter(ctx);
    RewritePatternSet patterns(ctx);

    // Populate MemRef → LLVM struct type conversion so that the memref
    // operands of tensor execute ops are convertible.
    populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);

    // Add Ada300HW-specific lowering patterns.
    mlir::buddy::populateAda300HWToLLVMPatterns(converter, patterns);

    // Mark all Ada300HW ops as illegal; LLVM dialect ops are legal.
    LLVMConversionTarget target(*ctx);
    target.addIllegalDialect<Ada300HWDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<memref::MemRefDialect>();

    if (failed(applyPartialConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

void mlir::buddy::registerLowerAda300HWToLLVMPass() {
  PassRegistration<LowerAda300HWToLLVMPass>();
}
