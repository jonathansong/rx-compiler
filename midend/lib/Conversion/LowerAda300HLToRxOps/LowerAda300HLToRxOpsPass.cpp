//===- LowerAda300HLToRxOpsPass.cpp - Ada300HL → rx-ops library calls ----===//
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
// Implements the Ada300HL → rx-ops library call lowering pass.
//
// This is an alternative to the Ada300HW / ISA-backend paths.  Instead of
// emitting hardware instructions or LLVM intrinsics, every Ada300HL op is
// replaced by a `llvm.call` to a flat-ABI C wrapper in the rx-ops operator
// library (rx_ops_bridge.c / rx_ops_bridge.h).
//
// Bridge functions are declared as `llvm.func` (LLVM::LLVMFuncOp) rather
// than `func.func`.  This is intentional: the `-llvm-request-c-wrappers`
// pass only operates on `func.func` ops, so using `llvm.func` ensures the
// bridge symbols are never given `_mlir_ciface_` prefixes.
//
// Pointer arguments are obtained via:
//   memref.extract_aligned_pointer_as_index → arith.index_cast (i64)
//   → llvm.inttoptr (!llvm.ptr)
//
// Lowering summary:
//
//   ada300hl.pwnl {func=exp}
//     → llvm.call @rxops_bridge_exp_f32(out_ptr, in_ptr, n)
//
//   ada300hl.pwnl {func=sqrt}
//     → llvm.call @rxops_bridge_sqrt_f32(out_ptr, in_ptr, n)
//
//   ada300hl.pwnl {func=log}
//     → llvm.call @rxops_bridge_log_f32(out_ptr, in_ptr, n)
//
//   ada300hl.pwnl {func=rsqrt}
//     → llvm.call @rxops_bridge_rsqrt_f32(out_ptr, in_ptr, n)
//
//   ada300hl.tensor_mma %C, %A, %B {...}
//     → llvm.call @rxops_bridge_matmul_f32(C_ptr, A_ptr, B_ptr, M, N, K)
//
//   ada300hl.tensor_sync  → erased
//   ada300hl.copy_to_sram / copy_from_sram → memref.copy
//
// Prerequisites (must run before this pass):
//   - one-shot-bufferize: all tensors must be memrefs.
//   - normalize-memrefs (recommended): identity-layout memrefs are simpler.
//
// Standard finishing passes to run AFTER this pass:
//   -convert-linalg-to-loops
//   -expand-strided-metadata
//   -lower-affine
//   -convert-vector-to-llvm
//   -convert-math-to-llvm -convert-math-to-libm
//   -convert-scf-to-cf -convert-cf-to-llvm
//   -convert-arith-to-llvm
//   -finalize-memref-to-llvm
//   -llvm-request-c-wrappers
//   -convert-func-to-llvm
//   -reconcile-unrealized-casts
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Ada300HL/Ada300HLDialect.h"
#include "Ada300HL/Ada300HLOps.h"
#include "Ada300HL/Transforms.h"

using namespace mlir;
using namespace ::buddy::ada300hl;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {

/// Return (or insert) an external `llvm.func` declaration for `sym`.
/// Using llvm.func (LLVM::LLVMFuncOp) rather than func.func ensures that
/// the `-llvm-request-c-wrappers` pass does not add `llvm.emit_c_interface`
/// to these declarations, so calls are emitted with plain symbol names.
static LLVM::LLVMFuncOp getOrInsertBridgeDecl(OpBuilder &b, ModuleOp mod,
                                               StringRef sym,
                                               ArrayRef<Type> argTys,
                                               Type retTy) {
  if (auto existing = mod.lookupSymbol<LLVM::LLVMFuncOp>(sym))
    return existing;

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(mod.getBody());
  auto fnTy = LLVM::LLVMFunctionType::get(retTy, argTys);
  return b.create<LLVM::LLVMFuncOp>(mod.getLoc(), sym, fnTy,
                                     LLVM::Linkage::External);
}

/// Extract the base pointer of a ranked memref as an `llvm.ptr`.
/// Steps: extract_aligned_pointer_as_index → index_cast i64 → inttoptr ptr.
static Value extractPtr(OpBuilder &b, Location loc, Value memref) {
  Value idx = b.create<memref::ExtractAlignedPointerAsIndexOp>(
      loc, b.getIndexType(), memref);
  Value i64Val = b.create<arith::IndexCastOp>(loc, b.getI64Type(), idx);
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  return b.create<LLVM::IntToPtrOp>(loc, ptrTy, i64Val);
}

/// Return the total number of elements in a static-shape ranked memref as an
/// i64 constant.  For dynamic shapes returns nullptr (caller must handle).
static Value staticElementCount(OpBuilder &b, Location loc,
                                 MemRefType mty) {
  int64_t total = 1;
  for (int64_t d : mty.getShape()) {
    if (ShapedType::isDynamic(d))
      return nullptr;
    total *= d;
  }
  return b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(total));
}

/// Return a static dimension of a ranked memref as an i64 constant, or
/// nullptr when the dimension is dynamic.
static Value staticDim(OpBuilder &b, Location loc, MemRefType mty, int axis) {
  int64_t d = mty.getShape()[axis];
  if (ShapedType::isDynamic(d))
    return nullptr;
  return b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(d));
}

} // namespace

//===----------------------------------------------------------------------===//
// Pattern: ada300hl.pwnl → rxops_bridge_<func>_f32(out, in, n)
//===----------------------------------------------------------------------===//

namespace {

/// Maps Ada300HL NonlinearFunc to the corresponding bridge symbol name.
static StringRef pwnlBridgeSym(NonlinearFunc f) {
  switch (f) {
  case NonlinearFunc::exp:   return "rxops_bridge_exp_f32";
  case NonlinearFunc::sqrt:  return "rxops_bridge_sqrt_f32";
  case NonlinearFunc::log:   return "rxops_bridge_log_f32";
  case NonlinearFunc::rsqrt: return "rxops_bridge_rsqrt_f32";
  default:
    // Other functions (sin, cos, div, ...) are not yet bridged.
    return "";
  }
}

/// Lower ada300hl.pwnl to a flat bridge call.
///
/// This pattern handles both forms of pwnl:
///
///  1. Vector form (after math-to-ada300hl + linalg-vectorize):
///       %y = ada300hl.pwnl %x {func=exp} : vector<Nxf32> -> vector<Nxf32>
///     The pattern allocates a temporary memref, stores the vector into it,
///     calls the bridge, loads the result back, and replaces the vector result.
///
///  2. Memref form (if pwnl was emitted directly on buffers):
///       ada300hl.pwnl %src_memref {func=exp} : memref<Nxf32> -> memref<Nxf32>
///     The pattern calls the bridge directly on the src/dst memrefs.
///
/// Only f32 element types are bridged at this time.
struct PwnlToRxOpsCall : public OpRewritePattern<PwnlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PwnlOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Only float32 is bridged for now.
    Type elemTy;
    if (auto vty = dyn_cast<VectorType>(op.getInput().getType()))
      elemTy = vty.getElementType();
    else if (auto mty = dyn_cast<MemRefType>(op.getInput().getType()))
      elemTy = mty.getElementType();
    else
      return rewriter.notifyMatchFailure(op, "unrecognised input type");

    if (!elemTy.isF32())
      return rewriter.notifyMatchFailure(op, "only f32 supported in rx-ops bridge");

    StringRef sym = pwnlBridgeSym(op.getFuncAttr().getValue());
    if (sym.empty())
      return rewriter.notifyMatchFailure(op,
          "no rx-ops bridge for this nonlinear function");

    // Ensure the bridge function is declared in the module.
    auto mod = op->getParentOfType<ModuleOp>();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type i64 = rewriter.getI64Type();
    Type i32 = rewriter.getI32Type();
    auto llvmFn = getOrInsertBridgeDecl(rewriter, mod, sym,
                                        {ptrTy, ptrTy, i64}, i32);

    // -----------------------------------------------------------------------
    // Vector form: allocate buffers, spill/fill around the bridge call.
    // -----------------------------------------------------------------------
    if (auto vecTy = dyn_cast<VectorType>(op.getInput().getType())) {
      // Only 1-D vectors for now.
      if (vecTy.getRank() != 1)
        return rewriter.notifyMatchFailure(op,
            "only 1-D vector pwnl supported for rx-ops bridge");

      int64_t n = vecTy.getShape()[0];
      if (ShapedType::isDynamic(n))
        return rewriter.notifyMatchFailure(op, "dynamic vector size not supported");

      auto memTy = MemRefType::get({n}, elemTy);
      Value nVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(n));

      // Allocate temporary buffers on the stack.
      Value inBuf  = rewriter.create<memref::AllocaOp>(loc, memTy);
      Value outBuf = rewriter.create<memref::AllocaOp>(loc, memTy);

      // Store input vector into inBuf.
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      rewriter.create<vector::StoreOp>(loc, op.getInput(), inBuf,
                                       ValueRange{zero});

      // Extract pointers and call bridge.
      Value inPtr  = extractPtr(rewriter, loc, inBuf);
      Value outPtr = extractPtr(rewriter, loc, outBuf);
      rewriter.create<LLVM::CallOp>(loc, llvmFn, ValueRange{outPtr, inPtr, nVal});

      // Load result vector from outBuf.
      Value result = rewriter.create<vector::LoadOp>(loc, vecTy, outBuf,
                                                     ValueRange{zero});
      rewriter.replaceOp(op, result);
      return success();
    }

    // -----------------------------------------------------------------------
    // Memref form: extract pointers directly.
    // -----------------------------------------------------------------------
    auto inMemTy = cast<MemRefType>(op.getInput().getType());
    Value n = staticElementCount(rewriter, loc, inMemTy);
    if (!n)
      return rewriter.notifyMatchFailure(op, "dynamic shape not supported");

    Value inPtr  = extractPtr(rewriter, loc, op.getInput());
    Value outPtr = extractPtr(rewriter, loc, op.getResult());

    rewriter.create<LLVM::CallOp>(loc, llvmFn, ValueRange{outPtr, inPtr, n});
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern: ada300hl.tensor_mma →
//   rxops_bridge_matmul_f32(C_ptr, A_ptr, B_ptr, M, N, K)
//===----------------------------------------------------------------------===//

namespace {

struct TensorMmaToRxOpsCall : public OpRewritePattern<TensorMmaOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorMmaOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto dstTy = dyn_cast<MemRefType>(op.getDst().getType());
    auto actTy = dyn_cast<MemRefType>(op.getAct().getType());
    auto whtTy = dyn_cast<MemRefType>(op.getWht().getType());
    if (!dstTy || !actTy || !whtTy)
      return rewriter.notifyMatchFailure(op, "expected memref operands");

    if (!dstTy.getElementType().isF32())
      return rewriter.notifyMatchFailure(op, "only f32 supported in rx-ops bridge");

    // Expect 2-D memrefs: dst [M,N], act [M,K], wht [K,N].
    if (actTy.getRank() != 2 || whtTy.getRank() != 2 || dstTy.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "expected rank-2 memrefs");

    // Extract M, N, K from static shapes.
    Value M = staticDim(rewriter, loc, actTy, 0);
    Value K = staticDim(rewriter, loc, actTy, 1);
    Value N = staticDim(rewriter, loc, whtTy, 1);
    if (!M || !N || !K)
      return rewriter.notifyMatchFailure(op, "dynamic dimensions not supported");

    // Extract raw pointers.
    Value cPtr = extractPtr(rewriter, loc, op.getDst());
    Value aPtr = extractPtr(rewriter, loc, op.getAct());
    Value bPtr = extractPtr(rewriter, loc, op.getWht());

    // Ensure the bridge function is declared.
    auto mod = op->getParentOfType<ModuleOp>();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type i64 = rewriter.getI64Type();
    Type i32 = rewriter.getI32Type();
    StringRef sym = "rxops_bridge_matmul_f32";
    auto llvmFn = getOrInsertBridgeDecl(rewriter, mod, sym,
                                        {ptrTy, ptrTy, ptrTy, i64, i64, i64},
                                        i32);

    // Emit the call.
    rewriter.create<LLVM::CallOp>(loc, llvmFn,
                                  ValueRange{cPtr, aPtr, bPtr, M, N, K});

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern: ada300hl.tensor_sync → erase
// (synchronisation is inside the library call; no explicit sync needed)
//===----------------------------------------------------------------------===//

namespace {

struct TensorSyncErase : public OpRewritePattern<TensorSyncOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TensorSyncOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern: ada300hl.copy_to_sram / copy_from_sram → memref.copy
// (SRAM staging is handled inside rxops_bridge_matmul_f32; these copies
//  become no-ops that the MLIR copy-elimination pass can fold away, but
//  keeping them as memref.copy is safe and correct.)
//===----------------------------------------------------------------------===//

namespace {

struct CopyToSramErase : public OpRewritePattern<CopyToSramOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CopyToSramOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<memref::CopyOp>(op.getLoc(), op.getSrc(), op.getDst());
    rewriter.eraseOp(op);
    return success();
  }
};

struct CopyFromSramErase : public OpRewritePattern<CopyFromSramOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CopyFromSramOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<memref::CopyOp>(op.getLoc(), op.getSrc(), op.getDst());
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace {

struct LowerAda300HLToRxOpsPass
    : public PassWrapper<LowerAda300HLToRxOpsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerAda300HLToRxOpsPass)

  StringRef getArgument() const override { return "lower-ada300hl-to-rxops"; }

  StringRef getDescription() const override {
    return "Lower Ada300HL ops to rx-ops operator-library calls "
           "(rxops_bridge_* wrappers)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect,
                    arith::ArithDialect,
                    memref::MemRefDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<PwnlToRxOpsCall,
                 TensorMmaToRxOpsCall,
                 TensorSyncErase,
                 CopyToSramErase,
                 CopyFromSramErase>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace buddy {

void registerLowerAda300HLToRxOpsPass() {
  PassRegistration<LowerAda300HLToRxOpsPass>();
}

} // namespace buddy
} // namespace mlir
