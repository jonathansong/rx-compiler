// RVV Matmul Kernel — runnable demo using the rvv dialect
//
// This is the RUNNABLE companion to rvv-matmul-static.mlir.
// It implements the 4 × VLMAX outer-product microkernel from the design doc
// (Section 3.2) using ops that buddy-mlir supports today.
//
// Key design mapping:
//   rvv.setvl         → vsetvli   (configure VL at runtime)
//   rvv.load          → vle32.v   (load B strip)
//   rvv.store         → vse32.v   (store accumulator)
//   vector.broadcast  → vfmv.v.f  (broadcast A scalar to vector)
//   vector.fma        → vfmadd.vv (fused multiply-add)
//
// NOTE: rvv.mul is integer-only (vmul.vx); there is no rvv.fmul / rvv.fmacc
// in the current dialect.  We use vector.broadcast + vector.fma as a substitute
// which generates vfmadd.vv instead of the ideal vfmacc.vf.  Adding rvv.fmacc
// (Step 5 in the design plan) would replace this with a single instruction.
//
// Buffers use flat 1-D memrefs to avoid subview complexity.
// The caller pre-packs A/B and zeros C before calling this kernel.
//
// To lower and compile for RVV:
//
//   buddy-opt rvv-matmul-kernel.mlir \
//     -lower-rvv \
//     -convert-vector-to-llvm \
//     -convert-scf-to-cf \
//     -convert-cf-to-llvm \
//     -convert-math-to-llvm \
//     -convert-arith-to-llvm \
//     -convert-func-to-llvm \
//     -finalize-memref-to-llvm \
//     -reconcile-unrealized-casts \
//   | buddy-translate --buddy-to-llvmir \
//   | llc -mtriple=riscv64-unknown-linux-gnu \
//          -mattr=+m,+f,+d,+v -O2 -o rvv-matmul-kernel.s

// SEW=32 (FP32), LMUL=1 encoding for rvv.setvl:
//   sew  = 2  (log2(32) - 3)
//   lmul = 0  (LMUL=1)

// Inner microkernel: C rows += A[0..3] × B (4 × vl outer product).
// All arguments are flat 1-D memrefs; caller slices them from the packed layout.
func.func @rvv_mmt4d_4xvl(
    %a0 : memref<?xf32>,   // A_pack row 0  [K scalars]
    %a1 : memref<?xf32>,   // A_pack row 1
    %a2 : memref<?xf32>,   // A_pack row 2
    %a3 : memref<?xf32>,   // A_pack row 3
    %b  : memref<?xf32>,   // B_pack row k  [N floats, strip-mined in N]
    %c0 : memref<?xf32>,   // C_pack row 0  [N floats, accumulator]
    %c1 : memref<?xf32>,
    %c2 : memref<?xf32>,
    %c3 : memref<?xf32>,
    %K  : index,           // reduction depth
    %N  : index) {         // output width

  %zero  = arith.constant 0 : index
  %one   = arith.constant 1 : index
  %sew   = arith.constant 2 : index   // SEW=32
  %lmul  = arith.constant 0 : index   // LMUL=1

  // Strip-mine the N dimension: each iteration processes vl elements
  %_avl, %_col = scf.while (%avl = %N, %col = %zero)
      : (index, index) -> (index, index) {
    %cond = arith.cmpi sgt, %avl, %zero : index
    scf.condition(%cond) %avl, %col : index, index
  } do {
  ^bb0(%avl : index, %col : index):

    %vl = rvv.setvl %avl, %sew, %lmul : index   // → vsetvli

    // Load initial C accumulators for this strip  → vle32.v
    %cv0 = rvv.load %c0[%col], %vl : memref<?xf32>, vector<[4]xf32>, index
    %cv1 = rvv.load %c1[%col], %vl : memref<?xf32>, vector<[4]xf32>, index
    %cv2 = rvv.load %c2[%col], %vl : memref<?xf32>, vector<[4]xf32>, index
    %cv3 = rvv.load %c3[%col], %vl : memref<?xf32>, vector<[4]xf32>, index

    // Reduction over K: 4 × vl outer-product accumulate
    %acc0, %acc1, %acc2, %acc3 = scf.for %k = %zero to %K step %one
        iter_args(%r0 = %cv0, %r1 = %cv1, %r2 = %cv2, %r3 = %cv3)
        -> (vector<[4]xf32>, vector<[4]xf32>, vector<[4]xf32>, vector<[4]xf32>) {

      // Load B strip  → vle32.v
      %bv = rvv.load %b[%col], %vl : memref<?xf32>, vector<[4]xf32>, index

      // Load 4 A scalars for this k
      %s0 = memref.load %a0[%k] : memref<?xf32>
      %s1 = memref.load %a1[%k] : memref<?xf32>
      %s2 = memref.load %a2[%k] : memref<?xf32>
      %s3 = memref.load %a3[%k] : memref<?xf32>

      // Broadcast each A scalar to a vector  → vfmv.v.f
      %sv0 = vector.broadcast %s0 : f32 to vector<[4]xf32>
      %sv1 = vector.broadcast %s1 : f32 to vector<[4]xf32>
      %sv2 = vector.broadcast %s2 : f32 to vector<[4]xf32>
      %sv3 = vector.broadcast %s3 : f32 to vector<[4]xf32>

      // Fused multiply-add: r_i = bv * sv_i + r_i  → vfmadd.vv
      // (ideal: vfmacc.vf when rvv.fmacc is added to the dialect)
      %n0 = vector.fma %bv, %sv0, %r0 : vector<[4]xf32>
      %n1 = vector.fma %bv, %sv1, %r1 : vector<[4]xf32>
      %n2 = vector.fma %bv, %sv2, %r2 : vector<[4]xf32>
      %n3 = vector.fma %bv, %sv3, %r3 : vector<[4]xf32>

      scf.yield %n0, %n1, %n2, %n3
          : vector<[4]xf32>, vector<[4]xf32>, vector<[4]xf32>, vector<[4]xf32>
    }

    // Store accumulators back  → vse32.v
    rvv.store %acc0, %c0[%col], %vl : vector<[4]xf32>, memref<?xf32>, index
    rvv.store %acc1, %c1[%col], %vl : vector<[4]xf32>, memref<?xf32>, index
    rvv.store %acc2, %c2[%col], %vl : vector<[4]xf32>, memref<?xf32>, index
    rvv.store %acc3, %c3[%col], %vl : vector<[4]xf32>, memref<?xf32>, index

    // Advance to next strip
    %new_col = arith.addi %col, %vl : index
    %new_avl = arith.subi %avl, %vl : index
    scf.yield %new_avl, %new_col : index, index
  }

  return
}
//
//   A_pack_row_i: flat view of row i of A_pack  → [K] scalars
//   B_pack_row_k: flat view of row k of B_pack  → [N] vector-loadable
//   C_pack_row_i: flat view of row i of C_pack  → [N] accumulator
//
// To lower and compile for RVV:
//
//   buddy-opt rvv-matmul-kernel.mlir \
//     -lower-rvv \
//     -convert-vector-to-llvm \
//     -convert-scf-to-cf \
//     -convert-cf-to-llvm \
//     -convert-math-to-llvm \
//     -convert-arith-to-llvm \
//     -convert-func-to-llvm \
//     -finalize-memref-to-llvm \
//     -reconcile-unrealized-casts \
//   | buddy-translate --buddy-to-llvmir \
//   | llc -mtriple=riscv64-unknown-linux-gnu \
//          -mattr=+m,+f,+d,+v -O2 -o rvv-matmul-kernel.s
