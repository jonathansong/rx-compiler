// RVV Static Packed Matmul — 16x16x16 FP32 (tensor-level reference)
//
// ── How this file is derived ─────────────────────────────────────────────────
//
// Start from rvv-matmul-input.mlir (plain linalg.matmul) and run:
//
//   mlir-opt rvv-matmul-input.mlir \
//     --linalg-block-pack-matmul="block-factors=4,4,1"
//
// That pass rewrites linalg.matmul into the pack / generic / unpack form
// shown below.  You can also write it by hand once you know the tile shapes.
//
// ── Tile shape ────────────────────────────────────────────────────────────────
//   Assumes VLEN=128, FP32, LMUL=1 → VLMAX=4
//   m_r = 4,  n_r = 4 (= VLMAX),  k_r = 1
//
// ── Packed tensor shapes ──────────────────────────────────────────────────────
//   A_pack : [4 x 16 x 4 x 1]  (M_o x K_o x m_r x k_r)
//   B_pack : [4 x 16 x 4 x 1]  (N_o x K_o x n_r x k_r)
//   C_pack : [4 x  4 x 4 x 4]  (M_o x N_o x m_r x n_r)
//
// ── Current lowering status ───────────────────────────────────────────────────
// This file documents the TARGET representation.
// linalg.pack does not yet have a complete bufferization path in the current
// buddy-mlir build (one-shot-bufferize rejects it without allow-unknown-ops,
// and allow-unknown-ops leaves bufferization.to_tensor unconverted).
//
// For a RUNNABLE demo that compiles to RVV assembly, see:
//   rvv-matmul-kernel.mlir   (uses rvv dialect directly — works today)
//
// VLEN parameterization: change block-factors=<mr,nr,kr> to match hardware.
//   VLEN=128 → VLMAX=4  → block-factors=4,4,1
//   VLEN=256 → VLMAX=8  → block-factors=4,8,1
//   VLEN=512 → VLMAX=16 → block-factors=4,16,1

#map  = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

func.func @matmul(%A: tensor<16x16xf32>,
                  %B: tensor<16x16xf32>,
                  %C: tensor<16x16xf32>) -> tensor<16x16xf32> {

  // ── Pack A: [16 x 16] → [4 x 16 x 4 x 1]  (M_o x K_o x m_r x k_r) ─────────
  %A_buf = tensor.empty() : tensor<4x16x4x1xf32>
  %A_pack = linalg.pack %A
      outer_dims_perm = [0, 1]
      inner_dims_pos  = [0, 1]
      inner_tiles     = [4, 1]
      into %A_buf
      : tensor<16x16xf32> -> tensor<4x16x4x1xf32>

  // ── Pack B: [16 x 16] → [4 x 16 x 4 x 1]  (N_o x K_o x n_r x k_r) ─────────
  // outer_dims_perm=[1,0] transposes K and N so the outer layout is [N_o, K_o].
  // inner_dims_pos=[1,0]  picks N (dim 1) then K (dim 0) for inner tiles.
  %B_buf = tensor.empty() : tensor<4x16x4x1xf32>
  %B_pack = linalg.pack %B
      outer_dims_perm = [1, 0]
      inner_dims_pos  = [1, 0]
      inner_tiles     = [4, 1]
      into %B_buf
      : tensor<16x16xf32> -> tensor<4x16x4x1xf32>

  // ── Pack C: [16 x 16] → [4 x 4 x 4 x 4]  (M_o x N_o x m_r x n_r) ──────────
  %C_buf = tensor.empty() : tensor<4x4x4x4xf32>
  %C_pack = linalg.pack %C
      inner_dims_pos = [0, 1]
      inner_tiles    = [4, 4]
      into %C_buf
      : tensor<16x16xf32> -> tensor<4x4x4x4xf32>

  // ── Packed matmul (mmt4d-equivalent linalg.generic) ──────────────────────────
  // Indexing: C[m,n,m0,n0] += A[m,k,m0,k0] * B[n,k,n0,k0]
  // This is what linalg.mmt4d computes; expressed here as a generic for clarity.
  // LLVM will auto-vectorize the k0/n0 inner loops to vfmacc.vf when targeting
  // -mattr=+v.
  %C_result = linalg.generic {
    indexing_maps  = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction",
                      "parallel", "parallel", "reduction"]
  }
  ins(%A_pack, %B_pack : tensor<4x16x4x1xf32>, tensor<4x16x4x1xf32>)
  outs(%C_pack : tensor<4x4x4x4xf32>) {
  ^bb0(%a : f32, %b : f32, %c : f32):
    %mul = arith.mulf %a, %b : f32
    %acc = arith.addf %c, %mul : f32
    linalg.yield %acc : f32
  } -> tensor<4x4x4x4xf32>

  // ── Unpack C: [4 x 4 x 4 x 4] → [16 x 16] ───────────────────────────────────
  %C_out = linalg.unpack %C_result
      inner_dims_pos = [0, 1]
      inner_tiles    = [4, 4]
      into %C
      : tensor<4x4x4x4xf32> -> tensor<16x16xf32>

  return %C_out : tensor<16x16xf32>
}
