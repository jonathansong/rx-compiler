// Plain linalg.matmul — the starting point.
//
// To derive the packed (static) form, run:
//
//   mlir-opt rvv-matmul-input.mlir \
//     --linalg-block-pack-matmul="block-factors=4,4,1"
//
// The output is rvv-matmul-static.mlir.
//
// Tile shape assumes VLEN=128, FP32, LMUL=1 → VLMAX=4:
//   m_r = 4
//   n_r = 4  (= VLMAX)
//   k_r = 1

func.func @matmul(%A: tensor<16x16xf32>,
                  %B: tensor<16x16xf32>,
                  %C: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = linalg.matmul
       ins(%A, %B : tensor<16x16xf32>, tensor<16x16xf32>)
       outs(%C : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}
