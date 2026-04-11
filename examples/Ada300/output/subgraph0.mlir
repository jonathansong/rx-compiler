module {
  func.func @subgraph0(%arg0: tensor<128x64xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1x64xf32>, %arg3: tensor<64x128xf32>, %arg4: tensor<64xf32>) -> tensor<1x64xf32> {
    %0 = tosa.transpose %arg0 {perms = array<i32: 1, 0>} : (tensor<128x64xf32>) -> tensor<64x128xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<1x128xf32>
    %1 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg2, %0 : tensor<1x64xf32>, tensor<64x128xf32>) outs(%cst : tensor<1x128xf32>) -> tensor<1x128xf32>
    %2 = tosa.const_shape  {values = dense<[1, 128]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.reshape %arg1, %2 : (tensor<128xf32>, !tosa.shape<2>) -> tensor<1x128xf32>
    %4 = tosa.add %3, %1 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
    %5 = math.exp %4 : tensor<1x128xf32>
    %6 = math.sqrt %5 : tensor<1x128xf32>
    %7 = tosa.transpose %arg3 {perms = array<i32: 1, 0>} : (tensor<64x128xf32>) -> tensor<128x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x64xf32>
    %8 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%6, %7 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%cst_0 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %9 = tosa.const_shape  {values = dense<[1, 64]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %10 = tosa.reshape %arg4, %9 : (tensor<64xf32>, !tosa.shape<2>) -> tensor<1x64xf32>
    %11 = tosa.add %10, %8 : (tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<1x64xf32>
    return %11 : tensor<1x64xf32>
  }
}

