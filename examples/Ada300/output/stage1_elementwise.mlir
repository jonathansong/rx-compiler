#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @subgraph0(%arg0: tensor<128x64xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1x64xf32>, %arg3: tensor<64x128xf32>, %arg4: tensor<64xf32>) -> tensor<1x64xf32> {
    %0 = tensor.empty() : tensor<64x128xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<128x64xf32>) outs(%0 : tensor<64x128xf32>) permutation = [1, 0] 
    %cst = arith.constant dense<0.000000e+00> : tensor<1x128xf32>
    %1 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg2, %transposed : tensor<1x64xf32>, tensor<64x128xf32>) outs(%cst : tensor<1x128xf32>) -> tensor<1x128xf32>
    %2 = tosa.const_shape  {values = dense<[1, 128]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %expanded = tensor.expand_shape %arg1 [[0, 1]] output_shape [1, 128] : tensor<128xf32> into tensor<1x128xf32>
    %3 = tensor.empty() : tensor<1x128xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded, %1 : tensor<1x128xf32>, tensor<1x128xf32>) outs(%3 : tensor<1x128xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %12 = arith.addf %in, %in_3 : f32
      linalg.yield %12 : f32
    } -> tensor<1x128xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<1x128xf32>) outs(%4 : tensor<1x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %12 = math.exp %in : f32
      linalg.yield %12 : f32
    } -> tensor<1x128xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<1x128xf32>) outs(%5 : tensor<1x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %12 = math.sqrt %in : f32
      linalg.yield %12 : f32
    } -> tensor<1x128xf32>
    %7 = tensor.empty() : tensor<128x64xf32>
    %transposed_0 = linalg.transpose ins(%arg3 : tensor<64x128xf32>) outs(%7 : tensor<128x64xf32>) permutation = [1, 0] 
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1x64xf32>
    %8 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%6, %transposed_0 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%cst_1 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %9 = tosa.const_shape  {values = dense<[1, 64]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %expanded_2 = tensor.expand_shape %arg4 [[0, 1]] output_shape [1, 64] : tensor<64xf32> into tensor<1x64xf32>
    %10 = tensor.empty() : tensor<1x64xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_2, %8 : tensor<1x64xf32>, tensor<1x64xf32>) outs(%10 : tensor<1x64xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %12 = arith.addf %in, %in_3 : f32
      linalg.yield %12 : f32
    } -> tensor<1x64xf32>
    return %11 : tensor<1x64xf32>
  }
}

