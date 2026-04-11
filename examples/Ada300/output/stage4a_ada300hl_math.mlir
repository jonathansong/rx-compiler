#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  memref.global "private" constant @__constant_1x64xf32 : memref<1x64xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x128xf32 : memref<1x128xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  func.func @subgraph0(%arg0: memref<128x64xf32, strided<[?, ?], offset: ?>>, %arg1: memref<128xf32, strided<[?], offset: ?>>, %arg2: memref<1x64xf32, strided<[?, ?], offset: ?>>, %arg3: memref<64x128xf32, strided<[?, ?], offset: ?>>, %arg4: memref<64xf32, strided<[?], offset: ?>>) -> memref<1x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32>
    linalg.transpose ins(%arg0 : memref<128x64xf32, strided<[?, ?], offset: ?>>) outs(%alloc : memref<64x128xf32>) permutation = [1, 0] 
    %0 = memref.get_global @__constant_1x128xf32 : memref<1x128xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    memref.copy %0, %alloc_0 : memref<1x128xf32> to memref<1x128xf32>
    linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg2, %alloc : memref<1x64xf32, strided<[?, ?], offset: ?>>, memref<64x128xf32>) outs(%alloc_0 : memref<1x128xf32>)
    %expand_shape = memref.expand_shape %arg1 [[0, 1]] output_shape [1, 128] : memref<128xf32, strided<[?], offset: ?>> into memref<1x128xf32, strided<[?, ?], offset: ?>>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%expand_shape, %alloc_0 : memref<1x128xf32, strided<[?, ?], offset: ?>>, memref<1x128xf32>) outs(%alloc_1 : memref<1x128xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %6 = arith.addf %in, %in_6 : f32
      linalg.yield %6 : f32
    }
    %1 = vector.transfer_read %alloc_1[%c0, %c0], %cst {in_bounds = [true]} : memref<1x128xf32>, vector<128xf32>
    %2 = ada300hl.pwnl %1 {func = #ada300hl.nlfunc<exp>, segments = #ada300hl.segments<seg16>} : vector<128xf32> -> vector<128xf32>
    vector.transfer_write %2, %alloc_1[%c0, %c0] {in_bounds = [true]} : vector<128xf32>, memref<1x128xf32>
    %3 = vector.transfer_read %alloc_1[%c0, %c0], %cst {in_bounds = [true]} : memref<1x128xf32>, vector<128xf32>
    %4 = ada300hl.pwnl %3 {func = #ada300hl.nlfunc<sqrt>, segments = #ada300hl.segments<seg16>} : vector<128xf32> -> vector<128xf32>
    vector.transfer_write %4, %alloc_1[%c0, %c0] {in_bounds = [true]} : vector<128xf32>, memref<1x128xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32>
    linalg.transpose ins(%arg3 : memref<64x128xf32, strided<[?, ?], offset: ?>>) outs(%alloc_2 : memref<128x64xf32>) permutation = [1, 0] 
    %5 = memref.get_global @__constant_1x64xf32 : memref<1x64xf32>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x64xf32>
    memref.copy %5, %alloc_3 : memref<1x64xf32> to memref<1x64xf32>
    linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%alloc_1, %alloc_2 : memref<1x128xf32>, memref<128x64xf32>) outs(%alloc_3 : memref<1x64xf32>)
    %expand_shape_4 = memref.expand_shape %arg4 [[0, 1]] output_shape [1, 64] : memref<64xf32, strided<[?], offset: ?>> into memref<1x64xf32, strided<[?, ?], offset: ?>>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x64xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_4, %alloc_3 : memref<1x64xf32, strided<[?, ?], offset: ?>>, memref<1x64xf32>) outs(%alloc_5 : memref<1x64xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %6 = arith.addf %in, %in_6 : f32
      linalg.yield %6 : f32
    }
    memref.dealloc %alloc : memref<64x128xf32>
    memref.dealloc %alloc_0 : memref<1x128xf32>
    memref.dealloc %alloc_1 : memref<1x128xf32>
    memref.dealloc %alloc_2 : memref<128x64xf32>
    memref.dealloc %alloc_3 : memref<1x64xf32>
    return %alloc_5 : memref<1x64xf32>
  }
}

