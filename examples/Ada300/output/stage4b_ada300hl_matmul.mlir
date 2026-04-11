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
    %alloc_1 = memref.alloc() : memref<1x64xf32>
    %alloc_2 = memref.alloc() : memref<64x128xf32>
    ada300hl.copy_to_sram %arg2, %alloc_1 : memref<1x64xf32, strided<[?, ?], offset: ?>>, memref<1x64xf32>
    ada300hl.copy_to_sram %alloc, %alloc_2 : memref<64x128xf32>, memref<64x128xf32>
    ada300hl.tensor_mma %alloc_0, %alloc_1, %alloc_2 {acc_size = 64 : i32, act_type = #ada300hl.dtype<fp16>, blk_cnt_a = 1 : i32, blk_cnt_w = 1 : i32, col_size = 128 : i32, mode = #ada300hl.tensor_mode<inner>, out_type = #ada300hl.dtype<fp16>, rhs_transposed = false, row_size = 1 : i32, wht_type = #ada300hl.dtype<fp16>} : memref<1x128xf32>, memref<1x64xf32>, memref<64x128xf32>
    ada300hl.tensor_sync
    memref.dealloc %alloc_1 : memref<1x64xf32>
    memref.dealloc %alloc_2 : memref<64x128xf32>
    %expand_shape = memref.expand_shape %arg1 [[0, 1]] output_shape [1, 128] : memref<128xf32, strided<[?], offset: ?>> into memref<1x128xf32, strided<[?, ?], offset: ?>>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%expand_shape, %alloc_0 : memref<1x128xf32, strided<[?, ?], offset: ?>>, memref<1x128xf32>) outs(%alloc_3 : memref<1x128xf32>) {
    ^bb0(%in: f32, %in_10: f32, %out: f32):
      %6 = arith.addf %in, %in_10 : f32
      linalg.yield %6 : f32
    }
    %1 = vector.transfer_read %alloc_3[%c0, %c0], %cst {in_bounds = [true]} : memref<1x128xf32>, vector<128xf32>
    %2 = ada300hl.pwnl %1 {func = #ada300hl.nlfunc<exp>, segments = #ada300hl.segments<seg16>} : vector<128xf32> -> vector<128xf32>
    vector.transfer_write %2, %alloc_3[%c0, %c0] {in_bounds = [true]} : vector<128xf32>, memref<1x128xf32>
    %3 = vector.transfer_read %alloc_3[%c0, %c0], %cst {in_bounds = [true]} : memref<1x128xf32>, vector<128xf32>
    %4 = ada300hl.pwnl %3 {func = #ada300hl.nlfunc<sqrt>, segments = #ada300hl.segments<seg16>} : vector<128xf32> -> vector<128xf32>
    vector.transfer_write %4, %alloc_3[%c0, %c0] {in_bounds = [true]} : vector<128xf32>, memref<1x128xf32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32>
    linalg.transpose ins(%arg3 : memref<64x128xf32, strided<[?, ?], offset: ?>>) outs(%alloc_4 : memref<128x64xf32>) permutation = [1, 0] 
    %5 = memref.get_global @__constant_1x64xf32 : memref<1x64xf32>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x64xf32>
    memref.copy %5, %alloc_5 : memref<1x64xf32> to memref<1x64xf32>
    %alloc_6 = memref.alloc() : memref<1x128xf32>
    %alloc_7 = memref.alloc() : memref<128x64xf32>
    ada300hl.copy_to_sram %alloc_3, %alloc_6 : memref<1x128xf32>, memref<1x128xf32>
    ada300hl.copy_to_sram %alloc_4, %alloc_7 : memref<128x64xf32>, memref<128x64xf32>
    ada300hl.tensor_mma %alloc_5, %alloc_6, %alloc_7 {acc_size = 128 : i32, act_type = #ada300hl.dtype<fp16>, blk_cnt_a = 1 : i32, blk_cnt_w = 1 : i32, col_size = 64 : i32, mode = #ada300hl.tensor_mode<inner>, out_type = #ada300hl.dtype<fp16>, rhs_transposed = false, row_size = 1 : i32, wht_type = #ada300hl.dtype<fp16>} : memref<1x64xf32>, memref<1x128xf32>, memref<128x64xf32>
    ada300hl.tensor_sync
    memref.dealloc %alloc_6 : memref<1x128xf32>
    memref.dealloc %alloc_7 : memref<128x64xf32>
    %expand_shape_8 = memref.expand_shape %arg4 [[0, 1]] output_shape [1, 64] : memref<64xf32, strided<[?], offset: ?>> into memref<1x64xf32, strided<[?, ?], offset: ?>>
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x64xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_8, %alloc_5 : memref<1x64xf32, strided<[?, ?], offset: ?>>, memref<1x64xf32>) outs(%alloc_9 : memref<1x64xf32>) {
    ^bb0(%in: f32, %in_10: f32, %out: f32):
      %6 = arith.addf %in, %in_10 : f32
      linalg.yield %6 : f32
    }
    memref.dealloc %alloc : memref<64x128xf32>
    memref.dealloc %alloc_0 : memref<1x128xf32>
    memref.dealloc %alloc_3 : memref<1x128xf32>
    memref.dealloc %alloc_4 : memref<128x64xf32>
    memref.dealloc %alloc_5 : memref<1x64xf32>
    return %alloc_9 : memref<1x64xf32>
  }
}

