#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  llvm.func @rxops_bridge_exp_f32(!llvm.ptr, !llvm.ptr, i64) -> i32
  llvm.func @rxops_bridge_sqrt_f32(!llvm.ptr, !llvm.ptr, i64) -> i32
  llvm.func @rxops_bridge_matmul_f32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
  memref.global "private" constant @__constant_1x64xf32 : memref<1x64xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x128xf32 : memref<1x128xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  func.func @subgraph0(%arg0: memref<128x64xf32, strided<[?, ?], offset: ?>>, %arg1: memref<128xf32, strided<[?], offset: ?>>, %arg2: memref<1x64xf32, strided<[?, ?], offset: ?>>, %arg3: memref<64x128xf32, strided<[?, ?], offset: ?>>, %arg4: memref<64xf32, strided<[?], offset: ?>>) -> memref<1x64xf32> {
    %c128_i64 = arith.constant 128 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32>
    linalg.transpose ins(%arg0 : memref<128x64xf32, strided<[?, ?], offset: ?>>) outs(%alloc : memref<64x128xf32>) permutation = [1, 0] 
    %0 = memref.get_global @__constant_1x128xf32 : memref<1x128xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    memref.copy %0, %alloc_0 : memref<1x128xf32> to memref<1x128xf32>
    %alloc_1 = memref.alloc() : memref<1x64xf32>
    %alloc_2 = memref.alloc() : memref<64x128xf32>
    memref.copy %arg2, %alloc_1 : memref<1x64xf32, strided<[?, ?], offset: ?>> to memref<1x64xf32>
    memref.copy %alloc, %alloc_2 : memref<64x128xf32> to memref<64x128xf32>
    %intptr = memref.extract_aligned_pointer_as_index %alloc_0 : memref<1x128xf32> -> index
    %1 = arith.index_cast %intptr : index to i64
    %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
    %intptr_3 = memref.extract_aligned_pointer_as_index %alloc_1 : memref<1x64xf32> -> index
    %3 = arith.index_cast %intptr_3 : index to i64
    %4 = llvm.inttoptr %3 : i64 to !llvm.ptr
    %intptr_4 = memref.extract_aligned_pointer_as_index %alloc_2 : memref<64x128xf32> -> index
    %5 = arith.index_cast %intptr_4 : index to i64
    %6 = llvm.inttoptr %5 : i64 to !llvm.ptr
    %7 = llvm.call @rxops_bridge_matmul_f32(%2, %4, %6, %c1_i64, %c128_i64, %c64_i64) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    memref.dealloc %alloc_1 : memref<1x64xf32>
    memref.dealloc %alloc_2 : memref<64x128xf32>
    %expand_shape = memref.expand_shape %arg1 [[0, 1]] output_shape [1, 128] : memref<128xf32, strided<[?], offset: ?>> into memref<1x128xf32, strided<[?, ?], offset: ?>>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%expand_shape, %alloc_0 : memref<1x128xf32, strided<[?, ?], offset: ?>>, memref<1x128xf32>) outs(%alloc_5 : memref<1x128xf32>) {
    ^bb0(%in: f32, %in_22: f32, %out: f32):
      %30 = arith.addf %in, %in_22 : f32
      linalg.yield %30 : f32
    }
    %8 = vector.transfer_read %alloc_5[%c0, %c0], %cst {in_bounds = [true]} : memref<1x128xf32>, vector<128xf32>
    %alloca = memref.alloca() : memref<128xf32>
    %alloca_6 = memref.alloca() : memref<128xf32>
    vector.store %8, %alloca[%c0] : memref<128xf32>, vector<128xf32>
    %intptr_7 = memref.extract_aligned_pointer_as_index %alloca : memref<128xf32> -> index
    %9 = arith.index_cast %intptr_7 : index to i64
    %10 = llvm.inttoptr %9 : i64 to !llvm.ptr
    %intptr_8 = memref.extract_aligned_pointer_as_index %alloca_6 : memref<128xf32> -> index
    %11 = arith.index_cast %intptr_8 : index to i64
    %12 = llvm.inttoptr %11 : i64 to !llvm.ptr
    %13 = llvm.call @rxops_bridge_exp_f32(%12, %10, %c128_i64) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %14 = vector.load %alloca_6[%c0] : memref<128xf32>, vector<128xf32>
    vector.transfer_write %14, %alloc_5[%c0, %c0] {in_bounds = [true]} : vector<128xf32>, memref<1x128xf32>
    %15 = vector.transfer_read %alloc_5[%c0, %c0], %cst {in_bounds = [true]} : memref<1x128xf32>, vector<128xf32>
    %alloca_9 = memref.alloca() : memref<128xf32>
    %alloca_10 = memref.alloca() : memref<128xf32>
    vector.store %15, %alloca_9[%c0] : memref<128xf32>, vector<128xf32>
    %intptr_11 = memref.extract_aligned_pointer_as_index %alloca_9 : memref<128xf32> -> index
    %16 = arith.index_cast %intptr_11 : index to i64
    %17 = llvm.inttoptr %16 : i64 to !llvm.ptr
    %intptr_12 = memref.extract_aligned_pointer_as_index %alloca_10 : memref<128xf32> -> index
    %18 = arith.index_cast %intptr_12 : index to i64
    %19 = llvm.inttoptr %18 : i64 to !llvm.ptr
    %20 = llvm.call @rxops_bridge_sqrt_f32(%19, %17, %c128_i64) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %21 = vector.load %alloca_10[%c0] : memref<128xf32>, vector<128xf32>
    vector.transfer_write %21, %alloc_5[%c0, %c0] {in_bounds = [true]} : vector<128xf32>, memref<1x128xf32>
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32>
    linalg.transpose ins(%arg3 : memref<64x128xf32, strided<[?, ?], offset: ?>>) outs(%alloc_13 : memref<128x64xf32>) permutation = [1, 0] 
    %22 = memref.get_global @__constant_1x64xf32 : memref<1x64xf32>
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x64xf32>
    memref.copy %22, %alloc_14 : memref<1x64xf32> to memref<1x64xf32>
    %alloc_15 = memref.alloc() : memref<1x128xf32>
    %alloc_16 = memref.alloc() : memref<128x64xf32>
    memref.copy %alloc_5, %alloc_15 : memref<1x128xf32> to memref<1x128xf32>
    memref.copy %alloc_13, %alloc_16 : memref<128x64xf32> to memref<128x64xf32>
    %intptr_17 = memref.extract_aligned_pointer_as_index %alloc_14 : memref<1x64xf32> -> index
    %23 = arith.index_cast %intptr_17 : index to i64
    %24 = llvm.inttoptr %23 : i64 to !llvm.ptr
    %intptr_18 = memref.extract_aligned_pointer_as_index %alloc_15 : memref<1x128xf32> -> index
    %25 = arith.index_cast %intptr_18 : index to i64
    %26 = llvm.inttoptr %25 : i64 to !llvm.ptr
    %intptr_19 = memref.extract_aligned_pointer_as_index %alloc_16 : memref<128x64xf32> -> index
    %27 = arith.index_cast %intptr_19 : index to i64
    %28 = llvm.inttoptr %27 : i64 to !llvm.ptr
    %29 = llvm.call @rxops_bridge_matmul_f32(%24, %26, %28, %c1_i64, %c64_i64, %c128_i64) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    memref.dealloc %alloc_15 : memref<1x128xf32>
    memref.dealloc %alloc_16 : memref<128x64xf32>
    %expand_shape_20 = memref.expand_shape %arg4 [[0, 1]] output_shape [1, 64] : memref<64xf32, strided<[?], offset: ?>> into memref<1x64xf32, strided<[?, ?], offset: ?>>
    %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<1x64xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_20, %alloc_14 : memref<1x64xf32, strided<[?, ?], offset: ?>>, memref<1x64xf32>) outs(%alloc_21 : memref<1x64xf32>) {
    ^bb0(%in: f32, %in_22: f32, %out: f32):
      %30 = arith.addf %in, %in_22 : f32
      linalg.yield %30 : f32
    }
    memref.dealloc %alloc : memref<64x128xf32>
    memref.dealloc %alloc_0 : memref<1x128xf32>
    memref.dealloc %alloc_5 : memref<1x128xf32>
    memref.dealloc %alloc_13 : memref<128x64xf32>
    memref.dealloc %alloc_14 : memref<1x64xf32>
    return %alloc_21 : memref<1x64xf32>
  }
}

