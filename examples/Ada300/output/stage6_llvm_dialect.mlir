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
    %1 = builtin.unrealized_conversion_cast %alloc_0 : memref<1x128xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    memref.copy %0, %alloc_0 : memref<1x128xf32> to memref<1x128xf32>
    %alloc_1 = memref.alloc() : memref<1x64xf32>
    %2 = builtin.unrealized_conversion_cast %alloc_1 : memref<1x64xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %alloc_2 = memref.alloc() : memref<64x128xf32>
    %3 = builtin.unrealized_conversion_cast %alloc_2 : memref<64x128xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    memref.copy %arg2, %alloc_1 : memref<1x64xf32, strided<[?, ?], offset: ?>> to memref<1x64xf32>
    memref.copy %alloc, %alloc_2 : memref<64x128xf32> to memref<64x128xf32>
    llvm.inline_asm has_side_effects asm_dialect = att "gmm.cfg 1, 128, 64, 0", ""  : () -> ()
    llvm.inline_asm has_side_effects asm_dialect = att "gmm.type 0, 0, 0", ""  : () -> ()
    llvm.inline_asm has_side_effects asm_dialect = att "gmm.iter 1, 1", ""  : () -> ()
    %4 = llvm.extractvalue %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.extractvalue %2[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.inline_asm has_side_effects asm_dialect = att "gmma.mm $0, $1, $2", "r,r,r" %4, %5, %6 : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.inline_asm has_side_effects asm_dialect = att "tcsync", ""  : () -> ()
    memref.dealloc %alloc_1 : memref<1x64xf32>
    memref.dealloc %alloc_2 : memref<64x128xf32>
    %expand_shape = memref.expand_shape %arg1 [[0, 1]] output_shape [1, 128] : memref<128xf32, strided<[?], offset: ?>> into memref<1x128xf32, strided<[?, ?], offset: ?>>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%expand_shape, %alloc_0 : memref<1x128xf32, strided<[?, ?], offset: ?>>, memref<1x128xf32>) outs(%alloc_3 : memref<1x128xf32>) {
    ^bb0(%in: f32, %in_10: f32, %out: f32):
      %18 = arith.addf %in, %in_10 : f32
      linalg.yield %18 : f32
    }
    %7 = vector.transfer_read %alloc_3[%c0, %c0], %cst {in_bounds = [true]} : memref<1x128xf32>, vector<128xf32>
    %8 = llvm.inline_asm asm_dialect = att "vfpwnl.exp.16 $0, $1", "=vr,vr" %7 : (vector<128xf32>) -> vector<128xf32>
    vector.transfer_write %8, %alloc_3[%c0, %c0] {in_bounds = [true]} : vector<128xf32>, memref<1x128xf32>
    %9 = vector.transfer_read %alloc_3[%c0, %c0], %cst {in_bounds = [true]} : memref<1x128xf32>, vector<128xf32>
    %10 = llvm.inline_asm asm_dialect = att "vfpwnl.sqrt.16 $0, $1", "=vr,vr" %9 : (vector<128xf32>) -> vector<128xf32>
    vector.transfer_write %10, %alloc_3[%c0, %c0] {in_bounds = [true]} : vector<128xf32>, memref<1x128xf32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32>
    linalg.transpose ins(%arg3 : memref<64x128xf32, strided<[?, ?], offset: ?>>) outs(%alloc_4 : memref<128x64xf32>) permutation = [1, 0] 
    %11 = memref.get_global @__constant_1x64xf32 : memref<1x64xf32>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x64xf32>
    %12 = builtin.unrealized_conversion_cast %alloc_5 : memref<1x64xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    memref.copy %11, %alloc_5 : memref<1x64xf32> to memref<1x64xf32>
    %alloc_6 = memref.alloc() : memref<1x128xf32>
    %13 = builtin.unrealized_conversion_cast %alloc_6 : memref<1x128xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %alloc_7 = memref.alloc() : memref<128x64xf32>
    %14 = builtin.unrealized_conversion_cast %alloc_7 : memref<128x64xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    memref.copy %alloc_3, %alloc_6 : memref<1x128xf32> to memref<1x128xf32>
    memref.copy %alloc_4, %alloc_7 : memref<128x64xf32> to memref<128x64xf32>
    llvm.inline_asm has_side_effects asm_dialect = att "gmm.cfg 1, 64, 128, 0", ""  : () -> ()
    llvm.inline_asm has_side_effects asm_dialect = att "gmm.type 0, 0, 0", ""  : () -> ()
    llvm.inline_asm has_side_effects asm_dialect = att "gmm.iter 1, 1", ""  : () -> ()
    %15 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.extractvalue %13[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.inline_asm has_side_effects asm_dialect = att "gmma.mm $0, $1, $2", "r,r,r" %15, %16, %17 : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.inline_asm has_side_effects asm_dialect = att "tcsync", ""  : () -> ()
    memref.dealloc %alloc_6 : memref<1x128xf32>
    memref.dealloc %alloc_7 : memref<128x64xf32>
    %expand_shape_8 = memref.expand_shape %arg4 [[0, 1]] output_shape [1, 64] : memref<64xf32, strided<[?], offset: ?>> into memref<1x64xf32, strided<[?, ?], offset: ?>>
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x64xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_8, %alloc_5 : memref<1x64xf32, strided<[?, ?], offset: ?>>, memref<1x64xf32>) outs(%alloc_9 : memref<1x64xf32>) {
    ^bb0(%in: f32, %in_10: f32, %out: f32):
      %18 = arith.addf %in, %in_10 : f32
      linalg.yield %18 : f32
    }
    memref.dealloc %alloc : memref<64x128xf32>
    memref.dealloc %alloc_0 : memref<1x128xf32>
    memref.dealloc %alloc_3 : memref<1x128xf32>
    memref.dealloc %alloc_4 : memref<128x64xf32>
    memref.dealloc %alloc_5 : memref<1x64xf32>
    return %alloc_9 : memref<1x64xf32>
  }
}

