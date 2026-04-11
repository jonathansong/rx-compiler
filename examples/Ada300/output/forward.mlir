module {
  func.func private @subgraph0(memref<128x64xf32, strided<[64, 1], offset: ?>>, memref<128xf32, strided<[1], offset: ?>>, memref<1x64xf32, strided<[64, 1], offset: ?>>, memref<64x128xf32, strided<[128, 1], offset: ?>>, memref<64xf32, strided<[1], offset: ?>>) -> memref<1x64xf32>
  func.func @forward(%arg0: memref<16576xf32>, %arg1: memref<1x64xf32>) -> memref<1x64xf32> {
    %subview = memref.subview %arg0[0] [8192] [1] : memref<16576xf32> to memref<8192xf32>
    %expand_shape = memref.expand_shape %subview [[0, 1]] output_shape [128, 64] : memref<8192xf32> into memref<128x64xf32>
    %subview_0 = memref.subview %arg0[8192] [128] [1] : memref<16576xf32> to memref<128xf32, strided<[1], offset: 8192>>
    %subview_1 = memref.subview %arg0[8320] [8192] [1] : memref<16576xf32> to memref<8192xf32, strided<[1], offset: 8320>>
    %expand_shape_2 = memref.expand_shape %subview_1 [[0, 1]] output_shape [64, 128] : memref<8192xf32, strided<[1], offset: 8320>> into memref<64x128xf32, strided<[128, 1], offset: 8320>>
    %subview_3 = memref.subview %arg0[16512] [64] [1] : memref<16576xf32> to memref<64xf32, strided<[1], offset: 16512>>
    %cast = memref.cast %expand_shape : memref<128x64xf32> to memref<128x64xf32, strided<[64, 1], offset: ?>>
    %cast_4 = memref.cast %subview_0 : memref<128xf32, strided<[1], offset: 8192>> to memref<128xf32, strided<[1], offset: ?>>
    %cast_5 = memref.cast %arg1 : memref<1x64xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>
    %cast_6 = memref.cast %expand_shape_2 : memref<64x128xf32, strided<[128, 1], offset: 8320>> to memref<64x128xf32, strided<[128, 1], offset: ?>>
    %cast_7 = memref.cast %subview_3 : memref<64xf32, strided<[1], offset: 16512>> to memref<64xf32, strided<[1], offset: ?>>
    %0 = call @subgraph0(%cast, %cast_4, %cast_5, %cast_6, %cast_7) : (memref<128x64xf32, strided<[64, 1], offset: ?>>, memref<128xf32, strided<[1], offset: ?>>, memref<1x64xf32, strided<[64, 1], offset: ?>>, memref<64x128xf32, strided<[128, 1], offset: ?>>, memref<64xf32, strided<[1], offset: ?>>) -> memref<1x64xf32>
    return %0 : memref<1x64xf32>
  }
}

