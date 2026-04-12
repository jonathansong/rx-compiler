module {
  llvm.func @free(!llvm.ptr)
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_1x64xf32(dense<0.000000e+00> : tensor<1x64xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<1 x array<64 x f32>>
  llvm.mlir.global private constant @__constant_1x128xf32(dense<0.000000e+00> : tensor<1x128xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<1 x array<128 x f32>>
  llvm.func @subgraph0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: !llvm.ptr, %arg13: !llvm.ptr, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: !llvm.ptr, %arg20: !llvm.ptr, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: !llvm.ptr, %arg27: !llvm.ptr, %arg28: i64, %arg29: i64, %arg30: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg26, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg27, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg28, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg29, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg30, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg19, %6[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.insertvalue %arg20, %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %9 = llvm.insertvalue %arg21, %8[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.insertvalue %arg22, %9[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %arg24, %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg23, %11[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg25, %12[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.insertvalue %arg12, %14[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.insertvalue %arg13, %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.insertvalue %arg14, %16[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.insertvalue %arg15, %17[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.insertvalue %arg17, %18[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %arg16, %19[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %arg18, %20[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %23 = llvm.insertvalue %arg7, %22[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.insertvalue %arg8, %23[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %25 = llvm.insertvalue %arg9, %24[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.insertvalue %arg10, %25[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %27 = llvm.insertvalue %arg11, %26[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %29 = llvm.insertvalue %arg0, %28[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.insertvalue %arg1, %29[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.insertvalue %arg2, %30[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.insertvalue %arg3, %31[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.insertvalue %arg5, %32[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.insertvalue %arg4, %33[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.insertvalue %arg6, %34[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.mlir.constant(128 : index) : i64
    %37 = llvm.mlir.constant(1 : index) : i64
    %38 = llvm.mlir.constant(64 : index) : i64
    %39 = llvm.mlir.constant(0 : index) : i64
    %40 = llvm.mlir.constant(64 : index) : i64
    %41 = llvm.mlir.constant(128 : index) : i64
    %42 = llvm.mlir.constant(1 : index) : i64
    %43 = llvm.mlir.constant(8192 : index) : i64
    %44 = llvm.mlir.zero : !llvm.ptr
    %45 = llvm.getelementptr %44[%43] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %46 = llvm.ptrtoint %45 : !llvm.ptr to i64
    %47 = llvm.mlir.constant(64 : index) : i64
    %48 = llvm.add %46, %47 : i64
    %49 = llvm.call @malloc(%48) : (i64) -> !llvm.ptr
    %50 = llvm.ptrtoint %49 : !llvm.ptr to i64
    %51 = llvm.mlir.constant(1 : index) : i64
    %52 = llvm.sub %47, %51 : i64
    %53 = llvm.add %50, %52 : i64
    %54 = llvm.urem %53, %47 : i64
    %55 = llvm.sub %53, %54 : i64
    %56 = llvm.inttoptr %55 : i64 to !llvm.ptr
    %57 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %58 = llvm.insertvalue %49, %57[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.insertvalue %56, %58[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %60 = llvm.mlir.constant(0 : index) : i64
    %61 = llvm.insertvalue %60, %59[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.insertvalue %40, %61[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.insertvalue %41, %62[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.insertvalue %41, %63[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.insertvalue %42, %64[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%39 : i64)
  ^bb1(%66: i64):  // 2 preds: ^bb0, ^bb5
    %67 = llvm.icmp "slt" %66, %38 : i64
    llvm.cond_br %67, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%39 : i64)
  ^bb3(%68: i64):  // 2 preds: ^bb2, ^bb4
    %69 = llvm.icmp "slt" %68, %36 : i64
    llvm.cond_br %69, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %70 = llvm.extractvalue %35[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.extractvalue %35[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.getelementptr %70[%71] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %73 = llvm.extractvalue %35[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = llvm.mul %68, %73 overflow<nsw, nuw> : i64
    %75 = llvm.extractvalue %35[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %76 = llvm.mul %66, %75 overflow<nsw, nuw> : i64
    %77 = llvm.add %74, %76 overflow<nsw, nuw> : i64
    %78 = llvm.getelementptr inbounds|nuw %72[%77] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %79 = llvm.load %78 : !llvm.ptr -> f32
    %80 = llvm.extractvalue %65[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %81 = llvm.mlir.constant(128 : index) : i64
    %82 = llvm.mul %66, %81 overflow<nsw, nuw> : i64
    %83 = llvm.add %82, %68 overflow<nsw, nuw> : i64
    %84 = llvm.getelementptr inbounds|nuw %80[%83] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %79, %84 : f32, !llvm.ptr
    %85 = llvm.add %68, %37 : i64
    llvm.br ^bb3(%85 : i64)
  ^bb5:  // pred: ^bb3
    %86 = llvm.add %66, %37 : i64
    llvm.br ^bb1(%86 : i64)
  ^bb6:  // pred: ^bb1
    %87 = llvm.mlir.constant(1 : index) : i64
    %88 = llvm.mlir.constant(128 : index) : i64
    %89 = llvm.mlir.constant(1 : index) : i64
    %90 = llvm.mlir.constant(128 : index) : i64
    %91 = llvm.mlir.zero : !llvm.ptr
    %92 = llvm.getelementptr %91[%90] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %93 = llvm.ptrtoint %92 : !llvm.ptr to i64
    %94 = llvm.mlir.addressof @__constant_1x128xf32 : !llvm.ptr
    %95 = llvm.getelementptr %94[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<128 x f32>>
    %96 = llvm.mlir.constant(3735928559 : index) : i64
    %97 = llvm.inttoptr %96 : i64 to !llvm.ptr
    %98 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %99 = llvm.insertvalue %97, %98[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %100 = llvm.insertvalue %95, %99[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.mlir.constant(0 : index) : i64
    %102 = llvm.insertvalue %101, %100[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %103 = llvm.insertvalue %87, %102[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %104 = llvm.insertvalue %88, %103[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %105 = llvm.insertvalue %88, %104[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.insertvalue %89, %105[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %107 = llvm.mlir.constant(1 : index) : i64
    %108 = llvm.mlir.constant(128 : index) : i64
    %109 = llvm.mlir.constant(1 : index) : i64
    %110 = llvm.mlir.constant(128 : index) : i64
    %111 = llvm.mlir.zero : !llvm.ptr
    %112 = llvm.getelementptr %111[%110] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %113 = llvm.ptrtoint %112 : !llvm.ptr to i64
    %114 = llvm.mlir.constant(64 : index) : i64
    %115 = llvm.add %113, %114 : i64
    %116 = llvm.call @malloc(%115) : (i64) -> !llvm.ptr
    %117 = llvm.ptrtoint %116 : !llvm.ptr to i64
    %118 = llvm.mlir.constant(1 : index) : i64
    %119 = llvm.sub %114, %118 : i64
    %120 = llvm.add %117, %119 : i64
    %121 = llvm.urem %120, %114 : i64
    %122 = llvm.sub %120, %121 : i64
    %123 = llvm.inttoptr %122 : i64 to !llvm.ptr
    %124 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %125 = llvm.insertvalue %116, %124[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %126 = llvm.insertvalue %123, %125[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %127 = llvm.mlir.constant(0 : index) : i64
    %128 = llvm.insertvalue %127, %126[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %129 = llvm.insertvalue %107, %128[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %130 = llvm.insertvalue %108, %129[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.insertvalue %108, %130[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.insertvalue %109, %131[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %133 = builtin.unrealized_conversion_cast %132 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<1x128xf32>
    %134 = llvm.mlir.constant(1 : index) : i64
    %135 = llvm.extractvalue %106[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.mul %134, %135 : i64
    %137 = llvm.extractvalue %106[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %138 = llvm.mul %136, %137 : i64
    %139 = llvm.mlir.zero : !llvm.ptr
    %140 = llvm.getelementptr %139[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %141 = llvm.ptrtoint %140 : !llvm.ptr to i64
    %142 = llvm.mul %138, %141 : i64
    %143 = llvm.extractvalue %106[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %144 = llvm.extractvalue %106[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %145 = llvm.getelementptr %143[%144] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %146 = llvm.extractvalue %132[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %147 = llvm.extractvalue %132[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %148 = llvm.getelementptr %146[%147] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%148, %145, %142) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %149 = llvm.mlir.constant(1 : index) : i64
    %150 = llvm.mlir.constant(64 : index) : i64
    %151 = llvm.mlir.constant(1 : index) : i64
    %152 = llvm.mlir.constant(64 : index) : i64
    %153 = llvm.mlir.zero : !llvm.ptr
    %154 = llvm.getelementptr %153[%152] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %155 = llvm.ptrtoint %154 : !llvm.ptr to i64
    %156 = llvm.call @malloc(%155) : (i64) -> !llvm.ptr
    %157 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %158 = llvm.insertvalue %156, %157[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %159 = llvm.insertvalue %156, %158[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %160 = llvm.mlir.constant(0 : index) : i64
    %161 = llvm.insertvalue %160, %159[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %162 = llvm.insertvalue %149, %161[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %163 = llvm.insertvalue %150, %162[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %164 = llvm.insertvalue %150, %163[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %165 = llvm.insertvalue %151, %164[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %166 = builtin.unrealized_conversion_cast %165 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<1x64xf32>
    %167 = llvm.mlir.constant(64 : index) : i64
    %168 = llvm.mlir.constant(128 : index) : i64
    %169 = llvm.mlir.constant(1 : index) : i64
    %170 = llvm.mlir.constant(8192 : index) : i64
    %171 = llvm.mlir.zero : !llvm.ptr
    %172 = llvm.getelementptr %171[%170] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %173 = llvm.ptrtoint %172 : !llvm.ptr to i64
    %174 = llvm.call @malloc(%173) : (i64) -> !llvm.ptr
    %175 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %176 = llvm.insertvalue %174, %175[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %177 = llvm.insertvalue %174, %176[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %178 = llvm.mlir.constant(0 : index) : i64
    %179 = llvm.insertvalue %178, %177[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %180 = llvm.insertvalue %167, %179[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %181 = llvm.insertvalue %168, %180[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %182 = llvm.insertvalue %168, %181[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %183 = llvm.insertvalue %169, %182[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %184 = builtin.unrealized_conversion_cast %183 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<64x128xf32>
    %185 = llvm.intr.stacksave : !llvm.ptr
    %186 = llvm.mlir.constant(2 : i64) : i64
    %187 = llvm.mlir.constant(1 : index) : i64
    %188 = llvm.alloca %187 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %21, %188 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %189 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %190 = llvm.insertvalue %186, %189[0] : !llvm.struct<(i64, ptr)> 
    %191 = llvm.insertvalue %188, %190[1] : !llvm.struct<(i64, ptr)> 
    %192 = llvm.mlir.constant(2 : i64) : i64
    %193 = llvm.mlir.constant(1 : index) : i64
    %194 = llvm.alloca %193 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %165, %194 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %195 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %196 = llvm.insertvalue %192, %195[0] : !llvm.struct<(i64, ptr)> 
    %197 = llvm.insertvalue %194, %196[1] : !llvm.struct<(i64, ptr)> 
    %198 = llvm.mlir.constant(1 : index) : i64
    %199 = llvm.alloca %198 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %191, %199 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %200 = llvm.alloca %198 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %197, %200 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %201 = llvm.mlir.zero : !llvm.ptr
    %202 = llvm.getelementptr %201[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %203 = llvm.ptrtoint %202 : !llvm.ptr to i64
    llvm.call @memrefCopy(%203, %199, %200) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %185 : !llvm.ptr
    %204 = llvm.mlir.constant(1 : index) : i64
    %205 = llvm.extractvalue %65[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %206 = llvm.mul %204, %205 : i64
    %207 = llvm.extractvalue %65[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %208 = llvm.mul %206, %207 : i64
    %209 = llvm.mlir.zero : !llvm.ptr
    %210 = llvm.getelementptr %209[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %211 = llvm.ptrtoint %210 : !llvm.ptr to i64
    %212 = llvm.mul %208, %211 : i64
    %213 = llvm.extractvalue %65[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %214 = llvm.extractvalue %65[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %215 = llvm.getelementptr %213[%214] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %216 = llvm.extractvalue %183[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %217 = llvm.extractvalue %183[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %218 = llvm.getelementptr %216[%217] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%218, %215, %212) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    ada300hw.set_gmm_cfg {acc_size = 64 : i32, col_size = 128 : i32, mode = #ada300hl.tensor_mode<inner>, row_size = 1 : i32}
    ada300hw.set_gmm_type {act_type = #ada300hl.dtype<fp16>, out_type = #ada300hl.dtype<fp16>, wht_type = #ada300hl.dtype<fp16>}
    ada300hw.set_gmm_iter {blk_cnt_a = 1 : i32, blk_cnt_w = 1 : i32}
    ada300hw.gmma_mm %133, %166, %184 : memref<1x128xf32>, memref<1x64xf32>, memref<64x128xf32>
    ada300hw.tcsync
    %219 = llvm.extractvalue %165[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%219) : (!llvm.ptr) -> ()
    %220 = llvm.extractvalue %183[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%220) : (!llvm.ptr) -> ()
    %221 = llvm.extractvalue %27[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %222 = llvm.extractvalue %27[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %223 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %224 = llvm.insertvalue %221, %223[0] : !llvm.struct<(ptr, ptr, i64)> 
    %225 = llvm.insertvalue %222, %224[1] : !llvm.struct<(ptr, ptr, i64)> 
    %226 = llvm.mlir.constant(0 : index) : i64
    %227 = llvm.insertvalue %226, %225[2] : !llvm.struct<(ptr, ptr, i64)> 
    %228 = llvm.extractvalue %27[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %229 = llvm.extractvalue %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %230 = llvm.extractvalue %27[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %231 = llvm.mul %230, %36 overflow<nsw> : i64
    %232 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %233 = llvm.extractvalue %227[0] : !llvm.struct<(ptr, ptr, i64)> 
    %234 = llvm.extractvalue %227[1] : !llvm.struct<(ptr, ptr, i64)> 
    %235 = llvm.insertvalue %233, %232[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %236 = llvm.insertvalue %234, %235[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %237 = llvm.insertvalue %228, %236[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %238 = llvm.mlir.constant(1 : index) : i64
    %239 = llvm.insertvalue %238, %237[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %240 = llvm.insertvalue %231, %239[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %241 = llvm.mlir.constant(128 : index) : i64
    %242 = llvm.insertvalue %241, %240[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %243 = llvm.insertvalue %230, %242[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %244 = llvm.mlir.constant(1 : index) : i64
    %245 = llvm.mlir.constant(128 : index) : i64
    %246 = llvm.mlir.constant(1 : index) : i64
    %247 = llvm.mlir.constant(128 : index) : i64
    %248 = llvm.mlir.zero : !llvm.ptr
    %249 = llvm.getelementptr %248[%247] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %250 = llvm.ptrtoint %249 : !llvm.ptr to i64
    %251 = llvm.mlir.constant(64 : index) : i64
    %252 = llvm.add %250, %251 : i64
    %253 = llvm.call @malloc(%252) : (i64) -> !llvm.ptr
    %254 = llvm.ptrtoint %253 : !llvm.ptr to i64
    %255 = llvm.mlir.constant(1 : index) : i64
    %256 = llvm.sub %251, %255 : i64
    %257 = llvm.add %254, %256 : i64
    %258 = llvm.urem %257, %251 : i64
    %259 = llvm.sub %257, %258 : i64
    %260 = llvm.inttoptr %259 : i64 to !llvm.ptr
    %261 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %262 = llvm.insertvalue %253, %261[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %263 = llvm.insertvalue %260, %262[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %264 = llvm.mlir.constant(0 : index) : i64
    %265 = llvm.insertvalue %264, %263[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %266 = llvm.insertvalue %244, %265[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %267 = llvm.insertvalue %245, %266[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %268 = llvm.insertvalue %245, %267[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %269 = llvm.insertvalue %246, %268[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb7(%39 : i64)
  ^bb7(%270: i64):  // 2 preds: ^bb6, ^bb11
    %271 = llvm.icmp "slt" %270, %37 : i64
    llvm.cond_br %271, ^bb8, ^bb12
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%39 : i64)
  ^bb9(%272: i64):  // 2 preds: ^bb8, ^bb10
    %273 = llvm.icmp "slt" %272, %36 : i64
    llvm.cond_br %273, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %274 = llvm.extractvalue %243[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %275 = llvm.extractvalue %243[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %276 = llvm.getelementptr %274[%275] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %277 = llvm.extractvalue %243[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %278 = llvm.mul %270, %277 overflow<nsw, nuw> : i64
    %279 = llvm.extractvalue %243[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %280 = llvm.mul %272, %279 overflow<nsw, nuw> : i64
    %281 = llvm.add %278, %280 overflow<nsw, nuw> : i64
    %282 = llvm.getelementptr inbounds|nuw %276[%281] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %283 = llvm.load %282 : !llvm.ptr -> f32
    %284 = llvm.extractvalue %132[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %285 = llvm.mlir.constant(128 : index) : i64
    %286 = llvm.mul %270, %285 overflow<nsw, nuw> : i64
    %287 = llvm.add %286, %272 overflow<nsw, nuw> : i64
    %288 = llvm.getelementptr inbounds|nuw %284[%287] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %289 = llvm.load %288 : !llvm.ptr -> f32
    %290 = llvm.fadd %283, %289 : f32
    %291 = llvm.extractvalue %269[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %292 = llvm.mlir.constant(128 : index) : i64
    %293 = llvm.mul %270, %292 overflow<nsw, nuw> : i64
    %294 = llvm.add %293, %272 overflow<nsw, nuw> : i64
    %295 = llvm.getelementptr inbounds|nuw %291[%294] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %290, %295 : f32, !llvm.ptr
    %296 = llvm.add %272, %37 : i64
    llvm.br ^bb9(%296 : i64)
  ^bb11:  // pred: ^bb9
    %297 = llvm.add %270, %37 : i64
    llvm.br ^bb7(%297 : i64)
  ^bb12:  // pred: ^bb7
    %298 = llvm.extractvalue %269[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %299 = llvm.mlir.constant(128 : index) : i64
    %300 = llvm.mul %39, %299 : i64
    %301 = llvm.add %300, %39 : i64
    %302 = llvm.getelementptr %298[%301] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %303 = llvm.load %302 {alignment = 4 : i64} : !llvm.ptr -> vector<128xf32>
    %304 = ada300hw.vfpwnl %303 {func = #ada300hl.nlfunc<exp>, masked = false, segments = #ada300hl.segments<seg16>} : vector<128xf32> -> vector<128xf32>
    %305 = llvm.extractvalue %269[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %306 = llvm.mlir.constant(128 : index) : i64
    %307 = llvm.mul %39, %306 : i64
    %308 = llvm.add %307, %39 : i64
    %309 = llvm.getelementptr %305[%308] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %304, %309 {alignment = 4 : i64} : vector<128xf32>, !llvm.ptr
    %310 = llvm.extractvalue %269[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %311 = llvm.mlir.constant(128 : index) : i64
    %312 = llvm.mul %39, %311 : i64
    %313 = llvm.add %312, %39 : i64
    %314 = llvm.getelementptr %310[%313] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %315 = llvm.load %314 {alignment = 4 : i64} : !llvm.ptr -> vector<128xf32>
    %316 = ada300hw.vfpwnl %315 {func = #ada300hl.nlfunc<sqrt>, masked = false, segments = #ada300hl.segments<seg16>} : vector<128xf32> -> vector<128xf32>
    %317 = llvm.extractvalue %269[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %318 = llvm.mlir.constant(128 : index) : i64
    %319 = llvm.mul %39, %318 : i64
    %320 = llvm.add %319, %39 : i64
    %321 = llvm.getelementptr %317[%320] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %316, %321 {alignment = 4 : i64} : vector<128xf32>, !llvm.ptr
    %322 = llvm.mlir.constant(128 : index) : i64
    %323 = llvm.mlir.constant(64 : index) : i64
    %324 = llvm.mlir.constant(1 : index) : i64
    %325 = llvm.mlir.constant(8192 : index) : i64
    %326 = llvm.mlir.zero : !llvm.ptr
    %327 = llvm.getelementptr %326[%325] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %328 = llvm.ptrtoint %327 : !llvm.ptr to i64
    %329 = llvm.mlir.constant(64 : index) : i64
    %330 = llvm.add %328, %329 : i64
    %331 = llvm.call @malloc(%330) : (i64) -> !llvm.ptr
    %332 = llvm.ptrtoint %331 : !llvm.ptr to i64
    %333 = llvm.mlir.constant(1 : index) : i64
    %334 = llvm.sub %329, %333 : i64
    %335 = llvm.add %332, %334 : i64
    %336 = llvm.urem %335, %329 : i64
    %337 = llvm.sub %335, %336 : i64
    %338 = llvm.inttoptr %337 : i64 to !llvm.ptr
    %339 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %340 = llvm.insertvalue %331, %339[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %341 = llvm.insertvalue %338, %340[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %342 = llvm.mlir.constant(0 : index) : i64
    %343 = llvm.insertvalue %342, %341[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %344 = llvm.insertvalue %322, %343[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %345 = llvm.insertvalue %323, %344[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %346 = llvm.insertvalue %323, %345[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %347 = llvm.insertvalue %324, %346[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb13(%39 : i64)
  ^bb13(%348: i64):  // 2 preds: ^bb12, ^bb17
    %349 = llvm.icmp "slt" %348, %36 : i64
    llvm.cond_br %349, ^bb14, ^bb18
  ^bb14:  // pred: ^bb13
    llvm.br ^bb15(%39 : i64)
  ^bb15(%350: i64):  // 2 preds: ^bb14, ^bb16
    %351 = llvm.icmp "slt" %350, %38 : i64
    llvm.cond_br %351, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    %352 = llvm.extractvalue %13[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %353 = llvm.extractvalue %13[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %354 = llvm.getelementptr %352[%353] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %355 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %356 = llvm.mul %350, %355 overflow<nsw, nuw> : i64
    %357 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %358 = llvm.mul %348, %357 overflow<nsw, nuw> : i64
    %359 = llvm.add %356, %358 overflow<nsw, nuw> : i64
    %360 = llvm.getelementptr inbounds|nuw %354[%359] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %361 = llvm.load %360 : !llvm.ptr -> f32
    %362 = llvm.extractvalue %347[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %363 = llvm.mlir.constant(64 : index) : i64
    %364 = llvm.mul %348, %363 overflow<nsw, nuw> : i64
    %365 = llvm.add %364, %350 overflow<nsw, nuw> : i64
    %366 = llvm.getelementptr inbounds|nuw %362[%365] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %361, %366 : f32, !llvm.ptr
    %367 = llvm.add %350, %37 : i64
    llvm.br ^bb15(%367 : i64)
  ^bb17:  // pred: ^bb15
    %368 = llvm.add %348, %37 : i64
    llvm.br ^bb13(%368 : i64)
  ^bb18:  // pred: ^bb13
    %369 = llvm.mlir.constant(1 : index) : i64
    %370 = llvm.mlir.constant(64 : index) : i64
    %371 = llvm.mlir.constant(1 : index) : i64
    %372 = llvm.mlir.constant(64 : index) : i64
    %373 = llvm.mlir.zero : !llvm.ptr
    %374 = llvm.getelementptr %373[%372] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %375 = llvm.ptrtoint %374 : !llvm.ptr to i64
    %376 = llvm.mlir.addressof @__constant_1x64xf32 : !llvm.ptr
    %377 = llvm.getelementptr %376[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<64 x f32>>
    %378 = llvm.mlir.constant(3735928559 : index) : i64
    %379 = llvm.inttoptr %378 : i64 to !llvm.ptr
    %380 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %381 = llvm.insertvalue %379, %380[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %382 = llvm.insertvalue %377, %381[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %383 = llvm.mlir.constant(0 : index) : i64
    %384 = llvm.insertvalue %383, %382[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %385 = llvm.insertvalue %369, %384[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %386 = llvm.insertvalue %370, %385[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %387 = llvm.insertvalue %370, %386[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %388 = llvm.insertvalue %371, %387[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %389 = llvm.mlir.constant(1 : index) : i64
    %390 = llvm.mlir.constant(64 : index) : i64
    %391 = llvm.mlir.constant(1 : index) : i64
    %392 = llvm.mlir.constant(64 : index) : i64
    %393 = llvm.mlir.zero : !llvm.ptr
    %394 = llvm.getelementptr %393[%392] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %395 = llvm.ptrtoint %394 : !llvm.ptr to i64
    %396 = llvm.mlir.constant(64 : index) : i64
    %397 = llvm.add %395, %396 : i64
    %398 = llvm.call @malloc(%397) : (i64) -> !llvm.ptr
    %399 = llvm.ptrtoint %398 : !llvm.ptr to i64
    %400 = llvm.mlir.constant(1 : index) : i64
    %401 = llvm.sub %396, %400 : i64
    %402 = llvm.add %399, %401 : i64
    %403 = llvm.urem %402, %396 : i64
    %404 = llvm.sub %402, %403 : i64
    %405 = llvm.inttoptr %404 : i64 to !llvm.ptr
    %406 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %407 = llvm.insertvalue %398, %406[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %408 = llvm.insertvalue %405, %407[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %409 = llvm.mlir.constant(0 : index) : i64
    %410 = llvm.insertvalue %409, %408[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %411 = llvm.insertvalue %389, %410[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %412 = llvm.insertvalue %390, %411[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %413 = llvm.insertvalue %390, %412[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %414 = llvm.insertvalue %391, %413[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %415 = builtin.unrealized_conversion_cast %414 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<1x64xf32>
    %416 = llvm.mlir.constant(1 : index) : i64
    %417 = llvm.extractvalue %388[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %418 = llvm.mul %416, %417 : i64
    %419 = llvm.extractvalue %388[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %420 = llvm.mul %418, %419 : i64
    %421 = llvm.mlir.zero : !llvm.ptr
    %422 = llvm.getelementptr %421[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %423 = llvm.ptrtoint %422 : !llvm.ptr to i64
    %424 = llvm.mul %420, %423 : i64
    %425 = llvm.extractvalue %388[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %426 = llvm.extractvalue %388[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %427 = llvm.getelementptr %425[%426] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %428 = llvm.extractvalue %414[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %429 = llvm.extractvalue %414[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %430 = llvm.getelementptr %428[%429] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%430, %427, %424) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %431 = llvm.mlir.constant(1 : index) : i64
    %432 = llvm.mlir.constant(128 : index) : i64
    %433 = llvm.mlir.constant(1 : index) : i64
    %434 = llvm.mlir.constant(128 : index) : i64
    %435 = llvm.mlir.zero : !llvm.ptr
    %436 = llvm.getelementptr %435[%434] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %437 = llvm.ptrtoint %436 : !llvm.ptr to i64
    %438 = llvm.call @malloc(%437) : (i64) -> !llvm.ptr
    %439 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %440 = llvm.insertvalue %438, %439[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %441 = llvm.insertvalue %438, %440[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %442 = llvm.mlir.constant(0 : index) : i64
    %443 = llvm.insertvalue %442, %441[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %444 = llvm.insertvalue %431, %443[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %445 = llvm.insertvalue %432, %444[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %446 = llvm.insertvalue %432, %445[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %447 = llvm.insertvalue %433, %446[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %448 = builtin.unrealized_conversion_cast %447 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<1x128xf32>
    %449 = llvm.mlir.constant(128 : index) : i64
    %450 = llvm.mlir.constant(64 : index) : i64
    %451 = llvm.mlir.constant(1 : index) : i64
    %452 = llvm.mlir.constant(8192 : index) : i64
    %453 = llvm.mlir.zero : !llvm.ptr
    %454 = llvm.getelementptr %453[%452] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %455 = llvm.ptrtoint %454 : !llvm.ptr to i64
    %456 = llvm.call @malloc(%455) : (i64) -> !llvm.ptr
    %457 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %458 = llvm.insertvalue %456, %457[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %459 = llvm.insertvalue %456, %458[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %460 = llvm.mlir.constant(0 : index) : i64
    %461 = llvm.insertvalue %460, %459[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %462 = llvm.insertvalue %449, %461[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %463 = llvm.insertvalue %450, %462[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %464 = llvm.insertvalue %450, %463[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %465 = llvm.insertvalue %451, %464[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %466 = builtin.unrealized_conversion_cast %465 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<128x64xf32>
    %467 = llvm.mlir.constant(1 : index) : i64
    %468 = llvm.extractvalue %269[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %469 = llvm.mul %467, %468 : i64
    %470 = llvm.extractvalue %269[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %471 = llvm.mul %469, %470 : i64
    %472 = llvm.mlir.zero : !llvm.ptr
    %473 = llvm.getelementptr %472[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %474 = llvm.ptrtoint %473 : !llvm.ptr to i64
    %475 = llvm.mul %471, %474 : i64
    %476 = llvm.extractvalue %269[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %477 = llvm.extractvalue %269[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %478 = llvm.getelementptr %476[%477] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %479 = llvm.extractvalue %447[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %480 = llvm.extractvalue %447[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %481 = llvm.getelementptr %479[%480] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%481, %478, %475) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %482 = llvm.mlir.constant(1 : index) : i64
    %483 = llvm.extractvalue %347[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %484 = llvm.mul %482, %483 : i64
    %485 = llvm.extractvalue %347[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %486 = llvm.mul %484, %485 : i64
    %487 = llvm.mlir.zero : !llvm.ptr
    %488 = llvm.getelementptr %487[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %489 = llvm.ptrtoint %488 : !llvm.ptr to i64
    %490 = llvm.mul %486, %489 : i64
    %491 = llvm.extractvalue %347[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %492 = llvm.extractvalue %347[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %493 = llvm.getelementptr %491[%492] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %494 = llvm.extractvalue %465[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %495 = llvm.extractvalue %465[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %496 = llvm.getelementptr %494[%495] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%496, %493, %490) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    ada300hw.set_gmm_cfg {acc_size = 128 : i32, col_size = 64 : i32, mode = #ada300hl.tensor_mode<inner>, row_size = 1 : i32}
    ada300hw.set_gmm_type {act_type = #ada300hl.dtype<fp16>, out_type = #ada300hl.dtype<fp16>, wht_type = #ada300hl.dtype<fp16>}
    ada300hw.set_gmm_iter {blk_cnt_a = 1 : i32, blk_cnt_w = 1 : i32}
    ada300hw.gmma_mm %415, %448, %466 : memref<1x64xf32>, memref<1x128xf32>, memref<128x64xf32>
    ada300hw.tcsync
    %497 = llvm.extractvalue %447[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%497) : (!llvm.ptr) -> ()
    %498 = llvm.extractvalue %465[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%498) : (!llvm.ptr) -> ()
    %499 = llvm.extractvalue %5[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %500 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %501 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %502 = llvm.insertvalue %499, %501[0] : !llvm.struct<(ptr, ptr, i64)> 
    %503 = llvm.insertvalue %500, %502[1] : !llvm.struct<(ptr, ptr, i64)> 
    %504 = llvm.mlir.constant(0 : index) : i64
    %505 = llvm.insertvalue %504, %503[2] : !llvm.struct<(ptr, ptr, i64)> 
    %506 = llvm.extractvalue %5[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %507 = llvm.extractvalue %5[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %508 = llvm.extractvalue %5[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %509 = llvm.mul %508, %38 overflow<nsw> : i64
    %510 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %511 = llvm.extractvalue %505[0] : !llvm.struct<(ptr, ptr, i64)> 
    %512 = llvm.extractvalue %505[1] : !llvm.struct<(ptr, ptr, i64)> 
    %513 = llvm.insertvalue %511, %510[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %514 = llvm.insertvalue %512, %513[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %515 = llvm.insertvalue %506, %514[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %516 = llvm.mlir.constant(1 : index) : i64
    %517 = llvm.insertvalue %516, %515[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %518 = llvm.insertvalue %509, %517[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %519 = llvm.mlir.constant(64 : index) : i64
    %520 = llvm.insertvalue %519, %518[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %521 = llvm.insertvalue %508, %520[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %522 = llvm.mlir.constant(1 : index) : i64
    %523 = llvm.mlir.constant(64 : index) : i64
    %524 = llvm.mlir.constant(1 : index) : i64
    %525 = llvm.mlir.constant(64 : index) : i64
    %526 = llvm.mlir.zero : !llvm.ptr
    %527 = llvm.getelementptr %526[%525] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %528 = llvm.ptrtoint %527 : !llvm.ptr to i64
    %529 = llvm.mlir.constant(64 : index) : i64
    %530 = llvm.add %528, %529 : i64
    %531 = llvm.call @malloc(%530) : (i64) -> !llvm.ptr
    %532 = llvm.ptrtoint %531 : !llvm.ptr to i64
    %533 = llvm.mlir.constant(1 : index) : i64
    %534 = llvm.sub %529, %533 : i64
    %535 = llvm.add %532, %534 : i64
    %536 = llvm.urem %535, %529 : i64
    %537 = llvm.sub %535, %536 : i64
    %538 = llvm.inttoptr %537 : i64 to !llvm.ptr
    %539 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %540 = llvm.insertvalue %531, %539[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %541 = llvm.insertvalue %538, %540[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %542 = llvm.mlir.constant(0 : index) : i64
    %543 = llvm.insertvalue %542, %541[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %544 = llvm.insertvalue %522, %543[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %545 = llvm.insertvalue %523, %544[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %546 = llvm.insertvalue %523, %545[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %547 = llvm.insertvalue %524, %546[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb19(%39 : i64)
  ^bb19(%548: i64):  // 2 preds: ^bb18, ^bb23
    %549 = llvm.icmp "slt" %548, %37 : i64
    llvm.cond_br %549, ^bb20, ^bb24
  ^bb20:  // pred: ^bb19
    llvm.br ^bb21(%39 : i64)
  ^bb21(%550: i64):  // 2 preds: ^bb20, ^bb22
    %551 = llvm.icmp "slt" %550, %38 : i64
    llvm.cond_br %551, ^bb22, ^bb23
  ^bb22:  // pred: ^bb21
    %552 = llvm.extractvalue %521[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %553 = llvm.extractvalue %521[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %554 = llvm.getelementptr %552[%553] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %555 = llvm.extractvalue %521[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %556 = llvm.mul %548, %555 overflow<nsw, nuw> : i64
    %557 = llvm.extractvalue %521[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %558 = llvm.mul %550, %557 overflow<nsw, nuw> : i64
    %559 = llvm.add %556, %558 overflow<nsw, nuw> : i64
    %560 = llvm.getelementptr inbounds|nuw %554[%559] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %561 = llvm.load %560 : !llvm.ptr -> f32
    %562 = llvm.extractvalue %414[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %563 = llvm.mlir.constant(64 : index) : i64
    %564 = llvm.mul %548, %563 overflow<nsw, nuw> : i64
    %565 = llvm.add %564, %550 overflow<nsw, nuw> : i64
    %566 = llvm.getelementptr inbounds|nuw %562[%565] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %567 = llvm.load %566 : !llvm.ptr -> f32
    %568 = llvm.fadd %561, %567 : f32
    %569 = llvm.extractvalue %547[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %570 = llvm.mlir.constant(64 : index) : i64
    %571 = llvm.mul %548, %570 overflow<nsw, nuw> : i64
    %572 = llvm.add %571, %550 overflow<nsw, nuw> : i64
    %573 = llvm.getelementptr inbounds|nuw %569[%572] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %568, %573 : f32, !llvm.ptr
    %574 = llvm.add %550, %37 : i64
    llvm.br ^bb21(%574 : i64)
  ^bb23:  // pred: ^bb21
    %575 = llvm.add %548, %37 : i64
    llvm.br ^bb19(%575 : i64)
  ^bb24:  // pred: ^bb19
    %576 = llvm.extractvalue %65[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%576) : (!llvm.ptr) -> ()
    %577 = llvm.extractvalue %132[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%577) : (!llvm.ptr) -> ()
    %578 = llvm.extractvalue %269[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%578) : (!llvm.ptr) -> ()
    %579 = llvm.extractvalue %347[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%579) : (!llvm.ptr) -> ()
    %580 = llvm.extractvalue %414[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%580) : (!llvm.ptr) -> ()
    llvm.return %547 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
}

