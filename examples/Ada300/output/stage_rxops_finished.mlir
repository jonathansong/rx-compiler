module {
  llvm.func @free(!llvm.ptr)
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @rxops_bridge_exp_f32(!llvm.ptr, !llvm.ptr, i64) -> i32
  llvm.func @rxops_bridge_sqrt_f32(!llvm.ptr, !llvm.ptr, i64) -> i32
  llvm.func @rxops_bridge_matmul_f32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
  llvm.mlir.global private constant @__constant_1x64xf32(dense<0.000000e+00> : tensor<1x64xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<1 x array<64 x f32>>
  llvm.mlir.global private constant @__constant_1x128xf32(dense<0.000000e+00> : tensor<1x128xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<1 x array<128 x f32>>
  llvm.func @subgraph0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: !llvm.ptr, %arg13: !llvm.ptr, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: !llvm.ptr, %arg20: !llvm.ptr, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: !llvm.ptr, %arg27: !llvm.ptr, %arg28: i64, %arg29: i64, %arg30: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> attributes {llvm.emit_c_interface} {
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
    %39 = llvm.mlir.constant(128 : i64) : i64
    %40 = llvm.mlir.constant(64 : i64) : i64
    %41 = llvm.mlir.constant(1 : i64) : i64
    %42 = llvm.mlir.constant(0 : index) : i64
    %43 = llvm.mlir.constant(64 : index) : i64
    %44 = llvm.mlir.constant(128 : index) : i64
    %45 = llvm.mlir.constant(1 : index) : i64
    %46 = llvm.mlir.constant(8192 : index) : i64
    %47 = llvm.mlir.zero : !llvm.ptr
    %48 = llvm.getelementptr %47[%46] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %49 = llvm.ptrtoint %48 : !llvm.ptr to i64
    %50 = llvm.mlir.constant(64 : index) : i64
    %51 = llvm.add %49, %50 : i64
    %52 = llvm.call @malloc(%51) : (i64) -> !llvm.ptr
    %53 = llvm.ptrtoint %52 : !llvm.ptr to i64
    %54 = llvm.mlir.constant(1 : index) : i64
    %55 = llvm.sub %50, %54 : i64
    %56 = llvm.add %53, %55 : i64
    %57 = llvm.urem %56, %50 : i64
    %58 = llvm.sub %56, %57 : i64
    %59 = llvm.inttoptr %58 : i64 to !llvm.ptr
    %60 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %61 = llvm.insertvalue %52, %60[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.insertvalue %59, %61[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.mlir.constant(0 : index) : i64
    %64 = llvm.insertvalue %63, %62[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.insertvalue %43, %64[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.insertvalue %44, %65[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.insertvalue %44, %66[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.insertvalue %45, %67[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%42 : i64)
  ^bb1(%69: i64):  // 2 preds: ^bb0, ^bb5
    %70 = llvm.icmp "slt" %69, %38 : i64
    llvm.cond_br %70, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%42 : i64)
  ^bb3(%71: i64):  // 2 preds: ^bb2, ^bb4
    %72 = llvm.icmp "slt" %71, %36 : i64
    llvm.cond_br %72, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %73 = llvm.extractvalue %35[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = llvm.extractvalue %35[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %75 = llvm.getelementptr %73[%74] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %76 = llvm.extractvalue %35[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %77 = llvm.mul %71, %76 overflow<nsw, nuw> : i64
    %78 = llvm.extractvalue %35[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %79 = llvm.mul %69, %78 overflow<nsw, nuw> : i64
    %80 = llvm.add %77, %79 overflow<nsw, nuw> : i64
    %81 = llvm.getelementptr inbounds|nuw %75[%80] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %82 = llvm.load %81 : !llvm.ptr -> f32
    %83 = llvm.extractvalue %68[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %84 = llvm.mlir.constant(128 : index) : i64
    %85 = llvm.mul %69, %84 overflow<nsw, nuw> : i64
    %86 = llvm.add %85, %71 overflow<nsw, nuw> : i64
    %87 = llvm.getelementptr inbounds|nuw %83[%86] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %82, %87 : f32, !llvm.ptr
    %88 = llvm.add %71, %37 : i64
    llvm.br ^bb3(%88 : i64)
  ^bb5:  // pred: ^bb3
    %89 = llvm.add %69, %37 : i64
    llvm.br ^bb1(%89 : i64)
  ^bb6:  // pred: ^bb1
    %90 = llvm.mlir.constant(1 : index) : i64
    %91 = llvm.mlir.constant(128 : index) : i64
    %92 = llvm.mlir.constant(1 : index) : i64
    %93 = llvm.mlir.constant(128 : index) : i64
    %94 = llvm.mlir.zero : !llvm.ptr
    %95 = llvm.getelementptr %94[%93] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %96 = llvm.ptrtoint %95 : !llvm.ptr to i64
    %97 = llvm.mlir.addressof @__constant_1x128xf32 : !llvm.ptr
    %98 = llvm.getelementptr %97[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<128 x f32>>
    %99 = llvm.mlir.constant(3735928559 : index) : i64
    %100 = llvm.inttoptr %99 : i64 to !llvm.ptr
    %101 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %102 = llvm.insertvalue %100, %101[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %103 = llvm.insertvalue %98, %102[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %104 = llvm.mlir.constant(0 : index) : i64
    %105 = llvm.insertvalue %104, %103[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.insertvalue %90, %105[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %107 = llvm.insertvalue %91, %106[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.insertvalue %91, %107[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %109 = llvm.insertvalue %92, %108[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %110 = llvm.mlir.constant(1 : index) : i64
    %111 = llvm.mlir.constant(128 : index) : i64
    %112 = llvm.mlir.constant(1 : index) : i64
    %113 = llvm.mlir.constant(128 : index) : i64
    %114 = llvm.mlir.zero : !llvm.ptr
    %115 = llvm.getelementptr %114[%113] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %116 = llvm.ptrtoint %115 : !llvm.ptr to i64
    %117 = llvm.mlir.constant(64 : index) : i64
    %118 = llvm.add %116, %117 : i64
    %119 = llvm.call @malloc(%118) : (i64) -> !llvm.ptr
    %120 = llvm.ptrtoint %119 : !llvm.ptr to i64
    %121 = llvm.mlir.constant(1 : index) : i64
    %122 = llvm.sub %117, %121 : i64
    %123 = llvm.add %120, %122 : i64
    %124 = llvm.urem %123, %117 : i64
    %125 = llvm.sub %123, %124 : i64
    %126 = llvm.inttoptr %125 : i64 to !llvm.ptr
    %127 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %128 = llvm.insertvalue %119, %127[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %129 = llvm.insertvalue %126, %128[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %130 = llvm.mlir.constant(0 : index) : i64
    %131 = llvm.insertvalue %130, %129[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.insertvalue %110, %131[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %133 = llvm.insertvalue %111, %132[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %134 = llvm.insertvalue %111, %133[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %135 = llvm.insertvalue %112, %134[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.mlir.constant(1 : index) : i64
    %137 = llvm.extractvalue %109[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %138 = llvm.mul %136, %137 : i64
    %139 = llvm.extractvalue %109[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %140 = llvm.mul %138, %139 : i64
    %141 = llvm.mlir.zero : !llvm.ptr
    %142 = llvm.getelementptr %141[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %143 = llvm.ptrtoint %142 : !llvm.ptr to i64
    %144 = llvm.mul %140, %143 : i64
    %145 = llvm.extractvalue %109[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %146 = llvm.extractvalue %109[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %147 = llvm.getelementptr %145[%146] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %148 = llvm.extractvalue %135[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %149 = llvm.extractvalue %135[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %150 = llvm.getelementptr %148[%149] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%150, %147, %144) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %151 = llvm.mlir.constant(1 : index) : i64
    %152 = llvm.mlir.constant(64 : index) : i64
    %153 = llvm.mlir.constant(1 : index) : i64
    %154 = llvm.mlir.constant(64 : index) : i64
    %155 = llvm.mlir.zero : !llvm.ptr
    %156 = llvm.getelementptr %155[%154] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %157 = llvm.ptrtoint %156 : !llvm.ptr to i64
    %158 = llvm.call @malloc(%157) : (i64) -> !llvm.ptr
    %159 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %160 = llvm.insertvalue %158, %159[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %161 = llvm.insertvalue %158, %160[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %162 = llvm.mlir.constant(0 : index) : i64
    %163 = llvm.insertvalue %162, %161[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %164 = llvm.insertvalue %151, %163[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %165 = llvm.insertvalue %152, %164[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %166 = llvm.insertvalue %152, %165[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %167 = llvm.insertvalue %153, %166[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %168 = llvm.mlir.constant(64 : index) : i64
    %169 = llvm.mlir.constant(128 : index) : i64
    %170 = llvm.mlir.constant(1 : index) : i64
    %171 = llvm.mlir.constant(8192 : index) : i64
    %172 = llvm.mlir.zero : !llvm.ptr
    %173 = llvm.getelementptr %172[%171] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %174 = llvm.ptrtoint %173 : !llvm.ptr to i64
    %175 = llvm.call @malloc(%174) : (i64) -> !llvm.ptr
    %176 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %177 = llvm.insertvalue %175, %176[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %178 = llvm.insertvalue %175, %177[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %179 = llvm.mlir.constant(0 : index) : i64
    %180 = llvm.insertvalue %179, %178[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %181 = llvm.insertvalue %168, %180[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %182 = llvm.insertvalue %169, %181[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %183 = llvm.insertvalue %169, %182[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %184 = llvm.insertvalue %170, %183[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
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
    llvm.store %167, %194 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
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
    %205 = llvm.extractvalue %68[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %206 = llvm.mul %204, %205 : i64
    %207 = llvm.extractvalue %68[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %208 = llvm.mul %206, %207 : i64
    %209 = llvm.mlir.zero : !llvm.ptr
    %210 = llvm.getelementptr %209[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %211 = llvm.ptrtoint %210 : !llvm.ptr to i64
    %212 = llvm.mul %208, %211 : i64
    %213 = llvm.extractvalue %68[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %214 = llvm.extractvalue %68[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %215 = llvm.getelementptr %213[%214] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %216 = llvm.extractvalue %184[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %217 = llvm.extractvalue %184[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %218 = llvm.getelementptr %216[%217] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%218, %215, %212) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %219 = llvm.extractvalue %135[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %220 = llvm.ptrtoint %219 : !llvm.ptr to i64
    %221 = llvm.inttoptr %220 : i64 to !llvm.ptr
    %222 = llvm.extractvalue %167[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %223 = llvm.ptrtoint %222 : !llvm.ptr to i64
    %224 = llvm.inttoptr %223 : i64 to !llvm.ptr
    %225 = llvm.extractvalue %184[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %226 = llvm.ptrtoint %225 : !llvm.ptr to i64
    %227 = llvm.inttoptr %226 : i64 to !llvm.ptr
    %228 = llvm.call @rxops_bridge_matmul_f32(%221, %224, %227, %41, %39, %40) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    %229 = llvm.extractvalue %167[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%229) : (!llvm.ptr) -> ()
    %230 = llvm.extractvalue %184[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%230) : (!llvm.ptr) -> ()
    %231 = llvm.extractvalue %27[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %232 = llvm.extractvalue %27[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %233 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %234 = llvm.insertvalue %231, %233[0] : !llvm.struct<(ptr, ptr, i64)> 
    %235 = llvm.insertvalue %232, %234[1] : !llvm.struct<(ptr, ptr, i64)> 
    %236 = llvm.mlir.constant(0 : index) : i64
    %237 = llvm.insertvalue %236, %235[2] : !llvm.struct<(ptr, ptr, i64)> 
    %238 = llvm.extractvalue %27[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %239 = llvm.extractvalue %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %240 = llvm.extractvalue %27[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %241 = llvm.mul %240, %36 overflow<nsw> : i64
    %242 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %243 = llvm.extractvalue %237[0] : !llvm.struct<(ptr, ptr, i64)> 
    %244 = llvm.extractvalue %237[1] : !llvm.struct<(ptr, ptr, i64)> 
    %245 = llvm.insertvalue %243, %242[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %246 = llvm.insertvalue %244, %245[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %247 = llvm.insertvalue %238, %246[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %248 = llvm.mlir.constant(1 : index) : i64
    %249 = llvm.insertvalue %248, %247[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %250 = llvm.insertvalue %241, %249[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %251 = llvm.mlir.constant(128 : index) : i64
    %252 = llvm.insertvalue %251, %250[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %253 = llvm.insertvalue %240, %252[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %254 = llvm.mlir.constant(1 : index) : i64
    %255 = llvm.mlir.constant(128 : index) : i64
    %256 = llvm.mlir.constant(1 : index) : i64
    %257 = llvm.mlir.constant(128 : index) : i64
    %258 = llvm.mlir.zero : !llvm.ptr
    %259 = llvm.getelementptr %258[%257] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %260 = llvm.ptrtoint %259 : !llvm.ptr to i64
    %261 = llvm.mlir.constant(64 : index) : i64
    %262 = llvm.add %260, %261 : i64
    %263 = llvm.call @malloc(%262) : (i64) -> !llvm.ptr
    %264 = llvm.ptrtoint %263 : !llvm.ptr to i64
    %265 = llvm.mlir.constant(1 : index) : i64
    %266 = llvm.sub %261, %265 : i64
    %267 = llvm.add %264, %266 : i64
    %268 = llvm.urem %267, %261 : i64
    %269 = llvm.sub %267, %268 : i64
    %270 = llvm.inttoptr %269 : i64 to !llvm.ptr
    %271 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %272 = llvm.insertvalue %263, %271[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %273 = llvm.insertvalue %270, %272[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %274 = llvm.mlir.constant(0 : index) : i64
    %275 = llvm.insertvalue %274, %273[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %276 = llvm.insertvalue %254, %275[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %277 = llvm.insertvalue %255, %276[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %278 = llvm.insertvalue %255, %277[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %279 = llvm.insertvalue %256, %278[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb7(%42 : i64)
  ^bb7(%280: i64):  // 2 preds: ^bb6, ^bb11
    %281 = llvm.icmp "slt" %280, %37 : i64
    llvm.cond_br %281, ^bb8, ^bb12
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%42 : i64)
  ^bb9(%282: i64):  // 2 preds: ^bb8, ^bb10
    %283 = llvm.icmp "slt" %282, %36 : i64
    llvm.cond_br %283, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %284 = llvm.extractvalue %253[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %285 = llvm.extractvalue %253[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %286 = llvm.getelementptr %284[%285] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %287 = llvm.extractvalue %253[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %288 = llvm.mul %280, %287 overflow<nsw, nuw> : i64
    %289 = llvm.extractvalue %253[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %290 = llvm.mul %282, %289 overflow<nsw, nuw> : i64
    %291 = llvm.add %288, %290 overflow<nsw, nuw> : i64
    %292 = llvm.getelementptr inbounds|nuw %286[%291] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %293 = llvm.load %292 : !llvm.ptr -> f32
    %294 = llvm.extractvalue %135[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %295 = llvm.mlir.constant(128 : index) : i64
    %296 = llvm.mul %280, %295 overflow<nsw, nuw> : i64
    %297 = llvm.add %296, %282 overflow<nsw, nuw> : i64
    %298 = llvm.getelementptr inbounds|nuw %294[%297] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %299 = llvm.load %298 : !llvm.ptr -> f32
    %300 = llvm.fadd %293, %299 : f32
    %301 = llvm.extractvalue %279[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %302 = llvm.mlir.constant(128 : index) : i64
    %303 = llvm.mul %280, %302 overflow<nsw, nuw> : i64
    %304 = llvm.add %303, %282 overflow<nsw, nuw> : i64
    %305 = llvm.getelementptr inbounds|nuw %301[%304] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %300, %305 : f32, !llvm.ptr
    %306 = llvm.add %282, %37 : i64
    llvm.br ^bb9(%306 : i64)
  ^bb11:  // pred: ^bb9
    %307 = llvm.add %280, %37 : i64
    llvm.br ^bb7(%307 : i64)
  ^bb12:  // pred: ^bb7
    %308 = llvm.extractvalue %279[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %309 = llvm.mlir.constant(128 : index) : i64
    %310 = llvm.mul %42, %309 : i64
    %311 = llvm.add %310, %42 : i64
    %312 = llvm.getelementptr %308[%311] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %313 = llvm.load %312 {alignment = 4 : i64} : !llvm.ptr -> vector<128xf32>
    %314 = llvm.mlir.constant(128 : index) : i64
    %315 = llvm.mlir.constant(1 : index) : i64
    %316 = llvm.alloca %314 x f32 : (i64) -> !llvm.ptr
    %317 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %318 = llvm.insertvalue %316, %317[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %319 = llvm.insertvalue %316, %318[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %320 = llvm.mlir.constant(0 : index) : i64
    %321 = llvm.insertvalue %320, %319[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %322 = llvm.insertvalue %314, %321[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %323 = llvm.insertvalue %315, %322[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %324 = llvm.mlir.constant(128 : index) : i64
    %325 = llvm.mlir.constant(1 : index) : i64
    %326 = llvm.alloca %324 x f32 : (i64) -> !llvm.ptr
    %327 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %328 = llvm.insertvalue %326, %327[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %329 = llvm.insertvalue %326, %328[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %330 = llvm.mlir.constant(0 : index) : i64
    %331 = llvm.insertvalue %330, %329[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %332 = llvm.insertvalue %324, %331[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %333 = llvm.insertvalue %325, %332[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %334 = llvm.extractvalue %323[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %335 = llvm.getelementptr %334[%42] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %313, %335 {alignment = 4 : i64} : vector<128xf32>, !llvm.ptr
    %336 = llvm.extractvalue %323[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %337 = llvm.ptrtoint %336 : !llvm.ptr to i64
    %338 = llvm.inttoptr %337 : i64 to !llvm.ptr
    %339 = llvm.extractvalue %333[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %340 = llvm.ptrtoint %339 : !llvm.ptr to i64
    %341 = llvm.inttoptr %340 : i64 to !llvm.ptr
    %342 = llvm.call @rxops_bridge_exp_f32(%341, %338, %39) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %343 = llvm.extractvalue %333[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %344 = llvm.getelementptr %343[%42] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %345 = llvm.load %344 {alignment = 4 : i64} : !llvm.ptr -> vector<128xf32>
    %346 = llvm.extractvalue %279[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %347 = llvm.mlir.constant(128 : index) : i64
    %348 = llvm.mul %42, %347 : i64
    %349 = llvm.add %348, %42 : i64
    %350 = llvm.getelementptr %346[%349] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %345, %350 {alignment = 4 : i64} : vector<128xf32>, !llvm.ptr
    %351 = llvm.extractvalue %279[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %352 = llvm.mlir.constant(128 : index) : i64
    %353 = llvm.mul %42, %352 : i64
    %354 = llvm.add %353, %42 : i64
    %355 = llvm.getelementptr %351[%354] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %356 = llvm.load %355 {alignment = 4 : i64} : !llvm.ptr -> vector<128xf32>
    %357 = llvm.mlir.constant(128 : index) : i64
    %358 = llvm.mlir.constant(1 : index) : i64
    %359 = llvm.alloca %357 x f32 : (i64) -> !llvm.ptr
    %360 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %361 = llvm.insertvalue %359, %360[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %362 = llvm.insertvalue %359, %361[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %363 = llvm.mlir.constant(0 : index) : i64
    %364 = llvm.insertvalue %363, %362[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %365 = llvm.insertvalue %357, %364[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %366 = llvm.insertvalue %358, %365[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %367 = llvm.mlir.constant(128 : index) : i64
    %368 = llvm.mlir.constant(1 : index) : i64
    %369 = llvm.alloca %367 x f32 : (i64) -> !llvm.ptr
    %370 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %371 = llvm.insertvalue %369, %370[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %372 = llvm.insertvalue %369, %371[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %373 = llvm.mlir.constant(0 : index) : i64
    %374 = llvm.insertvalue %373, %372[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %375 = llvm.insertvalue %367, %374[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %376 = llvm.insertvalue %368, %375[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %377 = llvm.extractvalue %366[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %378 = llvm.getelementptr %377[%42] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %356, %378 {alignment = 4 : i64} : vector<128xf32>, !llvm.ptr
    %379 = llvm.extractvalue %366[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %380 = llvm.ptrtoint %379 : !llvm.ptr to i64
    %381 = llvm.inttoptr %380 : i64 to !llvm.ptr
    %382 = llvm.extractvalue %376[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %383 = llvm.ptrtoint %382 : !llvm.ptr to i64
    %384 = llvm.inttoptr %383 : i64 to !llvm.ptr
    %385 = llvm.call @rxops_bridge_sqrt_f32(%384, %381, %39) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %386 = llvm.extractvalue %376[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %387 = llvm.getelementptr %386[%42] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %388 = llvm.load %387 {alignment = 4 : i64} : !llvm.ptr -> vector<128xf32>
    %389 = llvm.extractvalue %279[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %390 = llvm.mlir.constant(128 : index) : i64
    %391 = llvm.mul %42, %390 : i64
    %392 = llvm.add %391, %42 : i64
    %393 = llvm.getelementptr %389[%392] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %388, %393 {alignment = 4 : i64} : vector<128xf32>, !llvm.ptr
    %394 = llvm.mlir.constant(128 : index) : i64
    %395 = llvm.mlir.constant(64 : index) : i64
    %396 = llvm.mlir.constant(1 : index) : i64
    %397 = llvm.mlir.constant(8192 : index) : i64
    %398 = llvm.mlir.zero : !llvm.ptr
    %399 = llvm.getelementptr %398[%397] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %400 = llvm.ptrtoint %399 : !llvm.ptr to i64
    %401 = llvm.mlir.constant(64 : index) : i64
    %402 = llvm.add %400, %401 : i64
    %403 = llvm.call @malloc(%402) : (i64) -> !llvm.ptr
    %404 = llvm.ptrtoint %403 : !llvm.ptr to i64
    %405 = llvm.mlir.constant(1 : index) : i64
    %406 = llvm.sub %401, %405 : i64
    %407 = llvm.add %404, %406 : i64
    %408 = llvm.urem %407, %401 : i64
    %409 = llvm.sub %407, %408 : i64
    %410 = llvm.inttoptr %409 : i64 to !llvm.ptr
    %411 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %412 = llvm.insertvalue %403, %411[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %413 = llvm.insertvalue %410, %412[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %414 = llvm.mlir.constant(0 : index) : i64
    %415 = llvm.insertvalue %414, %413[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %416 = llvm.insertvalue %394, %415[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %417 = llvm.insertvalue %395, %416[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %418 = llvm.insertvalue %395, %417[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %419 = llvm.insertvalue %396, %418[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb13(%42 : i64)
  ^bb13(%420: i64):  // 2 preds: ^bb12, ^bb17
    %421 = llvm.icmp "slt" %420, %36 : i64
    llvm.cond_br %421, ^bb14, ^bb18
  ^bb14:  // pred: ^bb13
    llvm.br ^bb15(%42 : i64)
  ^bb15(%422: i64):  // 2 preds: ^bb14, ^bb16
    %423 = llvm.icmp "slt" %422, %38 : i64
    llvm.cond_br %423, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    %424 = llvm.extractvalue %13[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %425 = llvm.extractvalue %13[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %426 = llvm.getelementptr %424[%425] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %427 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %428 = llvm.mul %422, %427 overflow<nsw, nuw> : i64
    %429 = llvm.extractvalue %13[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %430 = llvm.mul %420, %429 overflow<nsw, nuw> : i64
    %431 = llvm.add %428, %430 overflow<nsw, nuw> : i64
    %432 = llvm.getelementptr inbounds|nuw %426[%431] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %433 = llvm.load %432 : !llvm.ptr -> f32
    %434 = llvm.extractvalue %419[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %435 = llvm.mlir.constant(64 : index) : i64
    %436 = llvm.mul %420, %435 overflow<nsw, nuw> : i64
    %437 = llvm.add %436, %422 overflow<nsw, nuw> : i64
    %438 = llvm.getelementptr inbounds|nuw %434[%437] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %433, %438 : f32, !llvm.ptr
    %439 = llvm.add %422, %37 : i64
    llvm.br ^bb15(%439 : i64)
  ^bb17:  // pred: ^bb15
    %440 = llvm.add %420, %37 : i64
    llvm.br ^bb13(%440 : i64)
  ^bb18:  // pred: ^bb13
    %441 = llvm.mlir.constant(1 : index) : i64
    %442 = llvm.mlir.constant(64 : index) : i64
    %443 = llvm.mlir.constant(1 : index) : i64
    %444 = llvm.mlir.constant(64 : index) : i64
    %445 = llvm.mlir.zero : !llvm.ptr
    %446 = llvm.getelementptr %445[%444] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %447 = llvm.ptrtoint %446 : !llvm.ptr to i64
    %448 = llvm.mlir.addressof @__constant_1x64xf32 : !llvm.ptr
    %449 = llvm.getelementptr %448[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<64 x f32>>
    %450 = llvm.mlir.constant(3735928559 : index) : i64
    %451 = llvm.inttoptr %450 : i64 to !llvm.ptr
    %452 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %453 = llvm.insertvalue %451, %452[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %454 = llvm.insertvalue %449, %453[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %455 = llvm.mlir.constant(0 : index) : i64
    %456 = llvm.insertvalue %455, %454[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %457 = llvm.insertvalue %441, %456[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %458 = llvm.insertvalue %442, %457[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %459 = llvm.insertvalue %442, %458[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %460 = llvm.insertvalue %443, %459[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %461 = llvm.mlir.constant(1 : index) : i64
    %462 = llvm.mlir.constant(64 : index) : i64
    %463 = llvm.mlir.constant(1 : index) : i64
    %464 = llvm.mlir.constant(64 : index) : i64
    %465 = llvm.mlir.zero : !llvm.ptr
    %466 = llvm.getelementptr %465[%464] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %467 = llvm.ptrtoint %466 : !llvm.ptr to i64
    %468 = llvm.mlir.constant(64 : index) : i64
    %469 = llvm.add %467, %468 : i64
    %470 = llvm.call @malloc(%469) : (i64) -> !llvm.ptr
    %471 = llvm.ptrtoint %470 : !llvm.ptr to i64
    %472 = llvm.mlir.constant(1 : index) : i64
    %473 = llvm.sub %468, %472 : i64
    %474 = llvm.add %471, %473 : i64
    %475 = llvm.urem %474, %468 : i64
    %476 = llvm.sub %474, %475 : i64
    %477 = llvm.inttoptr %476 : i64 to !llvm.ptr
    %478 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %479 = llvm.insertvalue %470, %478[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %480 = llvm.insertvalue %477, %479[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %481 = llvm.mlir.constant(0 : index) : i64
    %482 = llvm.insertvalue %481, %480[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %483 = llvm.insertvalue %461, %482[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %484 = llvm.insertvalue %462, %483[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %485 = llvm.insertvalue %462, %484[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %486 = llvm.insertvalue %463, %485[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %487 = llvm.mlir.constant(1 : index) : i64
    %488 = llvm.extractvalue %460[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %489 = llvm.mul %487, %488 : i64
    %490 = llvm.extractvalue %460[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %491 = llvm.mul %489, %490 : i64
    %492 = llvm.mlir.zero : !llvm.ptr
    %493 = llvm.getelementptr %492[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %494 = llvm.ptrtoint %493 : !llvm.ptr to i64
    %495 = llvm.mul %491, %494 : i64
    %496 = llvm.extractvalue %460[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %497 = llvm.extractvalue %460[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %498 = llvm.getelementptr %496[%497] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %499 = llvm.extractvalue %486[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %500 = llvm.extractvalue %486[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %501 = llvm.getelementptr %499[%500] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%501, %498, %495) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %502 = llvm.mlir.constant(1 : index) : i64
    %503 = llvm.mlir.constant(128 : index) : i64
    %504 = llvm.mlir.constant(1 : index) : i64
    %505 = llvm.mlir.constant(128 : index) : i64
    %506 = llvm.mlir.zero : !llvm.ptr
    %507 = llvm.getelementptr %506[%505] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %508 = llvm.ptrtoint %507 : !llvm.ptr to i64
    %509 = llvm.call @malloc(%508) : (i64) -> !llvm.ptr
    %510 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %511 = llvm.insertvalue %509, %510[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %512 = llvm.insertvalue %509, %511[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %513 = llvm.mlir.constant(0 : index) : i64
    %514 = llvm.insertvalue %513, %512[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %515 = llvm.insertvalue %502, %514[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %516 = llvm.insertvalue %503, %515[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %517 = llvm.insertvalue %503, %516[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %518 = llvm.insertvalue %504, %517[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %519 = llvm.mlir.constant(128 : index) : i64
    %520 = llvm.mlir.constant(64 : index) : i64
    %521 = llvm.mlir.constant(1 : index) : i64
    %522 = llvm.mlir.constant(8192 : index) : i64
    %523 = llvm.mlir.zero : !llvm.ptr
    %524 = llvm.getelementptr %523[%522] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %525 = llvm.ptrtoint %524 : !llvm.ptr to i64
    %526 = llvm.call @malloc(%525) : (i64) -> !llvm.ptr
    %527 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %528 = llvm.insertvalue %526, %527[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %529 = llvm.insertvalue %526, %528[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %530 = llvm.mlir.constant(0 : index) : i64
    %531 = llvm.insertvalue %530, %529[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %532 = llvm.insertvalue %519, %531[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %533 = llvm.insertvalue %520, %532[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %534 = llvm.insertvalue %520, %533[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %535 = llvm.insertvalue %521, %534[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %536 = llvm.mlir.constant(1 : index) : i64
    %537 = llvm.extractvalue %279[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %538 = llvm.mul %536, %537 : i64
    %539 = llvm.extractvalue %279[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %540 = llvm.mul %538, %539 : i64
    %541 = llvm.mlir.zero : !llvm.ptr
    %542 = llvm.getelementptr %541[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %543 = llvm.ptrtoint %542 : !llvm.ptr to i64
    %544 = llvm.mul %540, %543 : i64
    %545 = llvm.extractvalue %279[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %546 = llvm.extractvalue %279[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %547 = llvm.getelementptr %545[%546] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %548 = llvm.extractvalue %518[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %549 = llvm.extractvalue %518[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %550 = llvm.getelementptr %548[%549] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%550, %547, %544) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %551 = llvm.mlir.constant(1 : index) : i64
    %552 = llvm.extractvalue %419[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %553 = llvm.mul %551, %552 : i64
    %554 = llvm.extractvalue %419[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %555 = llvm.mul %553, %554 : i64
    %556 = llvm.mlir.zero : !llvm.ptr
    %557 = llvm.getelementptr %556[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %558 = llvm.ptrtoint %557 : !llvm.ptr to i64
    %559 = llvm.mul %555, %558 : i64
    %560 = llvm.extractvalue %419[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %561 = llvm.extractvalue %419[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %562 = llvm.getelementptr %560[%561] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %563 = llvm.extractvalue %535[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %564 = llvm.extractvalue %535[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %565 = llvm.getelementptr %563[%564] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%565, %562, %559) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %566 = llvm.extractvalue %486[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %567 = llvm.ptrtoint %566 : !llvm.ptr to i64
    %568 = llvm.inttoptr %567 : i64 to !llvm.ptr
    %569 = llvm.extractvalue %518[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %570 = llvm.ptrtoint %569 : !llvm.ptr to i64
    %571 = llvm.inttoptr %570 : i64 to !llvm.ptr
    %572 = llvm.extractvalue %535[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %573 = llvm.ptrtoint %572 : !llvm.ptr to i64
    %574 = llvm.inttoptr %573 : i64 to !llvm.ptr
    %575 = llvm.call @rxops_bridge_matmul_f32(%568, %571, %574, %41, %40, %39) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    %576 = llvm.extractvalue %518[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%576) : (!llvm.ptr) -> ()
    %577 = llvm.extractvalue %535[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%577) : (!llvm.ptr) -> ()
    %578 = llvm.extractvalue %5[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %579 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %580 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %581 = llvm.insertvalue %578, %580[0] : !llvm.struct<(ptr, ptr, i64)> 
    %582 = llvm.insertvalue %579, %581[1] : !llvm.struct<(ptr, ptr, i64)> 
    %583 = llvm.mlir.constant(0 : index) : i64
    %584 = llvm.insertvalue %583, %582[2] : !llvm.struct<(ptr, ptr, i64)> 
    %585 = llvm.extractvalue %5[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %586 = llvm.extractvalue %5[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %587 = llvm.extractvalue %5[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %588 = llvm.mul %587, %38 overflow<nsw> : i64
    %589 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %590 = llvm.extractvalue %584[0] : !llvm.struct<(ptr, ptr, i64)> 
    %591 = llvm.extractvalue %584[1] : !llvm.struct<(ptr, ptr, i64)> 
    %592 = llvm.insertvalue %590, %589[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %593 = llvm.insertvalue %591, %592[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %594 = llvm.insertvalue %585, %593[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %595 = llvm.mlir.constant(1 : index) : i64
    %596 = llvm.insertvalue %595, %594[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %597 = llvm.insertvalue %588, %596[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %598 = llvm.mlir.constant(64 : index) : i64
    %599 = llvm.insertvalue %598, %597[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %600 = llvm.insertvalue %587, %599[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %601 = llvm.mlir.constant(1 : index) : i64
    %602 = llvm.mlir.constant(64 : index) : i64
    %603 = llvm.mlir.constant(1 : index) : i64
    %604 = llvm.mlir.constant(64 : index) : i64
    %605 = llvm.mlir.zero : !llvm.ptr
    %606 = llvm.getelementptr %605[%604] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %607 = llvm.ptrtoint %606 : !llvm.ptr to i64
    %608 = llvm.mlir.constant(64 : index) : i64
    %609 = llvm.add %607, %608 : i64
    %610 = llvm.call @malloc(%609) : (i64) -> !llvm.ptr
    %611 = llvm.ptrtoint %610 : !llvm.ptr to i64
    %612 = llvm.mlir.constant(1 : index) : i64
    %613 = llvm.sub %608, %612 : i64
    %614 = llvm.add %611, %613 : i64
    %615 = llvm.urem %614, %608 : i64
    %616 = llvm.sub %614, %615 : i64
    %617 = llvm.inttoptr %616 : i64 to !llvm.ptr
    %618 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %619 = llvm.insertvalue %610, %618[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %620 = llvm.insertvalue %617, %619[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %621 = llvm.mlir.constant(0 : index) : i64
    %622 = llvm.insertvalue %621, %620[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %623 = llvm.insertvalue %601, %622[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %624 = llvm.insertvalue %602, %623[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %625 = llvm.insertvalue %602, %624[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %626 = llvm.insertvalue %603, %625[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb19(%42 : i64)
  ^bb19(%627: i64):  // 2 preds: ^bb18, ^bb23
    %628 = llvm.icmp "slt" %627, %37 : i64
    llvm.cond_br %628, ^bb20, ^bb24
  ^bb20:  // pred: ^bb19
    llvm.br ^bb21(%42 : i64)
  ^bb21(%629: i64):  // 2 preds: ^bb20, ^bb22
    %630 = llvm.icmp "slt" %629, %38 : i64
    llvm.cond_br %630, ^bb22, ^bb23
  ^bb22:  // pred: ^bb21
    %631 = llvm.extractvalue %600[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %632 = llvm.extractvalue %600[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %633 = llvm.getelementptr %631[%632] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %634 = llvm.extractvalue %600[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %635 = llvm.mul %627, %634 overflow<nsw, nuw> : i64
    %636 = llvm.extractvalue %600[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %637 = llvm.mul %629, %636 overflow<nsw, nuw> : i64
    %638 = llvm.add %635, %637 overflow<nsw, nuw> : i64
    %639 = llvm.getelementptr inbounds|nuw %633[%638] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %640 = llvm.load %639 : !llvm.ptr -> f32
    %641 = llvm.extractvalue %486[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %642 = llvm.mlir.constant(64 : index) : i64
    %643 = llvm.mul %627, %642 overflow<nsw, nuw> : i64
    %644 = llvm.add %643, %629 overflow<nsw, nuw> : i64
    %645 = llvm.getelementptr inbounds|nuw %641[%644] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %646 = llvm.load %645 : !llvm.ptr -> f32
    %647 = llvm.fadd %640, %646 : f32
    %648 = llvm.extractvalue %626[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %649 = llvm.mlir.constant(64 : index) : i64
    %650 = llvm.mul %627, %649 overflow<nsw, nuw> : i64
    %651 = llvm.add %650, %629 overflow<nsw, nuw> : i64
    %652 = llvm.getelementptr inbounds|nuw %648[%651] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %647, %652 : f32, !llvm.ptr
    %653 = llvm.add %629, %37 : i64
    llvm.br ^bb21(%653 : i64)
  ^bb23:  // pred: ^bb21
    %654 = llvm.add %627, %37 : i64
    llvm.br ^bb19(%654 : i64)
  ^bb24:  // pred: ^bb19
    %655 = llvm.extractvalue %68[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%655) : (!llvm.ptr) -> ()
    %656 = llvm.extractvalue %135[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%656) : (!llvm.ptr) -> ()
    %657 = llvm.extractvalue %279[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%657) : (!llvm.ptr) -> ()
    %658 = llvm.extractvalue %419[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%658) : (!llvm.ptr) -> ()
    %659 = llvm.extractvalue %486[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%659) : (!llvm.ptr) -> ()
    llvm.return %626 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
  llvm.func @_mlir_ciface_subgraph0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.load %arg3 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.extractvalue %14[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.extractvalue %14[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.extractvalue %14[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.extractvalue %14[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.extractvalue %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.extractvalue %14[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.load %arg4 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %23 = llvm.extractvalue %22[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.extractvalue %22[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.extractvalue %22[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.extractvalue %22[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.extractvalue %22[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.extractvalue %22[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.load %arg5 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %31 = llvm.extractvalue %30[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.extractvalue %30[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %33 = llvm.extractvalue %30[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.extractvalue %30[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.extractvalue %30[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = llvm.call @subgraph0(%1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %15, %16, %17, %18, %19, %20, %21, %23, %24, %25, %26, %27, %28, %29, %31, %32, %33, %34, %35) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.store %36, %arg0 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    llvm.return
  }
}

