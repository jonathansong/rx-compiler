module attributes {llvm.data_layout = "", module.chip = "ALL", module.platform = "ONNX", module.state = "TOSA_F32", module.top_run_mode = "STATIC", module.weight_file = "/workspace/rx-mlir-main/examples/Ada300-hetero/output/subgraph0_top_weight.npz"} {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @rxops_bridge_exp_f32(!llvm.ptr, !llvm.ptr, i64) -> i32
  llvm.func @rxops_bridge_sqrt_f32(!llvm.ptr, !llvm.ptr, i64) -> i32
  llvm.func @rxops_bridge_ada300_matmul_f32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
  llvm.func @rxops_bridge_ada300_add_f32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
  llvm.func @subgraph0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: !llvm.ptr, %arg22: !llvm.ptr, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: !llvm.ptr, %arg29: !llvm.ptr, %arg30: i64, %arg31: i64, %arg32: i64, %arg33: i64, %arg34: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.insertvalue %arg8, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %arg9, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg10, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg12, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg11, %13[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg13, %14[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %arg14, %16[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.insertvalue %arg15, %17[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.insertvalue %arg16, %18[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %arg17, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %arg19, %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %arg18, %21[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %arg20, %22[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %25 = llvm.insertvalue %arg21, %24[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.insertvalue %arg22, %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.insertvalue %arg23, %26[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.insertvalue %arg24, %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.insertvalue %arg26, %28[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.insertvalue %arg25, %29[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.insertvalue %arg27, %30[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %33 = llvm.insertvalue %arg28, %32[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.insertvalue %arg29, %33[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.insertvalue %arg30, %34[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.insertvalue %arg31, %35[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %arg33, %36[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %arg32, %37[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.insertvalue %arg34, %38[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.mlir.constant(1 : i64) : i64
    %41 = llvm.mlir.constant(128 : i64) : i64
    %42 = llvm.mlir.constant(64 : i64) : i64
    %43 = llvm.mlir.constant(1 : index) : i64
    %44 = llvm.mlir.constant(64 : index) : i64
    %45 = llvm.mlir.constant(1 : index) : i64
    %46 = llvm.mlir.constant(64 : index) : i64
    %47 = llvm.mlir.null : !llvm.ptr
    %48 = llvm.getelementptr %47[64] : (!llvm.ptr) -> !llvm.ptr, f32
    %49 = llvm.ptrtoint %48 : !llvm.ptr to i64
    %50 = llvm.call @malloc(%49) : (i64) -> !llvm.ptr
    %51 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %52 = llvm.insertvalue %50, %51[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.insertvalue %50, %52[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.mlir.constant(0 : index) : i64
    %55 = llvm.insertvalue %54, %53[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.insertvalue %43, %55[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.insertvalue %44, %56[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %58 = llvm.insertvalue %44, %57[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.insertvalue %45, %58[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %60 = llvm.intr.stacksave : !llvm.ptr
    %61 = llvm.mlir.constant(2 : i64) : i64
    %62 = llvm.mlir.constant(1 : index) : i64
    %63 = llvm.alloca %62 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %39, %63 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %64 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %65 = llvm.insertvalue %61, %64[0] : !llvm.struct<(i64, ptr)> 
    %66 = llvm.insertvalue %63, %65[1] : !llvm.struct<(i64, ptr)> 
    %67 = llvm.mlir.constant(2 : i64) : i64
    %68 = llvm.mlir.constant(1 : index) : i64
    %69 = llvm.alloca %68 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %59, %69 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %70 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %71 = llvm.insertvalue %67, %70[0] : !llvm.struct<(i64, ptr)> 
    %72 = llvm.insertvalue %69, %71[1] : !llvm.struct<(i64, ptr)> 
    %73 = llvm.mlir.constant(1 : index) : i64
    %74 = llvm.alloca %73 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %66, %74 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %75 = llvm.alloca %73 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %72, %75 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %76 = llvm.mlir.constant(4 : index) : i64
    llvm.call @memrefCopy(%76, %74, %75) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %60 : !llvm.ptr
    %77 = llvm.mlir.constant(64 : index) : i64
    %78 = llvm.mlir.constant(128 : index) : i64
    %79 = llvm.mlir.constant(1 : index) : i64
    %80 = llvm.mlir.constant(8192 : index) : i64
    %81 = llvm.mlir.null : !llvm.ptr
    %82 = llvm.getelementptr %81[8192] : (!llvm.ptr) -> !llvm.ptr, f32
    %83 = llvm.ptrtoint %82 : !llvm.ptr to i64
    %84 = llvm.call @malloc(%83) : (i64) -> !llvm.ptr
    %85 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %86 = llvm.insertvalue %84, %85[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %87 = llvm.insertvalue %84, %86[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %88 = llvm.mlir.constant(0 : index) : i64
    %89 = llvm.insertvalue %88, %87[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %90 = llvm.insertvalue %77, %89[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %91 = llvm.insertvalue %78, %90[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %92 = llvm.insertvalue %78, %91[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %93 = llvm.insertvalue %79, %92[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %94 = llvm.intr.stacksave : !llvm.ptr
    %95 = llvm.mlir.constant(2 : i64) : i64
    %96 = llvm.mlir.constant(1 : index) : i64
    %97 = llvm.alloca %96 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %7, %97 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %98 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %99 = llvm.insertvalue %95, %98[0] : !llvm.struct<(i64, ptr)> 
    %100 = llvm.insertvalue %97, %99[1] : !llvm.struct<(i64, ptr)> 
    %101 = llvm.mlir.constant(2 : i64) : i64
    %102 = llvm.mlir.constant(1 : index) : i64
    %103 = llvm.alloca %102 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %93, %103 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %104 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %105 = llvm.insertvalue %101, %104[0] : !llvm.struct<(i64, ptr)> 
    %106 = llvm.insertvalue %103, %105[1] : !llvm.struct<(i64, ptr)> 
    %107 = llvm.mlir.constant(1 : index) : i64
    %108 = llvm.alloca %107 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %100, %108 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %109 = llvm.alloca %107 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %106, %109 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %110 = llvm.mlir.constant(4 : index) : i64
    llvm.call @memrefCopy(%110, %108, %109) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %94 : !llvm.ptr
    %111 = llvm.mlir.constant(1 : index) : i64
    %112 = llvm.mlir.constant(128 : index) : i64
    %113 = llvm.mlir.constant(1 : index) : i64
    %114 = llvm.mlir.constant(128 : index) : i64
    %115 = llvm.mlir.null : !llvm.ptr
    %116 = llvm.getelementptr %115[128] : (!llvm.ptr) -> !llvm.ptr, f32
    %117 = llvm.ptrtoint %116 : !llvm.ptr to i64
    %118 = llvm.call @malloc(%117) : (i64) -> !llvm.ptr
    %119 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %120 = llvm.insertvalue %118, %119[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %121 = llvm.insertvalue %118, %120[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %122 = llvm.mlir.constant(0 : index) : i64
    %123 = llvm.insertvalue %122, %121[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %124 = llvm.insertvalue %111, %123[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %125 = llvm.insertvalue %112, %124[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %126 = llvm.insertvalue %112, %125[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %127 = llvm.insertvalue %113, %126[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %128 = llvm.ptrtoint %118 : !llvm.ptr to i64
    %129 = llvm.inttoptr %128 : i64 to !llvm.ptr
    %130 = llvm.ptrtoint %50 : !llvm.ptr to i64
    %131 = llvm.inttoptr %130 : i64 to !llvm.ptr
    %132 = llvm.ptrtoint %84 : !llvm.ptr to i64
    %133 = llvm.inttoptr %132 : i64 to !llvm.ptr
    %134 = llvm.call @rxops_bridge_ada300_matmul_f32(%129, %131, %133, %40, %41, %42) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    %135 = llvm.mlir.constant(1 : index) : i64
    %136 = llvm.mlir.constant(128 : index) : i64
    %137 = llvm.mlir.constant(1 : index) : i64
    %138 = llvm.mlir.constant(128 : index) : i64
    %139 = llvm.mlir.null : !llvm.ptr
    %140 = llvm.getelementptr %139[128] : (!llvm.ptr) -> !llvm.ptr, f32
    %141 = llvm.ptrtoint %140 : !llvm.ptr to i64
    %142 = llvm.call @malloc(%141) : (i64) -> !llvm.ptr
    %143 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %144 = llvm.insertvalue %142, %143[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %145 = llvm.insertvalue %142, %144[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %146 = llvm.mlir.constant(0 : index) : i64
    %147 = llvm.insertvalue %146, %145[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %148 = llvm.insertvalue %135, %147[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %149 = llvm.insertvalue %136, %148[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %150 = llvm.insertvalue %136, %149[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %151 = llvm.insertvalue %137, %150[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %152 = llvm.intr.stacksave : !llvm.ptr
    %153 = llvm.mlir.constant(2 : i64) : i64
    %154 = llvm.mlir.constant(1 : index) : i64
    %155 = llvm.alloca %154 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %15, %155 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %156 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %157 = llvm.insertvalue %153, %156[0] : !llvm.struct<(i64, ptr)> 
    %158 = llvm.insertvalue %155, %157[1] : !llvm.struct<(i64, ptr)> 
    %159 = llvm.mlir.constant(2 : i64) : i64
    %160 = llvm.mlir.constant(1 : index) : i64
    %161 = llvm.alloca %160 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %151, %161 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %162 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %163 = llvm.insertvalue %159, %162[0] : !llvm.struct<(i64, ptr)> 
    %164 = llvm.insertvalue %161, %163[1] : !llvm.struct<(i64, ptr)> 
    %165 = llvm.mlir.constant(1 : index) : i64
    %166 = llvm.alloca %165 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %158, %166 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %167 = llvm.alloca %165 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %164, %167 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %168 = llvm.mlir.constant(4 : index) : i64
    llvm.call @memrefCopy(%168, %166, %167) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %152 : !llvm.ptr
    %169 = llvm.mlir.constant(1 : index) : i64
    %170 = llvm.mlir.constant(128 : index) : i64
    %171 = llvm.mlir.constant(1 : index) : i64
    %172 = llvm.mlir.constant(128 : index) : i64
    %173 = llvm.mlir.null : !llvm.ptr
    %174 = llvm.getelementptr %173[128] : (!llvm.ptr) -> !llvm.ptr, f32
    %175 = llvm.ptrtoint %174 : !llvm.ptr to i64
    %176 = llvm.call @malloc(%175) : (i64) -> !llvm.ptr
    %177 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %178 = llvm.insertvalue %176, %177[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %179 = llvm.insertvalue %176, %178[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %180 = llvm.mlir.constant(0 : index) : i64
    %181 = llvm.insertvalue %180, %179[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %182 = llvm.insertvalue %169, %181[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %183 = llvm.insertvalue %170, %182[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %184 = llvm.insertvalue %170, %183[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %185 = llvm.insertvalue %171, %184[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %186 = llvm.ptrtoint %142 : !llvm.ptr to i64
    %187 = llvm.inttoptr %186 : i64 to !llvm.ptr
    %188 = llvm.ptrtoint %118 : !llvm.ptr to i64
    %189 = llvm.inttoptr %188 : i64 to !llvm.ptr
    %190 = llvm.ptrtoint %176 : !llvm.ptr to i64
    %191 = llvm.inttoptr %190 : i64 to !llvm.ptr
    %192 = llvm.call @rxops_bridge_ada300_add_f32(%191, %187, %189, %41) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
    %193 = llvm.mlir.constant(1 : index) : i64
    %194 = llvm.mlir.constant(128 : index) : i64
    %195 = llvm.mlir.constant(1 : index) : i64
    %196 = llvm.mlir.constant(128 : index) : i64
    %197 = llvm.mlir.null : !llvm.ptr
    %198 = llvm.getelementptr %197[128] : (!llvm.ptr) -> !llvm.ptr, f32
    %199 = llvm.ptrtoint %198 : !llvm.ptr to i64
    %200 = llvm.call @malloc(%199) : (i64) -> !llvm.ptr
    %201 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %202 = llvm.insertvalue %200, %201[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %203 = llvm.insertvalue %200, %202[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %204 = llvm.mlir.constant(0 : index) : i64
    %205 = llvm.insertvalue %204, %203[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %206 = llvm.insertvalue %193, %205[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %207 = llvm.insertvalue %194, %206[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %208 = llvm.insertvalue %194, %207[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %209 = llvm.insertvalue %195, %208[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %210 = llvm.ptrtoint %176 : !llvm.ptr to i64
    %211 = llvm.inttoptr %210 : i64 to !llvm.ptr
    %212 = llvm.ptrtoint %200 : !llvm.ptr to i64
    %213 = llvm.inttoptr %212 : i64 to !llvm.ptr
    %214 = llvm.call @rxops_bridge_exp_f32(%213, %211, %41) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %215 = llvm.mlir.constant(1 : index) : i64
    %216 = llvm.mlir.constant(128 : index) : i64
    %217 = llvm.mlir.constant(1 : index) : i64
    %218 = llvm.mlir.constant(128 : index) : i64
    %219 = llvm.mlir.null : !llvm.ptr
    %220 = llvm.getelementptr %219[128] : (!llvm.ptr) -> !llvm.ptr, f32
    %221 = llvm.ptrtoint %220 : !llvm.ptr to i64
    %222 = llvm.call @malloc(%221) : (i64) -> !llvm.ptr
    %223 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %224 = llvm.insertvalue %222, %223[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %225 = llvm.insertvalue %222, %224[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %226 = llvm.mlir.constant(0 : index) : i64
    %227 = llvm.insertvalue %226, %225[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %228 = llvm.insertvalue %215, %227[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %229 = llvm.insertvalue %216, %228[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %230 = llvm.insertvalue %216, %229[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %231 = llvm.insertvalue %217, %230[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %232 = llvm.ptrtoint %200 : !llvm.ptr to i64
    %233 = llvm.inttoptr %232 : i64 to !llvm.ptr
    %234 = llvm.ptrtoint %222 : !llvm.ptr to i64
    %235 = llvm.inttoptr %234 : i64 to !llvm.ptr
    %236 = llvm.call @rxops_bridge_sqrt_f32(%235, %233, %41) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %237 = llvm.mlir.constant(128 : index) : i64
    %238 = llvm.mlir.constant(64 : index) : i64
    %239 = llvm.mlir.constant(1 : index) : i64
    %240 = llvm.mlir.constant(8192 : index) : i64
    %241 = llvm.mlir.null : !llvm.ptr
    %242 = llvm.getelementptr %241[8192] : (!llvm.ptr) -> !llvm.ptr, f32
    %243 = llvm.ptrtoint %242 : !llvm.ptr to i64
    %244 = llvm.call @malloc(%243) : (i64) -> !llvm.ptr
    %245 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %246 = llvm.insertvalue %244, %245[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %247 = llvm.insertvalue %244, %246[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %248 = llvm.mlir.constant(0 : index) : i64
    %249 = llvm.insertvalue %248, %247[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %250 = llvm.insertvalue %237, %249[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %251 = llvm.insertvalue %238, %250[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %252 = llvm.insertvalue %238, %251[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %253 = llvm.insertvalue %239, %252[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %254 = llvm.intr.stacksave : !llvm.ptr
    %255 = llvm.mlir.constant(2 : i64) : i64
    %256 = llvm.mlir.constant(1 : index) : i64
    %257 = llvm.alloca %256 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %23, %257 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %258 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %259 = llvm.insertvalue %255, %258[0] : !llvm.struct<(i64, ptr)> 
    %260 = llvm.insertvalue %257, %259[1] : !llvm.struct<(i64, ptr)> 
    %261 = llvm.mlir.constant(2 : i64) : i64
    %262 = llvm.mlir.constant(1 : index) : i64
    %263 = llvm.alloca %262 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %253, %263 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %264 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %265 = llvm.insertvalue %261, %264[0] : !llvm.struct<(i64, ptr)> 
    %266 = llvm.insertvalue %263, %265[1] : !llvm.struct<(i64, ptr)> 
    %267 = llvm.mlir.constant(1 : index) : i64
    %268 = llvm.alloca %267 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %260, %268 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %269 = llvm.alloca %267 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %266, %269 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %270 = llvm.mlir.constant(4 : index) : i64
    llvm.call @memrefCopy(%270, %268, %269) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %254 : !llvm.ptr
    %271 = llvm.mlir.constant(1 : index) : i64
    %272 = llvm.mlir.constant(64 : index) : i64
    %273 = llvm.mlir.constant(1 : index) : i64
    %274 = llvm.mlir.constant(64 : index) : i64
    %275 = llvm.mlir.null : !llvm.ptr
    %276 = llvm.getelementptr %275[64] : (!llvm.ptr) -> !llvm.ptr, f32
    %277 = llvm.ptrtoint %276 : !llvm.ptr to i64
    %278 = llvm.call @malloc(%277) : (i64) -> !llvm.ptr
    %279 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %280 = llvm.insertvalue %278, %279[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %281 = llvm.insertvalue %278, %280[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %282 = llvm.mlir.constant(0 : index) : i64
    %283 = llvm.insertvalue %282, %281[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %284 = llvm.insertvalue %271, %283[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %285 = llvm.insertvalue %272, %284[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %286 = llvm.insertvalue %272, %285[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %287 = llvm.insertvalue %273, %286[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %288 = llvm.ptrtoint %278 : !llvm.ptr to i64
    %289 = llvm.inttoptr %288 : i64 to !llvm.ptr
    %290 = llvm.ptrtoint %222 : !llvm.ptr to i64
    %291 = llvm.inttoptr %290 : i64 to !llvm.ptr
    %292 = llvm.ptrtoint %244 : !llvm.ptr to i64
    %293 = llvm.inttoptr %292 : i64 to !llvm.ptr
    %294 = llvm.call @rxops_bridge_ada300_matmul_f32(%289, %291, %293, %40, %42, %41) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    %295 = llvm.mlir.constant(1 : index) : i64
    %296 = llvm.mlir.constant(64 : index) : i64
    %297 = llvm.mlir.constant(1 : index) : i64
    %298 = llvm.mlir.constant(64 : index) : i64
    %299 = llvm.mlir.null : !llvm.ptr
    %300 = llvm.getelementptr %299[64] : (!llvm.ptr) -> !llvm.ptr, f32
    %301 = llvm.ptrtoint %300 : !llvm.ptr to i64
    %302 = llvm.call @malloc(%301) : (i64) -> !llvm.ptr
    %303 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %304 = llvm.insertvalue %302, %303[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %305 = llvm.insertvalue %302, %304[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %306 = llvm.mlir.constant(0 : index) : i64
    %307 = llvm.insertvalue %306, %305[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %308 = llvm.insertvalue %295, %307[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %309 = llvm.insertvalue %296, %308[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %310 = llvm.insertvalue %296, %309[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %311 = llvm.insertvalue %297, %310[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %312 = llvm.intr.stacksave : !llvm.ptr
    %313 = llvm.mlir.constant(2 : i64) : i64
    %314 = llvm.mlir.constant(1 : index) : i64
    %315 = llvm.alloca %314 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %31, %315 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %316 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %317 = llvm.insertvalue %313, %316[0] : !llvm.struct<(i64, ptr)> 
    %318 = llvm.insertvalue %315, %317[1] : !llvm.struct<(i64, ptr)> 
    %319 = llvm.mlir.constant(2 : i64) : i64
    %320 = llvm.mlir.constant(1 : index) : i64
    %321 = llvm.alloca %320 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %311, %321 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %322 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %323 = llvm.insertvalue %319, %322[0] : !llvm.struct<(i64, ptr)> 
    %324 = llvm.insertvalue %321, %323[1] : !llvm.struct<(i64, ptr)> 
    %325 = llvm.mlir.constant(1 : index) : i64
    %326 = llvm.alloca %325 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %318, %326 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %327 = llvm.alloca %325 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %324, %327 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %328 = llvm.mlir.constant(4 : index) : i64
    llvm.call @memrefCopy(%328, %326, %327) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %312 : !llvm.ptr
    %329 = llvm.mlir.constant(1 : index) : i64
    %330 = llvm.mlir.constant(64 : index) : i64
    %331 = llvm.mlir.constant(1 : index) : i64
    %332 = llvm.mlir.constant(64 : index) : i64
    %333 = llvm.mlir.null : !llvm.ptr
    %334 = llvm.getelementptr %333[64] : (!llvm.ptr) -> !llvm.ptr, f32
    %335 = llvm.ptrtoint %334 : !llvm.ptr to i64
    %336 = llvm.call @malloc(%335) : (i64) -> !llvm.ptr
    %337 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %338 = llvm.insertvalue %336, %337[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %339 = llvm.insertvalue %336, %338[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %340 = llvm.mlir.constant(0 : index) : i64
    %341 = llvm.insertvalue %340, %339[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %342 = llvm.insertvalue %329, %341[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %343 = llvm.insertvalue %330, %342[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %344 = llvm.insertvalue %330, %343[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %345 = llvm.insertvalue %331, %344[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %346 = llvm.ptrtoint %302 : !llvm.ptr to i64
    %347 = llvm.inttoptr %346 : i64 to !llvm.ptr
    %348 = llvm.ptrtoint %278 : !llvm.ptr to i64
    %349 = llvm.inttoptr %348 : i64 to !llvm.ptr
    %350 = llvm.ptrtoint %336 : !llvm.ptr to i64
    %351 = llvm.inttoptr %350 : i64 to !llvm.ptr
    %352 = llvm.call @rxops_bridge_ada300_add_f32(%351, %347, %349, %42) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
    llvm.return %345 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
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
    %8 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.extractvalue %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.extractvalue %8[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.load %arg3 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.extractvalue %16[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.extractvalue %16[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.extractvalue %16[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.extractvalue %16[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.extractvalue %16[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.extractvalue %16[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.load %arg4 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %25 = llvm.extractvalue %24[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.extractvalue %24[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.extractvalue %24[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.extractvalue %24[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.extractvalue %24[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.extractvalue %24[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.extractvalue %24[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.load %arg5 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %33 = llvm.extractvalue %32[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.extractvalue %32[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.extractvalue %32[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.extractvalue %32[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.extractvalue %32[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.extractvalue %32[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.extractvalue %32[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.call @subgraph0(%1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %14, %15, %17, %18, %19, %20, %21, %22, %23, %25, %26, %27, %28, %29, %30, %31, %33, %34, %35, %36, %37, %38, %39) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.store %40, %arg0 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    llvm.return
  }
}

