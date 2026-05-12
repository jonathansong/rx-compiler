; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

declare i32 @rxops_bridge_exp_f32(ptr, ptr, i64)

declare i32 @rxops_bridge_sqrt_f32(ptr, ptr, i64)

declare i32 @rxops_bridge_matmul_f32(ptr, ptr, ptr, i64, i64, i64)

declare i32 @rxops_bridge_add_f32(ptr, ptr, ptr, i64)

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @subgraph0(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, ptr %21, ptr %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27, ptr %28, ptr %29, i64 %30, i64 %31, i64 %32, i64 %33, i64 %34) {
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %0, 0
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, ptr %1, 1
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 %2, 2
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 %3, 3, 0
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 %5, 4, 0
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, i64 %4, 3, 1
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, i64 %6, 4, 1
  %43 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %7, 0
  %44 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, ptr %8, 1
  %45 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, i64 %9, 2
  %46 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, i64 %10, 3, 0
  %47 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, i64 %12, 4, 0
  %48 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %47, i64 %11, 3, 1
  %49 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %48, i64 %13, 4, 1
  %50 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %14, 0
  %51 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, ptr %15, 1
  %52 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %51, i64 %16, 2
  %53 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, i64 %17, 3, 0
  %54 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, i64 %19, 4, 0
  %55 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, i64 %18, 3, 1
  %56 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, i64 %20, 4, 1
  %57 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %21, 0
  %58 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, ptr %22, 1
  %59 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, i64 %23, 2
  %60 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %59, i64 %24, 3, 0
  %61 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, i64 %26, 4, 0
  %62 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %61, i64 %25, 3, 1
  %63 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, i64 %27, 4, 1
  %64 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %28, 0
  %65 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %64, ptr %29, 1
  %66 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %65, i64 %30, 2
  %67 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %66, i64 %31, 3, 0
  %68 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %67, i64 %33, 4, 0
  %69 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %68, i64 %32, 3, 1
  %70 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %69, i64 %34, 4, 1
  %71 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 64) to i64))
  %72 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %71, 0
  %73 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %72, ptr %71, 1
  %74 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %73, i64 0, 2
  %75 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %74, i64 1, 3, 0
  %76 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %75, i64 64, 3, 1
  %77 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %76, i64 64, 4, 0
  %78 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %77, i64 1, 4, 1
  %79 = call ptr @llvm.stacksave()
  %80 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, ptr %80, align 8
  %81 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %80, 1
  %82 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %78, ptr %82, align 8
  %83 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %82, 1
  %84 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %81, ptr %84, align 8
  %85 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %83, ptr %85, align 8
  call void @memrefCopy(i64 4, ptr %84, ptr %85)
  call void @llvm.stackrestore(ptr %79)
  %86 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 8192) to i64))
  %87 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %86, 0
  %88 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, ptr %86, 1
  %89 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %88, i64 0, 2
  %90 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %89, i64 64, 3, 0
  %91 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %90, i64 128, 3, 1
  %92 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %91, i64 128, 4, 0
  %93 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %92, i64 1, 4, 1
  %94 = call ptr @llvm.stacksave()
  %95 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, ptr %95, align 8
  %96 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %95, 1
  %97 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %93, ptr %97, align 8
  %98 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %97, 1
  %99 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %96, ptr %99, align 8
  %100 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %98, ptr %100, align 8
  call void @memrefCopy(i64 4, ptr %99, ptr %100)
  call void @llvm.stackrestore(ptr %94)
  %101 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 128) to i64))
  %102 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %101, 0
  %103 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %102, ptr %101, 1
  %104 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %103, i64 0, 2
  %105 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %104, i64 1, 3, 0
  %106 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %105, i64 128, 3, 1
  %107 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %106, i64 128, 4, 0
  %108 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %107, i64 1, 4, 1
  %109 = ptrtoint ptr %101 to i64
  %110 = inttoptr i64 %109 to ptr
  %111 = ptrtoint ptr %71 to i64
  %112 = inttoptr i64 %111 to ptr
  %113 = ptrtoint ptr %86 to i64
  %114 = inttoptr i64 %113 to ptr
  %115 = call i32 @rxops_bridge_matmul_f32(ptr %110, ptr %112, ptr %114, i64 1, i64 128, i64 64)
  %116 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 128) to i64))
  %117 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %116, 0
  %118 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %117, ptr %116, 1
  %119 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %118, i64 0, 2
  %120 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, i64 1, 3, 0
  %121 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %120, i64 128, 3, 1
  %122 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %121, i64 128, 4, 0
  %123 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %122, i64 1, 4, 1
  %124 = call ptr @llvm.stacksave()
  %125 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %49, ptr %125, align 8
  %126 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %125, 1
  %127 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %123, ptr %127, align 8
  %128 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %127, 1
  %129 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %126, ptr %129, align 8
  %130 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %128, ptr %130, align 8
  call void @memrefCopy(i64 4, ptr %129, ptr %130)
  call void @llvm.stackrestore(ptr %124)
  %131 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 128) to i64))
  %132 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %131, 0
  %133 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %132, ptr %131, 1
  %134 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %133, i64 0, 2
  %135 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %134, i64 1, 3, 0
  %136 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %135, i64 128, 3, 1
  %137 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %136, i64 128, 4, 0
  %138 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %137, i64 1, 4, 1
  %139 = ptrtoint ptr %116 to i64
  %140 = inttoptr i64 %139 to ptr
  %141 = ptrtoint ptr %101 to i64
  %142 = inttoptr i64 %141 to ptr
  %143 = ptrtoint ptr %131 to i64
  %144 = inttoptr i64 %143 to ptr
  %145 = call i32 @rxops_bridge_add_f32(ptr %144, ptr %140, ptr %142, i64 128)
  %146 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 128) to i64))
  %147 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %146, 0
  %148 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %147, ptr %146, 1
  %149 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %148, i64 0, 2
  %150 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %149, i64 1, 3, 0
  %151 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %150, i64 128, 3, 1
  %152 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, i64 128, 4, 0
  %153 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, i64 1, 4, 1
  %154 = ptrtoint ptr %131 to i64
  %155 = inttoptr i64 %154 to ptr
  %156 = ptrtoint ptr %146 to i64
  %157 = inttoptr i64 %156 to ptr
  %158 = call i32 @rxops_bridge_exp_f32(ptr %157, ptr %155, i64 128)
  %159 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 128) to i64))
  %160 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %159, 0
  %161 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %160, ptr %159, 1
  %162 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %161, i64 0, 2
  %163 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %162, i64 1, 3, 0
  %164 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %163, i64 128, 3, 1
  %165 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %164, i64 128, 4, 0
  %166 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %165, i64 1, 4, 1
  %167 = ptrtoint ptr %146 to i64
  %168 = inttoptr i64 %167 to ptr
  %169 = ptrtoint ptr %159 to i64
  %170 = inttoptr i64 %169 to ptr
  %171 = call i32 @rxops_bridge_sqrt_f32(ptr %170, ptr %168, i64 128)
  %172 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 8192) to i64))
  %173 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %172, 0
  %174 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %173, ptr %172, 1
  %175 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %174, i64 0, 2
  %176 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %175, i64 128, 3, 0
  %177 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %176, i64 64, 3, 1
  %178 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %177, i64 64, 4, 0
  %179 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %178, i64 1, 4, 1
  %180 = call ptr @llvm.stacksave()
  %181 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %56, ptr %181, align 8
  %182 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %181, 1
  %183 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %179, ptr %183, align 8
  %184 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %183, 1
  %185 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %182, ptr %185, align 8
  %186 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %184, ptr %186, align 8
  call void @memrefCopy(i64 4, ptr %185, ptr %186)
  call void @llvm.stackrestore(ptr %180)
  %187 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 64) to i64))
  %188 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %187, 0
  %189 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %188, ptr %187, 1
  %190 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %189, i64 0, 2
  %191 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %190, i64 1, 3, 0
  %192 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %191, i64 64, 3, 1
  %193 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %192, i64 64, 4, 0
  %194 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %193, i64 1, 4, 1
  %195 = ptrtoint ptr %187 to i64
  %196 = inttoptr i64 %195 to ptr
  %197 = ptrtoint ptr %159 to i64
  %198 = inttoptr i64 %197 to ptr
  %199 = ptrtoint ptr %172 to i64
  %200 = inttoptr i64 %199 to ptr
  %201 = call i32 @rxops_bridge_matmul_f32(ptr %196, ptr %198, ptr %200, i64 1, i64 64, i64 128)
  %202 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 64) to i64))
  %203 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %202, 0
  %204 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %203, ptr %202, 1
  %205 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %204, i64 0, 2
  %206 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %205, i64 1, 3, 0
  %207 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %206, i64 64, 3, 1
  %208 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %207, i64 64, 4, 0
  %209 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %208, i64 1, 4, 1
  %210 = call ptr @llvm.stacksave()
  %211 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %63, ptr %211, align 8
  %212 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %211, 1
  %213 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %209, ptr %213, align 8
  %214 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %213, 1
  %215 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %212, ptr %215, align 8
  %216 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %214, ptr %216, align 8
  call void @memrefCopy(i64 4, ptr %215, ptr %216)
  call void @llvm.stackrestore(ptr %210)
  %217 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 64) to i64))
  %218 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %217, 0
  %219 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %218, ptr %217, 1
  %220 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %219, i64 0, 2
  %221 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %220, i64 1, 3, 0
  %222 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %221, i64 64, 3, 1
  %223 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %222, i64 64, 4, 0
  %224 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %223, i64 1, 4, 1
  %225 = ptrtoint ptr %202 to i64
  %226 = inttoptr i64 %225 to ptr
  %227 = ptrtoint ptr %187 to i64
  %228 = inttoptr i64 %227 to ptr
  %229 = ptrtoint ptr %217 to i64
  %230 = inttoptr i64 %229 to ptr
  %231 = call i32 @rxops_bridge_add_f32(ptr %230, ptr %226, ptr %228, i64 64)
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %224
}

define void @_mlir_ciface_subgraph0(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5) {
  %7 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %1, align 8
  %8 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, 0
  %9 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, 1
  %10 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, 2
  %11 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, 3, 0
  %12 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, 3, 1
  %13 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, 4, 0
  %14 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, 4, 1
  %15 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %2, align 8
  %16 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 0
  %17 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 1
  %18 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 2
  %19 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 3, 0
  %20 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 3, 1
  %21 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 4, 0
  %22 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 4, 1
  %23 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %3, align 8
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, 0
  %25 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, 1
  %26 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, 2
  %27 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, 3, 0
  %28 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, 3, 1
  %29 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, 4, 0
  %30 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, 4, 1
  %31 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %4, align 8
  %32 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 0
  %33 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 1
  %34 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 2
  %35 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 3, 0
  %36 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 3, 1
  %37 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 4, 0
  %38 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, 4, 1
  %39 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %5, align 8
  %40 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 0
  %41 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 1
  %42 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 2
  %43 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 3, 0
  %44 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 3, 1
  %45 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 4, 0
  %46 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 4, 1
  %47 = call { ptr, ptr, i64, [2 x i64], [2 x i64] } @subgraph0(ptr %8, ptr %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, ptr %16, ptr %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, ptr %24, ptr %25, i64 %26, i64 %27, i64 %28, i64 %29, i64 %30, ptr %32, ptr %33, i64 %34, i64 %35, i64 %36, i64 %37, i64 %38, ptr %40, ptr %41, i64 %42, i64 %43, i64 %44, i64 %45, i64 %46)
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %47, ptr %0, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave() #0

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore(ptr) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
