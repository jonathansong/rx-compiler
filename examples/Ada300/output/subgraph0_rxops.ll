; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@__constant_1x64xf32 = private constant [1 x [64 x float]] zeroinitializer, align 64
@__constant_1x128xf32 = private constant [1 x [128 x float]] zeroinitializer, align 64

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

declare ptr @malloc(i64)

declare i32 @rxops_bridge_exp_f32(ptr, ptr, i64)

declare i32 @rxops_bridge_sqrt_f32(ptr, ptr, i64)

declare i32 @rxops_bridge_matmul_f32(ptr, ptr, ptr, i64, i64, i64)

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @subgraph0(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, ptr %12, ptr %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, ptr %19, ptr %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, ptr %26, ptr %27, i64 %28, i64 %29, i64 %30) {
  %32 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %26, 0
  %33 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, ptr %27, 1
  %34 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, i64 %28, 2
  %35 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %34, i64 %29, 3, 0
  %36 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %35, i64 %30, 4, 0
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %19, 0
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, ptr %20, 1
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 %21, 2
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 %22, 3, 0
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, i64 %24, 4, 0
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, i64 %23, 3, 1
  %43 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, i64 %25, 4, 1
  %44 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %12, 0
  %45 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, ptr %13, 1
  %46 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, i64 %14, 2
  %47 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, i64 %15, 3, 0
  %48 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %47, i64 %17, 4, 0
  %49 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %48, i64 %16, 3, 1
  %50 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %49, i64 %18, 4, 1
  %51 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %7, 0
  %52 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %51, ptr %8, 1
  %53 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %52, i64 %9, 2
  %54 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %53, i64 %10, 3, 0
  %55 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %54, i64 %11, 4, 0
  %56 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %0, 0
  %57 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %56, ptr %1, 1
  %58 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, i64 %2, 2
  %59 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, i64 %3, 3, 0
  %60 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %59, i64 %5, 4, 0
  %61 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, i64 %4, 3, 1
  %62 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %61, i64 %6, 4, 1
  %63 = call ptr @malloc(i64 32832)
  %64 = ptrtoint ptr %63 to i64
  %65 = add i64 %64, 63
  %66 = urem i64 %65, 64
  %67 = sub i64 %65, %66
  %68 = inttoptr i64 %67 to ptr
  %69 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %63, 0
  %70 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %69, ptr %68, 1
  %71 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, i64 0, 2
  %72 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %71, i64 64, 3, 0
  %73 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %72, i64 128, 3, 1
  %74 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %73, i64 128, 4, 0
  %75 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %74, i64 1, 4, 1
  br label %76

76:                                               ; preds = %99, %31
  %77 = phi i64 [ %100, %99 ], [ 0, %31 ]
  %78 = icmp slt i64 %77, 64
  br i1 %78, label %79, label %101

79:                                               ; preds = %76
  br label %80

80:                                               ; preds = %83, %79
  %81 = phi i64 [ %98, %83 ], [ 0, %79 ]
  %82 = icmp slt i64 %81, 128
  br i1 %82, label %83, label %99

83:                                               ; preds = %80
  %84 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, 1
  %85 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, 2
  %86 = getelementptr float, ptr %84, i64 %85
  %87 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, 4, 0
  %88 = mul nuw nsw i64 %81, %87
  %89 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, 4, 1
  %90 = mul nuw nsw i64 %77, %89
  %91 = add nuw nsw i64 %88, %90
  %92 = getelementptr inbounds nuw float, ptr %86, i64 %91
  %93 = load float, ptr %92, align 4
  %94 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %75, 1
  %95 = mul nuw nsw i64 %77, 128
  %96 = add nuw nsw i64 %95, %81
  %97 = getelementptr inbounds nuw float, ptr %94, i64 %96
  store float %93, ptr %97, align 4
  %98 = add i64 %81, 1
  br label %80

99:                                               ; preds = %80
  %100 = add i64 %77, 1
  br label %76

101:                                              ; preds = %76
  %102 = call ptr @malloc(i64 576)
  %103 = ptrtoint ptr %102 to i64
  %104 = add i64 %103, 63
  %105 = urem i64 %104, 64
  %106 = sub i64 %104, %105
  %107 = inttoptr i64 %106 to ptr
  %108 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %102, 0
  %109 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %108, ptr %107, 1
  %110 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %109, i64 0, 2
  %111 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %110, i64 1, 3, 0
  %112 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, i64 128, 3, 1
  %113 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, i64 128, 4, 0
  %114 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %113, i64 1, 4, 1
  %115 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, 1
  %116 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, 2
  %117 = getelementptr float, ptr %115, i64 %116
  call void @llvm.memcpy.p0.p0.i64(ptr %117, ptr @__constant_1x128xf32, i64 512, i1 false)
  %118 = call ptr @malloc(i64 256)
  %119 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %118, 0
  %120 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, ptr %118, 1
  %121 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %120, i64 0, 2
  %122 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %121, i64 1, 3, 0
  %123 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %122, i64 64, 3, 1
  %124 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %123, i64 64, 4, 0
  %125 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %124, i64 1, 4, 1
  %126 = call ptr @malloc(i64 32768)
  %127 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %126, 0
  %128 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %127, ptr %126, 1
  %129 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %128, i64 0, 2
  %130 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %129, i64 64, 3, 0
  %131 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %130, i64 128, 3, 1
  %132 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %131, i64 128, 4, 0
  %133 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %132, i64 1, 4, 1
  %134 = call ptr @llvm.stacksave.p0()
  %135 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, ptr %135, align 8
  %136 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %135, 1
  %137 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %125, ptr %137, align 8
  %138 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %137, 1
  %139 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %136, ptr %139, align 8
  %140 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %138, ptr %140, align 8
  call void @memrefCopy(i64 4, ptr %139, ptr %140)
  call void @llvm.stackrestore.p0(ptr %134)
  %141 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %75, 3, 0
  %142 = mul i64 1, %141
  %143 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %75, 3, 1
  %144 = mul i64 %142, %143
  %145 = mul i64 %144, 4
  %146 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %75, 1
  %147 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %75, 2
  %148 = getelementptr float, ptr %146, i64 %147
  %149 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %133, 1
  %150 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %133, 2
  %151 = getelementptr float, ptr %149, i64 %150
  call void @llvm.memcpy.p0.p0.i64(ptr %151, ptr %148, i64 %145, i1 false)
  %152 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, 1
  %153 = ptrtoint ptr %152 to i64
  %154 = inttoptr i64 %153 to ptr
  %155 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %125, 1
  %156 = ptrtoint ptr %155 to i64
  %157 = inttoptr i64 %156 to ptr
  %158 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %133, 1
  %159 = ptrtoint ptr %158 to i64
  %160 = inttoptr i64 %159 to ptr
  %161 = call i32 @rxops_bridge_matmul_f32(ptr %154, ptr %157, ptr %160, i64 1, i64 128, i64 64)
  %162 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %125, 0
  call void @free(ptr %162)
  %163 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %133, 0
  call void @free(ptr %163)
  %164 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 0
  %165 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 1
  %166 = insertvalue { ptr, ptr, i64 } poison, ptr %164, 0
  %167 = insertvalue { ptr, ptr, i64 } %166, ptr %165, 1
  %168 = insertvalue { ptr, ptr, i64 } %167, i64 0, 2
  %169 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 2
  %170 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 3, 0
  %171 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 4, 0
  %172 = mul nsw i64 %171, 128
  %173 = extractvalue { ptr, ptr, i64 } %168, 0
  %174 = extractvalue { ptr, ptr, i64 } %168, 1
  %175 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %173, 0
  %176 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %175, ptr %174, 1
  %177 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %176, i64 %169, 2
  %178 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %177, i64 1, 3, 0
  %179 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %178, i64 %172, 4, 0
  %180 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %179, i64 128, 3, 1
  %181 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %180, i64 %171, 4, 1
  %182 = call ptr @malloc(i64 576)
  %183 = ptrtoint ptr %182 to i64
  %184 = add i64 %183, 63
  %185 = urem i64 %184, 64
  %186 = sub i64 %184, %185
  %187 = inttoptr i64 %186 to ptr
  %188 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %182, 0
  %189 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %188, ptr %187, 1
  %190 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %189, i64 0, 2
  %191 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %190, i64 1, 3, 0
  %192 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %191, i64 128, 3, 1
  %193 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %192, i64 128, 4, 0
  %194 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %193, i64 1, 4, 1
  br label %195

195:                                              ; preds = %224, %101
  %196 = phi i64 [ %225, %224 ], [ 0, %101 ]
  %197 = icmp slt i64 %196, 1
  br i1 %197, label %198, label %226

198:                                              ; preds = %195
  br label %199

199:                                              ; preds = %202, %198
  %200 = phi i64 [ %223, %202 ], [ 0, %198 ]
  %201 = icmp slt i64 %200, 128
  br i1 %201, label %202, label %224

202:                                              ; preds = %199
  %203 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %181, 1
  %204 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %181, 2
  %205 = getelementptr float, ptr %203, i64 %204
  %206 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %181, 4, 0
  %207 = mul nuw nsw i64 %196, %206
  %208 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %181, 4, 1
  %209 = mul nuw nsw i64 %200, %208
  %210 = add nuw nsw i64 %207, %209
  %211 = getelementptr inbounds nuw float, ptr %205, i64 %210
  %212 = load float, ptr %211, align 4
  %213 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, 1
  %214 = mul nuw nsw i64 %196, 128
  %215 = add nuw nsw i64 %214, %200
  %216 = getelementptr inbounds nuw float, ptr %213, i64 %215
  %217 = load float, ptr %216, align 4
  %218 = fadd float %212, %217
  %219 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %194, 1
  %220 = mul nuw nsw i64 %196, 128
  %221 = add nuw nsw i64 %220, %200
  %222 = getelementptr inbounds nuw float, ptr %219, i64 %221
  store float %218, ptr %222, align 4
  %223 = add i64 %200, 1
  br label %199

224:                                              ; preds = %199
  %225 = add i64 %196, 1
  br label %195

226:                                              ; preds = %195
  %227 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %194, 1
  %228 = getelementptr float, ptr %227, i64 0
  %229 = load <128 x float>, ptr %228, align 4
  %230 = alloca float, i64 128, align 4
  %231 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %230, 0
  %232 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %231, ptr %230, 1
  %233 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %232, i64 0, 2
  %234 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %233, i64 128, 3, 0
  %235 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %234, i64 1, 4, 0
  %236 = alloca float, i64 128, align 4
  %237 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %236, 0
  %238 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %237, ptr %236, 1
  %239 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %238, i64 0, 2
  %240 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %239, i64 128, 3, 0
  %241 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %240, i64 1, 4, 0
  %242 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %235, 1
  %243 = getelementptr float, ptr %242, i64 0
  store <128 x float> %229, ptr %243, align 4
  %244 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %235, 1
  %245 = ptrtoint ptr %244 to i64
  %246 = inttoptr i64 %245 to ptr
  %247 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %241, 1
  %248 = ptrtoint ptr %247 to i64
  %249 = inttoptr i64 %248 to ptr
  %250 = call i32 @rxops_bridge_exp_f32(ptr %249, ptr %246, i64 128)
  %251 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %241, 1
  %252 = getelementptr float, ptr %251, i64 0
  %253 = load <128 x float>, ptr %252, align 4
  %254 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %194, 1
  %255 = getelementptr float, ptr %254, i64 0
  store <128 x float> %253, ptr %255, align 4
  %256 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %194, 1
  %257 = getelementptr float, ptr %256, i64 0
  %258 = load <128 x float>, ptr %257, align 4
  %259 = alloca float, i64 128, align 4
  %260 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %259, 0
  %261 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %260, ptr %259, 1
  %262 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %261, i64 0, 2
  %263 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %262, i64 128, 3, 0
  %264 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %263, i64 1, 4, 0
  %265 = alloca float, i64 128, align 4
  %266 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %265, 0
  %267 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %266, ptr %265, 1
  %268 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %267, i64 0, 2
  %269 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %268, i64 128, 3, 0
  %270 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %269, i64 1, 4, 0
  %271 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %264, 1
  %272 = getelementptr float, ptr %271, i64 0
  store <128 x float> %258, ptr %272, align 4
  %273 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %264, 1
  %274 = ptrtoint ptr %273 to i64
  %275 = inttoptr i64 %274 to ptr
  %276 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %270, 1
  %277 = ptrtoint ptr %276 to i64
  %278 = inttoptr i64 %277 to ptr
  %279 = call i32 @rxops_bridge_sqrt_f32(ptr %278, ptr %275, i64 128)
  %280 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %270, 1
  %281 = getelementptr float, ptr %280, i64 0
  %282 = load <128 x float>, ptr %281, align 4
  %283 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %194, 1
  %284 = getelementptr float, ptr %283, i64 0
  store <128 x float> %282, ptr %284, align 4
  %285 = call ptr @malloc(i64 32832)
  %286 = ptrtoint ptr %285 to i64
  %287 = add i64 %286, 63
  %288 = urem i64 %287, 64
  %289 = sub i64 %287, %288
  %290 = inttoptr i64 %289 to ptr
  %291 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %285, 0
  %292 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %291, ptr %290, 1
  %293 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %292, i64 0, 2
  %294 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %293, i64 128, 3, 0
  %295 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %294, i64 64, 3, 1
  %296 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %295, i64 64, 4, 0
  %297 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %296, i64 1, 4, 1
  br label %298

298:                                              ; preds = %321, %226
  %299 = phi i64 [ %322, %321 ], [ 0, %226 ]
  %300 = icmp slt i64 %299, 128
  br i1 %300, label %301, label %323

301:                                              ; preds = %298
  br label %302

302:                                              ; preds = %305, %301
  %303 = phi i64 [ %320, %305 ], [ 0, %301 ]
  %304 = icmp slt i64 %303, 64
  br i1 %304, label %305, label %321

305:                                              ; preds = %302
  %306 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, 1
  %307 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, 2
  %308 = getelementptr float, ptr %306, i64 %307
  %309 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, 4, 0
  %310 = mul nuw nsw i64 %303, %309
  %311 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, 4, 1
  %312 = mul nuw nsw i64 %299, %311
  %313 = add nuw nsw i64 %310, %312
  %314 = getelementptr inbounds nuw float, ptr %308, i64 %313
  %315 = load float, ptr %314, align 4
  %316 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %297, 1
  %317 = mul nuw nsw i64 %299, 64
  %318 = add nuw nsw i64 %317, %303
  %319 = getelementptr inbounds nuw float, ptr %316, i64 %318
  store float %315, ptr %319, align 4
  %320 = add i64 %303, 1
  br label %302

321:                                              ; preds = %302
  %322 = add i64 %299, 1
  br label %298

323:                                              ; preds = %298
  %324 = call ptr @malloc(i64 320)
  %325 = ptrtoint ptr %324 to i64
  %326 = add i64 %325, 63
  %327 = urem i64 %326, 64
  %328 = sub i64 %326, %327
  %329 = inttoptr i64 %328 to ptr
  %330 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %324, 0
  %331 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %330, ptr %329, 1
  %332 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %331, i64 0, 2
  %333 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %332, i64 1, 3, 0
  %334 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %333, i64 64, 3, 1
  %335 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %334, i64 64, 4, 0
  %336 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %335, i64 1, 4, 1
  %337 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %336, 1
  %338 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %336, 2
  %339 = getelementptr float, ptr %337, i64 %338
  call void @llvm.memcpy.p0.p0.i64(ptr %339, ptr @__constant_1x64xf32, i64 256, i1 false)
  %340 = call ptr @malloc(i64 512)
  %341 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %340, 0
  %342 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %341, ptr %340, 1
  %343 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %342, i64 0, 2
  %344 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %343, i64 1, 3, 0
  %345 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %344, i64 128, 3, 1
  %346 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %345, i64 128, 4, 0
  %347 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %346, i64 1, 4, 1
  %348 = call ptr @malloc(i64 32768)
  %349 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %348, 0
  %350 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %349, ptr %348, 1
  %351 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %350, i64 0, 2
  %352 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %351, i64 128, 3, 0
  %353 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %352, i64 64, 3, 1
  %354 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %353, i64 64, 4, 0
  %355 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %354, i64 1, 4, 1
  %356 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %194, 3, 0
  %357 = mul i64 1, %356
  %358 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %194, 3, 1
  %359 = mul i64 %357, %358
  %360 = mul i64 %359, 4
  %361 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %194, 1
  %362 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %194, 2
  %363 = getelementptr float, ptr %361, i64 %362
  %364 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %347, 1
  %365 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %347, 2
  %366 = getelementptr float, ptr %364, i64 %365
  call void @llvm.memcpy.p0.p0.i64(ptr %366, ptr %363, i64 %360, i1 false)
  %367 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %297, 3, 0
  %368 = mul i64 1, %367
  %369 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %297, 3, 1
  %370 = mul i64 %368, %369
  %371 = mul i64 %370, 4
  %372 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %297, 1
  %373 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %297, 2
  %374 = getelementptr float, ptr %372, i64 %373
  %375 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %355, 1
  %376 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %355, 2
  %377 = getelementptr float, ptr %375, i64 %376
  call void @llvm.memcpy.p0.p0.i64(ptr %377, ptr %374, i64 %371, i1 false)
  %378 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %336, 1
  %379 = ptrtoint ptr %378 to i64
  %380 = inttoptr i64 %379 to ptr
  %381 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %347, 1
  %382 = ptrtoint ptr %381 to i64
  %383 = inttoptr i64 %382 to ptr
  %384 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %355, 1
  %385 = ptrtoint ptr %384 to i64
  %386 = inttoptr i64 %385 to ptr
  %387 = call i32 @rxops_bridge_matmul_f32(ptr %380, ptr %383, ptr %386, i64 1, i64 64, i64 128)
  %388 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %347, 0
  call void @free(ptr %388)
  %389 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %355, 0
  call void @free(ptr %389)
  %390 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 0
  %391 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 1
  %392 = insertvalue { ptr, ptr, i64 } poison, ptr %390, 0
  %393 = insertvalue { ptr, ptr, i64 } %392, ptr %391, 1
  %394 = insertvalue { ptr, ptr, i64 } %393, i64 0, 2
  %395 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 2
  %396 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 3, 0
  %397 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 4, 0
  %398 = mul nsw i64 %397, 64
  %399 = extractvalue { ptr, ptr, i64 } %394, 0
  %400 = extractvalue { ptr, ptr, i64 } %394, 1
  %401 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %399, 0
  %402 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %401, ptr %400, 1
  %403 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %402, i64 %395, 2
  %404 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %403, i64 1, 3, 0
  %405 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %404, i64 %398, 4, 0
  %406 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %405, i64 64, 3, 1
  %407 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %406, i64 %397, 4, 1
  %408 = call ptr @malloc(i64 320)
  %409 = ptrtoint ptr %408 to i64
  %410 = add i64 %409, 63
  %411 = urem i64 %410, 64
  %412 = sub i64 %410, %411
  %413 = inttoptr i64 %412 to ptr
  %414 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %408, 0
  %415 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %414, ptr %413, 1
  %416 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %415, i64 0, 2
  %417 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %416, i64 1, 3, 0
  %418 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %417, i64 64, 3, 1
  %419 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %418, i64 64, 4, 0
  %420 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %419, i64 1, 4, 1
  br label %421

421:                                              ; preds = %450, %323
  %422 = phi i64 [ %451, %450 ], [ 0, %323 ]
  %423 = icmp slt i64 %422, 1
  br i1 %423, label %424, label %452

424:                                              ; preds = %421
  br label %425

425:                                              ; preds = %428, %424
  %426 = phi i64 [ %449, %428 ], [ 0, %424 ]
  %427 = icmp slt i64 %426, 64
  br i1 %427, label %428, label %450

428:                                              ; preds = %425
  %429 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %407, 1
  %430 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %407, 2
  %431 = getelementptr float, ptr %429, i64 %430
  %432 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %407, 4, 0
  %433 = mul nuw nsw i64 %422, %432
  %434 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %407, 4, 1
  %435 = mul nuw nsw i64 %426, %434
  %436 = add nuw nsw i64 %433, %435
  %437 = getelementptr inbounds nuw float, ptr %431, i64 %436
  %438 = load float, ptr %437, align 4
  %439 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %336, 1
  %440 = mul nuw nsw i64 %422, 64
  %441 = add nuw nsw i64 %440, %426
  %442 = getelementptr inbounds nuw float, ptr %439, i64 %441
  %443 = load float, ptr %442, align 4
  %444 = fadd float %438, %443
  %445 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %420, 1
  %446 = mul nuw nsw i64 %422, 64
  %447 = add nuw nsw i64 %446, %426
  %448 = getelementptr inbounds nuw float, ptr %445, i64 %447
  store float %444, ptr %448, align 4
  %449 = add i64 %426, 1
  br label %425

450:                                              ; preds = %425
  %451 = add i64 %422, 1
  br label %421

452:                                              ; preds = %421
  %453 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %75, 0
  call void @free(ptr %453)
  %454 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, 0
  call void @free(ptr %454)
  %455 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %194, 0
  call void @free(ptr %455)
  %456 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %297, 0
  call void @free(ptr %456)
  %457 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %336, 0
  call void @free(ptr %457)
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %420
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
  %15 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %2, align 8
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %15, 0
  %17 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %15, 1
  %18 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %15, 2
  %19 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %15, 3, 0
  %20 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %15, 4, 0
  %21 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %3, align 8
  %22 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 0
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 1
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 2
  %25 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 3, 0
  %26 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 3, 1
  %27 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 4, 0
  %28 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 4, 1
  %29 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %4, align 8
  %30 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 0
  %31 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 1
  %32 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 2
  %33 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 3, 0
  %34 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 3, 1
  %35 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 4, 0
  %36 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, 4, 1
  %37 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %5, align 8
  %38 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 0
  %39 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %40 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 2
  %41 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 3, 0
  %42 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 4, 0
  %43 = call { ptr, ptr, i64, [2 x i64], [2 x i64] } @subgraph0(ptr %8, ptr %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, ptr %16, ptr %17, i64 %18, i64 %19, i64 %20, ptr %22, ptr %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28, ptr %30, ptr %31, i64 %32, i64 %33, i64 %34, i64 %35, i64 %36, ptr %38, ptr %39, i64 %40, i64 %41, i64 %42)
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, ptr %0, align 8
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #1

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
