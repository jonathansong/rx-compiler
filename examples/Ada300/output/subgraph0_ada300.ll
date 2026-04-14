; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@__constant_1x64xf32 = private constant [1 x [64 x float]] zeroinitializer, align 64
@__constant_1x128xf32 = private constant [1 x [128 x float]] zeroinitializer, align 64

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

declare ptr @malloc(i64)

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
  call void asm sideeffect "gmm.cfg 1, 128, 64, 0", ""()
  call void asm sideeffect "gmm.type 0, 0, 0", ""()
  call void asm sideeffect "gmm.iter 1, 1", ""()
  %152 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, 1
  %153 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %125, 1
  %154 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %133, 1
  call void asm sideeffect "gmma.mm $0, $1, $2", "r,r,r"(ptr %152, ptr %153, ptr %154)
  call void asm sideeffect "tcsync", ""()
  %155 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %125, 0
  call void @free(ptr %155)
  %156 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %133, 0
  call void @free(ptr %156)
  %157 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 0
  %158 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 1
  %159 = insertvalue { ptr, ptr, i64 } poison, ptr %157, 0
  %160 = insertvalue { ptr, ptr, i64 } %159, ptr %158, 1
  %161 = insertvalue { ptr, ptr, i64 } %160, i64 0, 2
  %162 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 2
  %163 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 3, 0
  %164 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 4, 0
  %165 = mul nsw i64 %164, 128
  %166 = extractvalue { ptr, ptr, i64 } %161, 0
  %167 = extractvalue { ptr, ptr, i64 } %161, 1
  %168 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %166, 0
  %169 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %168, ptr %167, 1
  %170 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %169, i64 %162, 2
  %171 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %170, i64 1, 3, 0
  %172 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %171, i64 %165, 4, 0
  %173 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %172, i64 128, 3, 1
  %174 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %173, i64 %164, 4, 1
  %175 = call ptr @malloc(i64 576)
  %176 = ptrtoint ptr %175 to i64
  %177 = add i64 %176, 63
  %178 = urem i64 %177, 64
  %179 = sub i64 %177, %178
  %180 = inttoptr i64 %179 to ptr
  %181 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %175, 0
  %182 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %181, ptr %180, 1
  %183 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %182, i64 0, 2
  %184 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %183, i64 1, 3, 0
  %185 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %184, i64 128, 3, 1
  %186 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %185, i64 128, 4, 0
  %187 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %186, i64 1, 4, 1
  br label %188

188:                                              ; preds = %217, %101
  %189 = phi i64 [ %218, %217 ], [ 0, %101 ]
  %190 = icmp slt i64 %189, 1
  br i1 %190, label %191, label %219

191:                                              ; preds = %188
  br label %192

192:                                              ; preds = %195, %191
  %193 = phi i64 [ %216, %195 ], [ 0, %191 ]
  %194 = icmp slt i64 %193, 128
  br i1 %194, label %195, label %217

195:                                              ; preds = %192
  %196 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %174, 1
  %197 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %174, 2
  %198 = getelementptr float, ptr %196, i64 %197
  %199 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %174, 4, 0
  %200 = mul nuw nsw i64 %189, %199
  %201 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %174, 4, 1
  %202 = mul nuw nsw i64 %193, %201
  %203 = add nuw nsw i64 %200, %202
  %204 = getelementptr inbounds nuw float, ptr %198, i64 %203
  %205 = load float, ptr %204, align 4
  %206 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, 1
  %207 = mul nuw nsw i64 %189, 128
  %208 = add nuw nsw i64 %207, %193
  %209 = getelementptr inbounds nuw float, ptr %206, i64 %208
  %210 = load float, ptr %209, align 4
  %211 = fadd float %205, %210
  %212 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %187, 1
  %213 = mul nuw nsw i64 %189, 128
  %214 = add nuw nsw i64 %213, %193
  %215 = getelementptr inbounds nuw float, ptr %212, i64 %214
  store float %211, ptr %215, align 4
  %216 = add i64 %193, 1
  br label %192

217:                                              ; preds = %192
  %218 = add i64 %189, 1
  br label %188

219:                                              ; preds = %188
  %220 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %187, 1
  %221 = getelementptr float, ptr %220, i64 0
  %222 = load <128 x float>, ptr %221, align 4
  %223 = call <128 x float> asm "vfpwnl.exp.16 $0, $1", "=vr,vr"(<128 x float> %222)
  %224 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %187, 1
  %225 = getelementptr float, ptr %224, i64 0
  store <128 x float> %223, ptr %225, align 4
  %226 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %187, 1
  %227 = getelementptr float, ptr %226, i64 0
  %228 = load <128 x float>, ptr %227, align 4
  %229 = call <128 x float> asm "vfpwnl.sqrt.16 $0, $1", "=vr,vr"(<128 x float> %228)
  %230 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %187, 1
  %231 = getelementptr float, ptr %230, i64 0
  store <128 x float> %229, ptr %231, align 4
  %232 = call ptr @malloc(i64 32832)
  %233 = ptrtoint ptr %232 to i64
  %234 = add i64 %233, 63
  %235 = urem i64 %234, 64
  %236 = sub i64 %234, %235
  %237 = inttoptr i64 %236 to ptr
  %238 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %232, 0
  %239 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %238, ptr %237, 1
  %240 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %239, i64 0, 2
  %241 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %240, i64 128, 3, 0
  %242 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %241, i64 64, 3, 1
  %243 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %242, i64 64, 4, 0
  %244 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %243, i64 1, 4, 1
  br label %245

245:                                              ; preds = %268, %219
  %246 = phi i64 [ %269, %268 ], [ 0, %219 ]
  %247 = icmp slt i64 %246, 128
  br i1 %247, label %248, label %270

248:                                              ; preds = %245
  br label %249

249:                                              ; preds = %252, %248
  %250 = phi i64 [ %267, %252 ], [ 0, %248 ]
  %251 = icmp slt i64 %250, 64
  br i1 %251, label %252, label %268

252:                                              ; preds = %249
  %253 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, 1
  %254 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, 2
  %255 = getelementptr float, ptr %253, i64 %254
  %256 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, 4, 0
  %257 = mul nuw nsw i64 %250, %256
  %258 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, 4, 1
  %259 = mul nuw nsw i64 %246, %258
  %260 = add nuw nsw i64 %257, %259
  %261 = getelementptr inbounds nuw float, ptr %255, i64 %260
  %262 = load float, ptr %261, align 4
  %263 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %244, 1
  %264 = mul nuw nsw i64 %246, 64
  %265 = add nuw nsw i64 %264, %250
  %266 = getelementptr inbounds nuw float, ptr %263, i64 %265
  store float %262, ptr %266, align 4
  %267 = add i64 %250, 1
  br label %249

268:                                              ; preds = %249
  %269 = add i64 %246, 1
  br label %245

270:                                              ; preds = %245
  %271 = call ptr @malloc(i64 320)
  %272 = ptrtoint ptr %271 to i64
  %273 = add i64 %272, 63
  %274 = urem i64 %273, 64
  %275 = sub i64 %273, %274
  %276 = inttoptr i64 %275 to ptr
  %277 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %271, 0
  %278 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %277, ptr %276, 1
  %279 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %278, i64 0, 2
  %280 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %279, i64 1, 3, 0
  %281 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %280, i64 64, 3, 1
  %282 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %281, i64 64, 4, 0
  %283 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %282, i64 1, 4, 1
  %284 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %283, 1
  %285 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %283, 2
  %286 = getelementptr float, ptr %284, i64 %285
  call void @llvm.memcpy.p0.p0.i64(ptr %286, ptr @__constant_1x64xf32, i64 256, i1 false)
  %287 = call ptr @malloc(i64 512)
  %288 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %287, 0
  %289 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %288, ptr %287, 1
  %290 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %289, i64 0, 2
  %291 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %290, i64 1, 3, 0
  %292 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %291, i64 128, 3, 1
  %293 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %292, i64 128, 4, 0
  %294 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %293, i64 1, 4, 1
  %295 = call ptr @malloc(i64 32768)
  %296 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %295, 0
  %297 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %296, ptr %295, 1
  %298 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %297, i64 0, 2
  %299 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %298, i64 128, 3, 0
  %300 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %299, i64 64, 3, 1
  %301 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %300, i64 64, 4, 0
  %302 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %301, i64 1, 4, 1
  %303 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %187, 3, 0
  %304 = mul i64 1, %303
  %305 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %187, 3, 1
  %306 = mul i64 %304, %305
  %307 = mul i64 %306, 4
  %308 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %187, 1
  %309 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %187, 2
  %310 = getelementptr float, ptr %308, i64 %309
  %311 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %294, 1
  %312 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %294, 2
  %313 = getelementptr float, ptr %311, i64 %312
  call void @llvm.memcpy.p0.p0.i64(ptr %313, ptr %310, i64 %307, i1 false)
  %314 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %244, 3, 0
  %315 = mul i64 1, %314
  %316 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %244, 3, 1
  %317 = mul i64 %315, %316
  %318 = mul i64 %317, 4
  %319 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %244, 1
  %320 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %244, 2
  %321 = getelementptr float, ptr %319, i64 %320
  %322 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %302, 1
  %323 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %302, 2
  %324 = getelementptr float, ptr %322, i64 %323
  call void @llvm.memcpy.p0.p0.i64(ptr %324, ptr %321, i64 %318, i1 false)
  call void asm sideeffect "gmm.cfg 1, 64, 128, 0", ""()
  call void asm sideeffect "gmm.type 0, 0, 0", ""()
  call void asm sideeffect "gmm.iter 1, 1", ""()
  %325 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %283, 1
  %326 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %294, 1
  %327 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %302, 1
  call void asm sideeffect "gmma.mm $0, $1, $2", "r,r,r"(ptr %325, ptr %326, ptr %327)
  call void asm sideeffect "tcsync", ""()
  %328 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %294, 0
  call void @free(ptr %328)
  %329 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %302, 0
  call void @free(ptr %329)
  %330 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 0
  %331 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 1
  %332 = insertvalue { ptr, ptr, i64 } poison, ptr %330, 0
  %333 = insertvalue { ptr, ptr, i64 } %332, ptr %331, 1
  %334 = insertvalue { ptr, ptr, i64 } %333, i64 0, 2
  %335 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 2
  %336 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 3, 0
  %337 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 4, 0
  %338 = mul nsw i64 %337, 64
  %339 = extractvalue { ptr, ptr, i64 } %334, 0
  %340 = extractvalue { ptr, ptr, i64 } %334, 1
  %341 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %339, 0
  %342 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %341, ptr %340, 1
  %343 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %342, i64 %335, 2
  %344 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %343, i64 1, 3, 0
  %345 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %344, i64 %338, 4, 0
  %346 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %345, i64 64, 3, 1
  %347 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %346, i64 %337, 4, 1
  %348 = call ptr @malloc(i64 320)
  %349 = ptrtoint ptr %348 to i64
  %350 = add i64 %349, 63
  %351 = urem i64 %350, 64
  %352 = sub i64 %350, %351
  %353 = inttoptr i64 %352 to ptr
  %354 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %348, 0
  %355 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %354, ptr %353, 1
  %356 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %355, i64 0, 2
  %357 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %356, i64 1, 3, 0
  %358 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %357, i64 64, 3, 1
  %359 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %358, i64 64, 4, 0
  %360 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %359, i64 1, 4, 1
  br label %361

361:                                              ; preds = %390, %270
  %362 = phi i64 [ %391, %390 ], [ 0, %270 ]
  %363 = icmp slt i64 %362, 1
  br i1 %363, label %364, label %392

364:                                              ; preds = %361
  br label %365

365:                                              ; preds = %368, %364
  %366 = phi i64 [ %389, %368 ], [ 0, %364 ]
  %367 = icmp slt i64 %366, 64
  br i1 %367, label %368, label %390

368:                                              ; preds = %365
  %369 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %347, 1
  %370 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %347, 2
  %371 = getelementptr float, ptr %369, i64 %370
  %372 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %347, 4, 0
  %373 = mul nuw nsw i64 %362, %372
  %374 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %347, 4, 1
  %375 = mul nuw nsw i64 %366, %374
  %376 = add nuw nsw i64 %373, %375
  %377 = getelementptr inbounds nuw float, ptr %371, i64 %376
  %378 = load float, ptr %377, align 4
  %379 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %283, 1
  %380 = mul nuw nsw i64 %362, 64
  %381 = add nuw nsw i64 %380, %366
  %382 = getelementptr inbounds nuw float, ptr %379, i64 %381
  %383 = load float, ptr %382, align 4
  %384 = fadd float %378, %383
  %385 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %360, 1
  %386 = mul nuw nsw i64 %362, 64
  %387 = add nuw nsw i64 %386, %366
  %388 = getelementptr inbounds nuw float, ptr %385, i64 %387
  store float %384, ptr %388, align 4
  %389 = add i64 %366, 1
  br label %365

390:                                              ; preds = %365
  %391 = add i64 %362, 1
  br label %361

392:                                              ; preds = %361
  %393 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %75, 0
  call void @free(ptr %393)
  %394 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, 0
  call void @free(ptr %394)
  %395 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %187, 0
  call void @free(ptr %395)
  %396 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %244, 0
  call void @free(ptr %396)
  %397 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %283, 0
  call void @free(ptr %397)
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %360
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
