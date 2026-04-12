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
  call void @llvm.riscv.ada300.set.gmm.cfg(i64 4227073, i64 0)
  call void @llvm.riscv.ada300.set.gmm.type(i64 0, i64 0)
  call void @llvm.riscv.ada300.set.gmm.iter(i64 1, i64 1)
  %152 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, 1
  %153 = ptrtoint ptr %152 to i64
  %154 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %125, 1
  %155 = ptrtoint ptr %154 to i64
  %156 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %133, 1
  %157 = ptrtoint ptr %156 to i64
  call void @llvm.riscv.ada300.gmma.mm(i64 %153, i64 %155, i64 %157)
  call void @llvm.riscv.ada300.tcsync()
  %158 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %125, 0
  call void @free(ptr %158)
  %159 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %133, 0
  call void @free(ptr %159)
  %160 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 0
  %161 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 1
  %162 = insertvalue { ptr, ptr, i64 } poison, ptr %160, 0
  %163 = insertvalue { ptr, ptr, i64 } %162, ptr %161, 1
  %164 = insertvalue { ptr, ptr, i64 } %163, i64 0, 2
  %165 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 2
  %166 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 3, 0
  %167 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, 4, 0
  %168 = mul nsw i64 %167, 128
  %169 = extractvalue { ptr, ptr, i64 } %164, 0
  %170 = extractvalue { ptr, ptr, i64 } %164, 1
  %171 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %169, 0
  %172 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %171, ptr %170, 1
  %173 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %172, i64 %165, 2
  %174 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %173, i64 1, 3, 0
  %175 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %174, i64 %168, 4, 0
  %176 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %175, i64 128, 3, 1
  %177 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %176, i64 %167, 4, 1
  %178 = call ptr @malloc(i64 576)
  %179 = ptrtoint ptr %178 to i64
  %180 = add i64 %179, 63
  %181 = urem i64 %180, 64
  %182 = sub i64 %180, %181
  %183 = inttoptr i64 %182 to ptr
  %184 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %178, 0
  %185 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %184, ptr %183, 1
  %186 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %185, i64 0, 2
  %187 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %186, i64 1, 3, 0
  %188 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %187, i64 128, 3, 1
  %189 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %188, i64 128, 4, 0
  %190 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %189, i64 1, 4, 1
  br label %191

191:                                              ; preds = %220, %101
  %192 = phi i64 [ %221, %220 ], [ 0, %101 ]
  %193 = icmp slt i64 %192, 1
  br i1 %193, label %194, label %222

194:                                              ; preds = %191
  br label %195

195:                                              ; preds = %198, %194
  %196 = phi i64 [ %219, %198 ], [ 0, %194 ]
  %197 = icmp slt i64 %196, 128
  br i1 %197, label %198, label %220

198:                                              ; preds = %195
  %199 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %177, 1
  %200 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %177, 2
  %201 = getelementptr float, ptr %199, i64 %200
  %202 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %177, 4, 0
  %203 = mul nuw nsw i64 %192, %202
  %204 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %177, 4, 1
  %205 = mul nuw nsw i64 %196, %204
  %206 = add nuw nsw i64 %203, %205
  %207 = getelementptr inbounds nuw float, ptr %201, i64 %206
  %208 = load float, ptr %207, align 4
  %209 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, 1
  %210 = mul nuw nsw i64 %192, 128
  %211 = add nuw nsw i64 %210, %196
  %212 = getelementptr inbounds nuw float, ptr %209, i64 %211
  %213 = load float, ptr %212, align 4
  %214 = fadd float %208, %213
  %215 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %190, 1
  %216 = mul nuw nsw i64 %192, 128
  %217 = add nuw nsw i64 %216, %196
  %218 = getelementptr inbounds nuw float, ptr %215, i64 %217
  store float %214, ptr %218, align 4
  %219 = add i64 %196, 1
  br label %195

220:                                              ; preds = %195
  %221 = add i64 %192, 1
  br label %191

222:                                              ; preds = %191
  %223 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %190, 1
  %224 = getelementptr float, ptr %223, i64 0
  %225 = load <128 x float>, ptr %224, align 4
  %226 = alloca <128 x float>, align 512
  store <128 x float> %225, ptr %226, align 512
  %227 = ptrtoint ptr %226 to i64
  %228 = call i64 @llvm.riscv.ada300.vfpwnl(i64 %227, i64 0, i64 16)
  %229 = inttoptr i64 %228 to ptr
  %230 = load <128 x float>, ptr %229, align 512
  %231 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %190, 1
  %232 = getelementptr float, ptr %231, i64 0
  store <128 x float> %230, ptr %232, align 4
  %233 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %190, 1
  %234 = getelementptr float, ptr %233, i64 0
  %235 = load <128 x float>, ptr %234, align 4
  %236 = alloca <128 x float>, align 512
  store <128 x float> %235, ptr %236, align 512
  %237 = ptrtoint ptr %236 to i64
  %238 = call i64 @llvm.riscv.ada300.vfpwnl(i64 %237, i64 2, i64 16)
  %239 = inttoptr i64 %238 to ptr
  %240 = load <128 x float>, ptr %239, align 512
  %241 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %190, 1
  %242 = getelementptr float, ptr %241, i64 0
  store <128 x float> %240, ptr %242, align 4
  %243 = call ptr @malloc(i64 32832)
  %244 = ptrtoint ptr %243 to i64
  %245 = add i64 %244, 63
  %246 = urem i64 %245, 64
  %247 = sub i64 %245, %246
  %248 = inttoptr i64 %247 to ptr
  %249 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %243, 0
  %250 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %249, ptr %248, 1
  %251 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %250, i64 0, 2
  %252 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %251, i64 128, 3, 0
  %253 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %252, i64 64, 3, 1
  %254 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %253, i64 64, 4, 0
  %255 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %254, i64 1, 4, 1
  br label %256

256:                                              ; preds = %279, %222
  %257 = phi i64 [ %280, %279 ], [ 0, %222 ]
  %258 = icmp slt i64 %257, 128
  br i1 %258, label %259, label %281

259:                                              ; preds = %256
  br label %260

260:                                              ; preds = %263, %259
  %261 = phi i64 [ %278, %263 ], [ 0, %259 ]
  %262 = icmp slt i64 %261, 64
  br i1 %262, label %263, label %279

263:                                              ; preds = %260
  %264 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, 1
  %265 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, 2
  %266 = getelementptr float, ptr %264, i64 %265
  %267 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, 4, 0
  %268 = mul nuw nsw i64 %261, %267
  %269 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, 4, 1
  %270 = mul nuw nsw i64 %257, %269
  %271 = add nuw nsw i64 %268, %270
  %272 = getelementptr inbounds nuw float, ptr %266, i64 %271
  %273 = load float, ptr %272, align 4
  %274 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %255, 1
  %275 = mul nuw nsw i64 %257, 64
  %276 = add nuw nsw i64 %275, %261
  %277 = getelementptr inbounds nuw float, ptr %274, i64 %276
  store float %273, ptr %277, align 4
  %278 = add i64 %261, 1
  br label %260

279:                                              ; preds = %260
  %280 = add i64 %257, 1
  br label %256

281:                                              ; preds = %256
  %282 = call ptr @malloc(i64 320)
  %283 = ptrtoint ptr %282 to i64
  %284 = add i64 %283, 63
  %285 = urem i64 %284, 64
  %286 = sub i64 %284, %285
  %287 = inttoptr i64 %286 to ptr
  %288 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %282, 0
  %289 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %288, ptr %287, 1
  %290 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %289, i64 0, 2
  %291 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %290, i64 1, 3, 0
  %292 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %291, i64 64, 3, 1
  %293 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %292, i64 64, 4, 0
  %294 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %293, i64 1, 4, 1
  %295 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %294, 1
  %296 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %294, 2
  %297 = getelementptr float, ptr %295, i64 %296
  call void @llvm.memcpy.p0.p0.i64(ptr %297, ptr @__constant_1x64xf32, i64 256, i1 false)
  %298 = call ptr @malloc(i64 512)
  %299 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %298, 0
  %300 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %299, ptr %298, 1
  %301 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %300, i64 0, 2
  %302 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %301, i64 1, 3, 0
  %303 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %302, i64 128, 3, 1
  %304 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %303, i64 128, 4, 0
  %305 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %304, i64 1, 4, 1
  %306 = call ptr @malloc(i64 32768)
  %307 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %306, 0
  %308 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %307, ptr %306, 1
  %309 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %308, i64 0, 2
  %310 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %309, i64 128, 3, 0
  %311 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %310, i64 64, 3, 1
  %312 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %311, i64 64, 4, 0
  %313 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %312, i64 1, 4, 1
  %314 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %190, 3, 0
  %315 = mul i64 1, %314
  %316 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %190, 3, 1
  %317 = mul i64 %315, %316
  %318 = mul i64 %317, 4
  %319 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %190, 1
  %320 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %190, 2
  %321 = getelementptr float, ptr %319, i64 %320
  %322 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %305, 1
  %323 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %305, 2
  %324 = getelementptr float, ptr %322, i64 %323
  call void @llvm.memcpy.p0.p0.i64(ptr %324, ptr %321, i64 %318, i1 false)
  %325 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %255, 3, 0
  %326 = mul i64 1, %325
  %327 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %255, 3, 1
  %328 = mul i64 %326, %327
  %329 = mul i64 %328, 4
  %330 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %255, 1
  %331 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %255, 2
  %332 = getelementptr float, ptr %330, i64 %331
  %333 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %313, 1
  %334 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %313, 2
  %335 = getelementptr float, ptr %333, i64 %334
  call void @llvm.memcpy.p0.p0.i64(ptr %335, ptr %332, i64 %329, i1 false)
  call void @llvm.riscv.ada300.set.gmm.cfg(i64 8404993, i64 0)
  call void @llvm.riscv.ada300.set.gmm.type(i64 0, i64 0)
  call void @llvm.riscv.ada300.set.gmm.iter(i64 1, i64 1)
  %336 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %294, 1
  %337 = ptrtoint ptr %336 to i64
  %338 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %305, 1
  %339 = ptrtoint ptr %338 to i64
  %340 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %313, 1
  %341 = ptrtoint ptr %340 to i64
  call void @llvm.riscv.ada300.gmma.mm(i64 %337, i64 %339, i64 %341)
  call void @llvm.riscv.ada300.tcsync()
  %342 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %305, 0
  call void @free(ptr %342)
  %343 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %313, 0
  call void @free(ptr %343)
  %344 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 0
  %345 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 1
  %346 = insertvalue { ptr, ptr, i64 } poison, ptr %344, 0
  %347 = insertvalue { ptr, ptr, i64 } %346, ptr %345, 1
  %348 = insertvalue { ptr, ptr, i64 } %347, i64 0, 2
  %349 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 2
  %350 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 3, 0
  %351 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, 4, 0
  %352 = mul nsw i64 %351, 64
  %353 = extractvalue { ptr, ptr, i64 } %348, 0
  %354 = extractvalue { ptr, ptr, i64 } %348, 1
  %355 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %353, 0
  %356 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %355, ptr %354, 1
  %357 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %356, i64 %349, 2
  %358 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %357, i64 1, 3, 0
  %359 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %358, i64 %352, 4, 0
  %360 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %359, i64 64, 3, 1
  %361 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %360, i64 %351, 4, 1
  %362 = call ptr @malloc(i64 320)
  %363 = ptrtoint ptr %362 to i64
  %364 = add i64 %363, 63
  %365 = urem i64 %364, 64
  %366 = sub i64 %364, %365
  %367 = inttoptr i64 %366 to ptr
  %368 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %362, 0
  %369 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %368, ptr %367, 1
  %370 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %369, i64 0, 2
  %371 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %370, i64 1, 3, 0
  %372 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %371, i64 64, 3, 1
  %373 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %372, i64 64, 4, 0
  %374 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %373, i64 1, 4, 1
  br label %375

375:                                              ; preds = %404, %281
  %376 = phi i64 [ %405, %404 ], [ 0, %281 ]
  %377 = icmp slt i64 %376, 1
  br i1 %377, label %378, label %406

378:                                              ; preds = %375
  br label %379

379:                                              ; preds = %382, %378
  %380 = phi i64 [ %403, %382 ], [ 0, %378 ]
  %381 = icmp slt i64 %380, 64
  br i1 %381, label %382, label %404

382:                                              ; preds = %379
  %383 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %361, 1
  %384 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %361, 2
  %385 = getelementptr float, ptr %383, i64 %384
  %386 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %361, 4, 0
  %387 = mul nuw nsw i64 %376, %386
  %388 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %361, 4, 1
  %389 = mul nuw nsw i64 %380, %388
  %390 = add nuw nsw i64 %387, %389
  %391 = getelementptr inbounds nuw float, ptr %385, i64 %390
  %392 = load float, ptr %391, align 4
  %393 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %294, 1
  %394 = mul nuw nsw i64 %376, 64
  %395 = add nuw nsw i64 %394, %380
  %396 = getelementptr inbounds nuw float, ptr %393, i64 %395
  %397 = load float, ptr %396, align 4
  %398 = fadd float %392, %397
  %399 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %374, 1
  %400 = mul nuw nsw i64 %376, 64
  %401 = add nuw nsw i64 %400, %380
  %402 = getelementptr inbounds nuw float, ptr %399, i64 %401
  store float %398, ptr %402, align 4
  %403 = add i64 %380, 1
  br label %379

404:                                              ; preds = %379
  %405 = add i64 %376, 1
  br label %375

406:                                              ; preds = %375
  %407 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %75, 0
  call void @free(ptr %407)
  %408 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, 0
  call void @free(ptr %408)
  %409 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %190, 0
  call void @free(ptr %409)
  %410 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %255, 0
  call void @free(ptr %410)
  %411 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %294, 0
  call void @free(ptr %411)
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %374
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #1

; Function Attrs: nounwind
declare void @llvm.riscv.ada300.set.gmm.cfg(i64, i64) #2

; Function Attrs: nounwind
declare void @llvm.riscv.ada300.set.gmm.type(i64, i64) #2

; Function Attrs: nounwind
declare void @llvm.riscv.ada300.set.gmm.iter(i64, i64) #2

; Function Attrs: nounwind
declare void @llvm.riscv.ada300.gmma.mm(i64, i64, i64) #2

; Function Attrs: nounwind
declare void @llvm.riscv.ada300.tcsync() #2

; Function Attrs: nounwind
declare i64 @llvm.riscv.ada300.vfpwnl(i64, i64, i64) #2

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nocallback nofree nosync nounwind willreturn }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
