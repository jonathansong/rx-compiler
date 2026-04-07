# ADA300 MLIR Dialect Design Document

## 1. 目标

本文档给出一套面向 ADA300 Vector / Tensor ISA 的 MLIR dialect 设计方案，重点覆盖：

- dialect 分层
- ops 设计
- attributes 设计
- types 设计
- memory / memory space 建模
- verifier 与 side effects
- lowering pipeline
- 最小可行实现路径

该设计基于 ADA300 ISA 文档中的几个关键事实：

- ADA300 ISA 以 **RISC-V 64IF + RISC-V Vector v1.0** 为基础，vector 指令使用 `OP-V` 等 RVV 风格编码。
- VLEN 固定为 2048，LMUL 固定为 1，由软件适配 SEW 和寄存器内元素个数。
- 非线性函数通过 `vfpwnl` 指令和 P/S/B 分段参数表实现，支持 `sqrt / rsqrt / div / log / exp`，并支持 16 段和 32 段近似。
- Tensor Core 指令是 **基于 SRAM 的长周期指令**，依赖 `gmm_* / gmv_*` 配置寄存器和 `tcsync` 同步语义。

---

## 2. 设计原则

### 2.1 不建议只做“汇编一比一映射”

如果所有 op 都直接映射成底层 ISA 指令，IR 会过早失去高层语义，优化空间会明显变小。

例如：

- `math.exp` 如果直接变成 `vfpwnl`，会丢失“这是指数函数”的高层语义。
- `linalg.matmul` 如果立刻拆成 `set_gmm_cfg + set_gmm_type + gmma.mm + tcsync`，则调度、融合和合法化更难做。

因此建议采用 **双层 dialect**：

1. **高层 `Ada300HL` dialect**：表达硬件相关但仍可优化的语义。
2. **低层 `Ada300HW` dialect**：尽量接近真实硬件协议和指令编码。

### 2.2 向量路径和 Tensor 路径分开建模

ADA300 ISA 中存在两类不同执行模型：

- **Vector 路径**：RVV 风格，偏纯计算 op。
- **Tensor 路径**：配置寄存器 + 长时延执行 + 同步等待。

两者在 MLIR 中不应被建模成同一类 op。

### 2.3 明确 memory space 与数据布局

Tensor 指令强依赖：

- SRAM 地址
- 目标地址可为 VR
- row/col/blk 计数
- block 化布局

因此设计中必须显式考虑：

- 普通内存
- SRAM
- 向量寄存器/VR 背后的存储空间
- block layout 和 transpose 语义

---

## 3. 推荐的 dialect 分层

## 3.1 高层：`Ada300HL`

这个 dialect 负责表达"硬件相关语义"，但避免直接暴露具体编码细节。

适合放入的 op：

- `ada300hl.pwnl`（涵盖 exp/log/sqrt/rsqrt/div，通过 `func` 属性选择）
- `ada300hl.cvt`
- `ada300hl.vmul_mixed`
- `ada300hl.tensor_mma`
- `ada300hl.tensor_sync`
- `ada300hl.pack`
- `ada300hl.unpack`

特点：

- 更接近 `math` / `vector` / `linalg` / `tosa` / `stablehlo` 的语义
- 便于做 pattern rewrite、fusion、合法化和成本模型选择
- 可通过 attributes 表达段数、数据类型、layout 等

## 3.2 低层：`Ada300HW`

这个 dialect 负责表达接近真实硬件的协议和 ISA 语义。

适合放入的 op：

- `ada300hw.vfpwnl`
- `ada300hw.vfcvt`
- `ada300hw.vfmul_low`
- `ada300hw.vfmul_high`
- `ada300hw.set_gmm_cfg`
- `ada300hw.set_gmm_type`
- `ada300hw.set_gmm_iter`
- `ada300hw.gmm_mm`
- `ada300hw.gmma_mm`
- `ada300hw.gmma_mt`
- `ada300hw.gmv_mm`
- `ada300hw.gmva_mm`
- `ada300hw.gmva_mt`
- `ada300hw.tcsync`

特点：

- 接近硬件协议
- 容易映射到 LLVM intrinsic / inline asm / custom backend instruction
- 适合建模 side effect、config register 和同步边界

---

## 4. Dialect 设计

## 4.1 `Ada300HL` dialect

### 定位

高层硬件语义 dialect。

### 目标

- 保留指数、分段非线性、混合精度乘法、Tensor MMA 等语义
- 延后对真实 ISA 协议的展开
- 为后续 pass 留出优化空间

### 依赖接口

建议实现或使用：

- `InferTypeOpInterface`
- `MemoryEffectOpInterface`（仅对有副作用 op）
- `ConditionallySpeculatable`
- `OpAsmOpInterface`
- `DestinationStyleOpInterface`（若 tensor op 采用 destination-style）

## 4.2 `Ada300HW` dialect

### 定位

低层 ISA / 协处理器协议 dialect。

### 目标

- 精确表达 `vfpwnl`、`gmm/gmma/gmv/gmva` 和 `tcsync`
- 显式表示 config register 协议
- 为最终 codegen 提供稳定中间层

### 建议接口

- `MemoryEffectOpInterface`
- `OpAsmOpInterface`
- `SymbolUserOpInterface`（若 P/S/B 表或配置模板用 symbol 表达）

---

## 5. Attributes 设计

为了避免把关键信息编码进字符串，推荐使用 enum / structured attributes。

## 5.1 非线性函数属性

```tablegen
ADA300_NonlinearFuncAttr ::= exp | log | sqrt | rsqrt | div
```

用途：

- `ada300hl.pwnl` 通过 `func = #ada300hl.nlfunc<exp>` 表达指数函数，无需单独的 `exp` op
- `ada300hw.vfpwnl` 同样显式携带该属性

## 5.2 分段数属性

```tablegen
ADA300_SegmentCountAttr ::= 16 | 32
```

用途：

- 描述 `vfpwnl` 的 16 段 / 32 段近似
- verifier 中限制只允许 16 和 32

## 5.3 Tensor 输出模式属性

```tablegen
ADA300_TensorModeAttr ::= inner | outer
```

用途：

- 对应 Tensor Core 内积输出模式 / 外积输出模式

## 5.4 Tensor 数据类型属性

```tablegen
ADA300_TensorDataTypeAttr ::= fp16 | bf16 | int16 | fp8 | int8 | int4
```

用途：

- 对应 `gmm_type` / `gmv_type` 的 `data_out_type / data_act_type / data_wht_type`

## 5.5 布局属性

```tablegen
ADA300_LayoutAttr ::= {
  block_m: i32,
  block_n: i32,
  block_k: i32,
  rhs_transposed: bool,
  major: enum<row_major, col_major, blocked>
}
```

用途：

- 描述 block layout
- 描述 `.mm` vs `.mt`
- 为 pack / unpack 与 tensor lowering 服务

## 5.6 Memory space 属性

```tablegen
ADA300_MemorySpaceAttr ::= global | sram | vr
```

用途：

- 区分普通内存、SRAM 和 VR-backed buffer

## 5.7 PWL 参数表属性（可选）

如果选择在 IR 中显式引用 P/S/B 表，可以定义：

```tablegen
ADA300_PwlTableAttr ::= {
  func: ADA300_NonlinearFuncAttr,
  segments: ADA300_SegmentCountAttr,
  table_symbol: SymbolRefAttr
}
```

更推荐将表建模成外部 symbol 或常量 buffer，而不是把全部 49 个或更多参数直接塞进 op 属性。

---

## 6. Types 设计

## 6.1 尽量复用内建类型

可直接复用的类型：

- `f32`
- `f16`
- `bf16`
- `i8`
- `i16`
- `i32`
- `vector<...>`
- `memref<...>`
- `tensor<...>`

## 6.2 自定义 FP8 类型（推荐）

如果需要严肃支持 `vfcvt.low/high.*` 和混合精度路径，建议定义：

```tablegen
!ada300hl.fp8<e4m3>
```

也可以在初版里先复用已有 FP8 支持（如果工具链已有对应类型），否则使用自定义 ADA300 类型更清晰。

## 6.3 I4 / 打包低精度类型

对于 INT4：

- 若上层仅表达数学语义，可使用 `i4`
- 若要表达 pack 语义，则建议通过 layout / pack op 显式建模，不把 pack 约束直接塞进基础类型

## 6.4 向量寄存器类型（可选）

如果你希望更接近 ISA，可以定义：

```tablegen
!ada300hl.vreg<elem_type, lanes>
```

但在大多数情况下，更建议继续使用 MLIR 标准 `vector<...>`，把 VLEN / SEW / packing 作为 lowering 约束，而不是一开始就引入过强的寄存器语义。

---

## 7. Memory 设计

这是 ADA300 dialect 设计里最重要的部分之一。

## 7.1 推荐的 memory space 方案

建议至少区分 3 类：

- `global`：普通 DRAM / host-visible memory
- `sram`：Tensor Core 使用的本地 SRAM
- `vr`：向量寄存器或向量寄存器支持的特殊存储空间

可以通过 memory space ID 或自定义 attribute 建模，例如：

```mlir
memref<128x128xf16, #ada300hl.memory_space<sram>>
memref<64x64x!ada300hl.fp8<e4m3>, #ada300hl.memory_space<global>>
memref<1xvector<64xf16>, #ada300hl.memory_space<vr>>
```

## 7.2 Tensor op 必须考虑 SRAM

由于 Tensor Core 指令的输入输出地址语义明确依赖 SRAM，因此 Tensor 路径要么：

- 在 `ada300hl.tensor_mma` 层显式要求输入输出 buffer 位于 `sram`
- 要么在 lowering 前插入 `copy_to_sram` / `copy_from_sram`

推荐新增：

- `ada300hl.copy_to_sram`
- `ada300hl.copy_from_sram`
- `ada300hl.copy_to_vr`
- `ada300hl.copy_from_vr`

## 7.3 Layout 与 memory 一起建模

Tensor Core 并不是对任意 row-major buffer 都直接可用。
因此推荐配套：

- `ada300hl.pack`
- `ada300hl.unpack`
- `ada300hl.layout_cast`

以保证 block 化 layout 在 lowering 过程中明确可见。

---

## 8. Ops 设计

## 8.1 高层 `Ada300HL` ops

### 8.1.1 `ada300hl.pwnl`

#### 语义

泛化分段非线性 op，对应 exp/log/sqrt/rsqrt/div。

#### 示例

```mlir
%y = ada300hl.pwnl %x {
  func = #ada300hl.nlfunc<exp>,
  segments = 32
} : vector<64xf32> -> vector<64xf32>
```

---

### 8.1.3 `ada300hl.cvt`

#### 语义

表达低精度 / 混合精度转换。

#### 示例

```mlir
%y = ada300hl.cvt %x {
  dst_type = #ada300hl.dtype<fp8>,
  low_high = #ada300hl.part<low>
} : vector<64xf16> -> vector<64x!ada300hl.fp8<e4m3>>
```

---

### 8.1.4 `ada300hl.vmul_mixed`

#### 语义

表达类似 `vfmul.low/high.*` 的混合精度乘法，但先不暴露底层编码。

#### 示例

```mlir
%z = ada300hl.vmul_mixed %a, %b {
  acc_type = f32,
  part = #ada300hl.part<low>
} : vector<64xbf16>, vector<64xbf16> -> vector<64xf32>
```

---

### 8.1.5 `ada300hl.tensor_mma`

#### 语义

表达 Tensor Core 矩阵乘 / 乘加。

#### 示例

```mlir
ada300hl.tensor_mma %dst, %a, %w {
  mode = #ada300hl.tensor_mode<inner>,
  rhs_transposed = false,
  row_size = 16,
  col_size = 16,
  acc_size = 16,
  out_type = #ada300hl.dtype<bf16>,
  act_type = #ada300hl.dtype<bf16>,
  wht_type = #ada300hl.dtype<bf16>,
  blk_cnt_a = 4,
  blk_cnt_w = 4
} : memref<...>, memref<...>, memref<...>
```

#### trait / interface

- 不是 `Pure`
- `MemoryEffectOpInterface`
- destination-style 更自然

---

### 8.1.6 `ada300hl.tensor_sync`

#### 语义

抽象同步 Tensor Core 完成状态。

#### 示例

```mlir
ada300hl.tensor_sync
```

---

### 8.1.7 `ada300hl.pack` / `ada300hl.unpack`

#### 语义

显式表达为 Tensor Core 需要的 block layout 做 pack / unpack。

#### 示例

```mlir
%packed = ada300hl.pack %src {
  layout = #ada300hl.layout<blocked, block_m = 16, block_n = 16, block_k = 16>
} : memref<64x64xf16> -> memref<4x4x16x16xf16, #ada300hl.memory_space<sram>>
```

---

## 8.2 低层 `Ada300HW` ops

### 8.2.1 `ada300hw.vfpwnl`

#### 语义

直接对应 `vfpwnl` 指令。

#### 示例

```mlir
%y = ada300hw.vfpwnl %x {
  func = #ada300hl.nlfunc<exp>,
  segments = 16,
  masked = false
} : vector<64xf16> -> vector<64xf16>
```

#### 说明

- `func` 对应 ISA 中立即数字段选择的 nonlinear type
- `segments` 对应 16 / 32 段

---

### 8.2.2 `ada300hw.vfcvt`

#### 语义

对应 `vfcvt.*` 指令族。

#### 示例

```mlir
%y = ada300hw.vfcvt %x {
  dst_type = #ada300hl.dtype<fp8>,
  part = #ada300hl.part<high>
} : vector<64xf16> -> vector<64x!ada300hl.fp8<e4m3>>
```

---

### 8.2.3 `ada300hw.vfmul_low` / `vfmul_high`

#### 语义

对应 `vfmul.low/high.*` 指令。

#### 示例

```mlir
%z = ada300hw.vfmul_low %a, %b {
  src_a_type = #ada300hl.dtype<bf16>,
  src_b_type = #ada300hl.dtype<bf16>,
  dst_type = #ada300hl.dtype<f32>
} : vector<64xbf16>, vector<64xbf16> -> vector<64xf32>
```

---

### 8.2.4 Tensor 配置 op

#### `ada300hw.set_gmm_cfg`

```mlir
ada300hw.set_gmm_cfg {
  row_size = 16,
  col_size = 16,
  acc_size = 16,
  mode = #ada300hl.tensor_mode<inner>
}
```

#### `ada300hw.set_gmm_type`

```mlir
ada300hw.set_gmm_type {
  out_type = #ada300hl.dtype<bf16>,
  act_type = #ada300hl.dtype<bf16>,
  wht_type = #ada300hl.dtype<bf16>
}
```

#### `ada300hw.set_gmm_iter`

```mlir
ada300hw.set_gmm_iter {
  blk_cnt_a = 4,
  blk_cnt_w = 4
}
```

这些 op 应视为有副作用，因为它们写入硬件配置寄存器。

---

### 8.2.5 `ada300hw.gmm_mm` / `gmma_mm` / `gmma_mt`

#### 语义

对应 Tensor Core 指令本身。

#### 示例

```mlir
ada300hw.gmma_mm %dst, %a, %w
  : memref<... , #ada300hl.memory_space<sram>>,
    memref<... , #ada300hl.memory_space<sram>>,
    memref<... , #ada300hl.memory_space<sram>>
```

#### 说明

- 要求输入输出地址处于合法 memory space
- 与前序 config op 共同决定执行行为

---

### 8.2.6 `ada300hw.tcsync`

#### 语义

对应 `tcsync`，阻塞直到完成位为 1。

#### trait / interface

- 显式 side effect
- 不可随意删去或移动

---

## 9. Verifier 设计

## 9.1 `ada300hl.pwnl`

需要检查：

- `segments` 只能为 16 或 32
- 输入输出元素类型是否在支持列表内
- 如果引用参数表，表的 `func` / `segments` 必须与 op 一致

## 9.2 `ada300hl.tensor_mma`

需要检查：

- `row_size / col_size / acc_size` 为正
- 数据类型与尺寸上限匹配
- `.mt` 语义与 `rhs_transposed` / layout 一致
- `blk_cnt_a / blk_cnt_w` 合法
- 输入输出 buffer 位于支持的 memory space

## 9.3 `ada300hw.vfcvt`

需要检查：

- `dst_type` 与输入类型组合是 ISA 支持的组合
- `part` 只能为 `low` 或 `high`

## 9.4 `ada300hw.gmm*`

需要检查：

- 是否在当前块中存在先行配置 op，或通过 SSA token 建模配置依赖
- 目标地址与输入地址类型、memory space、布局是否合法

---

## 10. Side Effects 与执行模型

## 10.1 可以视为纯计算的 op

通常可先视为纯 op：

- `ada300hl.pwnl`
- `ada300hl.cvt`
- `ada300hl.vmul_mixed`
- `ada300hw.vfpwnl`
- `ada300hw.vfcvt`

前提是它们仅从 SSA 输入产生 SSA 输出，不显式读写外部状态。

## 10.2 必须建模 side effect 的 op

必须建模成有副作用：

- `ada300hl.tensor_mma`
- `ada300hl.tensor_sync`
- `ada300hl.copy_to_sram`
- `ada300hl.copy_from_sram`
- `ada300hw.set_gmm_cfg`
- `ada300hw.set_gmm_type`
- `ada300hw.set_gmm_iter`
- `ada300hw.gmm_mm`
- `ada300hw.gmma_mm`
- `ada300hw.gmma_mt`
- `ada300hw.gmv_mm`
- `ada300hw.gmva_mm`
- `ada300hw.gmva_mt`
- `ada300hw.tcsync`

原因：

- 它们读写 SRAM / 寄存器状态
- 涉及长时延执行
- 涉及同步和顺序约束

---

## 11. 典型 lowering pipeline

推荐 pipeline：

```text
math / arith / vector / linalg / tosa / stablehlo
    ↓
Ada300HL
    ↓
Ada300HW
    ↓
LLVM dialect / custom intrinsic / inline asm
    ↓
RISC-V backend / ADA300 backend
```

## 11.1 `ada300hl.pwnl` lowering 路径

```text
math.exp / math.log / math.sqrt / math.rsqrt
  ↓  (MathToAda300HL)
ada300hl.pwnl { func = <exp|log|sqrt|rsqrt|div>, segments = 16/32 }
  ↓  (LowerAda300HLToAda300HW)
ada300hw.vfpwnl { func = <exp|log|sqrt|rsqrt|div>, segments = 16/32 }
  ↓  (LowerAda300HWToLLVM)
LLVM intrinsic / custom instruction emission
```

## 11.2 `linalg.matmul` lowering 路径

```text
linalg.matmul
  ↓
ada300hl.pack + ada300hl.tensor_mma + ada300hl.tensor_sync
  ↓
ada300hw.set_gmm_cfg
ada300hw.set_gmm_type
ada300hw.set_gmm_iter
ada300hw.gmma_mm or gmma_mt
ada300hw.tcsync
  ↓
LLVM / backend emission
```

---

## 12. 最小可行实现（MVP）

如果现在开始做，推荐第一阶段只实现以下内容。

## 12.1 Dialect

- `Ada300HL`
- `Ada300HW`

## 12.2 Attributes

- `NonlinearFuncAttr`
- `SegmentCountAttr`
- `TensorModeAttr`
- `TensorDataTypeAttr`
- `MemorySpaceAttr`
- `LayoutAttr`

## 12.3 Types

- 复用 `f16`, `bf16`, `f32`, `i8`, `i4`
- 可选实现 `!ada300hl.fp8<e4m3>`

## 12.4 Ops

- `ada300hl.pwnl`（`func` 属性选择 exp/log/sqrt/rsqrt/div）
- `ada300hl.tensor_mma`
- `ada300hl.pack`
- `ada300hw.vfpwnl`
- `ada300hw.set_gmm_cfg`
- `ada300hw.set_gmm_type`
- `ada300hw.set_gmm_iter`
- `ada300hw.gmma_mm`
- `ada300hw.tcsync`

## 12.5 Passes

- `MathToAda300HL`
- `LinalgToAda300HL`
- `Ada300HLToAda300HW`
- `Ada300HWToLLVM`

---

## 13. 建议的目录结构

```text
mlir/
  include/
    mlir/Dialect/Ada300HL/
      Ada300HLDialect.td
      Ada300HLOps.td
      Ada300HLAttrs.td
      Ada300HLTypes.td
      Ada300HLMemory.td
    mlir/Dialect/Ada300HW/
      Ada300HWDialect.td
      Ada300HWOps.td
      Ada300HWAttrs.td
  lib/
    Dialect/
      Ada300HL/
        Ada300HLDialect.cpp
        Ada300HLOps.cpp
        Ada300HLAttrs.cpp
        Ada300HLTypes.cpp
      Ada300HW/
        Ada300HWDialect.cpp
        Ada300HWOps.cpp
    Conversion/
      MathToAda300HL/
      LinalgToAda300HL/
      Ada300HLToAda300HW/
      Ada300HWToLLVM/
  test/
    Dialect/Ada300HL/
    Dialect/Ada300HW/
    Conversion/
```

---

## 14. 结论

对于 ADA300 这类基于 RVV 向量扩展、同时又包含 Tensor Core 长时延协处理器协议的架构，最合理的 MLIR 方案不是单层“指令方言”，而是：

- 用 **高层 `Ada300HL` dialect** 保存 `exp`、PWL、混合精度转换、tensor MMA、pack/unpack 等硬件相关语义；
- 用 **低层 `Ada300HW` dialect** 精确建模 `vfpwnl`、`vfcvt`、`gmm/gmma/gmv/gmva`、`gmm_*` 配置寄存器和 `tcsync`；
- 在 type、attribute、memory 和 layout 上提前建立清晰的语义边界；
- 将 pure vector path 与 side-effect-heavy Tensor path 分开建模。

这个分层可以同时兼顾：

- 上层 IR 的可优化性
- 硬件协议表达能力
- 后续 LLVM / backend codegen 的可落地性

