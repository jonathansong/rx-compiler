# ADA300 方言实现指南

本文档描述了两个 ADA300 MLIR 方言（`Ada300HL` 和 `Ada300HW`）以及将它们连接到更广泛的 MLIR/LLVM 流水线的四个转换 Pass 的实现。

---

## 目录

1. [概述](#概述)
2. [目录结构](#目录结构)
3. [Ada300HL 方言](#ada300hl-方言)
   - [属性](#属性)
   - [类型](#类型)
   - [内存约束](#内存约束)
   - [操作](#ada300hl-操作)
4. [Ada300HW 方言](#ada300hw-方言)
   - [操作](#ada300hw-操作)
5. [转换 Pass](#转换-pass)
   - [MathToAda300HL](#1-mathtoadabl300hl)
   - [LinalgToAda300HL](#2-linalgtoadabl300hl)
   - [LowerAda300HLToAda300HW](#3-lowerada300hltoadabl300hw)
   - [LowerAda300HWToLLVM](#4-lowerada300hwtollvm)
6. [完整流水线](#完整流水线)
7. [CMake 集成](#cmake-集成)

---

## 概述

ADA300 ISA 是基于 RISC-V 64IF + RVV v1.0 的处理器，具有两个关键硬件子系统：

- **向量单元** – 类 RVV 的寄存器到寄存器操作，包括分段线性非线性近似指令（`vfpwnl`）和混合精度算术。
- **张量核心** – 基于 SRAM 的长延迟矩阵加速器，由硬件配置寄存器（`gmm_cfg`、`gmm_type`、`gmm_iter`）和执行指令（`gmma_mm`、`gmma_mt` 等）驱动。

为支持对这两个子系统的优化和代码生成，两个 MLIR 方言按如下层次结构叠加：

```
math / arith / vector / linalg
        │
        ▼  MathToAda300HL / LinalgToAda300HL
   ada300hl  ← 高级硬件感知操作（可优化）
        │
        ▼  LowerAda300HLToAda300HW
   ada300hw  ← ISA 级操作（直接硬件协议）
        │
        ▼  LowerAda300HWToLLVM
   LLVM 方言 (llvm.inline_asm)
        │
        ▼
   RISC-V / ADA300 后端
```

---

## 目录结构

```
midend/
  include/
    Dialect/
      Ada300HL/
        Ada300HLDialect.td     # 方言注册 + 基础 Op 类
        Ada300HLAttrs.td       # 7 个枚举属性 + LayoutAttr
        Ada300HLTypes.td       # 自定义 Fp8Type（E4M3）
        Ada300HLMemory.td      # 内存空间类型约束
        Ada300HLOps.td         # 根 tablegen 文件；包含所有子文件
        Ada300HLDialect.h      # C++ 方言头文件
        Ada300HLOps.h          # C++ Op 声明
        Transforms.h           # 降低 Pass 的公共 API
      Ada300HW/
        Ada300HWDialect.td     # 方言注册 + 基础 Op 类
        Ada300HWAttrs.td       # 重新导出 Ada300HL 属性（共享命名空间）
        Ada300HWOps.td         # 根 tablegen 文件；14 个 ISA 级操作
        Ada300HWDialect.h      # C++ 方言头文件
        Ada300HWOps.h          # C++ Op 声明
        Transforms.h           # Ada300HW → LLVM Pass 的公共 API
  lib/
    Dialect/
      Ada300HL/
        Ada300HLDialect.cpp    # 方言注册（ops + types + attrs）
        Ada300HLAttrs.cpp      # 枚举 + AttrDef 实现
        Ada300HLTypes.cpp      # Fp8Type 实现
        Ada300HLOps.cpp        # Op 校验器
      Ada300HW/
        Ada300HWDialect.cpp    # 方言注册
        Ada300HWOps.cpp        # Op 校验器
    Conversion/
      MathToAda300HL/
        MathToAda300HLPass.cpp
        CMakeLists.txt
      LinalgToAda300HL/
        LinalgToAda300HLPass.cpp
        CMakeLists.txt
      LowerAda300HLToAda300HW/
        LowerAda300HLToAda300HWPass.cpp
        CMakeLists.txt
      LowerAda300HWToLLVM/
        LowerAda300HWToLLVMPass.cpp
        CMakeLists.txt
```

---

## Ada300HL 方言

**命名空间：** `::buddy::ada300hl`  
**助记符前缀：** `ada300hl`  
**根 tablegen 文件：** `Ada300HLOps.td`

高级方言在推迟 ISA 特定编码细节的同时捕获硬件相关语义。其操作适合进行模式重写、融合和基于代价模型的选择，然后再降低到 `Ada300HW`。

### 属性

所有属性均在 `Ada300HLAttrs.td` 中定义，并共享 `::buddy::ada300hl` C++ 命名空间。Ada300HW 的 Op 直接复用这些属性（`Ada300HWAttrs.td` 文件是一个薄的重新导出层）。

| 属性 | 助记符 | 取值 | 用途 |
|---|---|---|---|
| `NonlinearFuncAttr` | `#ada300hl.nlfunc<...>` | `exp`、`log`、`sqrt`、`rsqrt`、`div` | 选择 PWL 函数类型 |
| `SegmentCountAttr` | `#ada300hl.segments<...>` | `16`、`32` | PWL 近似分段数 |
| `TensorModeAttr` | `#ada300hl.tensor_mode<...>` | `inner`、`outer` | 张量核心累积模式 |
| `TensorDataTypeAttr` | `#ada300hl.dtype<...>` | `fp16`、`bf16`、`int16`、`fp8`、`int8`、`int4` | gmm/gmv 操作的操作数数据类型 |
| `MemorySpaceAttr` | `#ada300hl.memory_space<...>` | `global`、`sram`、`vr` | 内存域注解 |
| `PartAttr` | `#ada300hl.part<...>` | `low`、`high` | 子向量半部选择 |
| `MajorAttr` | `#ada300hl.major<...>` | `row_major`、`col_major`、`blocked` | 矩阵存储顺序 |
| `LayoutAttr` | `#ada300hl.layout<...>` | `{block_m, block_n, block_k, rhs_transposed, major}` | pack/unpack 的 Tile 布局 |

### 类型

在 `Ada300HLTypes.td` 中定义：

| 类型 | 助记符 | 描述 |
|---|---|---|
| `Fp8Type` | `!ada300hl.fp8<e4m3>` | 8 位浮点，E4M3 格式，用于混合精度路径 |

标准 MLIR 类型（`f16`、`bf16`、`f32`、`i8`、`i16`、`vector<...>`、`memref<...>`）直接复用，不作封装。

### 内存约束

`Ada300HLMemory.td` 定义了 tablegen 类型约束别名，用于操作签名中记录 memref 操作数必须来自哪个内存域：

| 约束 | 含义 |
|---|---|
| `Ada300HL_GlobalMemRef` | 全局 DRAM 中的 `memref<...>` |
| `Ada300HL_SRAMMemRef` | `memref<..., #ada300hl.memory_space<sram>>` |
| `Ada300HL_VRMemRef` | `memref<..., #ada300hl.memory_space<vr>>` |
| `Ada300HL_AnyMemRef` | 上述任意一种 |

### Ada300HL 操作

#### 向量非线性 / PWL 操作

| Op | Traits | 描述 | 示例 |
|---|---|---|---|
| `ada300hl.exp` | `Pure` | 通过 PWL 单元的逐元素 `exp(x)` | `%y = ada300hl.exp %x {segments = #ada300hl.segments<16>} : vector<64xf16> -> vector<64xf16>` |
| `ada300hl.pwnl` | `Pure` | 通用 PWL（exp/log/sqrt/rsqrt/div） | `%y = ada300hl.pwnl %x {func = #ada300hl.nlfunc<sqrt>, segments = #ada300hl.segments<32>} : ...` |
| `ada300hl.cvt` | `Pure` | 混合精度类型转换 | `%y = ada300hl.cvt %x {dst_type = #ada300hl.dtype<fp8>, part = #ada300hl.part<low>} : ...` |
| `ada300hl.vmul_mixed` | `Pure` | 混合精度向量乘法 | `%z = ada300hl.vmul_mixed %a, %b {acc_type = #ada300hl.dtype<fp16>, part = #ada300hl.part<low>} : ...` |

#### 张量核心操作

| Op | Traits | 描述 |
|---|---|---|
| `ada300hl.tensor_mma` | `MemRead`、`MemWrite` | 全语义张量核心矩阵乘法。携带模式、维度、数据类型和块计数，以便后续 Pass 做出明智的降低决策。 |
| `ada300hl.tensor_sync` | `MemRead`、`MemWrite` | 等待张量核心完成的抽象屏障。 |

`tensor_mma` 属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `mode` | `TensorModeAttr` | `inner` 或 `outer` 乘积 |
| `rhs_transposed` | `bool` | 是否使用 `gmma_mt`（转置权重） |
| `row_size`、`col_size`、`acc_size` | `i32` | 矩阵 Tile 维度 |
| `out_type`、`act_type`、`wht_type` | `TensorDataTypeAttr` | 硬件数据类型 |
| `blk_cnt_a`、`blk_cnt_w` | `i32` | SRAM 块迭代计数 |

#### 布局 / 数据移动操作

| Op | 描述 |
|---|---|
| `ada300hl.pack` | 将平坦矩阵缓冲区重新格式化为张量核心所需的块布局。 |
| `ada300hl.unpack` | `pack` 的逆操作；将块布局转换回平坦布局。 |
| `ada300hl.layout_cast` | 布局的视图式重新解释；无数据移动。 |
| `ada300hl.copy_to_sram` | 将缓冲区从全局/VR 内存复制到片上 SRAM。 |
| `ada300hl.copy_from_sram` | 将缓冲区从 SRAM 复制回全局/VR 内存。 |
| `ada300hl.copy_to_vr` | 复制到向量寄存器支持的存储区。 |
| `ada300hl.copy_from_vr` | 从向量寄存器支持的存储区复制出。 |

---

## Ada300HW 方言

**命名空间：** `::buddy::ada300hw`  
**助记符前缀：** `ada300hw`  
**根 tablegen 文件：** `Ada300HWOps.td`

低级 ISA 方言直接对 ADA300 硬件协议进行建模。其操作与机器指令具有 1:1 或接近 1:1 的对应关系，代码生成器必须保留这些操作。

属性与 Ada300HL 共享——`Ada300HWAttrs.td` 简单地包含 `Ada300HLAttrs.td`，因此两个方言中使用相同的 `#ada300hl.*` 属性语法。

### Ada300HW 操作

#### 向量操作

| Op | Traits | ISA 指令 | 描述 |
|---|---|---|---|
| `ada300hw.vfpwnl` | `Pure` | `vfpwnl.<func>.<segs>` | PWL 非线性函数。`func` 和 `segments` 选择近似表。 |
| `ada300hw.vfcvt` | `Pure` | `vfcvt.<part>.*` | 浮点精度转换（低/高半部）。 |
| `ada300hw.vfmul_low` | `Pure` | `vfmul.low.*` | 混合精度乘法到低子向量。 |
| `ada300hw.vfmul_high` | `Pure` | `vfmul.high.*` | 混合精度乘法到高子向量。 |

#### 张量核心配置操作

这些操作写入硬件配置寄存器，不得被重排序或消除。它们必须在对应的执行操作之前出现。

| Op | 寄存器 | 关键属性 |
|---|---|---|
| `ada300hw.set_gmm_cfg` | `gmm_cfg` | `row_size`、`col_size`、`acc_size`、`mode` |
| `ada300hw.set_gmm_type` | `gmm_type` | `out_type`、`act_type`、`wht_type` |
| `ada300hw.set_gmm_iter` | `gmm_iter` | `blk_cnt_a`、`blk_cnt_w` |

#### 张量核心执行操作

所有执行操作均带有 `MemRead` + `MemWrite` 副作用。

| Op | ISA 指令 | 描述 |
|---|---|---|
| `ada300hw.gmm_mm` | `gmm.mm` | 非累积矩阵乘法（清空目标）。 |
| `ada300hw.gmma_mm` | `gmma.mm` | 累积矩阵乘法（累加到目标）。 |
| `ada300hw.gmma_mt` | `gmma.mt` | 累积矩阵乘法，权重矩阵转置。 |
| `ada300hw.gmv_mm` | `gmv.mm` | 非累积矩阵向量乘法。 |
| `ada300hw.gmva_mm` | `gmva.mm` | 累积矩阵向量乘法。 |
| `ada300hw.gmva_mt` | `gmva.mt` | 累积矩阵向量乘法，矩阵转置。 |

#### 同步

| Op | ISA 指令 | 描述 |
|---|---|---|
| `ada300hw.tcsync` | `tcsync` | 轮询张量核心完成标志。阻塞直到硬件发出完成信号。 |

---

## 转换 Pass

### 1. MathToAda300HL

**Pass 参数：** `--math-to-ada300hl`  
**源文件：** `midend/lib/Conversion/MathToAda300HL/MathToAda300HLPass.cpp`  
**Pass 类：** `MathToAda300HLPass`（作用于 `func::FuncOp`）

将标准 `math` 方言中具有直接 ADA300 向量单元支持的操作替换为对应的高级 `ada300hl` 操作。只有向量类型的操作数会被降低；标量操作留给标准 arith 流水线处理。

| 输入 | 输出 | 备注 |
|---|---|---|
| `math.exp %v` | `ada300hl.exp %v {segments = 16}` | 默认 16 段表 |
| `math.log %v` | `ada300hl.pwnl %v {func=log, segments=16}` | |
| `math.sqrt %v` | `ada300hl.pwnl %v {func=sqrt, segments=16}` | |
| `math.rsqrt %v` | `ada300hl.pwnl %v {func=rsqrt, segments=16}` | |

默认段数（16）较为保守。后续调优 Pass 可将其增加到 32 以获得更高精度。

---

### 2. LinalgToAda300HL

**Pass 参数：** `--linalg-to-ada300hl`  
**源文件：** `midend/lib/Conversion/LinalgToAda300HL/LinalgToAda300HLPass.cpp`  
**Pass 类：** `LinalgToAda300HLPass`（作用于 `func::FuncOp`）

将已缓冲化的 `linalg.matmul` 转换为 Ada300HL 张量核心序列。该 Pass 运行前 IR 必须已完全缓冲化（memref 操作数）。

**`linalg.matmul ins(%A, %B) outs(%C)` 的展开模式：**

```mlir
%A_sram = memref.alloc(...)                    // SRAM 暂存缓冲区
%B_sram = memref.alloc(...)
ada300hl.copy_to_sram %A, %A_sram              // 将 A 暂存到 SRAM
ada300hl.copy_to_sram %B, %B_sram              // 将 B 暂存到 SRAM
ada300hl.tensor_mma %C, %A_sram, %B_sram {
    mode         = inner,
    rhs_transposed = false,
    row_size     = <M（静态）或 16（动态）>,
    col_size     = <N（静态）或 16（动态）>,
    acc_size     = <K（静态）或 16（动态）>,
    out_type     = <从 C 元素类型推断>,
    act_type     = <从 A 元素类型推断>,
    wht_type     = <从 B 元素类型推断>,
    blk_cnt_a    = 1,
    blk_cnt_w    = 1
}
ada300hl.tensor_sync
memref.dealloc %A_sram
memref.dealloc %B_sram
```

矩阵维度在静态已知时直接编码；否则使用保守默认值（16）。块计数从 1 开始，可由分块 Pass 增大。

SRAM 缓冲区地址由下游内存规划 Pass 解析；在此阶段它们是普通的 `memref.alloc`。

---

### 3. LowerAda300HLToAda300HW

**Pass 参数：** `--lower-ada300hl-to-ada300hw`  
**源文件：** `midend/lib/Conversion/LowerAda300HLToAda300HW/LowerAda300HLToAda300HWPass.cpp`  
**Pass 类：** `LowerAda300HLToAda300HWPass`（作用于 `func::FuncOp`）

将每个 `ada300hl` 操作降低到等价的 `ada300hw` ISA 级操作或操作序列。该 Pass 结束后，函数体中不再包含 `ada300hl` 操作。

#### 降低对照表

| Ada300HL Op | Ada300HW 输出 | 备注 |
|---|---|---|
| `ada300hl.exp` | `ada300hw.vfpwnl {func=exp, masked=false}` | `func` 固定为 `exp` |
| `ada300hl.pwnl` | `ada300hw.vfpwnl {func=<转发>, masked=false}` | `func` 属性转发 |
| `ada300hl.cvt` | `ada300hw.vfcvt` | `dst_type` 和 `part` 转发 |
| `ada300hl.vmul_mixed`（`part=low`） | `ada300hw.vfmul_low` | `src_a/b_type` 从 MLIR 元素类型推断 |
| `ada300hl.vmul_mixed`（`part=high`） | `ada300hw.vfmul_high` | |
| `ada300hl.tensor_mma` | `ada300hw.set_gmm_cfg` + `set_gmm_type` + `set_gmm_iter` + `gmma_mm` 或 `gmma_mt` | 1 → 4 个操作展开；`rhs_transposed = true` 时选择 `gmma_mt` |
| `ada300hl.tensor_sync` | `ada300hw.tcsync` | 直接替换 |
| `ada300hl.pack` | `memref.copy`（占位符） | 后端 DMA 代码生成负责数据移动 |
| `ada300hl.unpack` | `memref.copy`（占位符） | |
| `ada300hl.layout_cast` | 消除（源值转发） | 布局信息在降低过程中消耗 |
| `ada300hl.copy_to_sram` | `memref.copy`（占位符） | |
| `ada300hl.copy_from_sram` | `memref.copy`（占位符） | |
| `ada300hl.copy_to_vr` | `memref.copy`（占位符） | |
| `ada300hl.copy_from_vr` | `memref.copy`（占位符） | |

使用 `applyPatternsAndFoldGreedily` 进行模式应用。

---

### 4. LowerAda300HWToLLVM

**Pass 参数：** `--lower-ada300hw-to-llvm`  
**源文件：** `midend/lib/Conversion/LowerAda300HWToLLVM/LowerAda300HWToLLVMPass.cpp`  
**Pass 类：** `LowerAda300HWToLLVMPass`（作用于 `func::FuncOp`）

将 Ada300HW ISA 级操作降低到 `LLVM::InlineAsmOp` 操作。这允许 LLVM 后端通过现有的 RISC-V 内联汇编发射路径发出自定义 ADA300 指令字节序列，而无需为 ADA300 自定义扩展提供完整的 LLVM target 支持。

使用带 `LLVMConversionTarget` 的 `applyPartialConversion`；所有 `ada300hw` 操作被标记为非法，`llvm` 操作合法。也填充了 MemRef → LLVM 结构体类型转换模式，以便可以提取张量执行操作的指针操作数。

#### 指令编码约定

| Ada300HW Op | 内联汇编助记符 | 约束 |
|---|---|---|
| `vfpwnl` | `vfpwnl.<func>.<segments>  vd, vs1` | `=vr,vr` |
| `vfcvt` | `vfcvt.<part>  vd, vs1` | `=vr,vr` |
| `vfmul_low` | `vfmul.low  vd, vs1, vs2` | `=vr,vr,vr` |
| `vfmul_high` | `vfmul.high  vd, vs1, vs2` | `=vr,vr,vr` |
| `set_gmm_cfg` | `gmm.cfg <row>, <col>, <acc>, <mode>` | 副作用，无 SSA 操作数 |
| `set_gmm_type` | `gmm.type <out_dt>, <act_dt>, <wht_dt>` | 副作用，无 SSA 操作数 |
| `set_gmm_iter` | `gmm.iter <blk_a>, <blk_w>` | 副作用，无 SSA 操作数 |
| `gmm_mm` | `gmm.mm  $0, $1, $2` | `r,r,r`（GPR 指针） |
| `gmma_mm` | `gmma.mm  $0, $1, $2` | `r,r,r` |
| `gmma_mt` | `gmma.mt  $0, $1, $2` | `r,r,r` |
| `gmv_mm` | `gmv.mm  $0, $1, $2` | `r,r,r` |
| `gmva_mm` | `gmva.mm  $0, $1, $2` | `r,r,r` |
| `gmva_mt` | `gmva.mt  $0, $1, $2` | `r,r,r` |
| `tcsync` | `tcsync` | 副作用，无操作数 |

配置寄存器值（`row_size`、`col_size`、`mode`、数据类型代码等）是从操作属性派生的编译时常量；它们在转换时直接嵌入到汇编字符串中。

对于张量执行操作，对齐指针（LLVM memref 描述符结构体的字段索引 1）通过 `llvm.extractvalue` 提取，并作为 GPR（`"r"`）操作数传入。

---

## 完整流水线

包含 `linalg.matmul` 和向量 `math.exp` 的函数的完整编译流水线：

```
1.  linalg.matmul + math.exp   （标准 MLIR）
         │
         │  （缓冲化：one-shot-bufferize 或类似方式）
         │
2.  linalg.matmul + math.exp   （已缓冲化，memref 形式）
         │
         │  --math-to-ada300hl
         │
3.  linalg.matmul              （linalg）
    ada300hl.exp               （高级）
         │
         │  --linalg-to-ada300hl
         │
4.  ada300hl.copy_to_sram
    ada300hl.tensor_mma
    ada300hl.tensor_sync
    ada300hl.exp               （全部为 ada300hl）
         │
         │  --lower-ada300hl-to-ada300hw
         │
5.  ada300hw.set_gmm_cfg
    ada300hw.set_gmm_type
    ada300hw.set_gmm_iter
    ada300hw.gmma_mm
    ada300hw.tcsync
    ada300hw.vfpwnl            （全部为 ada300hw）
    memref.copy                （数据移动占位符）
         │
         │  --lower-ada300hw-to-llvm
         │  （+ finalize-memref-to-llvm）
         │
6.  llvm.inline_asm            （LLVM 方言，准备好供后端使用）
```

---

## CMake 集成

两个方言库和四个 Pass 库按如下方式接入 midend 构建系统：

### 方言库

```cmake
# midend/include/Dialect/Ada300HL/CMakeLists.txt
add_mlir_dialect(Ada300HLOps ada300hl)          # 生成 ops/types/dialect inc 文件
mlir_tablegen(Ada300HLOpsAttrs.h.inc  ...)      # 枚举 + 属性声明
mlir_tablegen(Ada300HLOpsEnums.cpp.inc ...)     # 枚举定义
add_public_tablegen_target(BuddyAda300HLAttrsIncGen)

# midend/lib/Dialect/Ada300HL/CMakeLists.txt
add_mlir_dialect_library(BuddyAda300HL
  Ada300HLDialect.cpp Ada300HLAttrs.cpp Ada300HLTypes.cpp Ada300HLOps.cpp
  DEPENDS MLIRAda300HLOpsIncGen BuddyAda300HLAttrsIncGen
  ...)

# midend/lib/Dialect/Ada300HW/CMakeLists.txt
add_mlir_dialect_library(BuddyAda300HW
  Ada300HWDialect.cpp Ada300HWOps.cpp
  DEPENDS MLIRAda300HWOpsIncGen BuddyAda300HLAttrsIncGen
  LINK_LIBS PUBLIC BuddyAda300HL ...)
```

### Pass 库

```cmake
# midend/lib/Conversion/CMakeLists.txt
add_subdirectory(MathToAda300HL)
add_subdirectory(LinalgToAda300HL)
add_subdirectory(LowerAda300HLToAda300HW)
add_subdirectory(LowerAda300HWToLLVM)
```

每个 Pass 目录包含一个独立的 `add_mlir_library(...)` 目标，链接相关方言库和标准 MLIR 基础设施（`MLIRPass`、`MLIRTransformUtils`、`MLIRFuncDialect` 等）。

### 关键生成文件

| 目标 | 由...生成 | 用于 |
|---|---|---|
| `MLIRAda300HLOpsIncGen` | `Ada300HLOps.td` | `Ada300HLOps.h`、`Ada300HLDialect.cpp`、`Ada300HLOps.cpp` |
| `BuddyAda300HLAttrsIncGen` | `Ada300HLOps.td` | `Ada300HLAttrs.h`、`Ada300HLDialect.h`、`Ada300HLAttrs.cpp` |
| `MLIRAda300HWOpsIncGen` | `Ada300HWOps.td` | `Ada300HWOps.h`、`Ada300HWDialect.cpp` |
