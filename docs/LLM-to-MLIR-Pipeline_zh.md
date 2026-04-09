# LLM 模型转换为 MLIR：流程分析

本文档追踪 `examples/BuddyQwen3/import-qwen3.py` 如何将 HuggingFace LLM（Qwen3-0.6B）转换为 MLIR。

---

## 总览

```
HuggingFace PyTorch 模型
        │
        ▼
  TorchDynamo 追踪  (torch._dynamo + AOT Autograd)
        │
        ▼
  Buddy Graph IR  (Op 节点 + TensorMeta)
        │
        ▼
  图优化 Pass  (eliminate_transpose, fuse_ops, …)
        │
        ▼
  GraphImporter  (遍历 IR，按节点调用 ops_registry)
        │
        ▼
  TOSA 方言 MLIR  (func.func 包含 tosa.* 算子)
        │
        ▼
  输出文件  (.mlir + .data)
```

---

## 第一步 — 加载 PyTorch 模型

**文件：** `examples/BuddyQwen3/import-qwen3.py`

```python
model = AutoModelForCausalLM.from_pretrained(model_path).eval().to(torch.float32)
model.config.use_cache = False
```

模型从本地 HuggingFace 快照（或从 Hub 下载）加载，设置为推理模式并转换为 float32 精度。`use_cache = False` 确保追踪时捕获完整的前向传播，不带 KV Cache 分支。

---

## 第二步 — 通过 `DynamoCompiler` 捕获计算图

**文件：** `frontend/Python/frontend.py`

```python
dynamo_compiler_prefill = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_prefill",
)
graphs_prefill = dynamo_compiler_prefill.importer(model, input_ids=..., ...)
```

`DynamoCompiler` 将 **`torch._dynamo`** 与 AOT Autograd 结合封装，使用符号张量/假张量运行模型，完成计算图追踪：

- 所有 PyTorch 算子通过 `inductor_decomp` 分解为细粒度的 ATen 算子。
- 针对两个阶段分别捕获独立的计算图：

| 计算图 | 输入形状 | 用途 |
|---|---|---|
| **Prefill（预填充）** | `[1, 1024]` | 处理完整的输入提示词 |
| **Decode（解码）** | `[1, 1]` | 每次生成一个 token，使用 `StaticCache` KV Cache |

结果是一组 `Graph` 对象（Buddy IR），每个对象包含一系列 `Op` 节点。

---

## 第三步 — 图优化 Pass

**相关文件：**
- `frontend/Python/graph/graph.py` — `Graph.perform()`、`Graph.fuse_ops()`
- `frontend/Python/graph/transform/` — 各 Pass 的具体实现

### 3a. 结构化简化（`perform`）

```python
graphs_prefill[0].perform([
    eliminate_transpose,
    eliminate_matmul_transpose_reshape,
])
```

| Pass | 作用 |
|---|---|
| `eliminate_transpose` | 删除效果相互抵消的冗余转置算子 |
| `eliminate_matmul_transpose_reshape` | 将转置+reshape 序列折叠进矩阵乘法本身 |

### 3b. 算子融合（`fuse_ops`）

```python
graphs_prefill[0].fuse_ops([
    simply_fuse,
    apply_classic_fusion,
    flash_attention_prefill,   # 仅 prefill
])
graphs_decode[0].fuse_ops([
    simply_fuse,
    apply_classic_fusion,
    gqa_attention_fusion,      # 仅 decode
])
```

| Pass | 作用 |
|---|---|
| `simply_fuse` | 将相邻的逐元素算子合并为单个节点 |
| `apply_classic_fusion` | 应用常见编译器融合模式（如将 bias-add 融入矩阵乘法） |
| `flash_attention_prefill` | 将多算子注意力子图替换为单个 `FlashAttentionForCpuPrefillOp` |
| `gqa_attention_fusion` | 将 Grouped Query Attention（decode 阶段）融合为单个算子 |

融合完成后，算子组被重命名并绑定到设备：

```python
graph_prefill.op_groups["subgraph0_prefill"] = ...
graph_prefill.group_map_device["subgraph0_prefill"] = DeviceType.CPU
```

---

## 第四步 — 下降到 MLIR（TOSA 方言）

### 4a. `GraphDriver` 拆分子图

**文件：** `frontend/Python/graph/graph_driver.py`

```python
driver_prefill = GraphDriver(graphs_prefill[0])
driver_prefill.subgraphs[0].lower_to_top_level_ir()
```

`GraphDriver.__init__` 调用 `build_subgraph_by_group()`，完成以下工作：
1. 从 `graph.op_groups` 中识别每个算子组。
2. 分析跨组数据流，确定各子图的输入/输出。
3. 为每个子图构建独立的 `Graph` 对象。

### 4b. `lower_to_top_level_ir()` — 创建 MLIR 模块

**文件：** `frontend/Python/graph/graph.py` — `Graph.lower_to_top_level_ir()`

```python
fx_importer = GraphImporter(
    self._body, self.params_shapes, self.inputs_shapes,
    self._func_name, self._ops_registry, ...
)
self._imported_module = fx_importer.import_graph()
```

在 `import_graph()` 内部，通过 MLIR Python 绑定（`buddy_mlir.ir`）创建 `ir.Module`。

**输入签名构建：**

```
TensorMeta { shape=[1,1024], dtype=f32 }
  → ir.RankedTensorType.get([1, 1024], ir.F32Type.get())
```

所有参数和动态输入均成为 `func.FuncOp` 的 ranked tensor 参数。

### 4c. 逐节点算子下降 — `_import_op()`

**文件：** `frontend/Python/graph/graph.py` — `GraphImporter._import_op()`

```python
op_ret = self._ops_registry[op_name](node, self._symbol_table)
```

对每个 `Op` 节点：
1. 在 `ops_registry` 中根据类名（如 `"AddOp"`）查找对应函数。
2. 调用该函数，传入节点和 `symbol_table`（名称 → `ir.Value` 的映射）。
3. 将返回的 `ir.OpResult` 写回 `symbol_table[(node.name, 0)]`。

后续算子通过 `symbol_table.get((arg_name, 0))` 获取其输入值。

### 4d. `tosa.ops_registry` — 完整算子映射表

**文件：** `frontend/Python/ops/tosa.py`

约 80 个 PyTorch/ATen 算子被映射到 TOSA（或 TOSA+Linalg）的实现：

```python
ops_registry = {
    "AddOp":       add_op,
    "MulOp":       mul_op,
    "RsqrtOp":     rsqrt_op,
    "EmbeddingOp": embedding_op,
    "BatchMatmulOp": bmm_op,
    "FlashAttentionForCpuPrefillOp": flash_attention_for_cpu_prefill_op,
    ...
}
```

### 4e. 典型算子下降模式示例

**简单一对一映射（`rsqrt_op`）：**
```python
input1 = symbol_table.get((str(node.args[0]), 0))
sizes  = ir.RankedTensorType(input1.type).shape
result = tosa.RsqrtOp(ir.RankedTensorType.get(sizes, element_type), input1)
```
→ 生成单个 `tosa.rsqrt` 算子。

**含数据类型广播（`add_op`）：**
```python
# 若类型不匹配则先插入 tosa.cast，再生成带广播 reshape 的 tosa.add
if input1_dtype != mlir_dtype:
    input1 = tosa.CastOp(..., input1).result
return _gen_arith_binary_op(input1, input2, tosa.AddOp)
```

**融合注意力（prefill）：**  
`FlashAttentionForCpuPrefillOp` 已由融合 Pass 插入。其下降函数生成单个融合的 `linalg`/`vector` 算子，将完整的 QKV 投影 + softmax + 输出投影合并在一个 MLIR 操作中。

### 4f. 生成的 MLIR 结构

```mlir
func.func @forward_prefill(%arg0: tensor<...xi64>, %arg1: tensor<...xf32>, ...) 
    -> tensor<1x1024x32000xf32> {
  %0  = tosa.reshape %arg1 ...
  %1  = tosa.matmul %0, %arg2 ...
  %2  = tosa.rsqrt %1 ...
  %3  = tosa.mul %1, %2 ...
  ...
  return %N : tensor<1x1024x32000xf32>
}
```

---

## 第五步 — 写出输出文件

**文件：** `examples/BuddyQwen3/import-qwen3.py`

| 文件 | 来源 | 内容 |
|---|---|---|
| `subgraph0_prefill_0_6b.mlir` | `driver_prefill.subgraphs[0]._imported_module` | Prefill 阶段的 TOSA 计算子图 |
| `forward_prefill_0_6b.mlir` | `driver_prefill.construct_main_graph(True)` | 调用 prefill 子图的顶层 `func.func` |
| `subgraph0_decode_0_6b.mlir` | `driver_decode.subgraphs[0]._imported_module` | Decode/生成阶段的 TOSA 计算子图 |
| `forward_decode_0_6b.mlir` | `driver_decode.construct_main_graph(True)` | 调用 decode 子图的顶层 `func.func` |
| `arg0_0_6b.data` | `dynamo_compiler_prefill.imported_params` | 所有模型权重拼接后的原始 float32 二进制文件 |

模型权重的序列化方式：
```python
all_param = numpy.concatenate([
    param.detach().to(torch.float32).numpy().reshape([-1])
    for param in params
])
all_param.tofile("arg0_0_6b.data")
```

---

## 关键源文件索引

| 文件 | 职责 |
|---|---|
| `examples/BuddyQwen3/import-qwen3.py` | 入口：加载模型，运行完整流程 |
| `frontend/Python/frontend.py` | `DynamoCompiler` — TorchDynamo 追踪前端 |
| `frontend/Python/graph/graph.py` | `Graph`、`GraphImporter` — IR 表示与 MLIR 生成 |
| `frontend/Python/graph/graph_driver.py` | `GraphDriver` — 子图拆分与主图构建 |
| `frontend/Python/graph/operation.py` | 所有 `Op` 节点类定义 |
| `frontend/Python/graph/transform/` | 图优化与算子融合 Pass |
| `frontend/Python/ops/tosa.py` | `ops_registry` — 约 80 个 PyTorch 算子到 TOSA 的下降函数 |
| `frontend/Python/graph/type.py` | `TensorMeta`、`TensorDType`、`DeviceType` |

---

## 进一步下降（不用于 AOT 文件输出）

`Graph.lower_to_llvm_ir()` 展示了 JIT 执行时使用的完整下降栈：

```
TOSA 方言
  → tosa-to-linalg-named
  → tosa-to-linalg
  → tosa-to-tensor / tosa-to-arith
  → bufferize（tensor → memref）
  → convert-to-llvm
  → LLVM IR / ExecutionEngine
```

对于 AOT 导出（BuddyQwen3 所使用的方式），流程止步于 TOSA 方言，后续由 `buddy-opt` 单独执行目标相关的下降。
