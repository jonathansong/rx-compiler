# LLM Model to MLIR: Pipeline Analysis

This document traces how `examples/BuddyQwen3/import-qwen3.py` converts a HuggingFace LLM (Qwen3-0.6B) into MLIR.

---

## Overview

```
HuggingFace PyTorch Model
        │
        ▼
  TorchDynamo Tracing  (torch._dynamo + AOT Autograd)
        │
        ▼
  Buddy Graph IR  (Op nodes + TensorMeta)
        │
        ▼
  Graph Optimization Passes  (eliminate_transpose, fuse_ops, …)
        │
        ▼
  GraphImporter  (walks IR, calls ops_registry per node)
        │
        ▼
  TOSA Dialect MLIR  (func.func with tosa.* ops)
        │
        ▼
  Output files  (.mlir + .data)
```

---

## Step 1 — Load the PyTorch Model

**File:** `examples/BuddyQwen3/import-qwen3.py`

```python
model = AutoModelForCausalLM.from_pretrained(model_path).eval().to(torch.float32)
model.config.use_cache = False
```

The model is loaded from a local HuggingFace snapshot (or downloaded), set to eval mode, and cast to float32. `use_cache = False` is set so tracing captures the full forward pass without KV-cache branching.

---

## Step 2 — Capture the Computation Graph via `DynamoCompiler`

**File:** `frontend/Python/frontend.py`

```python
dynamo_compiler_prefill = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_prefill",
)
graphs_prefill = dynamo_compiler_prefill.importer(model, input_ids=..., ...)
```

`DynamoCompiler` wraps **`torch._dynamo`** with AOT Autograd. It traces the model by running it with symbolic/fake tensors:

- All PyTorch ops are decomposed via `inductor_decomp` into fine-grained ATen ops.
- Two separate graphs are captured to specialise each phase:

| Graph | Input shape | Purpose |
|---|---|---|
| **Prefill** | `[1, 1024]` | Process the full prompt |
| **Decode** | `[1, 1]` | Generate one token at a time using `StaticCache` KV cache |

The result is a list of `Graph` objects (Buddy IR), each containing a sequence of `Op` nodes.

---

## Step 3 — Graph Optimization Passes

**Files:**
- `frontend/Python/graph/graph.py` — `Graph.perform()`, `Graph.fuse_ops()`
- `frontend/Python/graph/transform/` — individual pass implementations

### 3a. Structural simplifications (`perform`)

```python
graphs_prefill[0].perform([
    eliminate_transpose,
    eliminate_matmul_transpose_reshape,
])
```

| Pass | What it does |
|---|---|
| `eliminate_transpose` | Removes redundant transpose ops whose effect cancels out |
| `eliminate_matmul_transpose_reshape` | Folds transpose+reshape sequences into the matmul itself |

### 3b. Operator fusion (`fuse_ops`)

```python
graphs_prefill[0].fuse_ops([
    simply_fuse,
    apply_classic_fusion,
    flash_attention_prefill,   # prefill only
])
graphs_decode[0].fuse_ops([
    simply_fuse,
    apply_classic_fusion,
    gqa_attention_fusion,      # decode only
])
```

| Pass | What it does |
|---|---|
| `simply_fuse` | Merges adjacent elementwise ops into one node |
| `apply_classic_fusion` | Applies common compiler fusion patterns (e.g., bias-add into matmul) |
| `flash_attention_prefill` | Replaces the multi-op attention subgraph with a single `FlashAttentionForCpuPrefillOp` |
| `gqa_attention_fusion` | Fuses Grouped Query Attention (decode) into a single op |

After fusion, op groups are renamed and mapped to a device:

```python
graph_prefill.op_groups["subgraph0_prefill"] = ...
graph_prefill.group_map_device["subgraph0_prefill"] = DeviceType.CPU
```

---

## Step 4 — Lower to MLIR (TOSA Dialect)

### 4a. `GraphDriver` splits subgraphs

**File:** `frontend/Python/graph/graph_driver.py`

```python
driver_prefill = GraphDriver(graphs_prefill[0])
driver_prefill.subgraphs[0].lower_to_top_level_ir()
```

`GraphDriver.__init__` calls `build_subgraph_by_group()`, which:
1. Identifies each op group from `graph.op_groups`.
2. Determines cross-group data-flow to find subgraph inputs/outputs.
3. Constructs a separate `Graph` object per subgraph.

### 4b. `lower_to_top_level_ir()` — MLIR module creation

**File:** `frontend/Python/graph/graph.py` — `Graph.lower_to_top_level_ir()`

```python
fx_importer = GraphImporter(
    self._body, self.params_shapes, self.inputs_shapes,
    self._func_name, self._ops_registry, ...
)
self._imported_module = fx_importer.import_graph()
```

Inside `import_graph()`, an `ir.Module` is created via the MLIR Python bindings (`buddy_mlir.ir`).

**Input signature construction:**

```
TensorMeta { shape=[1,1024], dtype=f32 }
  → ir.RankedTensorType.get([1, 1024], ir.F32Type.get())
```

All parameters and dynamic inputs become ranked tensor arguments of a `func.FuncOp`.

### 4c. Node-by-node op lowering — `_import_op()`

**File:** `frontend/Python/graph/graph.py` — `GraphImporter._import_op()`

```python
op_ret = self._ops_registry[op_name](node, self._symbol_table)
```

For each `Op` node:
1. Look up its class name (e.g., `"AddOp"`) in `ops_registry`.
2. Call the registered Python function, passing the node and a `symbol_table` (name → `ir.Value`).
3. Store the returned `ir.OpResult` back into `symbol_table[(node.name, 0)]`.

Later ops retrieve their inputs by `symbol_table.get((arg_name, 0))`.

### 4d. `tosa.ops_registry` — the full op map

**File:** `frontend/Python/ops/tosa.py`

~80 PyTorch/ATen ops are mapped to TOSA (or TOSA+Linalg) implementations:

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

### 4e. Example op lowering patterns

**Simple 1-to-1 (`rsqrt_op`):**
```python
input1 = symbol_table.get((str(node.args[0]), 0))
sizes  = ir.RankedTensorType(input1.type).shape
result = tosa.RsqrtOp(ir.RankedTensorType.get(sizes, element_type), input1)
```
→ emits a single `tosa.rsqrt` op.

**With dtype broadcasting (`add_op`):**
```python
# Emit tosa.cast if dtypes mismatch, then tosa.add with broadcast reshape
if input1_dtype != mlir_dtype:
    input1 = tosa.CastOp(..., input1).result
return _gen_arith_binary_op(input1, input2, tosa.AddOp)
```

**Fused attention (prefill):**  
`FlashAttentionForCpuPrefillOp` was already inserted by the fusion pass. Its lowering emits a single fused `linalg`/`vector` op covering the full QKV projection + softmax + output projection in one MLIR operation.

### 4f. Resulting MLIR structure

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

## Step 5 — Write Output Files

**File:** `examples/BuddyQwen3/import-qwen3.py`

| File | Source | Contents |
|---|---|---|
| `subgraph0_prefill_0_6b.mlir` | `driver_prefill.subgraphs[0]._imported_module` | TOSA compute subgraph for the prefill pass |
| `forward_prefill_0_6b.mlir` | `driver_prefill.construct_main_graph(True)` | Top-level `func.func` calling the prefill subgraph |
| `subgraph0_decode_0_6b.mlir` | `driver_decode.subgraphs[0]._imported_module` | TOSA compute subgraph for the decode/generation pass |
| `forward_decode_0_6b.mlir` | `driver_decode.construct_main_graph(True)` | Top-level `func.func` calling the decode subgraph |
| `arg0_0_6b.data` | `dynamo_compiler_prefill.imported_params` | All model weights concatenated as raw binary float32 |

Model weights are serialized as:
```python
all_param = numpy.concatenate([
    param.detach().to(torch.float32).numpy().reshape([-1])
    for param in params
])
all_param.tofile("arg0_0_6b.data")
```

---

## Key Source Files

| File | Role |
|---|---|
| `examples/BuddyQwen3/import-qwen3.py` | Entry point: loads model, runs full pipeline |
| `frontend/Python/frontend.py` | `DynamoCompiler` — TorchDynamo tracing frontend |
| `frontend/Python/graph/graph.py` | `Graph`, `GraphImporter` — IR and MLIR emission |
| `frontend/Python/graph/graph_driver.py` | `GraphDriver` — subgraph splitting and main-graph construction |
| `frontend/Python/graph/operation.py` | All `Op` node class definitions |
| `frontend/Python/graph/transform/` | Graph optimization and fusion passes |
| `frontend/Python/ops/tosa.py` | `ops_registry` — ~80 PyTorch op → TOSA lowering functions |
| `frontend/Python/graph/type.py` | `TensorMeta`, `TensorDType`, `DeviceType` |

---

## Further Lowering (not used for AOT file output)

`Graph.lower_to_llvm_ir()` shows the full lowering stack used when executing JIT:

```
TOSA dialect
  → tosa-to-linalg-named
  → tosa-to-linalg
  → tosa-to-tensor / tosa-to-arith
  → bufferize (tensor → memref)
  → convert-to-llvm
  → LLVM IR / ExecutionEngine
```

For AOT export (used by BuddyQwen3), the pipeline stops at the TOSA dialect so that `buddy-opt` can apply further target-specific lowering separately.
