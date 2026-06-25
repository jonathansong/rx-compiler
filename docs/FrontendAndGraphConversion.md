# Buddy Compiler Python Frontend & Graph Conversion

## Overview

`frontend/Python/frontend.py` is the entry point of the Buddy Compiler Python frontend. Its primary responsibility is converting PyTorch models into MLIR modules via a two-stage pipeline:

1. **PyTorch model → FX Graph** (handled by TorchDynamo / `torch.export`)
2. **FX Graph → Buddy Graph → MLIR module** (handled by `DynamoCompiler`)

---

## Architecture

```
PyTorch model
    │
    ▼  dynamo.optimize  /  torch.export.export
FX Graph (Aten IR)
    │
    ▼  DynamoCompiler._compile_fx
Buddy Graph  (Op node list)
    │
    ▼  graph.perform(transforms)
Transformed Buddy Graph
    │
    ▼  graph.compile()
MLIR Module
    │
    ▼  ExecutionEngine (opt_level=3)
Native execution  →  torch.Tensor outputs
```

---

## Key Classes

### `DynamoCompiler`

The main frontend class. Registers as a custom backend for TorchDynamo and drives the full compilation pipeline.

| Attribute | Description |
|-----------|-------------|
| `_func_name` | MLIR function name (default `"forward"`) |
| `_ops_registry` | Dict mapping op names to MLIR lowering functions |
| `_ops_map` | Dict mapping `aten.*` op names to Buddy `Op` subclasses (~300+ entries) |
| `_imported_graphs` | List of compiled `Graph` objects |
| `_imported_params` | Map from `Graph` → flat parameter tensors |
| `_aot_autograd_decomposition` | Decomposition table passed to AOT Autograd |

#### Public Methods

| Method | Description |
|--------|-------------|
| `importer(model, *args)` | Import via `dynamo.optimize`; falls back to `make_fx` on list-index errors |
| `importer_by_export(module, *args)` | Import via `torch.export.export` (preserves argument order) |
| `__call__(gm, inputs)` | Callable backend interface for TorchDynamo |
| `dynamo_run()` | Return an execution callable for the last imported graph |

### `TorchCompileBackend`

A thin wrapper adapting `DynamoCompiler` to the `torch.compile(backend=...)` interface. Sets `return_type="buddy"` so the compiler returns an executable closure instead of the original forward.

### `make_default_torch_backend()` / `dynamo_compiler`

Module-level factory that creates a default `TorchCompileBackend` using the TOSA ops registry and inductor decompositions. The resulting `dynamo_compiler` instance can be used directly:

```python
import torch
from buddy.frontend.Python.frontend import dynamo_compiler

model_opt = torch.compile(model, backend=dynamo_compiler)
```

---

## FX Graph → Buddy Graph Conversion

The conversion takes place inside `_compile_fx`, specifically within the inner `_compiler` closure. It proceeds in five stages.

### Stage 1 — Node Classification

Before traversal, every FX node is assigned to one of four buckets by position index:

| Bucket | Condition | Buddy `NodeType` | Stored in `Graph` |
|--------|-----------|-----------------|-------------------|
| `param_nodes` | weight parameters | `FakeNode` | `Graph._fake_params` |
| `buffers_nodes` | registered buffers | `FakeNode` | `Graph._fake_params` |
| `input_nodes` | real runtime inputs | `InputNode` | `Graph._inputs` |
| `other_nodes` | compute / output nodes | `OtherNode` | `Graph._body` |

### Stage 2 — Per-Node Conversion

Each FX node is dispatched by its `op` field to produce a Buddy `Op` object:

```
node.op == "placeholder"       →  PlaceholderOp
node.op == "output"            →  OutputOp
node.target == operator.getitem →  GetItemOp
node.op == "get_attr"
  └─ name contains "_tensor_constant"  →  TensorConstantOp
node.op == "call_function"     →  _ops_map[target.__name__] lookup
```

The `_create_node` method populates every `Op` object with:

```python
buddy_node._name              = gm_node.name          # unique string id
buddy_node._arguments         = [arg1, arg2, ...]      # positional inputs
buddy_node._parents           = ["node_x", "node_y"]  # upstream node names
buddy_node._children          = ["node_z", ...]        # downstream node names
buddy_node._keyword_arguments = gm_node.kwargs
buddy_node._tensor_meta       = {
    "shape": <output shape>,
    "dtype": <TensorDType enum>,
}
```

### Stage 3 — Type Metadata Inference

Output shape and dtype are resolved through a fallback chain when `tensor_meta` is absent:

```
meta["tensor_meta"]          ← highest priority (set by AOT Autograd)
  └─ meta["val"]             ← concrete / fake tensor value
      └─ meta["example_value"]
          └─ op._schema.returns  ← SymInt/SymBool/int/bool return types
              └─ call_function target name
                 (comparison ops like ge/gt/le/lt/eq/ne → Bool)
```

For ops with multiple return values (`num_returns > 1`), both `node_shape` and `node_dtype` become tuples of tuples.

#### dtype Translation Table

| PyTorch dtype | `TensorDType` |
|---------------|--------------|
| `torch.float32` | `Float32` |
| `torch.float16` | `Float16` |
| `torch.bfloat16` | `BFloat16` |
| `torch.float64` | `Float64` |
| `torch.int64` | `Int64` |
| `torch.int32` | `Int32` |
| `torch.int8` | `Int8` |
| `torch.complex64` | `Complex64` |
| `torch.complex128` | `Complex128` |
| `torch.bool` | `Bool` |

### Stage 4 — Insertion into `Graph`

```python
graph.add_node(node=buddy_node, node_type=node_type)
```

`Graph.add_node` maintains three parallel data structures:

```
Graph._body        = [Op_0, Op_1, ..., Op_N]   # ordered list (topological)
Graph._fake_params = [0, 1, 2, ...]             # indices of parameter nodes
Graph._inputs      = [3, 4]                     # indices of input nodes
Graph.node_table   = {"add_1": <Op>, ...}       # name → Op lookup map
```

### Stage 5 — Graph Transforms

After all nodes are inserted, registered transform passes run over the whole graph:

```python
transform_list = [maxpool2d_simplify]
if self._enable_external_calls:
    transform_list.extend(RUNTIME_RNG_TRANSFORMS)
graph.perform(transform_list)
```

Each transform receives the `Graph` object and may add, remove, or replace nodes using `graph.delete_node` / `graph.displace_node`. This is the last simplification step before MLIR code generation.

---

## Node Mapping Summary

```
FX Graph node                               Buddy Op object
─────────────────────────────────────────────────────────────────
node.name       "add_tensor_1"         →    Op._name
node.op         "call_function"        →    (dispatch branch)
node.target     aten.add.Tensor        →    _ops_map["add.Tensor"] = AddOp
node.args       (x, y)                 →    Op._arguments = ["x", "y"]
                                            Op._parents   = ["x", "y"]
node.users      {z: ...}               →    Op._children  = ["z"]
meta["tensor_meta"].shape              →    Op._tensor_meta["shape"]
meta["tensor_meta"].dtype              →    Op._tensor_meta["dtype"]
node.kwargs     {}                     →    Op._keyword_arguments
```

---

## Execution Pipeline (after compilation)

Once `graph.compile()` produces an MLIR module, `_dynamo_run_for_graph` wires up the execution:

1. Load shared libraries: `libmlir_runner_utils`, `libmlir_c_runner_utils`, `libomp` (and optionally `libbuddy_rng_utils`)
2. Create an `ExecutionEngine` at `opt_level=3`
3. For each call, convert input `torch.Tensor` objects to NumPy arrays (bfloat16 is handled as uint16 bit patterns)
4. Pass memref descriptors via ctypes into `ee.invoke(func_name, ...)`
5. Read output memrefs back into NumPy arrays, then convert to `torch.Tensor`

---

## File Structure

```
frontend/Python/
├── frontend.py          # DynamoCompiler, TorchCompileBackend (this document)
├── graph/
│   ├── graph.py         # Graph class: node storage, compile(), transforms
│   ├── operation.py     # Op base class + all Op subclasses
│   ├── graph_driver.py  # GraphDriver: MLIR emission logic
│   ├── transform.py     # Graph transform passes
│   └── type.py          # TensorDType, TensorMeta, DeviceType enums
└── ops/
    ├── linalg/          # Linalg dialect lowering registry
    ├── tosa/            # TOSA dialect lowering registry
    ├── math/            # Math dialect lowering registry
    └── func/            # Func dialect lowering registry
```
