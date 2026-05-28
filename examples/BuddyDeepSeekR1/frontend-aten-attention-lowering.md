# Frontend Aten to MLIR Mapping and Attention Lowering

This note summarizes how the Buddy Python frontend imports TorchDynamo FX/Aten
graphs for the DeepSeekR1 example, how attention patterns are fused, and what
the fused ops lower to.

## 1. Import Flow

The DeepSeekR1 importer uses `DynamoCompiler` with the TOSA registry:

```python
dynamo_compiler_prefill = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_prefill",
)

dynamo_compiler_decode = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_decode",
)
```

The model is first traced by TorchDynamo/AOTAutograd into an FX graph whose
nodes are mostly Aten/Prims operations. The Buddy frontend then maps each FX
node name to a Buddy graph op through `DynamoCompiler._ops_map` in
`frontend/Python/frontend.py`.

The lowering path is:

```text
Torch model
  -> TorchDynamo FX graph in Aten/Prims form
  -> Buddy graph ops
  -> graph transforms/fusion
  -> MLIR module through ops_registry lowering
  -> CMake lowering pipeline
  -> LLVM IR/object file
```

## 2. Frontend Mapping

There are two mapping levels.

1. Aten FX op name to Buddy graph op class:

```python
"_scaled_dot_product_flash_attention_for_cpu.default":
    ScaledDotProductFlashAttentionForCpuOp
"index_put.default": IndexPutOp
"view.default": ViewOp
"clone.default": CloneOp
"expand.default": ExpandOp
"unsqueeze.default": UnsqueezeOp
"pow.Tensor_Scalar": PowOp
"mean.dim": MeanOp
"rsqrt.default": RsqrtOp
"mul.Tensor": MulOp
"add.Tensor": AddOp
```

2. Buddy graph op class to MLIR builder function:

```python
"ScaledDotProductFlashAttentionForCpuOp":
    scaled_dot_product_flash_attention_for_cpu_op
"FlashAttentionForCpuPrefillOp":
    flash_attention_for_cpu_prefill_op
"GQAAttentionFusedOp":
    gqa_attention_fused_op
"RsqrtOp": rsqrt_op
"MeanOp": mean_op
"MulOp": mul_op
"AddOp": add_op
```

The full frontend Aten name list lives in `DynamoCompiler._ops_map`. The full
Buddy-op-to-lowering-function list lives in `frontend/Python/ops/tosa.py`
`ops_registry`, plus the math/linalg/func registries merged by `DynamoCompiler`.

## 3. LLM-Relevant Aten Mapping

| FX/Aten name | Buddy op | Typical MLIR lowering |
| --- | --- | --- |
| `_scaled_dot_product_flash_attention_for_cpu.default` | `ScaledDotProductFlashAttentionForCpuOp` | Either generic SDPA lowering or replaced by fused attention ops |
| `index_put.default` | `IndexPutOp` | Cache update loops/memref writes |
| `view.default` | `ViewOp` | `tosa.reshape` |
| `clone.default` | `CloneOp` | Tensor copy/identity-style lowering |
| `expand.default` | `ExpandOp` | Broadcast/reshape-style lowering |
| `unsqueeze.default` | `UnsqueezeOp` | `tosa.reshape` |
| `permute.default` | `PermuteOp` | `tosa.transpose` |
| `transpose.int` | `TransposeOp` | `tosa.transpose` |
| `mm.default` | `MatmulOp` | `tosa.matmul`/linalg matmul path |
| `bmm.default` | `BatchMatmulOp` | batch matmul lowering |
| `add.Tensor` | `AddOp` | `tosa.add` |
| `mul.Tensor` | `MulOp` | `tosa.mul` |
| `sub.Tensor` | `SubOp` | `tosa.sub` |
| `div.Tensor` | `DivOp` | reciprocal + multiply or division lowering |
| `pow.Tensor_Scalar` | `PowOp` | math/TOSA composition |
| `mean.dim` | `MeanOp` | reduce-sum + scale |
| `rsqrt.default` | `RsqrtOp` | `tosa.rsqrt` or `math.rsqrt` depending registry |
| `silu.default` | `SiluOp` | sigmoid/multiply style lowering |
| `embedding.default` | `EmbeddingOp` | gather/index-style lowering |

RMSNorm is currently not raised to a dedicated `RMSNormOp`. It is imported as a
composition of elementary Aten ops, typically:

```text
pow(x, 2)
mean(..., dim=-1, keepdim=True)
add(eps)
rsqrt(...)
mul(x, rsqrt)
mul(weight)
```

Those pieces are lowered independently.

## 4. Attention Fusion

The DeepSeekR1 importer applies different fusion lists for prefill and decode:

```python
pattern_list_prefill = [
    simply_fuse,
    apply_classic_fusion,
    flash_attention_prefill,
]

pattern_list_decode = [
    simply_fuse,
    apply_classic_fusion,
    gqa_attention_fusion,
]
```

### Prefill

Prefill looks for:

```text
ScaledDotProductFlashAttentionForCpuOp
```

and replaces it with:

```text
FlashAttentionForCpuPrefillOp
```

This is done by `flash_attention_prefill()` and `replace_attention_op()` in
`frontend/Python/graph/transform/fuse_ops.py`.

### Decode with KV Cache

Decode looks for a KV-cache-aware GQA attention pattern:

```text
ScaledDotProductFlashAttentionForCpuOp
  K input: View <- Clone <- Expand <- Unsqueeze <- IndexPut
  V input: View <- Clone <- Expand <- Unsqueeze <- IndexPut
```

When the pattern matches, the pass replaces the SDPA node and its cache read
chain with:

```text
GQAAttentionFusedOp(query, k_cache, v_cache, ...)
```

This preserves the fact that attention reads from `k_cache` and `v_cache`,
instead of treating the cache path as unrelated view/clone/expand operations.

## 5. Generic SDPA Lowering

If `ScaledDotProductFlashAttentionForCpuOp` is not replaced by a fused op, its
generic lowering in `scaled_dot_product_flash_attention_for_cpu_op()` expands
attention into:

```text
reshape Q/K/V
Q @ K^T
scale
add mask
softmax
softmax @ V
reshape output
```

The emitted MLIR is a mixture of:

```text
tosa.reshape
linalg.batch_matmul_transpose_b
tosa.mul/add/sub/reduce_max/reduce_sum/log/matmul
math.exp
```

This path is simple but materializes larger intermediate tensors such as scores
and probabilities.

## 6. FlashAttentionForCpuPrefillOp Optimization

`flash_attention_for_cpu_prefill_op()` lowers prefill attention directly into
tiled loop/vector IR. It uses:

```text
Q block size  = 16
KV block size = 64
vector width  = 16
f32 intermediate accumulation
```

The algorithm keeps online-softmax state per query block:

```text
m_i: running row max
l_i: running row sum
accum: running output accumulator
```

Conceptually:

```text
for batch:
  for head:
    for q_block in Q:
      initialize m_i, l_i, accum
      for kv_block in K/V:
        score_tile = Q_block @ K_block
        m_new, l_new = online_softmax_update(score_tile)
        accum = rescale_old_accum + exp(score_tile - m_new) @ V_block
      output = accum / l_i
```

The main emitted dialects/ops are:

```text
bufferization.to_buffer / bufferization.to_tensor
memref.alloc / memref.load / memref.store
affine.for / affine.parallel
scf.for
vector.load / vector.store / vector.fma / vector.reduction / vector.splat
arith.addf / arith.subf / arith.mulf / arith.divf
arith.extf / arith.truncf / arith.cmpf / arith.select
math.exp
```

The important optimization is avoiding full materialization of:

```text
score = Q @ K^T
prob = softmax(score)
```

## 7. GQAAttentionFusedOp Optimization

`gqa_attention_fused_op()` lowers decode attention with explicit KV-cache
inputs:

```python
query = ...
k_cache = ...
v_cache = ...
```

It handles grouped-query attention by mapping query heads to KV heads:

```python
group_size = query_shape[1] // key_shape[1]
h_kv = h / group_size
```

The computation is:

```text
score = Q @ K_cache^T
score = score * scale + mask
prob = softmax(score)
out = prob @ V_cache
```

QK and PV are built with explicit loops and vector operations. Softmax is built
from TOSA/math operations.

The main emitted dialects/ops are:

```text
scf.for
tensor.extract / tensor.insert
vector.transfer_read / vector.transfer_write
vector.fma / vector.reduction / vector.splat
arith.extf / arith.truncf / arith.addf / arith.mulf / arith.divsi
tosa.mul / tosa.add / tosa.sub / tosa.reduce_max / tosa.reduce_sum
tosa.reshape / tosa.cast
math.exp / math.log
```

Compared with generic SDPA, the fused decode op keeps the KV-cache access
explicit and avoids lowering the cache path as unrelated `view/clone/expand`
noise.

## 8. Final Lowering Pipeline

The MLIR emitted by the frontend is not final. The DeepSeekR1 CMake pipeline
continues lowering through Buddy/MLIR/LLVM passes. The broad path is:

```text
TOSA/Linalg/Tensor/Affine/SCF/Vector/MemRef
  -> bufferization
  -> linalg/vector/matmul optimizations
  -> affine/scf lowering
  -> OpenMP lowering
  -> LLVM dialect
  -> LLVM IR
  -> object file
```

For the generated object files, the pipeline invokes:

```text
buddy-opt
mlir-opt
buddy-opt
mlir-translate -mlir-to-llvmir
llvm-as
llc
```

So the fused attention ops do not survive to the final binary. They are frontend
semantic anchors used to emit better loop/vector/cache-aware MLIR before the
normal compiler lowering pipeline takes over.
