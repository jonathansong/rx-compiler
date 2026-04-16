# LLM Compiler Optimizations in buddy-mlir

This document describes all LLM-related compiler optimizations implemented in buddy-mlir, organized by optimization level: graph-level transforms, KV cache strategies, attention-specific passes, and midend matmul optimizations.

---

## Table of Contents

1. [Supported Models](#1-supported-models)
2. [Compilation Pipeline Overview](#2-compilation-pipeline-overview)
3. [Graph-Level Optimizations](#3-graph-level-optimizations)
   - [Flash Attention Fusion](#31-flash-attention-fusion)
   - [GQA Attention Fusion](#32-gqa-attention-fusion)
   - [Transpose-Matmul Fusion](#33-transpose-matmul-fusion)
   - [Weight Transpose Elimination](#34-weight-transpose-elimination)
   - [Matmul-Transpose-Reshape Elimination](#35-matmul-transpose-reshape-elimination)
4. [KV Cache Optimizations](#4-kv-cache-optimizations)
   - [Static Cache Shape Specialization](#41-static-cache-shape-specialization)
   - [Prefill / Decode Split Compilation](#42-prefill--decode-split-compilation)
   - [Tiered KV Cache](#43-tiered-kv-cache)
   - [GQA KV Cache Broadcast Fusion](#44-gqa-kv-cache-broadcast-fusion)
   - [KV Cache as Explicit ABI Arguments](#45-kv-cache-as-explicit-abi-arguments)
5. [Attention-Specific Optimizations](#5-attention-specific-optimizations)
   - [Flash Attention MLIR Kernel](#51-flash-attention-mlir-kernel)
   - [GQA Decode Kernel](#52-gqa-decode-kernel)
   - [Decode-Phase Batch Matmul Passes](#53-decode-phase-batch-matmul-passes)
   - [Static Shapes Enable SIMD Vectorization](#54-static-shapes-enable-simd-vectorization)
6. [Quantization Optimizations](#6-quantization-optimizations)
   - [Weight-Only Quantization (w8a32, w8a16)](#61-weight-only-quantization-w8a32-w8a16)
   - [Int4 Weight Quantization (w4a16)](#62-int4-weight-quantization-w4a16)
   - [W8A8 Dynamic Quantization](#63-w8a8-dynamic-quantization)
   - [Fused Dequant+Matmul for Decode](#64-fused-dequantmatmul-for-decode)
7. [Midend MatMul Optimization Passes](#7-midend-matmul-optimization-passes)
8. [Runtime LLM Support](#8-runtime-llm-support)

---

## 1. Supported Models

| Model | Directory | Quantization | Features |
|---|---|---|---|
| LLaMA 2 (6.7B) | `examples/BuddyLlama/` | f32 | Prefill |
| DeepSeek-R1-Distill-Qwen-1.5B | `examples/BuddyDeepSeekR1/` | f32/f16/bf16, w4a16/w8a16/w8a32/w8a8 | Prefill+decode split, tiered KV cache |
| Qwen3-0.6B | `examples/BuddyQwen3/` | f32 | Prefill+decode split, GQA, flash attention |
| BERT | `examples/BuddyBert/` | f32 | Encoder |
| Whisper | `examples/BuddyWhisper/` | f32 | Audio transcription |
| Transformer | `examples/BuddyTransformer/` | f32 | Generic encoder-decoder |

`examples/BuddyNext/` contains standalone MLIR kernel benchmarks for every individual LLM sub-operator.

---

## 2. Compilation Pipeline Overview

```
HuggingFace PyTorch Model
     │ DynamoCompiler.importer()           [frontend/Python/frontend.py]
     ▼
Buddy Graph IR (ATen → op mapping)
     │ eliminate_transpose                  [weight transpose removal]
     │ eliminate_matmul_transpose_reshape   [weight reshape removal]
     │ flash_attention_prefill              [flash attention fusion]
     │ gqa_attention_fusion                 [GQA fusion]
     │ apply_classic_fusion                 [transpose+matmul fusion]
     │ w8a8 / w4a16 / w8a16                [optional quantization]
     ▼
TOSA MLIR (subgraph0.mlir)
     │ tosa-to-linalg-named
     │ tosa-to-linalg, tosa-to-tensor, tosa-to-arith
     ▼
Linalg MLIR
     │ matmul-parallel-vectorization-optimize
     │ batchmatmul-optimize / batchmatmul-scf-optimize
     │ dequant-matmul-vectorization-decode  [fused dequant+matmul]
     │ one-shot-bufferize
     │ convert-linalg-to-affine-loops
     │ affine-loop-fusion + affine-parallelize
     │ convert-scf-to-openmp
     │ convert-vector-to-llvm
     │ convert-func-to-llvm
     ▼
LLVM IR → native binary (.o)
     │
     ▼
Runtime: TextContainer tokenizer → _mlir_ciface_forward*() → argmax → next token
```

---

## 3. Graph-Level Optimizations

All graph passes operate on the Buddy Graph IR before MLIR lowering. They are applied in `frontend/Python/graph/transform/`.

### 3.1 Flash Attention Fusion

**File:** `frontend/Python/graph/transform/fuse_ops.py`  
**Pass:** `flash_attention_prefill(graph)`

PyTorch's `_scaled_dot_product_flash_attention_for_cpu` is captured as `ScaledDotProductFlashAttentionForCpuOp`. This pass replaces it with `FlashAttentionForCpuPrefillOp` — a single fused op that lowers to a block-tiled Flash Attention MLIR implementation instead of decomposing into separate `Q×K^T → softmax → ×V` tensor operations.

**Effect:** Eliminates the O(N²) intermediate attention score matrix; computes attention in tiles that fit in cache.

```python
# Usage in import scripts:
from buddy.compiler.graph.transform import flash_attention_prefill
graphs = dynamo_compiler.importer(model, ...)
graph_driver.generate_code(flash_attention_prefill)
```

### 3.2 GQA Attention Fusion

**File:** `frontend/Python/graph/transform/fuse_ops.py`  
**Pass:** `gqa_attention_fusion(graph)`

Grouped Query Attention (GQA) requires broadcasting K/V heads across multiple Q head groups. The raw PyTorch graph produces a chain of 8 intermediate ops per attention:

```
IndexPut → Unsqueeze → Expand → Clone → View
                                              ↘
                                        ScaledDotProductFlashAttentionForCpuOp
                                              ↗
IndexPut → Unsqueeze → Expand → Clone → View
```

The pass matches this pattern and collapses it into a single `GQAAttentionFusedOp`, eliminating all intermediate expanded tensors (e.g., `[1,2,1024,128]` broadcast to `[1,12,1024,128]`).

**Effect:** Removes ~8 intermediate nodes per attention layer, eliminates the temporary expanded K/V buffer allocation.

### 3.3 Transpose-Matmul Fusion

**File:** `frontend/Python/graph/transform/fuse_ops.py`  
**Pass:** `apply_classic_fusion(graph)` → `classic_fuse_check()`

Detects `permute([1,0]) → MatmulOp` patterns common in Q/K/V attention projections and fuses them into `TransposeMatmulFusedOp`. The weight is never materialized in transposed form — the matmul kernel reads it transposed directly.

**Effect:** Eliminates a full weight tensor copy for every linear projection that uses a transposed weight.

### 3.4 Weight Transpose Elimination

**File:** `frontend/Python/graph/transform/eliminate_weight_transpose.py`  
**Pass:** `eliminate_transpose(graph)`

Eliminates `TransposeOp`, `TOp` (2D `.t()`), and `PermuteOp` on model weight `PlaceholderOp`s that have only a single use. Instead of emitting a runtime transpose, the weight's shape metadata is modified in place before MLIR code generation.

**Effect:** Zero runtime cost for weight transposes — shape is corrected statically before any MLIR is emitted.

### 3.5 Matmul-Transpose-Reshape Elimination

**File:** `frontend/Python/graph/transform/eliminate_matmul_transpose_reshape.py`  
**Pass:** `eliminate_matmul_transpose_reshape(graph)`

Eliminates the pattern `weight → permute/transpose → reshape → matmul` that appears in attention head dimension splitting. Rewrites the weight shape metadata directly, removing the runtime permute and reshape chain.

**Effect:** Removes O(weight_size) memory operations per transformer layer during inference.

---

## 4. KV Cache Optimizations

### 4.1 Static Cache Shape Specialization

All LLM importers use HuggingFace `StaticCache` to fix KV cache tensor dimensions at compile time:

```python
past_key_values = StaticCache(config=model.config, max_cache_len=1024)
```

With fully static shapes, MLIR loop bounds and buffer sizes are compile-time constants, enabling:
- Aggressive loop unrolling in LLVM
- SIMD vectorization of attention inner loops
- Constant-folding of index arithmetic in KV cache scatter/gather

The KV update ops (`slice_scatter`, `index_put`, `as_strided_scatter`) are supported as first-class graph nodes in `DynamoCompiler` and lower to MLIR scatter/store operations.

### 4.2 Prefill / Decode Split Compilation

LLMs are compiled into two separate MLIR functions to avoid shape mismatches between the prefill and generation phases:

| Phase | Input shape | Matmul shape | KV operation |
|---|---|---|---|
| Prefill | `input_ids[1, seq_len]` | `[seq_len, head_dim]` | Writes all K/V entries at once |
| Decode | `input_ids[1, 1]` | `[1, head_dim]` (dot product) | Reads full cache, writes one entry |

The decode graph gets m=1 matmul passes applied (`matmul-vectorization-decode`, `batchmatmul-scf-optimize`) that exploit the `seq_len=1` structure for dot-product style kernels instead of full matrix kernels.

```python
# Separate compilation in import scripts:
dynamo_compiler_prefill = DynamoCompiler(func_name="forward_prefill", ...)
dynamo_compiler_decode  = DynamoCompiler(func_name="forward_decode",  ...)
```

### 4.3 Tiered KV Cache

**File:** `examples/BuddyDeepSeekR1/buddy-deepseek-r1-tiered-kv-cache-main.cpp`  
**Build:** `examples/BuddyDeepSeekR1/CMakeLists.txt`

The most significant KV cache optimization. Instead of always using a fixed `max_cache_len=1024` buffer (meaning attention always runs over 1024 positions even when only 45 tokens have been generated), the compiler generates **12 graph variants** — 6 prefill sizes + 6 decode sizes — for cache lengths `{32, 64, 128, 256, 512, 1024}`:

```cmake
set(KV_CACHE_SIZES 32 64 128 256 512 1024)
# Produces: forward_prefill_32.mlir, forward_decode_32.mlir, ...
#           forward_prefill_1024.mlir, forward_decode_1024.mlir
```

The runtime selects the smallest tier that fits the current context:

```cpp
size_t selectCacheSize(size_t currentPos) {
    for (size_t size : KV_CACHE_SIZES)
        if (currentPos < size) return size;  // pos=45 → uses cache_64
    return KV_CACHE_SIZES.back();
}
```

When context grows beyond the current tier, `copyKVCache<SrcLen, DstLen>()` copies the accumulated K/V state up to the next tier using `std::memcpy`.

**Effect:** Attention complexity is $O(n^2)$ in context length. Using `decode_64` instead of `decode_1024` for a 45-token context is approximately 16× less attention computation. Each tier's MLIR is specialized with the actual cache length as a compile-time constant.

### 4.4 GQA KV Cache Broadcast Fusion

For models with Grouped Query Attention (Qwen3, DeepSeekR1), K/V heads must be broadcast across Q head groups before attention. In `next-gqa-attention.mlir`, the KV cache broadcast is done via `tosa.add` with a zero tensor of the target shape, which the TOSA→Linalg lowering converts to a `linalg.generic` broadcast that the bufferization pass can overlap with the attention computation.

### 4.5 KV Cache as Explicit ABI Arguments

In the decode phase, all 56 KV cache tensors for DeepSeekR1 are passed as **explicit `MemRef<float,4>*` arguments** to the compiled MLIR function:

```cpp
void _mlir_ciface_forward_decode_64(
    DecodeContainer64 *result, MemRef<float,1> *params,
    MemRef<long long,2> *input_ids, MemRef<long long,1> *cache_position,
    MemRef<float,4> *kv0, MemRef<float,4> *kv1, ..., MemRef<float,4> *kv55);
```

This makes the KV cache ABI-visible memory that persists across decode steps with zero heap allocation or copy per token step. The same buffer is read and updated in place each iteration.

---

## 5. Attention-Specific Optimizations

### 5.1 Flash Attention MLIR Kernel

**File:** `examples/BuddyNext/next-flash-attention.mlir`

A complete hand-written MLIR implementation of the Flash Attention algorithm (Dao et al., 2022):

**Key properties:**
- **Block tiling:** `block_size_q = 32`, `block_size_kv = 32` — keeps Q/K/V blocks in L1 cache, eliminates the O(N²) intermediate score matrix from standard attention
- **Online softmax:** accumulates running max `m` and normalization factor `l` per Q-block; rescales the output accumulator as new KV blocks arrive — no recomputation pass needed
- **Parallelism:** outer `scf.parallel` over `(batch, q_block_idx)`, inner `scf.parallel` over output dimensions → compiles to OpenMP threads
- **Tested at seq_len=16384**

Algorithm:

```mlir
scf.parallel (%batch, %q_block_idx) ... {        // parallelized over batch × Q blocks
  // private: temp_o, temp_m, temp_l (in registers)
  scf.for %kv_start = %c0 to %seq_len step %block_size_kv {
    // 1. Compute scores = Q_block × K_block^T    (scf.parallel over i,j)
    // 2. new_m = max(old_m, row_max(scores))      (row-parallel)
    // 3. exp(scores - new_m), exp_sum             
    // 4. Rescale: old_o *= exp(old_m - new_m)    
    // 5. Accumulate: o += softmax_weights × V_block
    // 6. Update m, l statistics
  }
  // Normalize output: o /= l
}
```

**Additional flash attention files:**
- `next-flash-attention-prefill.mlir` — prefill-specific variant
- `next-attention-fusion.mlir` — fused attention variant

### 5.2 GQA Decode Kernel

**File:** `examples/BuddyNext/next-gqa-attention.mlir`

Implements GQA decode attention where KV heads (2) are fewer than Q heads (12):

1. KV cache `[1,2,1024,128]` broadcast to `[1,12,1024,128]` via zero-tensor add (compiler eliminates the materialization)
2. Attention scores: `linalg.batch_matmul_transpose_b` on `[12,1,128] × [12,1024,128]` → `[12,1,1024]`
3. Online log-sum-exp softmax in TOSA ops
4. Output: `tosa.matmul` for attention × V

**Additional attention kernel files in `examples/BuddyNext/`:**

| File | Description |
|---|---|
| `next-mhsa-core.mlir` | Multi-head self-attention core |
| `next-mhsa-qkv.mlir` | QKV projection |
| `next-mhsa-context.mlir` | Context aggregation |
| `next-attention.mlir` | Standard attention |
| `next-attention-loop.mlir` | Loop-structured attention |
| `next-gqa-attention-fusion.mlir` | Fused GQA variant |

### 5.3 Decode-Phase Batch Matmul Passes

For the token-generation loop (seq_len=1), attention `Q×K^T` and `attention×V` reduce to **batch matmuls with m=1**. Specialized midend passes exploit this structure:

| Pass | Description |
|---|---|
| `-batchmatmul-optimize` | Tiled+vectorized batch matmul (prefill QK^T, AV) |
| `-batchmatmul-scf-optimize` | SCF-loop batch matmul, decode-optimized |
| `-batchmatmul-trans-b-vec` | `linalg.batch_matmul_transpose_b` vectorization — matches attention score QK^T where K is stored transposed |
| `-batchmatmul-tile-optimize` | Tiled batch matmul with explicit register blocking |
| `-batchmatmul-vec-decode` | Decode-specific: exploits seq_len=1 for dot-product style kernel |

The decode pipeline in `CMakeLists.txt` applies these after bufferization:

```bash
buddy-opt forward_decode_64.mlir
    -one-shot-bufferize=...
    -matmul-vectorization-blis
    -batchmatmul-optimize
    -convert-linalg-to-affine-loops
    -affine-parallelize
    -convert-scf-to-openmp=...
```

### 5.4 Static Shapes Enable SIMD Vectorization

Because `StaticCache` and tiered cache sizes fix all attention tensor shapes at compile time, LLVM's backend can:
- Fully unroll attention inner reduction loops
- Emit AVX2/AVX-512 SIMD instructions for the dot products
- Eliminate all runtime bounds checks in the attention computation

---

## 6. Quantization Optimizations

All quantization passes operate on the Buddy Graph IR before MLIR lowering. Files are in `frontend/Python/graph/transform/quantization/`.

### 6.1 Weight-Only Quantization (w8a32, w8a16)

**Pass:** `weight_only_channel_wise(graph, dtype=...)` in `passes.py`  
**Implementation:** `weight_only_channel_wise.py` → `WeightOnlyQuantization`

Converts weight tensors to int8, inserts per-channel dequantization before matmul:

```
Original:  float_weight → MatmulOp
Quantized: int8_weight → CastOp(i8→f32) → MulOp(scale) → MatmulOp
```

- **w8a32**: activations remain f32, dequantized weight multiplied at f32 precision
- **w8a16**: activations remain f16

The quantization uses `ChannelWiseQuantizationConstraint` with per-output-channel scale factors stored alongside the model weights.

### 6.2 Int4 Weight Quantization (w4a16)

**Pass:** `weight_only_int4_f16_channel_wise(graph)` in `passes.py`  
**Implementation:** `WeightOnlyInt4F16Quantization`

Packs two int4 values per byte. The graph inserts:

```
int4_packed_weight → Int4UnpackOp → MulOp(scale_f16) → MatmulOp(f16)
```

Halves weight memory bandwidth compared to int8, at the cost of unpack overhead. This is particularly effective in decode where memory bandwidth is the bottleneck.

### 6.3 W8A8 Dynamic Quantization

**Pass:** `w8a8_channel_wise(graph)` in `passes.py`

Converts both weights (int8, static) and activations (int8, dynamic) to native integer matmul:

```
Original:  float_weight → MatmulOp(f32)
Quantized: int8_weight  → QuantizedMatmulOp(i8×i8→i32) → scale → f32
```

`_convert_to_quantized_matmul()` replaces the `CastOp→MulOp→MatmulOp` chain with `QuantizedMatmulOp` that uses the CPU's native 8-bit VNNI or SDOT instructions when available.

### 6.4 Fused Dequant+Matmul for Decode

**File:** `midend/lib/Conversion/MatMulOptimization/DequantMatMulVectorizationDecode.cpp`  
**Pass:** `-dequant-matmul-vectorization-decode`

The most impactful quantization optimization for token generation. Pattern-matches the three-op chain produced by w8a32 quantization:

```
linalg.generic(sitofp i8→f32)       // weight dequant
linalg.generic(mulf, per-channel)    // scale application  
linalg.matmul                        // matrix multiply
```

And fuses them into a single vectorized kernel that:
- Reads int8 weights directly (4× less memory bandwidth than f32)
- Casts and scales inside the reduction loop
- Applies per-channel scale after the accumulation completes
- Eliminates the intermediate f32 weight buffer entirely

The companion pass `-int4-dequant-matmul-vectorization-decode` handles the int4 packed variant.

**Memory bandwidth reduction:** For a decode step with a single output token, the weight matrix dominates memory access. Using int8 weights cuts weight bandwidth by 4× (f32 → i8), and fusing dequant into the matmul loop eliminates a full O(weight_size) buffer pass.

---

## 7. Midend MatMul Optimization Passes

These MLIR passes are applied after TOSA→Linalg lowering. All files are in `midend/lib/Conversion/MatMulOptimization/`.

| Pass Flag | File | Target Op | Use Case |
|---|---|---|---|
| `-matmul-optimize` | `MatMulOptimize.cpp` | `linalg.matmul` | Tiled with register blocking (kernelM×kernelN) |
| `-matmul-parallel-vectorization-optimize` | `MatMulParallelVectorization.cpp` | `linalg.matmul` | Multi-core parallel + SIMD (FFN prefill) |
| `-matmul-vec` | `MatMulVectorization.cpp` | `linalg.matmul` | Basic SIMD vectorization |
| `-matmul-vectorization-decode` | `MatMulVectorizationDecode.cpp` | `linalg.matmul` (m=1) | Token-step linear projections |
| `-matmul-trans-b-vec` | `MatMulTransposeBVec.cpp` | `linalg.matmul_transpose_b` | LLM Q/K/V projection (transposed weight) |
| `-matmul-trans-b-unroll-vec` | `MatMulTransposeBUnrollVec.cpp` | `linalg.matmul_transpose_b` | Unrolled+vectorized transpose-B |
| `-matmul-blis-vec` | `MatMulBlisVectorization.cpp` | `linalg.matmul` | BLIS-style 3-loop blocked matmul |
| `-matmul-amx` | `MatmulAMX.cpp` | `linalg.matmul` | Intel AMX tile instruction acceleration |
| `-batchmatmul-optimize` | `BatchMatMulOptimize.cpp` | `linalg.batch_matmul` | Attention QK^T and AV (prefill) |
| `-batchmatmul-tile-optimize` | `BatchMatMulTileOptimize.cpp` | `linalg.batch_matmul` | Tiled batch matmul |
| `-batchmatmul-scf-optimize` | `BatchMatMulSCFOptimize.cpp` | `linalg.batch_matmul` | SCF-loop batch matmul (decode) |
| `-batchmatmul-trans-b-vec` | `BatchMatMulTransBVec.cpp` | `linalg.batch_matmul_transpose_b` | Decode attention with transposed K cache |
| `-dequant-matmul-vectorization-decode` | `DequantMatMulVectorizationDecode.cpp` | `sitofp+mulf+matmul` | Fused w8a32 dequant+matmul (decode) |
| `-int4-dequant-matmul-vectorization-decode` | `Int4DequantMatMulVectorizationDecode.cpp` | `int4_unpack+mulf+matmul` | Fused w4a16 dequant+matmul (decode) |

### Decode vs. Prefill Pass Selection

The build system applies different pass pipelines for the two phases:

**Prefill** (large batch, high arithmetic intensity):
```bash
-matmul-parallel-vectorization-optimize
-batchmatmul-optimize
-affine-parallelize
-convert-scf-to-openmp
```

**Decode** (m=1, memory-bound):
```bash
-matmul-vectorization-blis        # or -dequant-matmul-vectorization-decode
-batchmatmul-optimize
-convert-scf-to-openmp
```

---

## 8. Runtime LLM Support

### TextContainer Tokenizers

**File:** `frontend/Interfaces/buddy/LLM/TextContainer.h`

`TextContainer<T, N>` extends `MemRef<T, N>` with multiple tokenizer implementations for different model families:

| Method | Model Family | Algorithm |
|---|---|---|
| `tokenizeBert(vocab, length)` | BERT | WordPiece |
| `tokenizeLlama(vocab, length)` | LLaMA | SentencePiece BPE |
| `tokenizeStableDiffusion(vocab, length)` | Stable Diffusion | BPE |
| `tokenizeDeepSeekR1(vocab, length)` | DeepSeekR1/Qwen | Tiktoken |

### Inference Runtime Pattern

The compiled MLIR exposes a C interface that the runtime calls directly:

```cpp
// Prefill: process the prompt
_mlir_ciface_forward_prefill_64(&prefill_container, &params, &input_tokens);
// Copy KV state to decode container
copyKVCache(prefill_container, decode_container, token_count);

// Decode: generate one token at a time
while (!done) {
    size_t cache_size = selectCacheSize(current_pos);   // tiered selection
    _mlir_ciface_forward_decode_64(&decode_container, &params,
                                    &input_ids, &cache_position,
                                    kv0, kv1, ..., kv55);
    next_token = argmax(decode_container.logits);
}
```

### BuddyNext Operator Kernels

`examples/BuddyNext/` provides complete, standalone MLIR benchmarks for every LLM operator, each with a full lowering pipeline to LLVM IR:

| Kernel | Operator |
|---|---|
| `next-rope.mlir` | RoPE (Rotary Position Encoding) |
| `next-flash-attention.mlir` | Flash Attention (parallel, block_size=32) |
| `next-gqa-attention.mlir` | GQA decode attention |
| `next-ffn.mlir` / `next-ffn-optimized.mlir` | Feed-forward network (SiLU gate + matmuls) |
| `next-norm.mlir` | RMSNorm (parallel matmul+batchmatmul) |
| `next-embedding.mlir` | Token embedding lookup |
| `next-positional-encoding.mlir` | Sinusoidal positional encoding |
| `next-sgemm*.mlir` | SGEMM variants (fixed/aligned/scalable) |
| `next-mmtb*.mlir` | Matmul-transpose-B for attention |
| `next-linalg-matmul-decode.mlir` | Decode-specific matmul (m=1) |
| `next-batchmatmul-decode-size.mlir` | Decode-phase batch matmul |
| `next-transpose-vec-manual.mlir` | Manually vectorized head transpose |
| `next-sigmoid.mlir` / `next-silu-fusion-test.mlir` | Activation functions |
