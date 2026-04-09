# BuddyQwen3 Architecture & Code Analysis

## Overview

The `BuddyQwen3` folder implements an **AOT (Ahead-of-Time) compilation pipeline** for Qwen3-0.6B inference using MLIR/LLVM. It produces a fully native CPU binary with zero Python dependencies at runtime.

## File Structure

```
BuddyQwen3/
├── import-qwen3.py            # Model import & graph compilation (Python)
├── CMakeLists.txt             # Multi-stage MLIR lowering pipeline
├── buddy-qwen3-0.6b-main.cpp  # C++ runtime inference loop
├── vocab.txt                  # Qwen3 tokenizer vocabulary (151,936 entries)
└── README.md                  # Build & run instructions
```

---

## Overall Architecture

```
PyTorch Model (HuggingFace Qwen/Qwen3-0.6B)
        ↓  import-qwen3.py
        │  DynamoCompiler + graph-level optimizations
        │
   MLIR files (TOSA dialect)       arg0_0_6b.data (raw f32 weights)
   ├── forward_prefill_0_6b.mlir
   ├── subgraph0_prefill_0_6b.mlir
   ├── forward_decode_0_6b.mlir
   └── subgraph0_decode_0_6b.mlir
        ↓  CMakeLists.txt
        │  buddy-opt → mlir-opt → mlir-translate → llc
        │
   Native .o files (vectorized, OpenMP-parallelized)
        ↓  linked via CMake
   buddy-qwen3-0.6b-run  (C++ binary)
        │  reads vocab.txt + arg0_0_6b.data
        ↓
   Interactive chat inference (prefill → decode loop)
```

---

## File Analysis

### `import-qwen3.py` — Model Import & Graph Compilation

This is the **entry point of the compilation pipeline**. It captures the PyTorch model's compute graph and lowers it to MLIR.

#### Key Steps

1. **Load the model** from HuggingFace or a local path:
   ```python
   model_path = os.environ.get("QWEN3-0.6B_MODEL_PATH") or "Qwen/Qwen3-0.6B"
   model = AutoModelForCausalLM.from_pretrained(model_path, torchscript=True).eval()
   ```

2. **Split inference into two phases** — a critical performance optimization:
   - **Prefill**: processes the full input prompt at once (`input_ids` shape `[1, 1024]`)
   - **Decode**: generates one token at a time (`input_ids` shape `[1, 1]`) using a static KV cache

3. **Use `DynamoCompiler`** (Buddy's PyTorch frontend) to trace and capture the compute graph via `torch.compile` / AOT autograd.

4. **Apply graph-level optimizations** before lowering to MLIR:

   | Pass | Effect |
   |---|---|
   | `eliminate_transpose` | Remove redundant transpose ops |
   | `eliminate_matmul_transpose_reshape` | Fuse/remove transposes around matmuls |
   | `flash_attention_prefill` | Fuse multi-head attention ops (prefill) |
   | `gqa_attention_fusion` | Fuse grouped-query attention (decode) |
   | `simply_fuse`, `apply_classic_fusion` | General op fusion |

5. **Output files** written to `--output-dir` (default `./`):
   - `forward_prefill_0_6b.mlir` — top-level function dispatching to subgraph (prefill)
   - `subgraph0_prefill_0_6b.mlir` — actual compute subgraph (prefill)
   - `forward_decode_0_6b.mlir` — top-level function (decode)
   - `subgraph0_decode_0_6b.mlir` — actual compute subgraph (decode)
   - `arg0_0_6b.data` — all model weights serialized as a flat raw `float32` binary

---

### `CMakeLists.txt` — Multi-Stage MLIR Lowering Pipeline

Drives compilation from MLIR down to native object files. All four MLIR files go through this pipeline:

```
MLIR (TOSA ops)
  → buddy-opt   (-simplify-tosa-reshape, custom buddy passes)
  → mlir-opt    (TOSA → Linalg → Arith/Tensor)
  → buddy-opt   (bufferization → affine loops → OpenMP → LLVM dialect)
  → mlir-translate  (-mlir-to-llvmir)
  → llvm-as → llc   (native .o, -O3, -relocation-model=pic)
```

#### Key Optimization Passes

| Pass | Applied To | Purpose |
|---|---|---|
| `-matmul-vectorization-blis` | prefill | BLIS-style tiled matrix multiplication |
| `-batchmatmul-optimize` | prefill | Optimize batched matmul |
| `-matmul-vectorization-decode=vector-size=128` | decode | SIMD-vectorized decode matmul |
| `-batch-matmul-vectorization-decode=vector-size=128` | decode | SIMD batched matmul for decode |
| `-batchmatmul-transpose-b-vectorization` | both | Vectorize transposed-B matmul |
| `-affine-parallelize` | both | Parallelize affine loops |
| `-convert-scf-to-openmp=num-threads=48` | both | Multi-threaded OpenMP (48 threads on x86; 32 on RVV) |
| `-eliminate-memref-copy` | decode | Remove redundant memory copies |
| `-assume-tight-memref-layout` / `-staticize-memref-layout` | decode | Enable static layout analysis |

#### Platform Detection

The build detects **RISC-V Vector (RVV)** at configure time:
- **x86/ARM**: 48 OpenMP threads, no RVV attrs
- **RISC-V**: 32 OpenMP threads, `mattr=+m,+d,+v`, `march=rv64gcv`

The import step (`import-qwen3.py`) is skipped on RVV platforms (pre-compiled MLIR is expected).

#### Static Library & Executable

The four `.o` files are bundled into a CMake static library `QWEN3_0_6B`, then linked into the final executable `buddy-qwen3-0.6b-run` along with `mlir_c_runner_utils` and `omp`.

---

### `buddy-qwen3-0.6b-main.cpp` — C++ Runtime Inference Loop

The executable that runs the compiled model entirely in C++.

#### Model Constants

```cpp
constexpr size_t ParamsSize     = 751,632,448;  // ~2.8 GB weights (float32)
constexpr size_t MaxVocabSize   = 151,936;
constexpr size_t MaxTokenLength = 1024;
constexpr size_t NUM_LAYERS     = 56;
constexpr size_t HeadNum        = 8;
constexpr size_t HiddenSize     = 128;
```

#### KV Cache Management

Each of the 56 transformer layers has a dedicated `MemRef<float, 4>` of shape `[1, HeadNum, MaxTokenLength, HiddenSize]`. They are grouped in a `MemRefContainer` struct, which:
- Holds 56 KV tensors as named fields (`kv0`…`kv55`) plus a `logits` tensor
- Exposes a `kv_ptrs` array for bulk operations
- Is passed directly to the MLIR-generated C interface functions

After prefill, `copy_kv_by_cache_position_block()` memcpy-copies the populated KV states from the prefill container into the decode container, transferring only the filled prefix (not the full 1024-slot buffer).

#### External MLIR Function Signatures

```cpp
// Prefill: full prompt → logits + KV cache populated
extern "C" void _mlir_ciface_forward_prefill(
    MemRefContainer *result, MemRef<float, 1> *params, Text<size_t, 2> *input_ids);

// Decode: single token + KV cache → next logit + updated KV cache
extern "C" void _mlir_ciface_forward_decode(
    MemRefContainer *result, MemRef<float, 1> *params,
    MemRef<long long, 2> *input_ids, MemRef<long long, 1> *cache_position,
    MemRef<float, 4> *kv0, ... /* 56 KV args */);
```

#### Inference Loop

```
1. getUserInput()          → read prompt from stdin
2. tokenizeInput()         → Text::tokenizeQwen3(vocab.txt)
3. loadParameters()        → read arg0_0_6b.data into flat MemRef<float,1>
4. forward_prefill()       → fill KV cache + get first token logits
5. copy_kv_by_cache_position_block()  → copy prefill KV → decode KV
6. for each decode step:
   a. forward_decode()     → single-token inference (updates KV cache in-place)
   b. argmax over logits   → next token index
   c. decode token string  → print + timing
   d. break on EOS or MaxTokenLength
7. print prefill/decode throughput (tokens/sec)
```

#### Greedy Decoding

Token selection uses simple argmax (`findMaxIndex`) — no sampling, temperature, or top-p. This is intentional for deterministic, low-latency inference.

---

### `vocab.txt` — Tokenizer Vocabulary

Contains all 151,936 Qwen3 token strings, one per line, indexed by position. Used by `Text<size_t, 2>::tokenizeQwen3()` and `loadVocab()` in the C++ runtime. Eliminates any Python/sentencepiece dependency at inference time.

---

## Data Flow Summary

```
vocab.txt ──────────────────────────────────────┐
                                                 ↓
User prompt ──→ tokenizeQwen3() ──→ Text<size_t,2> inputContainerPrefill
                                                 │
arg0_0_6b.data ──→ loadParameters() ──→ MemRef<float,1> ParamsContainer
                                                 │
                                    forward_prefill()
                                          │
                              MemRefContainer (kv0…kv55 + logits_prefill)
                                          │
                              copy_kv_by_cache_position_block()
                                          │
                              [decode loop] forward_decode()
                                          │
                              argmax ──→ token string ──→ stdout
```

## Performance Notes

- **Parallelism**: OpenMP with 48 threads (x86) parallelizes affine loops across cores
- **Vectorization**: BLIS tiling for prefill matmuls; 128-wide SIMD for decode matmuls
- **Memory**: KV cache is pre-allocated statically for the full `MaxTokenLength=1024` context
- **Prefill**: processes all 1024 tokens in a single forward pass (highly parallelizable)
- **Decode**: one token per call, KV cache updated in-place (latency-sensitive path)
- **Optional**: `mimalloc` can replace the system allocator for reduced allocation overhead
