# RVV Version Design and Implementation: Scalable Packed Layouts for Vector-Length-Agnostic ML Code Generation

## 0. Purpose

This document describes how to adapt the paper **“Scalable Packed Layouts for Vector-Length-Agnostic ML Code Generation”** from Arm SVE to **RISC-V Vector Extension (RVV)**.

The goal is to build an MLIR/IREE-style compiler path that lowers:

```text
linalg.matmul
  → linalg.pack
  → linalg.mmt4d
  → linalg.unpack
  → vector dialect with scalable vectors
  → LLVM/RISC-V vector lowering
  → RVV machine code
```

The key idea is:

```text
Do not treat vector length only as a late LLVM/RVV backend issue.

Instead, make vector length visible earlier as a layout-level and tiling-level abstraction.
```

For RVV, this means designing **VLEN/SEW/LMUL-aware packed layouts** and ensuring that `pack`, `mmt4d`, vectorization, bufferization, and lowering are all consistent with the selected RVV microkernel.

---

## 1. Background and Motivation

### 1.1 Problem

High-performance ML inference on CPUs depends heavily on optimized matrix multiplication. For LLM workloads, the dominant compute patterns include:

```text
Linear projection
QKᵀ
P · V
Feed-forward network matmul
Output projection
Batched contraction
GEMV/GEMM-like decode-time kernels
```

Classical SIMD code generation assumes a fixed vector length. For example, Arm NEON always has 128-bit vectors. This allows tile sizes such as:

```text
m_r = 8
n_r = 8
k_r = 1
```

to be chosen at compile time.

RVV is different. The hardware vector length is implementation-defined:

```text
VLEN = 128 / 256 / 512 / ... bits
```

The actual runtime vector length `vl` is configured by:

```text
vsetvl / vsetvli
```

and depends on:

```text
AVL: application vector length
SEW: selected element width
LMUL: vector register group multiplier
VLEN: hardware vector length
```

Therefore, a fixed packed layout may underutilize wider RVV hardware.

---

### 1.2 Core Insight

Instead of fixing the register tile shape statically, define the register tile shape as a function of RVV vector capacity:

```text
m_r = f_m(VLEN, SEW, LMUL)
n_r = f_n(VLEN, SEW, LMUL)
k_r = f_k(VLEN, SEW, LMUL)
```

For a simple FP32 RVV outer-product microkernel:

```text
m_r = 4
n_r = VLMAX
k_r = 1
```

or a wider version:

```text
m_r = 4
n_r = 2 × VLMAX
k_r = 1
```

where:

```text
VLMAX = maximum number of elements in one vector register group
       = (VLEN / SEW) × LMUL
```

For FP32 with LMUL=1:

```text
VLMAX = VLEN / 32
```

Examples:

```text
VLEN = 128-bit → VLMAX = 4 FP32 elements
VLEN = 256-bit → VLMAX = 8 FP32 elements
VLEN = 512-bit → VLMAX = 16 FP32 elements
```

---

## 2. Target Architecture Assumptions

### 2.1 RVV Baseline

Initial implementation target:

```text
ISA: RV64GCV
Vector ISA: RVV 1.0
Data type: FP32
Kernel type: outer-product matmul microkernel
Vector operations:
  vle32.v
  vse32.v
  vfmacc.vf
  vfmv.v.f / scalar broadcast as needed
  vsetvli / vsetvl
```

Optional future targets:

```text
BF16: Zvfbfmin / vendor extension / custom lowering
FP16: Zvfh
INT8: vector dot-product or custom extension
Custom NPU instruction: fixed or scalable matrix instruction
```

---

### 2.2 RVV-Specific Parameters

The compiler backend should model the following parameters:

```text
VLEN      hardware vector register length in bits
ELEN      maximum element width
SEW       selected element width
LMUL      vector register grouping factor
VLMAX     maximum runtime vector length for current SEW/LMUL
VL        actual runtime vector length selected by vsetvl
```

For code generation, distinguish:

```text
VLMAX: used to define the steady-state packed tile shape
vl:    runtime value used by RVV instructions
```

For core packed matmul, the preferred strategy is:

```text
Use VLMAX-sized tiles for the main computation.
Handle edge cases through padding or masked pack/unpack.
```

This keeps the inner microkernel regular.

---

## 3. RVV Scalable Packed Layout Design

### 3.1 Standard Packed Matmul Form

Original row-major matmul:

```text
A: M × K
B: K × N
C: M × N
```

Packed form:

```text
A_pack: M_o × K_o × m_r × k_r
B_pack: K_o × N_o × k_r × n_r
C_pack: M_o × N_o × m_r × n_r
```

where:

```text
M_o = ceil(M / m_r)
N_o = ceil(N / n_r)
K_o = ceil(K / k_r)
```

Indexing relation for A:

```text
A_pack[i_o, k_o, i_i, k_i]
=
A[i_o × m_r + i_i, k_o × k_r + k_i]
```

For B:

```text
B_pack[k_o, j_o, k_i, j_i]
=
B[k_o × k_r + k_i, j_o × n_r + j_i]
```

For C:

```text
C_pack[i_o, j_o, i_i, j_i]
=
C[i_o × m_r + i_i, j_o × n_r + j_i]
```

---

### 3.2 RVV Version A: Conservative FP32 Kernel

Recommended first prototype:

```text
m_r = 4
n_r = VLMAX
k_r = 1
```

Packed tensors:

```text
A_pack: M_o × K × 4 × 1
B_pack: K × N_o × 1 × VLMAX
C_pack: M_o × N_o × 4 × VLMAX
```

Why this is a good starting point:

```text
Register pressure is moderate.
Only 4 vector accumulators are needed.
One B vector is loaded per K iteration.
Simple mapping to vfmacc.vf.
Easy to debug.
Good first target for MLIR lowering.
```

Conceptual microkernel:

```text
for each packed C tile of shape 4 × VLMAX:
  initialize c0, c1, c2, c3 as RVV vector accumulators

  for k in 0..K:
    b = load B_pack[k, j_o, 0, 0:VLMAX]

    a0 = A_pack[i_o, k, 0, 0]
    a1 = A_pack[i_o, k, 1, 0]
    a2 = A_pack[i_o, k, 2, 0]
    a3 = A_pack[i_o, k, 3, 0]

    c0 += a0 × b
    c1 += a1 × b
    c2 += a2 × b
    c3 += a3 × b

  store c0..c3 into C_pack
```

RVV intrinsic-style pseudocode:

```c
void rvv_matmul_4xvl_f32(
    const float *A_pack,
    const float *B_pack,
    float *C_pack,
    int K,
    int n_valid) {
  size_t vl = __riscv_vsetvl_e32m1(n_valid);

  vfloat32m1_t c0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t c1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t c2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t c3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

  for (int k = 0; k < K; ++k) {
    vfloat32m1_t b = __riscv_vle32_v_f32m1(B_pack + k * vl, vl);

    float a0 = A_pack[k * 4 + 0];
    float a1 = A_pack[k * 4 + 1];
    float a2 = A_pack[k * 4 + 2];
    float a3 = A_pack[k * 4 + 3];

    c0 = __riscv_vfmacc_vf_f32m1(c0, a0, b, vl);
    c1 = __riscv_vfmacc_vf_f32m1(c1, a1, b, vl);
    c2 = __riscv_vfmacc_vf_f32m1(c2, a2, b, vl);
    c3 = __riscv_vfmacc_vf_f32m1(c3, a3, b, vl);
  }

  __riscv_vse32_v_f32m1(C_pack + 0 * vl, c0, vl);
  __riscv_vse32_v_f32m1(C_pack + 1 * vl, c1, vl);
  __riscv_vse32_v_f32m1(C_pack + 2 * vl, c2, vl);
  __riscv_vse32_v_f32m1(C_pack + 3 * vl, c3, vl);
}
```

In production compiler lowering, avoid hardcoding intrinsic calls too early. Instead, represent this first in MLIR `vector` dialect with scalable vector types.

---

### 3.3 RVV Version B: Wider FP32 Kernel

Second-stage optimization:

```text
m_r = 4
n_r = 2 × VLMAX
k_r = 1
```

Packed tensors:

```text
A_pack: M_o × K × 4 × 1
B_pack: K × N_o × 1 × 2VLMAX
C_pack: M_o × N_o × 4 × 2VLMAX
```

Microkernel state:

```text
8 vector accumulators:
  c00, c01
  c10, c11
  c20, c21
  c30, c31
```

Each row of C has two RVV vectors.

This improves reuse of A scalars:

```text
Each loaded A scalar updates two B vectors.
```

But it increases register pressure.

Use this only after the conservative version works.

---

### 3.4 RVV Version C: 8 × VLMAX Kernel

Alternative second-stage version:

```text
m_r = 8
n_r = VLMAX
k_r = 1
```

Microkernel state:

```text
8 vector accumulators.
```

Compared with `4 × 2VLMAX`:

```text
Same number of accumulators.
Different reuse pattern.
Better when M dimension is large and contiguous A access is efficient.
```

Benchmark both versions.

---

### 3.5 Choosing Tile Shapes

A practical heuristic:

| Data Type | LMUL | Candidate `(m_r, n_r, k_r)` | Accumulators | Notes |
|---|---:|---:|---:|---|
| FP32 | 1 | `(4, VLMAX, 1)` | 4 | Best first prototype |
| FP32 | 1 | `(4, 2VLMAX, 1)` | 8 | Better A reuse |
| FP32 | 1 | `(8, VLMAX, 1)` | 8 | More M rows |
| FP32 | 2 | `(4, VLMAX, 1)` | 4 groups | Safer register pressure |
| FP32 | 4 | `(2, VLMAX, 1)` | 2 groups | Conservative for large LMUL |
| BF16/FP16 | 1 | `(4, VLMAX, 1)` or `(8, VLMAX, 1)` | 4/8 | Higher VLMAX, more bandwidth pressure |
| INT8 | 1 | Depends on dot-product strategy | TBD | Needs different microkernel |

General selection principle:

```text
Choose the largest tile that:
  1. keeps accumulators in vector registers,
  2. avoids spilling,
  3. gives contiguous B loads,
  4. has simple C stores,
  5. allows efficient pack/unpack.
```

---

## 4. MLIR IR Design

### 4.1 High-Level Entry Point

Start from:

```mlir
func.func @matmul(%A: tensor<?x?xf32>,
                  %B: tensor<?x?xf32>,
                  %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul
       ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
       outs(%C : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
```

---

### 4.2 Transform to Packed Form

Conceptual target:

```mlir
%A_pack = linalg.pack %A
  inner_dims_pos = [0, 1]
  inner_tiles = [4, 1]
  : tensor<?x?xf32> -> tensor<?x?x4x1xf32>

%B_pack = linalg.pack %B
  inner_dims_pos = [0, 1]
  inner_tiles = [1, VLMAX]
  : tensor<?x?xf32> -> tensor<?x?x1x[VL]xf32>

%C_pack = linalg.mmt4d
  ins(%A_pack, %B_pack)
  outs(%C_init_pack)
  : tensor<?x?x4x1xf32>,
    tensor<?x?x1x[VL]xf32>
    -> tensor<?x?x4x[VL]xf32>

%C_out = linalg.unpack %C_pack
  : tensor<?x?x4x[VL]xf32> -> tensor<?x?xf32>
```

The exact MLIR syntax for scalable dimensions may require using `vector` types or dynamic dimensions rather than writing `[VL]` directly in tensor types. Treat the above as a design-level representation.

---

### 4.3 Representation Choices

There are three possible implementation strategies.

#### Strategy A: Reuse `linalg.pack` / `linalg.mmt4d` Directly

Pros:

```text
Maximum reuse of MLIR infrastructure.
Aligned with IREE-style data-tiled matmul.
Easier to integrate with existing transform dialect and vectorization.
```

Cons:

```text
May need patches if linalg.pack inner tile sizes must be static integers.
Scalable inner dimensions may not be fully supported everywhere.
```

#### Strategy B: Introduce RVV-Aware Layout Attributes

Example:

```mlir
#rvv.layout<mmt4d, mr = 4, nr = "vlmax", kr = 1, sew = 32, lmul = 1>
```

Attach to packed tensor or custom ops:

```mlir
%pB = rvv_pack %B {
  layout = #rvv.layout<mmt4d, mr = 4, nr = "vlmax", kr = 1>
}
```

Pros:

```text
Explicitly models RVV layout.
Good for custom backend experimentation.
```

Cons:

```text
Requires more custom passes.
Less reuse of existing linalg transformations.
```

#### Strategy C: Introduce a Hardware-Aware Dialect

Example:

```mlir
rvvhw.pack
rvvhw.mmt4d
rvvhw.unpack
rvvhw.matmul_microkernel
```

Pros:

```text
Clear boundary between generic linalg and RVV-specific lowering.
Useful if targeting both RVV and custom NPU instructions.
```

Cons:

```text
More engineering effort.
Need custom bufferization, vectorization, canonicalization.
```

Recommended path:

```text
Start with Strategy A.
Add Strategy B attributes only where MLIR needs extra information.
Move to Strategy C only if you need custom NPU integration or if upstream linalg is too restrictive.
```

---

## 5. Compiler Pipeline

### 5.1 Proposed Pass Pipeline

```text
Input MLIR / torch-mlir / StableHLO
  ↓
Canonicalize tensor operations
  ↓
Legalize to linalg
  ↓
Detect matmul / batch matmul / contraction
  ↓
Select RVV matmul config
  ↓
Convert linalg.matmul to scalable packed form
  ↓
Propagate packed layout through elementwise consumers/producers
  ↓
Tile outer loops for cache and parallelism
  ↓
Vectorize mmt4d inner computation
  ↓
Lower vector.transfer / vector.contract to RVV-compatible vector ops
  ↓
Bufferize packed tensors
  ↓
Lower to LLVM dialect
  ↓
LLVM RISC-V backend
  ↓
RVV binary
```

---

### 5.2 Pass 1: Matmul Detection

Detect:

```text
linalg.matmul
linalg.batch_matmul
linalg.generic with contraction indexing maps
attention contractions such as QKᵀ and P·V
```

For each contraction, extract:

```text
M, N, K
element type
transpose flags
batch dimensions
static/dynamic shape information
consumer/producer ops
```

Reject or defer:

```text
unsupported quantization
unsupported layouts
very small matmuls where packing overhead dominates
convolutions unless lowered through im2col
```

---

### 5.3 Pass 2: RVV Kernel Configuration Selection

Define a configuration object:

```c++
struct RVVMatmulConfig {
  Type elementType;
  int64_t sew;
  Rational lmul;
  int64_t mr;
  RVVScalableExpr nr;
  int64_t kr;

  int64_t numAccumulators;
  bool useTwoNVectors;
  bool usePackedA;
  bool usePackedB;
  bool usePackedC;

  StringRef microkernelName;
};
```

Example configurations:

```text
FP32 baseline:
  sew = 32
  lmul = 1
  mr = 4
  nr = VLMAX
  kr = 1
  accumulators = 4

FP32 wider:
  sew = 32
  lmul = 1
  mr = 4
  nr = 2 × VLMAX
  kr = 1
  accumulators = 8

FP32 M-wide:
  sew = 32
  lmul = 1
  mr = 8
  nr = VLMAX
  kr = 1
  accumulators = 8
```

Selection heuristic:

```text
if M and N are large:
  try wider configs
else if N is small:
  use smaller nr
else:
  use baseline 4 × VLMAX
```

For LLM decode-time GEMV-like kernels:

```text
batch = 1
M may be small
N may be large
K may be large
```

A `1 × VLMAX`, `2 × VLMAX`, or `4 × VLMAX` variant may be better than a large `m_r`.

---

### 5.4 Pass 3: Convert Matmul to Scalable Packed Mmt4d

Rewrite:

```mlir
linalg.matmul
```

to:

```text
pack A
pack B
initialize packed C
mmt4d on packed tensors
unpack C
```

Important details:

```text
A inner tile: [m_r, k_r]
B inner tile: [k_r, n_r]
C inner tile: [m_r, n_r]
```

For RVV baseline:

```text
A inner tile: [4, 1]
B inner tile: [1, VLMAX]
C inner tile: [4, VLMAX]
```

If MLIR cannot directly encode `VLMAX` as an inner tile, represent it as:

```text
dynamic dimension + layout attribute
or scalable vector type during vectorization
or custom rvv.pack op
```

---

### 5.5 Pass 4: Packed Layout Propagation

Try to avoid unnecessary:

```text
pack → mmt4d → unpack → elementwise
```

Rewrite to:

```text
pack → mmt4d → elementwise_on_packed → unpack
```

Good candidates:

```text
add
bias add
relu
gelu approximation
silu
mul
sub
clamp
pointwise quant/dequant
```

Harder candidates:

```text
softmax
layernorm
transpose
reshape with non-layout-preserving semantics
reduction across packed dimension
```

Example:

```text
Before:
  C = matmul(A, B)
  D = relu(C)

After:
  C_pack = mmt4d(pack(A), pack(B))
  D_pack = relu(C_pack)
  D = unpack(D_pack)
```

This amortizes packing cost and reduces memory traffic.

---

### 5.6 Pass 5: Outer Tiling and Parallelization

Separate two tiling levels:

```text
Register tile:
  m_r × n_r × k_r
  determined by RVV microkernel

Cache/thread tile:
  T_M × T_N × T_K
  determined by cache and thread scheduling
```

Example loop structure:

```c
for i_t in range(0, M_o, T_M):
  for j_t in range(0, N_o, T_N):
    for k_t in range(0, K_o, T_K):
      parallel / tiled region
        for i_o in T_M:
          for j_o in T_N:
            for k_o in T_K:
              microkernel(A_pack[i_o,k_o], B_pack[k_o,j_o], C_pack[i_o,j_o])
```

Heuristics:

```text
Keep B tiles cache-resident when possible.
Parallelize across M_o × N_o tiles.
Avoid over-parallelizing small decode-time GEMV.
Consider NUMA/cache topology if available.
```

---

### 5.7 Pass 6: Vectorization

The mmt4d inner computation should vectorize to operations like:

```mlir
vector.transfer_read
vector.transfer_write
vector.fma
vector.broadcast
vector.contract or explicit vector ops
```

Conceptual vector type:

```mlir
vector<[1]xf32>      // one scalable vector
vector<[2]xf32>      // two scalable vectors, if supported/represented as two ops
```

For the baseline `4 × VLMAX` kernel:

```text
C accumulators:
  vector<[1]xf32> c0
  vector<[1]xf32> c1
  vector<[1]xf32> c2
  vector<[1]xf32> c3
```

For the `4 × 2VLMAX` kernel:

```text
C accumulators:
  c00, c01
  c10, c11
  c20, c21
  c30, c31
```

Avoid generating:

```text
scalar remainder loops inside the microkernel
unnecessary masks on full packed tiles
gather/scatter for packed B
```

---

### 5.8 Pass 7: Bufferization and Memory Planning

Packed buffers may have runtime-dependent sizes.

For baseline:

```text
B_pack size = K × ceil(N / VLMAX) × VLMAX
```

Since `VLMAX` may only be known at runtime, memory allocation can be:

```text
runtime-sized allocation using vlenb/csr query
or compile to code that computes VLMAX at runtime
```

RVV can read vector length in bytes using:

```text
csrr t0, vlenb
```

Then:

```text
VLMAX = (vlenb * 8 / SEW) * LMUL
```

For FP32 LMUL=1:

```text
VLMAX = vlenb / 4
```

Memory planning should account for:

```text
padding to VLMAX
alignment
lifetime of packed buffers
reuse between subgraphs
temporary buffers for pack/unpack
```

For static embedded deployment, a runtime memory plan can still be deterministic:

```text
At program initialization:
  query vlenb
  compute packed buffer sizes
  allocate workspace
```

---

### 5.9 Pass 8: Lower to LLVM/RVV

Lower scalable vector operations to LLVM scalable vector IR, then to RISC-V vector instructions.

Expected LLVM/RVV-level pattern:

```text
vsetvli
vle32.v
vfmacc.vf
vse32.v
```

Desired assembly shape:

```asm
vsetvli t0, a0, e32, m1, ta, ma
vle32.v v0, (b_ptr)
flw ft0, 0(a_ptr)
vfmacc.vf v8, ft0, v0
vse32.v v8, (c_ptr)
```

For packed full tiles:

```text
The compiler should prefer full-tile vector operations.
Masking should mainly appear in pack/unpack edge handling.
```

---

## 6. Runtime Handling of VLMAX and Padding

### 6.1 Runtime VLMAX Query

At runtime:

```c
size_t get_vlmax_f32_m1() {
  return __riscv_vsetvlmax_e32m1();
}
```

or with inline assembly / CSR:

```c
size_t get_vlenb() {
  size_t vlenb;
  asm volatile ("csrr %0, vlenb" : "=r"(vlenb));
  return vlenb;
}
```

Then:

```text
VLMAX_f32_m1 = vlenb / 4
```

---

### 6.2 Packing with Padding

For each tile:

```text
if original index is in bounds:
  copy original value
else:
  write padding value
```

Padding values:

```text
A/B padding: 0
C padding: 0 or ignored depending on output semantics
```

This allows the microkernel to compute full tiles without inner boundary checks.

---

### 6.3 Unpacking

When unpacking C:

```text
if output index is in bounds:
  store C_pack value to C
else:
  ignore padding region
```

Unpack is the main place where edge conditions are handled.

---

## 7. RVV Microkernel Design Details

### 7.1 Baseline 4 × VLMAX FP32 Outer Product

Register usage:

```text
4 vector accumulators
1 vector B operand
scalar A values
temporary registers
```

Pseudo assembly:

```asm
# a0 = A_pack pointer
# a1 = B_pack pointer
# a2 = C_pack pointer
# a3 = K
# a4 = n_valid or VLMAX

vsetvli t0, a4, e32, m1, ta, ma

vmv.v.i v8, 0       # c0
vmv.v.i v9, 0       # c1
vmv.v.i v10, 0      # c2
vmv.v.i v11, 0      # c3

loop_k:
  vle32.v v0, (a1)       # b vector

  flw ft0, 0(a0)
  flw ft1, 4(a0)
  flw ft2, 8(a0)
  flw ft3, 12(a0)

  vfmacc.vf v8, ft0, v0
  vfmacc.vf v9, ft1, v0
  vfmacc.vf v10, ft2, v0
  vfmacc.vf v11, ft3, v0

  addi a0, a0, 16        # 4 floats
  add  a1, a1, t_bstride
  addi a3, a3, -1
  bnez a3, loop_k

vse32.v v8,  (a2)
vse32.v v9,  offset1(a2)
vse32.v v10, offset2(a2)
vse32.v v11, offset3(a2)
```

---

### 7.2 Accumulation Policy

For FP32:

```text
accumulator type = FP32
input type = FP32
```

For BF16/FP16 future versions:

```text
input type = BF16/FP16
accumulator type = FP32 if supported
```

For INT8 future versions:

```text
input type = INT8
accumulator type = INT32
```

---

### 7.3 Tail Policy

Prefer:

```text
ta, ma
```

for full packed tiles where inactive elements are irrelevant.

For unpacking or true tails:

```text
use masks or reduced vl
```

Do not rely on inactive lanes preserving values unless explicitly required.

---

## 8. Integration with IREE

### 8.1 IREE CPU Backend Hook Points

Possible integration points:

```text
Flow / Dispatch formation:
  detect matmul-like dispatches

Codegen configuration:
  select RVV matmul tile config

Linalg transformation:
  generate pack/mmt4d/unpack

LLVMCPU vector lowering:
  lower scalable vector ops to RISC-V vector-compatible LLVM IR

HAL / runtime:
  query vlenb if workspace size depends on VLMAX
```

---

### 8.2 Codegen Configuration Attribute

Example design:

```mlir
#iree_codegen.translation_info<
  RVVScalableMmt4D,
  workgroup_size = [...],
  subgroup_size = ...,
  configuration = {
    mr = 4,
    nr = "vlmax",
    kr = 1,
    sew = 32,
    lmul = 1,
    pack_lhs = true,
    pack_rhs = true,
    pack_result = true
  }
>
```

---

### 8.3 Transform Dialect Sketch

A transform sequence could be:

```mlir
transform.sequence failures(propagate) {
^bb0(%module_op: !transform.any_op):
  %matmuls = transform.structured.match ops{["linalg.matmul"]} in %module_op
  transform.annotate %matmuls "rvv.mmt4d.config" = "f32_mr4_nrvl_kr1"

  %packed = transform.apply_patterns to %matmuls {
    transform.apply_patterns.rvv.convert_matmul_to_scalable_mmt4d
  }

  transform.structured.tile_using_for %packed [T_M, T_N, T_K]
  transform.structured.vectorize %packed
}
```

This is conceptual. The exact transform ops depend on the current MLIR/IREE version.

---

## 9. Custom Dialect Option

If upstream MLIR/IREE does not support scalable packed dimensions directly, introduce a small RVV hardware-aware dialect.

### 9.1 Dialect Name

```text
rvvhw
```

### 9.2 Operations

#### `rvvhw.pack`

```mlir
%p = rvvhw.pack %src {
  operand = "rhs",
  mr = 4,
  nr = #rvv.scalable<vlmax>,
  kr = 1,
  padding_value = 0.0
}
```

#### `rvvhw.mmt4d`

```mlir
%cp = rvvhw.mmt4d %ap, %bp, %cinit {
  mr = 4,
  nr = #rvv.scalable<vlmax>,
  kr = 1,
  sew = 32,
  lmul = 1
}
```

#### `rvvhw.unpack`

```mlir
%out = rvvhw.unpack %cp {
  output_shape = [M, N]
}
```

#### `rvvhw.query_vlmax`

```mlir
%vlmax = rvvhw.query_vlmax { sew = 32, lmul = 1 } : index
```

### 9.3 Lowering Path

```text
rvvhw.pack
  → scf loops + vector stores or memref copies

rvvhw.mmt4d
  → vector dialect scalable ops
  → LLVM scalable vector IR
  → RVV

rvvhw.unpack
  → scf loops + vector loads/stores with boundary handling
```

This dialect can later be folded back into generic MLIR once scalable pack support is mature.

---

## 10. Testing Plan

### 10.1 Unit Tests

Create MLIR lit tests for:

```text
matmul detection
matmul → pack/mmt4d/unpack rewrite
layout attribute propagation
vectorization to scalable vector types
bufferization with dynamic VLMAX-dependent sizes
LLVM lowering containing scalable vector IR
```

Example check patterns:

```text
CHECK: linalg.pack
CHECK: linalg.mmt4d
CHECK: vector.transfer_read
CHECK: vector<[1]xf32>
CHECK: llvm.intr.vp
CHECK: vsetvli
CHECK: vfmacc
```

---

### 10.2 Correctness Tests

Test matrix sizes:

```text
Small:
  M,N,K = 1..16

Non-multiple of VLMAX:
  M = 5, N = VLMAX + 3, K = 7

Large:
  128 × 128 × 128
  512 × 512 × 512

Skinny:
  1 × K × N
  4 × K × N

LLM-like:
  M = batch × sequence
  K = hidden
  N = 3 × hidden or 4 × hidden
```

Compare against reference:

```text
numpy / Eigen / plain C matmul
```

Error tolerance:

```text
FP32: relative tolerance around 1e-5
```

---

### 10.3 Performance Tests

Benchmark:

```text
linalg.matmul baseline
fixed mmt4d
RVV scalable mmt4d
vendor BLAS if available
handwritten RVV microkernel
```

Metrics:

```text
latency
GFLOP/s
speedup
cache misses
instruction count
vector instruction count
packing overhead
unpacking overhead
```

Hardware variants:

```text
RVV VLEN=128
RVV VLEN=256
RVV VLEN=512 if available
QEMU for correctness
Spike or gem5 for instruction-level validation
real board for performance
```

---

## 11. Benchmark Workloads

### 11.1 Microbenchmarks

```text
Square GEMM:
  64, 128, 256, 512, 1024

LLM prefill-like:
  M = sequence length
  K = hidden size
  N = hidden size

LLM decode-like:
  M = 1
  K = hidden size
  N = hidden size

Skinny-K:
  M = 2048
  N = 2048
  K = 512
```

---

### 11.2 Model-Level Benchmarks

Good candidates:

```text
TinyLlama
SmolLM
MobileBERT
ViT
Whisper encoder blocks
Qwen small models
```

Important breakdown:

```text
matmul time
non-matmul time
pack/unpack time
layout-propagated elementwise time
memory-bound operator time
```

---

## 12. Expected Challenges

### 12.1 Scalable Tensor Dimensions

MLIR tensor types may not naturally express:

```text
tensor<?x?x4xVLMAXxf32>
```

Possible solutions:

```text
Use dynamic dimensions plus layout attributes.
Lower scalable dimensions only at vectorization stage.
Introduce custom rvvhw ops.
Patch linalg.pack/mmt4d to accept scalable tile descriptors.
```

---

### 12.2 Buffer Size Computation

Because packed buffer sizes depend on VLMAX:

```text
N_o = ceil(N / VLMAX)
```

The compiler must emit runtime shape computation.

Possible approach:

```text
%vlenb = query_vlenb()
%vlmax = %vlenb / sizeof(f32)
%No = ceildiv(%N, %vlmax)
%buffer_size = %K * %No * %vlmax * sizeof(f32)
```

---

### 12.3 Register Pressure

RVV LMUL affects physical register availability.

Example:

```text
LMUL=1:
  32 vector registers available logically

LMUL=4:
  only 8 register groups
```

Therefore, avoid large accumulator tiles with high LMUL.

---

### 12.4 Packing Overhead

Packing is not free.

Avoid using packed layout for:

```text
very small matmuls
one-time tiny operations
ops where packing cost exceeds compute benefit
```

Use heuristics:

```text
if M × N × K is below threshold:
  use direct vectorized matmul
else:
  use packed mmt4d
```

---

### 12.5 Decode-Time LLM Inference

During autoregressive decoding:

```text
M is often 1
```

Classic GEMM-style `m_r = 4 or 8` may be inefficient.

Need separate kernels:

```text
GEMV-style RVV kernel
batch-size-1 linear layer kernel
KV-cache-aware attention kernel
```

Do not assume the same packed GEMM kernel is optimal for both prefill and decode.

---

## 13. Development Roadmap

### Phase 1: Standalone RVV Microkernel

Deliverables:

```text
C intrinsic microkernel for 4 × VLMAX FP32
plain C pack/unpack
correctness tests
basic benchmarks
```

Goal:

```text
Validate layout and microkernel before MLIR integration.
```

---

### Phase 2: MLIR Prototype Pass

Deliverables:

```text
Pass: convert linalg.matmul to fixed-shape mmt4d-like form
Use dynamic dimension to represent VLMAX
Generate explicit loops
Lower to vector dialect
```

Goal:

```text
Get end-to-end MLIR → RVV assembly for one matmul.
```

---

### Phase 3: Scalable Layout Attribute

Deliverables:

```text
RVV layout attribute
kernel config selection
pack/mmt4d/unpack rewrite
basic vectorization support
```

Goal:

```text
Make layout explicit and reusable.
```

---

### Phase 4: IREE Integration

Deliverables:

```text
IREE codegen config
dispatch-level matmul selection
bufferization support
runtime VLMAX query
benchmark through iree-run-module
```

Goal:

```text
Run real model subgraphs through IREE.
```

---

### Phase 5: Layout Propagation and Fusion

Deliverables:

```text
elementwise-on-packed support
bias/relu/gelu fusion
pack/unpack sinking/hoisting
cost model
```

Goal:

```text
Reduce layout conversion overhead.
```

---

### Phase 6: BF16/FP16/INT8 and LLM-Specific Kernels

Deliverables:

```text
BF16/FP16 kernels
INT8 kernels if hardware supports efficient dot product
decode-time GEMV kernels
attention-specific layouts
KV-cache layout optimization
```

Goal:

```text
Make the system useful for production LLM inference.
```

---

## 14. Minimal Example: End-to-End Lowering

### 14.1 Input

```mlir
%0 = linalg.matmul
  ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
  outs(%C : tensor<?x?xf32>) -> tensor<?x?xf32>
```

### 14.2 After RVV Packed Rewrite

```text
%A_pack = pack_lhs(%A, mr=4, kr=1)
%B_pack = pack_rhs(%B, kr=1, nr=VLMAX)
%C_pack = mmt4d(%A_pack, %B_pack, mr=4, nr=VLMAX, kr=1)
%C_out  = unpack_result(%C_pack)
```

### 14.3 After Vectorization

```text
for each C tile:
  c0, c1, c2, c3 : scalable vector<f32>

  for k:
    b : scalable vector<f32>
    a0, a1, a2, a3 : scalar f32

    c0 = fma(a0, b, c0)
    c1 = fma(a1, b, c1)
    c2 = fma(a2, b, c2)
    c3 = fma(a3, b, c3)
```

### 14.4 Expected RVV Assembly Shape

```asm
vsetvli ..., e32, m1
vle32.v
flw
vfmacc.vf
vse32.v
```

---

## 15. Summary

The RVV adaptation of scalable packed layouts is feasible and natural.

The main design rule is:

```text
Derive packed layout from the RVV microkernel access pattern.
```

For the first implementation, use:

```text
FP32 baseline:
  m_r = 4
  n_r = VLMAX
  k_r = 1
```

Then represent this in the compiler as:

```text
A_pack: M_o × K_o × 4 × 1
B_pack: K_o × N_o × 1 × VLMAX
C_pack: M_o × N_o × 4 × VLMAX
```

The compiler pipeline should be:

```text
linalg.matmul
  → RVV scalable pack
  → RVV scalable mmt4d
  → vectorization with scalable vectors
  → LLVM/RVV lowering
  → vsetvli + vle/vfmacc/vse
```

The most important engineering challenges are:

```text
1. representing scalable packed dimensions in MLIR,
2. computing VLMAX-dependent buffer sizes,
3. controlling RVV register pressure,
4. reducing pack/unpack overhead through layout propagation,
5. adding separate strategies for LLM decode-time GEMV.
```

The result is a compiler design where RVV vector-length agnosticism is handled at the layout and tiling level, not only at instruction selection.
