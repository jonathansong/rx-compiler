# RVVMatmul Example

Demonstrates a scalable FP32 matrix-multiplication microkernel written in the
`rvv` dialect and compiled to RISC-V Vector (RVV) assembly via buddy-mlir.

The example follows the packed-layout design described in
[`docs/rvv_scalable_packed_layout_design.md`](../../docs/rvv_scalable_packed_layout_design.md).

## Files

| File | Purpose |
|------|---------|
| `rvv-matmul-input.mlir` | Baseline `linalg.matmul` (16×16×16 FP32) — starting point |
| `rvv-matmul-static.mlir` | Target packed form (`linalg.pack` / `linalg.generic` mmt4d) — documentation only, not yet lowerable |
| `rvv-matmul-kernel.mlir` | **Runnable** `rvv`-dialect microkernel (4 × VLMAX outer-product loop) |
| `makefile` | Build targets (see below) |

## Microkernel Design

The kernel implements a `4 × VLMAX` outer-product over a strip of K columns:

```
for strip in 0..N step VLMAX:
  vl = rvv.setvl(N - strip)          # vsetvli
  b_strip = rvv.load(B[strip], vl)   # vle32.v
  for k in 0..K:
    a_scalar = A[m, k]               # scalar load
    c_row[m] += a_scalar * b_strip   # vector.broadcast + vector.fma → vfmacc.vf
  rvv.store(C[strip], c_row, vl)     # vse32.v
```

LLVM fuses `vector.broadcast` + `vector.fma` into a single `vfmacc.vf`
instruction at compile time.

## Build

First build buddy-mlir according to the top-level
[`docs/BuildMethods.md`](../../docs/BuildMethods.md).

### Makefile targets

| Target | Description |
|--------|-------------|
| `make derive-packed` | Show packed form derived by `--linalg-block-pack-matmul` |
| `make derive-packed-save` | Save derived packed form to `rvv-matmul-derived.mlir` |
| `make kernel-lower` | Lower `rvv-matmul-kernel.mlir` → LLVM IR (`rvv-matmul-kernel.ll`) |
| `make kernel-asm` | Compile to RISC-V RVV assembly (`rvv-matmul-kernel.s`) |
| `make kernel-obj` | Compile to RISC-V object file (`rvv-matmul-kernel.o`) |

```
$ cd examples/RVVMatmul
$ make kernel-asm
```

Expected output (key RVV instructions):

```
vsetvli t4, a0, e32, m1, ta, ma
vle32.v v8, (s0)
vfmacc.vf v8, fa5, v16
vse32.v v8, (s0)
```

## Lowering Pipeline

The `rvv` dialect requires `buddy-opt` and `buddy-translate` (not plain
`mlir-opt`/`mlir-translate`):

```
buddy-opt rvv-matmul-kernel.mlir \
  -lower-rvv \
  -convert-vector-to-llvm \
  -convert-scf-to-cf \
  -convert-cf-to-llvm \
  -convert-math-to-llvm \
  -convert-arith-to-llvm \
  -convert-func-to-llvm \
  -finalize-memref-to-llvm \
  -reconcile-unrealized-casts \
| buddy-translate --buddy-to-llvmir \
| llc -mtriple=riscv64-unknown-linux-gnu -mattr=+m,+f,+d,+v -O2 -o kernel.s
```

## Known Limitations

- `linalg.pack` has no complete lowering path in the current buddy-mlir build;
  `rvv-matmul-static.mlir` is a documentation artifact, not runnable.
- `rvv.mul` / `rvv.add` are integer-only; FP32 accumulation uses
  `vector.broadcast` + `vector.fma` instead of a dedicated `rvv.fmacc` op.
- QEMU run target is not yet wired up (requires linking with the RISC-V sysroot).
