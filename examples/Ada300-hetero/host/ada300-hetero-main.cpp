//===- ada300-hetero-main.cpp - Ada300 heterogeneous host runner ----------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Host orchestrator for the Ada300 heterogeneous execution demo.
//
// Extends ada300-rxops-main.cpp (examples/Ada300) to run inference across
// two compute units:
//
//   Op         Backend          Location
//   ────────   ──────────────   ──────────────────────────────────
//   MatMul     RXOPS_ADA300     Ada300 RISC-V SNPU (QEMU, ivshmem)
//   Sqrt       RXOPS_ADA300     Ada300 RISC-V SNPU (QEMU, ivshmem)
//   Exp        RXOPS_C          Host x86 CPU
//   Add        RXOPS_C          Host x86 CPU
//
// Communication mechanism:
//   POSIX shared memory (/dev/shm/ada300_bar, 64 MiB) is used as a PCIe BAR
//   simulation.  QEMU maps the same file into the guest at ADA300_IVSHMEM_BASE.
//   The baremetal firmware polls the control-register block for commands.
//
// Usage:
//   # 1.  Start QEMU with ivshmem in a separate terminal:
//   #     qemu-system-riscv64 -machine ada300s_evk -cpu ada300s   \
//   #       -nographic -serial mon:stdio                           \
//   #       -object memory-backend-file,id=shmem0,share=on,        \
//   #              mem-path=/dev/shm/ada300_bar,size=64M           \
//   #       -kernel ada300-hetero-device.elf
//   #
//   # 2.  Run this host binary (QEMU must be booted before the first MatMul):
//   #     ada300-hetero-runner [input_value]
//   #
//   #     input_value – optional float to fill the 1×64 input (default 1.0).
//
// The shared-memory path can be overridden via the environment variable
// HETERO_SHM_NAME (default: "ada300_bar").
//
//===----------------------------------------------------------------------===//

#include <mlir/ExecutionEngine/CRunnerUtils.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

// Shared-memory transport API (hetero_shmem_host.c).
extern "C" {
#include "../include/hetero_shmem.h"
void *hetero_shmem_open(const char *name, size_t size);
void  hetero_shmem_close(void);
}

// ---------------------------------------------------------------------------
// memrefCopy: required by MLIR bufferization runtime.
// Signature matches MLIR's CRunnerUtils expectation.
// ---------------------------------------------------------------------------
extern "C" void memrefCopy(int64_t elemSize,
                            ::UnrankedMemRefType<char> *srcUnranked,
                            ::UnrankedMemRefType<char> *dstUnranked) {
  struct DescHdr {
    void   *basePtr;
    void   *data;
    int64_t offset;
    // int64_t sizes[rank]; int64_t strides[rank]; follow here
  };

  int64_t rank = srcUnranked->rank;
  auto *sHdr   = static_cast<DescHdr *>(srcUnranked->descriptor);
  auto *dHdr   = static_cast<DescHdr *>(dstUnranked->descriptor);

  const int64_t *sSizes = reinterpret_cast<const int64_t *>(sHdr + 1);
  const int64_t *sSt    = sSizes + rank;
  const int64_t *dSizes = reinterpret_cast<const int64_t *>(dHdr + 1);
  const int64_t *dSt    = dSizes + rank;

  int64_t total = 1;
  for (int64_t i = 0; i < rank; ++i)
    total *= sSizes[i];

  // Fast path: both src and dst are C-contiguous.
  bool srcContig = true, dstContig = true;
  {
    int64_t exp = 1;
    for (int64_t i = rank - 1; i >= 0; --i) {
      if (sSt[i] != exp) srcContig = false;
      if (dSt[i] != exp) dstContig = false;
      exp *= sSizes[i];
    }
  }
  if (srcContig && dstContig) {
    const char *s = static_cast<char *>(sHdr->data) + sHdr->offset * elemSize;
    char       *d = static_cast<char *>(dHdr->data) + dHdr->offset * elemSize;
    std::memcpy(d, s, static_cast<size_t>(total * elemSize));
    return;
  }

  // Slow path: general strided copy.
  const char *sBase = static_cast<char *>(sHdr->data);
  char       *dBase = static_cast<char *>(dHdr->data);
  for (int64_t idx = 0; idx < total; ++idx) {
    int64_t rem = idx;
    int64_t sOff = sHdr->offset, dOff = dHdr->offset;
    for (int64_t dim = rank - 1; dim >= 0; --dim) {
      int64_t coord = rem % sSizes[dim];
      rem /= sSizes[dim];
      sOff += coord * sSt[dim];
      dOff += coord * dSt[dim];
    }
    std::memcpy(dBase + dOff * elemSize, sBase + sOff * elemSize,
                static_cast<size_t>(elemSize));
  }
}

// ---------------------------------------------------------------------------
// Model constants
// ---------------------------------------------------------------------------
static constexpr int64_t BATCH   = 1;
static constexpr int64_t FC1_IN  = 64;
static constexpr int64_t HIDDEN  = 128;
static constexpr int64_t FC2_OUT = 64;

// ---------------------------------------------------------------------------
// Declaration of the MLIR-generated C-ABI wrapper (same as Ada300 example).
// Arg order (from subgraph0_top.mlir / ada300-import.py):
//   %arg0 fc1_weight^T  [FC1_IN  × HIDDEN]
//   %arg1 fc1_bias      [1       × HIDDEN]
//   %arg2 fc2_weight^T  [HIDDEN  × FC2_OUT]
//   %arg3 fc2_bias      [1       × FC2_OUT]
//   %arg4 input         [BATCH   × FC1_IN]
// ---------------------------------------------------------------------------
extern "C" void _mlir_ciface_subgraph0(
    StridedMemRefType<float, 2> *result,
    StridedMemRefType<float, 2> *fc1_weight,
    StridedMemRefType<float, 2> *fc1_bias,
    StridedMemRefType<float, 2> *fc2_weight,
    StridedMemRefType<float, 2> *fc2_bias,
    StridedMemRefType<float, 2> *input);

// ---------------------------------------------------------------------------
// Helper: build a 2-D StridedMemRefType over an existing buffer (row-major).
// ---------------------------------------------------------------------------
static StridedMemRefType<float, 2> makeRef2D(float *data,
                                              int64_t rows, int64_t cols) {
  StridedMemRefType<float, 2> m;
  m.basePtr    = data;
  m.data       = data;
  m.offset     = 0;
  m.sizes[0]   = rows;
  m.sizes[1]   = cols;
  m.strides[0] = cols;
  m.strides[1] = 1;
  return m;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  const float inputVal = (argc >= 2) ? std::stof(argv[1]) : 1.0f;

  // ---- Open shared memory (created here; QEMU maps it into the guest) ------
  const char *shmName = std::getenv("HETERO_SHM_NAME");
  if (!shmName) shmName = "ada300_bar";

  std::cout << "[hetero] Opening shared memory /" << shmName
            << " (" << (HETERO_SHM_SIZE / (1024 * 1024)) << " MiB)...\n";

  void *shmBase = hetero_shmem_open(shmName, HETERO_SHM_SIZE);
  if (!shmBase) {
    std::cerr << "[hetero] ERROR: failed to open shared memory.\n";
    return 1;
  }
  std::cout << "[hetero] Shared memory ready at " << shmBase << "\n";

  // ---- Allocate and initialise model buffers --------------------------------
  std::vector<float> fc1w(FC1_IN * HIDDEN,   0.01f);
  std::vector<float> fc1b(HIDDEN,             0.0f);
  std::vector<float> inp(BATCH  * FC1_IN,    inputVal);
  std::vector<float> fc2w(HIDDEN * FC2_OUT,  0.01f);
  std::vector<float> fc2b(FC2_OUT,            0.0f);

  // ---- Build StridedMemRefType descriptors ---------------------------------
  auto mFc1w = makeRef2D(fc1w.data(), FC1_IN, HIDDEN);
  auto mFc1b = makeRef2D(fc1b.data(), BATCH,  HIDDEN);
  auto mInp  = makeRef2D(inp.data(),  BATCH,  FC1_IN);
  auto mFc2w = makeRef2D(fc2w.data(), HIDDEN, FC2_OUT);
  auto mFc2b = makeRef2D(fc2b.data(), BATCH,  FC2_OUT);

  // The result descriptor is filled by the callee; caller frees result.data.
  StridedMemRefType<float, 2> result;
  std::memset(&result, 0, sizeof(result));

  // ---- Run inference -------------------------------------------------------
  std::cout << "[hetero] Running heterogeneous inference (input=" << inputVal << ")...\n";
  std::cout << "[hetero]   MatMul/Sqrt ops → Ada300 RISC-V SNPU (QEMU, ivshmem)\n";
  std::cout << "[hetero]   Exp/Add ops → host x86 (RXOPS_C)\n";

  const auto t0 = std::chrono::high_resolution_clock::now();

  _mlir_ciface_subgraph0(&result, &mFc1w, &mFc1b, &mFc2w, &mFc2b, &mInp);

  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "[hetero] Inference time: " << ms << " ms\n";

  // ---- Print output --------------------------------------------------------
  float *out = result.data;
  std::cout << "[hetero] Output (1×" << FC2_OUT << "):\n  [";
  for (int64_t i = 0; i < FC2_OUT; ++i) {
    std::cout << out[i];
    if (i + 1 < FC2_OUT) std::cout << ", ";
  }
  std::cout << "]\n";
  std::cout << "[hetero] Expected ≈ 1.7629 per element\n";

  // ---- Cleanup -------------------------------------------------------------
  std::free(out);
  hetero_shmem_close();
  return 0;
}
