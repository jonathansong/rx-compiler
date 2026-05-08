//===- ada300-rxops-main.cpp - Ada300 rx-ops path runner ------------------===//
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
// Runner for the Ada300 example compiled via the Top-dialect → rx-ops path
// (--convert-top-to-rxops in tpuc-opt).
//
// The MLIR function signature (subgraph0_top.mlir) after lowering:
//
//   func.func @subgraph0(
//       %arg0: tensor<64x128xf32>   fc1 weight^T   [in=64, hidden=128]
//       %arg1: tensor<1x128xf32>    fc1 bias
//       %arg2: tensor<1x64xf32>     input          [batch=1, in=64]
//       %arg3: tensor<128x64xf32>   fc2 weight^T   [hidden=128, out=64]
//       %arg4: tensor<1x64xf32>     fc2 bias
//   ) -> tensor<1x64xf32>
//
// After -llvm-request-c-wrappers the C-callable wrapper becomes:
//
//   void _mlir_ciface_subgraph0(
//       StridedMemRefType<float,2> *result,    // output [1x64]  (malloc'd)
//       StridedMemRefType<float,2> *fc1_weight,// [64x128]
//       StridedMemRefType<float,2> *fc1_bias,  // [1x128]
//       StridedMemRefType<float,2> *input,     // [1x64]
//       StridedMemRefType<float,2> *fc2_weight,// [128x64]
//       StridedMemRefType<float,2> *fc2_bias   // [1x64]
//   );
//
// Usage:
//   ada300-rxops-runner [input_value]
//
//   input_value – optional float to fill the 1x64 input tensor (default 1.0).
//
// All weights are initialised to small constants (0.01 for weight matrices,
// 0.0 for biases).  The output buffer is allocated by the MLIR function via
// malloc and freed by this runner after printing.
//
//===----------------------------------------------------------------------===//

#include <mlir/ExecutionEngine/CRunnerUtils.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

// ---------------------------------------------------------------------------
// memrefCopy: required by MLIR bufferization runtime.
// Signature (MLIR CRunnerUtils):
//   void memrefCopy(int64_t elemSize,
//                   UnrankedMemRefType<char> *src,
//                   UnrankedMemRefType<char> *dst);
//
// UnrankedMemRefType layout:
//   { int64_t rank; void *descriptor }
//
// The descriptor is a StridedMemRef laid out as:
//   { void *basePtr; void *data; int64_t offset;
//     int64_t sizes[rank]; int64_t strides[rank] }
//
// This implementation handles rank-0..N row-major copies.
// ---------------------------------------------------------------------------
extern "C" void memrefCopy(int64_t elemSize,
                            ::UnrankedMemRefType<char> *srcUnranked,
                            ::UnrankedMemRefType<char> *dstUnranked) {
  // Layout of the descriptor block (sizes/strides follow contiguously).
  struct DescHdr {
    void *basePtr;
    void *data;
    int64_t offset;
    // int64_t sizes[rank]; int64_t strides[rank]; // follow here
  };

  auto *src = srcUnranked;
  auto *dst = dstUnranked;
  int64_t rank = src->rank;

  auto *sHdr = static_cast<DescHdr *>(src->descriptor);
  auto *dHdr = static_cast<DescHdr *>(dst->descriptor);

  // Sizes start right after the three pointer/offset fields.
  const int64_t *sSizes =
      reinterpret_cast<const int64_t *>(sHdr + 1);
  const int64_t *sSt = sSizes + rank; // strides

  const int64_t *dSizes =
      reinterpret_cast<const int64_t *>(dHdr + 1);
  const int64_t *dSt = dSizes + rank;

  // Compute total number of elements.
  int64_t total = 1;
  for (int64_t i = 0; i < rank; ++i)
    total *= sSizes[i];

  // For the simple contiguous case (all unit strides from innermost dim),
  // use a flat memcpy.
  bool srcContiguous = true, dstContiguous = true;
  {
    int64_t expectedStride = 1;
    for (int64_t i = rank - 1; i >= 0; --i) {
      if (sSt[i] != expectedStride) { srcContiguous = false; break; }
      if (dSt[i] != expectedStride) { dstContiguous = false; break; }
      expectedStride *= sSizes[i];
    }
  }
  if (srcContiguous && dstContiguous) {
    char *s = static_cast<char *>(sHdr->data) + sHdr->offset * elemSize;
    char *d = static_cast<char *>(dHdr->data) + dHdr->offset * elemSize;
    std::memcpy(d, s, static_cast<size_t>(total * elemSize));
    return;
  }

  // General strided copy via flat index decomposition.
  char *sBase = static_cast<char *>(sHdr->data);
  char *dBase = static_cast<char *>(dHdr->data);
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
// Declaration of the MLIR-generated C-ABI wrapper.
// The result memref descriptor is passed as the first (output) argument;
// the callee allocates the backing buffer with malloc.
//
// Arg order matches func arg order in the generated subgraph0_top.mlir
// (produced by ada300-import.py via torch.export):
//   %arg0 fc1_weight^T [FC1_IN x HIDDEN]
//   %arg1 fc1_bias     [1 x HIDDEN]
//   %arg2 fc2_weight^T [HIDDEN x FC2_OUT]
//   %arg3 fc2_bias     [1 x FC2_OUT]
//   %arg4 input        [BATCH x FC1_IN]
// ---------------------------------------------------------------------------
extern "C" void _mlir_ciface_subgraph0(
    StridedMemRefType<float, 2> *result,
    StridedMemRefType<float, 2> *fc1_weight,
    StridedMemRefType<float, 2> *fc1_bias,
    StridedMemRefType<float, 2> *fc2_weight,
    StridedMemRefType<float, 2> *fc2_bias,
    StridedMemRefType<float, 2> *input);

// ---------------------------------------------------------------------------
// Helper: initialise a 2-D StridedMemRefType pointing at an existing buffer.
// Assumes row-major (C-contiguous) layout: stride[0]=cols, stride[1]=1.
// ---------------------------------------------------------------------------
static StridedMemRefType<float, 2> makeMemRef2D(float *data, int64_t rows,
                                                 int64_t cols) {
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

  // ---- Allocate and initialise input / weight buffers ----------------------
  // Weights: small positive constant (produces finite, non-trivial outputs).
  // Biases: zero.
  std::vector<float> fc1w(FC1_IN * HIDDEN,  0.01f);
  std::vector<float> fc1b(HIDDEN,            0.0f);
  std::vector<float> inp(BATCH   * FC1_IN,  inputVal);
  std::vector<float> fc2w(HIDDEN * FC2_OUT, 0.01f);
  std::vector<float> fc2b(FC2_OUT,           0.0f);

  // ---- Build StridedMemRefType descriptors ----------------------------------
  auto mFc1w = makeMemRef2D(fc1w.data(), FC1_IN, HIDDEN);
  auto mFc1b = makeMemRef2D(fc1b.data(), BATCH,  HIDDEN);
  auto mInp  = makeMemRef2D(inp.data(),  BATCH,  FC1_IN);
  auto mFc2w = makeMemRef2D(fc2w.data(), HIDDEN, FC2_OUT);
  auto mFc2b = makeMemRef2D(fc2b.data(), BATCH,  FC2_OUT);

  // The result descriptor is filled by the callee (output buffer malloc'd
  // inside the MLIR function; caller is responsible for freeing result.data).
  StridedMemRefType<float, 2> result;
  std::memset(&result, 0, sizeof(result));

  // ---- Run inference -------------------------------------------------------
  std::cout << "[Log] Running inference (input=" << inputVal << ")...\n";
  const auto t0 = std::chrono::high_resolution_clock::now();

  _mlir_ciface_subgraph0(&result, &mFc1w, &mFc1b, &mFc2w, &mFc2b, &mInp);

  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "[Log] Inference time: " << ms << " ms\n";

  // ---- Print output --------------------------------------------------------
  float *out = result.data;
  std::cout << "[Log] Output (1x" << FC2_OUT << "):\n  [";
  for (int64_t i = 0; i < FC2_OUT; ++i) {
    std::cout << out[i];
    if (i + 1 < FC2_OUT)
      std::cout << ", ";
  }
  std::cout << "]\n";

  std::free(out); // release the malloc'd output buffer
  return 0;
}

