# ===- CMakeLists.txt ----------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# CMake build rules for the Ada300 example.
#
# Lowering path: Top dialect → rx-ops library calls
#
#   subgraph0_top.mlir  (Top dialect, tensor-level)
#     ↓ tpuc-opt --convert-top-to-rxops
#         top.MatMul / top.Add / top.Exp / top.Sqrt / top.Input
#           → llvm.call @rxops_bridge_*_f32(...)
#     ↓ mlir-opt (one-shot-bufferize + finalize-memref-to-llvm + ...)
#         tensor args / bufferization ops → fully lowered LLVM dialect
#     ↓ mlir-translate --mlir-to-llvmir
#         → subgraph0_rxops.ll  (LLVM IR with rxops_bridge_* call sites)
#     ↓ llc -filetype=obj
#         → subgraph0_rxops.o
#     ↓ (+ rx_ops_bridge.c compiled → rx_ops_bridge.o)
#     ↓ (+ ada300-rxops-main.cpp compiled → ada300-rxops-main.o)
#     ↓ link all → ada300-rxops-runner
#
# Usage (standalone):
#   cd examples/Ada300 && mkdir -p build && cd build
#   cmake .. -DMLIR_DIR=/usr/local/lib/cmake/mlir
#   ninja ada300-rxops-run
#
# Usage (from repo root, as a subdirectory):
#   cmake -S . -B build && cmake --build build --target ada300-rxops-run
#
# Prerequisites:
#   - rx-ops static library pre-built:
#       cd third_party/rx-ops && mkdir -p build && cd build
#       cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release && ninja
#   - subgraph0_top.mlir present in examples/Ada300/output/
#
# ===---------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Standalone vs. subdirectory mode.
# When built standalone (cmake .. from examples/Ada300/build/), we need to
# call project() and find_package(MLIR) ourselves and derive repo paths.
# When included as add_subdirectory() from the repo root, those are already
# set by the parent project.
# ---------------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # ---- Standalone mode ------------------------------------------------
  project(Ada300RxOps CXX C)

  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Ada300: standalone build, MLIR found at ${MLIR_DIR}")

  # Repo root is two directories up from this file's directory.
  get_filename_component(_ADA300_REPO_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

  # tpuc-opt: prefer install/bin (built artefact), fall back to PATH.
  find_program(TPUC_OPT tpuc-opt
    HINTS "${_ADA300_REPO_ROOT}/install/bin"
          "${_ADA300_REPO_ROOT}/build/bin"
    DOC   "Path to tpuc-opt (rx-mlir)"
  )
  if(NOT TPUC_OPT)
    message(FATAL_ERROR "Ada300: tpuc-opt not found. "
      "Build rx-mlir first and re-run cmake with "
      "-DTPUC_OPT=/path/to/tpuc-opt")
  endif()
  message(STATUS "Ada300: tpuc-opt = ${TPUC_OPT}")

  set(RXOPS_SRC_DIR    "${_ADA300_REPO_ROOT}/third_party/rx-ops")
  set(RXOPS_BRIDGE_DIR "${_ADA300_REPO_ROOT}/examples/Ada300")
else()
  # ---- Subdirectory mode (default) ------------------------------------
  set(TPUC_OPT         ${CMAKE_BINARY_DIR}/bin/tpuc-opt)
  set(RXOPS_SRC_DIR    ${CMAKE_SOURCE_DIR}/third_party/rx-ops)
  set(RXOPS_BRIDGE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# ---------------------------------------------------------------------------
# Output directory – all generated files are written here.
# ---------------------------------------------------------------------------
set(ADA300_DUMP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/output)
file(MAKE_DIRECTORY ${ADA300_DUMP_DIR})

set(RXOPS_LIB ${RXOPS_SRC_DIR}/build/librx_ops.a)

# ---------------------------------------------------------------------------
# Option: enable the Ada300 rx-ops runner target (on by default).
# ---------------------------------------------------------------------------
option(ADA300_RXOPS "Build Ada300 Top-dialect → rx-ops example runner" ON)

if(ADA300_RXOPS)

  # In subdirectory mode, tpuc-opt is a build target we can depend on.
  # In standalone mode it is a pre-built file; use the file path instead.
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_ADA300_TPUC_OPT_DEP "${TPUC_OPT}")   # file dependency
  else()
    set(_ADA300_TPUC_OPT_DEP tpuc-opt)          # CMake target dependency
  endif()

  # -- Step 1: tpuc-opt --convert-top-to-rxops --------------------------------
  #   top.MatMul / top.Add / top.Exp / top.Sqrt / top.Input
  #     → llvm.call @rxops_bridge_*_f32(...)
  add_custom_command(
    OUTPUT ${ADA300_DUMP_DIR}/subgraph0_rxops.mlir
    COMMAND ${TPUC_OPT}
              --convert-top-to-rxops
              ${ADA300_DUMP_DIR}/subgraph0_top.mlir
              -o ${ADA300_DUMP_DIR}/subgraph0_rxops.mlir
    DEPENDS ${ADA300_DUMP_DIR}/subgraph0_top.mlir
            ${_ADA300_TPUC_OPT_DEP}
    COMMENT "Ada300 Step 1: --convert-top-to-rxops → subgraph0_rxops.mlir"
    VERBATIM
  )

  # -- Step 2: finish lowering to LLVM dialect (mlir-opt) ---------------------
  #   -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs"
  #       tensor function args → memref args; bufferization.to_memref → identity
  #   -finalize-memref-to-llvm
  #       memref.alloc → malloc; memref.extract_aligned_pointer → inttoptr
  #   -convert-arith-to-llvm   arith.index_cast etc. → LLVM dialect
  #   -llvm-request-c-wrappers emit _mlir_ciface_subgraph0 C-ABI wrapper
  #   -convert-func-to-llvm    func.func → llvm.func
  #   -reconcile-unrealized-casts  clean up type cast ops
  #
  # NOTE: CMake VERBATIM keeps literal `"` in unquoted arguments, so the
  # pass-option string must be stored in a variable and expanded with
  # `"${VAR}"` (quoted CMake expansion) to avoid stray `"` in the Ninja rule.
  set(_ADA300_BUFFERIZE_OPT
      "-one-shot-bufferize=bufferize-function-boundaries allow-return-allocs")
  add_custom_command(
    OUTPUT ${ADA300_DUMP_DIR}/subgraph0_rxops_finished.mlir
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              ${ADA300_DUMP_DIR}/subgraph0_rxops.mlir
              "${_ADA300_BUFFERIZE_OPT}"
              -finalize-memref-to-llvm
              -convert-arith-to-llvm
              -llvm-request-c-wrappers
              -convert-func-to-llvm
              -reconcile-unrealized-casts
              -o ${ADA300_DUMP_DIR}/subgraph0_rxops_finished.mlir
    DEPENDS ${ADA300_DUMP_DIR}/subgraph0_rxops.mlir
    COMMENT "Ada300 Step 2: finishing passes → subgraph0_rxops_finished.mlir"
    VERBATIM
  )

  # -- Step 3: translate MLIR LLVM dialect → LLVM IR (.ll) -------------------
  add_custom_command(
    OUTPUT ${ADA300_DUMP_DIR}/subgraph0_rxops.ll
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-translate
              --mlir-to-llvmir
              ${ADA300_DUMP_DIR}/subgraph0_rxops_finished.mlir
              -o ${ADA300_DUMP_DIR}/subgraph0_rxops.ll
    DEPENDS ${ADA300_DUMP_DIR}/subgraph0_rxops_finished.mlir
    COMMENT "Ada300 Step 3: mlir-translate → subgraph0_rxops.ll"
    VERBATIM
  )

  # -- Step 4: compile LLVM IR → native object file --------------------------
  # The MLIR-generated IR uses opaque pointers (LLVM 15+ style).
  # ${LLVM_TOOLS_BINARY_DIR}/llc (LLVM 18) is built without any backend
  # targets in this environment ("Registered Targets: (none)").
  # Workaround: convert the textual IR to LLVM bitcode first with llvm-as
  # (LLVM 18), then compile with the system llc which understands LLVM 18
  # bitcode via the -opaque-pointers flag.
  find_program(LLC_SYSTEM
    NAMES llc-18 llc-17 llc-16 llc-15 llc-14 llc
    HINTS
      /workspace/rx-llvm-main/build/bin
      /usr/lib/llvm-18/bin
      /usr/lib/llvm-17/bin
      /usr/lib/llvm-16/bin
      /usr/lib/llvm-15/bin
      /usr/lib/llvm-14/bin
    DOC "llc with X86 target (prefer rx-llvm-main/build/bin over system)"
    NO_DEFAULT_PATH
  )
  if(NOT LLC_SYSTEM)
    find_program(LLC_SYSTEM NAMES llc DOC "System llc with X86 target")
  endif()
  if(NOT LLC_SYSTEM)
    message(WARNING "Ada300: Could not find a system llc; Step 4 will likely fail.")
    set(LLC_SYSTEM "${LLVM_TOOLS_BINARY_DIR}/llc")
  endif()
  message(STATUS "Ada300: using LLC_SYSTEM=${LLC_SYSTEM} for LLVM IR → .o compilation")

  add_custom_command(
    OUTPUT ${ADA300_DUMP_DIR}/subgraph0_rxops.o
    # 4a: textual LLVM IR → LLVM bitcode (using LLVM 18 llvm-as which
    #     understands the MLIR-generated opaque-pointer IR).
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/llvm-as
              ${ADA300_DUMP_DIR}/subgraph0_rxops.ll
              -o ${ADA300_DUMP_DIR}/subgraph0_rxops.bc
    # 4b: LLVM bitcode → native object (system llc with -opaque-pointers).
    COMMAND ${LLC_SYSTEM}
              -opaque-pointers
              -filetype=obj
              -relocation-model=pic
              -O3
              ${ADA300_DUMP_DIR}/subgraph0_rxops.bc
              -o ${ADA300_DUMP_DIR}/subgraph0_rxops.o
    DEPENDS ${ADA300_DUMP_DIR}/subgraph0_rxops.ll
    COMMENT "Ada300 Step 4: llvm-as + llc → subgraph0_rxops.o"
    VERBATIM
  )

  # -- Step 5: compile the rx-ops bridge wrapper (C) -------------------------
  add_custom_command(
    OUTPUT ${ADA300_DUMP_DIR}/rx_ops_bridge.o
    COMMAND ${CMAKE_C_COMPILER}
              -O2
              -I${RXOPS_BRIDGE_DIR}
              -I${RXOPS_SRC_DIR}/include
              -I${RXOPS_SRC_DIR}/include/interface
              -c ${RXOPS_BRIDGE_DIR}/rx_ops_bridge.c
              -o ${ADA300_DUMP_DIR}/rx_ops_bridge.o
    DEPENDS ${RXOPS_BRIDGE_DIR}/rx_ops_bridge.c
            ${RXOPS_BRIDGE_DIR}/rx_ops_bridge.h
    COMMENT "Ada300 Step 5: compile rx_ops_bridge.c → rx_ops_bridge.o"
    VERBATIM
  )

  # -- Step 6: compile the runner (C++) --------------------------------------
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ada300-rxops-main.o
    COMMAND ${CMAKE_CXX_COMPILER}
              -O2
              -std=c++17
              -I${MLIR_INCLUDE_DIRS}
              -c ${CMAKE_CURRENT_SOURCE_DIR}/ada300-rxops-main.cpp
              -o ${CMAKE_CURRENT_BINARY_DIR}/ada300-rxops-main.o
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ada300-rxops-main.cpp
    COMMENT "Ada300 Step 6: compile ada300-rxops-main.cpp → ada300-rxops-main.o"
    VERBATIM
  )

  # -- Step 7: link everything → runner binary --------------------------------
  # -Wl,--whole-archive forces the linker to include all symbols from
  # librx_ops.a (needed for the RXOPS_C backend registration symbol).
  # -fopenmp resolves OpenMP symbols inside librx_ops.a.
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ada300-rxops-runner
    COMMAND ${CMAKE_CXX_COMPILER}
              ${ADA300_DUMP_DIR}/subgraph0_rxops.o
              ${ADA300_DUMP_DIR}/rx_ops_bridge.o
              ${CMAKE_CURRENT_BINARY_DIR}/ada300-rxops-main.o
              -Wl,--whole-archive ${RXOPS_LIB} -Wl,--no-whole-archive
              -lm -lstdc++ -fopenmp
              -o ${CMAKE_CURRENT_BINARY_DIR}/ada300-rxops-runner
    DEPENDS
      ${ADA300_DUMP_DIR}/subgraph0_rxops.o
      ${ADA300_DUMP_DIR}/rx_ops_bridge.o
      ${CMAKE_CURRENT_BINARY_DIR}/ada300-rxops-main.o
      ${RXOPS_LIB}
    COMMENT "Ada300 Step 7: link → ada300-rxops-runner"
    VERBATIM
  )

  # Convenience target: LLVM IR dump only
  add_custom_target(ada300-rxops-ll
    DEPENDS ${ADA300_DUMP_DIR}/subgraph0_rxops.ll
    COMMENT "Ada300: LLVM IR with rxops_bridge_* calls ready"
  )

  # Convenience target: fully linked runner binary
  add_custom_target(ada300-rxops-run
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ada300-rxops-runner
    COMMENT "Ada300: runner binary ready → ${CMAKE_CURRENT_BINARY_DIR}/ada300-rxops-runner"
  )

else()
  message(STATUS "Ada300: rx-ops runner disabled (cmake -DADA300_RXOPS=ON to enable)")
endif() # ADA300_RXOPS

# ===========================================================================
# Ada300 BAREMETAL target
#
# Cross-compiles the Ada300 example to RISC-V and links it with the
# ada300_snpu_sdk BSP (startup, UART, mr_heap) to produce an ELF that runs
# under QEMU's ada300s_evk machine.
#
# Prerequisites:
#   1. Ada300 LLVM toolchain at /opt/ada300-llvm
#      (riscv64-unknown-elf clang + llvm-objdump + lld)
#   2. ada300_snpu_sdk BSP at ADA300_SDK_ROOT (default: sibling repo path)
#   3. subgraph0_rxops.ll already generated (see the ada300-rxops-run target)
#
# Build (standalone — outputs always go to examples/Ada300/build/):
#   cd examples/Ada300 && cmake -S . -B build -DADA300_BAREMETAL=ON
#   cmake --build build --target ada300-baremetal
#
# Build (from repo root — outputs still go to examples/Ada300/build/):
#   cmake -S . -B build -DADA300_BAREMETAL=ON
#   cmake --build build --target ada300-baremetal
#
# Run in QEMU:
#   cmake --build build --target run-ada300-baremetal
#
# Or manually:
#   /opt/ada300-qemu/bin/qemu-system-riscv64 \
#       -machine ada300s_evk -cpu ada300s \
#       -nographic -serial mon:stdio \
#       -kernel examples/Ada300/build/ada300-baremetal.elf
#   (Exit: Ctrl-A, then x)
# ===========================================================================
option(ADA300_BAREMETAL
  "Cross-compile Ada300 example for RISC-V and run under QEMU ada300s_evk" OFF)

if(ADA300_BAREMETAL)

  # ---- Configurable paths --------------------------------------------------
  set(ADA300_LLVM_ROOT "/workspace/rx-llvm-main/build"
      CACHE PATH "Ada300 LLVM toolchain root (provides riscv64-unknown-elf clang)")

  # Default SDK root: sibling directory relative to the rx-mlir workspace.
  # In the dev container the layout is:
  #   /workspace/rx-mlir-main/        ← this repo
  #   /workspace/embed-system-main/ada300_snpu_sdk/
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # standalone build: examples/Ada300/ → resolve to /workspace/embed-system-main
    # CMAKE_CURRENT_SOURCE_DIR = /workspace/rx-mlir-main/examples/Ada300
    # Three levels up gets /workspace; then append the sibling repo.
    get_filename_component(_sdk_default
      "${CMAKE_CURRENT_SOURCE_DIR}/../../../embed-system-main/ada300_snpu_sdk"
      ABSOLUTE)
  else()
    # subdirectory build: CMAKE_SOURCE_DIR is the rx-mlir root
    get_filename_component(_sdk_default
      "${CMAKE_SOURCE_DIR}/../embed-system-main/ada300_snpu_sdk"
      ABSOLUTE)
  endif()

  set(ADA300_SDK_ROOT "${_sdk_default}"
      CACHE PATH "Path to ada300_snpu_sdk (BSP sources + linker script)")

  set(ADA300_QEMU "/opt/ada300-qemu/bin/qemu-system-riscv64"
      CACHE FILEPATH "QEMU binary for run-ada300-baremetal")

  # Validate paths
  if(NOT EXISTS "${ADA300_LLVM_ROOT}/bin/clang")
    message(FATAL_ERROR
      "Ada300 LLVM toolchain not found at '${ADA300_LLVM_ROOT}/bin/clang'. "
      "Pass -DADA300_LLVM_ROOT=/path/to/ada300-llvm to override.")
  endif()
  if(NOT EXISTS "${ADA300_SDK_ROOT}/bsp/ld/ada300s_evk.ld")
    message(FATAL_ERROR
      "ada300_snpu_sdk not found at '${ADA300_SDK_ROOT}'. "
      "Pass -DADA300_SDK_ROOT=/path/to/ada300_snpu_sdk to override.")
  endif()

  set(_ADA300_CC   "${ADA300_LLVM_ROOT}/bin/clang")
  set(_ADA300_OBJDUMP "${ADA300_LLVM_ROOT}/bin/llvm-objdump")

  # Newlib-compatible sysroot providing <stdlib.h>, <string.h> etc.
  # Defaults to the system picolibc install (apt: picolibc-riscv64-unknown-elf).
  # Override with -DADA300_NEWLIB_SYSROOT=/path if using a different newlib.
  set(ADA300_NEWLIB_SYSROOT "/usr/lib/picolibc/riscv64-unknown-elf"
      CACHE PATH "Picolibc/newlib sysroot for riscv64-unknown-elf (leave empty if none)")

  # ISA / ABI flags matching the ada300s QEMU CPU model
  set(_MARCH "rv64gcv_zfh_zvfh_zfbfmin0p8_zvfbfmin0p8_xadatmm_xadacv")
  set(_ADA300_ARCH_FLAGS
      --target=riscv64-unknown-elf
      -march=${_MARCH}
      -mabi=lp64d
      -mcmodel=medany
      -menable-experimental-extensions
      -ffreestanding
      -fno-builtin
      -fno-strict-aliasing)

  set(_LD_SCRIPT "${ADA300_SDK_ROOT}/bsp/ld/ada300s_evk.ld")

  # _BM_DIR   : source-side output/ directory — used only to READ the MLIR/LL
  #             files produced by the earlier ada300-rxops pipeline steps.
  # _BM_OBJ_DIR: dedicated build directory for all compiled objects and the
  #             final ELF.  Fixed to examples/Ada300/build/ regardless of
  #             whether this file is included as add_subdirectory() or built
  #             standalone, so artefacts never land in rx-mlir-main/build/.
  set(_BM_DIR "${ADA300_DUMP_DIR}")
  set(_BM_OBJ_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")
  file(MAKE_DIRECTORY "${_BM_OBJ_DIR}")

  # Convenience macro: compile a single C or ASM file with Ada300 flags.
  # Usage: ada300_compile(<output_obj> <src_file> [extra_includes...])
  #
  # -ffunction-sections / -fdata-sections: enables --gc-sections at link
  # time to eliminate any bridge functions whose ops are not compiled in
  # (e.g. rxops_bridge_log_f32 / rxops_bridge_rsqrt_f32 reference
  # rxops_log / rxops_rsqrt; those dispatcher stubs are NOT in the model
  # subgraph but the compiler still emits symbol references for them).
  macro(ada300_compile _out _src)
    set(_extra_incs ${ARGN})
    add_custom_command(
      OUTPUT "${_out}"
      COMMAND "${_ADA300_CC}"
                ${_ADA300_ARCH_FLAGS}
                -O2 -g
                -ffunction-sections
                -fdata-sections
                -DRXNN_BUILD_RTOS=1
                -I${ADA300_SDK_ROOT}/bsp/include
                -I${RXOPS_BRIDGE_DIR}
                -I${RXOPS_SRC_DIR}/include
                -I${RXOPS_SRC_DIR}/include/interface
                -I${RXOPS_SRC_DIR}/include/backend
                $<$<BOOL:${ADA300_NEWLIB_SYSROOT}>:-isystem${ADA300_NEWLIB_SYSROOT}/include>
                ${_extra_incs}
                -c "${_src}"
                -o "${_out}"
      DEPENDS "${_src}"
      COMMENT "Ada300 Baremetal: compile ${_src}"
      VERBATIM
    )
  endmacro()

  # ---------------------------------------------------------------------------
  # Step A: cross-compile subgraph0_rxops.ll → RISC-V object
  # The LLVM IR produced by mlir-translate has no embedded target triple,
  # so clang can retarget it cleanly.
  # ---------------------------------------------------------------------------
  add_custom_command(
    OUTPUT "${_BM_OBJ_DIR}/subgraph0_rxops_rv64.o"
    COMMAND "${_ADA300_CC}"
              ${_ADA300_ARCH_FLAGS}
              -O3
              -c "${_BM_DIR}/subgraph0_rxops.ll"
              -o "${_BM_OBJ_DIR}/subgraph0_rxops_rv64.o"
    DEPENDS "${_BM_DIR}/subgraph0_rxops.ll"
    COMMENT "Ada300 Baremetal Step A: clang → subgraph0_rxops_rv64.o (RISC-V)"
    VERBATIM
  )

  # ---------------------------------------------------------------------------
  # Step B: compile the Ada300 rx-ops bridge (RXOPS_ADA300 backend)
  # ---------------------------------------------------------------------------
  ada300_compile(
    "${_BM_OBJ_DIR}/rx_ops_bridge_ada300.o"
    "${CMAKE_CURRENT_SOURCE_DIR}/rx_ops_bridge_ada300.c")

  # ---------------------------------------------------------------------------
  # Step C: compile the baremetal main
  # ---------------------------------------------------------------------------
  ada300_compile(
    "${_BM_OBJ_DIR}/ada300-baremetal-main.o"
    "${CMAKE_CURRENT_SOURCE_DIR}/ada300-baremetal-main.c")

  # ---------------------------------------------------------------------------
  # Step D: compile BSP sources (startup.S + uart, printf, heap, bsp_main)
  # ---------------------------------------------------------------------------
  ada300_compile(
    "${_BM_OBJ_DIR}/bm_startup.o"
    "${ADA300_SDK_ROOT}/bsp/src/startup.S")

  set(_BSP_C_SRCS bsp_main.c uart.c mini_printf.c mr_heap.c)
  set(_BSP_OBJS "${_BM_OBJ_DIR}/bm_startup.o")

  # bsp_stubs.c provides mr_smp_secondary_entry (secondary harts spin on wfi)
  ada300_compile(
    "${_BM_OBJ_DIR}/bm_bsp_stubs.o"
    "${RXOPS_SRC_DIR}/src/bsp_stubs.c")
  list(APPEND _BSP_OBJS "${_BM_OBJ_DIR}/bm_bsp_stubs.o")
  foreach(_bsp_src ${_BSP_C_SRCS})
    get_filename_component(_bsp_base "${_bsp_src}" NAME_WE)
    set(_bsp_obj "${_BM_OBJ_DIR}/bm_${_bsp_base}.o")
    ada300_compile("${_bsp_obj}" "${ADA300_SDK_ROOT}/bsp/src/${_bsp_src}")
    list(APPEND _BSP_OBJS "${_bsp_obj}")
  endforeach()

  # ---------------------------------------------------------------------------
  # Step E: compile rx-ops Ada300 backend sources
  #
  # We compile exactly the sources needed:
  #   - rx_ops_register.c   callback table + lookup
  #   - log.c               logging (RXNN_BUILD_RTOS suppresses dirent/stat)
  #   - ops/{exp,sqrt,add,matmul}.c  operator dispatch for the 4 used ops
  #   - backend/ada300/setup.c + gemm.c + utils.c + rx_ops_memory.c
  #   - backend/ada300/fp32/*.c   fp32 Ada300 kernels (add/exp/sqrt + others)
  #   - backend/ada300/fp16/*.c   fp16 kernels (pulled in by setup.c refs)
  #   - backend/ada300/asm/*.S    vectorised SNPU assembly kernels
  # ---------------------------------------------------------------------------
  # rx_ops_register.c  – callback dispatch table + timing
  # log.c              – logging (RXNN_BUILD_RTOS stubs out clock_gettime)
  # ops/*.c            – operator dispatcher stubs that forward to backend
  #                      callbacks.  Include all four ops used by the model
  #                      (exp/sqrt/add/matmul) PLUS log and rsqrt because
  #                      rx_ops_bridge_ada300.c provides bridge wrappers for
  #                      them too; without these objects the linker would see
  #                      undefined references to rxops_log / rxops_rsqrt even
  #                      though those bridge functions are never called by the
  #                      current model.  --gc-sections (below) removes the
  #                      dead sections once -ffunction-sections is in effect.
  set(_RXOPS_COMMON_SRCS
      "${RXOPS_SRC_DIR}/src/rx_ops_register.c"
      "${RXOPS_SRC_DIR}/src/rx_ops_utils.c"
      "${RXOPS_SRC_DIR}/src/log.c"
      "${RXOPS_SRC_DIR}/src/ops/exp.c"
      "${RXOPS_SRC_DIR}/src/ops/sqrt.c"
      "${RXOPS_SRC_DIR}/src/ops/add.c"
      "${RXOPS_SRC_DIR}/src/ops/matmul.c"
      "${RXOPS_SRC_DIR}/src/ops/log.c"
      "${RXOPS_SRC_DIR}/src/ops/rsqrt.c")

  file(GLOB _RXOPS_ADA300_C_SRCS
      "${RXOPS_SRC_DIR}/src/backend/ada300/*.c"
      "${RXOPS_SRC_DIR}/src/backend/ada300/fp32/*.c"
      "${RXOPS_SRC_DIR}/src/backend/ada300/fp16/*.c")
  file(GLOB _RXOPS_ADA300_ASM_SRCS
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/*.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/convert/*.S")

  set(_RXOPS_OBJS)
  foreach(_rsrc ${_RXOPS_COMMON_SRCS} ${_RXOPS_ADA300_C_SRCS})
    get_filename_component(_rbase "${_rsrc}" NAME_WE)
    # Use an MD5 prefix to avoid basename collisions across subdirectories.
    string(MD5 _rhash "${_rsrc}")
    string(SUBSTRING "${_rhash}" 0 6 _rhash)
    set(_robj "${_BM_OBJ_DIR}/rxops_${_rbase}_${_rhash}.o")
    ada300_compile("${_robj}" "${_rsrc}"
        -I${RXOPS_SRC_DIR}/include/ada300
        -I${RXOPS_SRC_DIR}/src/backend/ada300)
    list(APPEND _RXOPS_OBJS "${_robj}")
  endforeach()

  foreach(_rsrc ${_RXOPS_ADA300_ASM_SRCS})
    get_filename_component(_rbase "${_rsrc}" NAME_WE)
    string(MD5 _rhash "${_rsrc}")
    string(SUBSTRING "${_rhash}" 0 6 _rhash)
    set(_robj "${_BM_OBJ_DIR}/rxops_asm_${_rbase}_${_rhash}.o")
    ada300_compile("${_robj}" "${_rsrc}"
        -I${RXOPS_SRC_DIR}/include/ada300)
    list(APPEND _RXOPS_OBJS "${_robj}")
  endforeach()

  # ---------------------------------------------------------------------------
  # Step F: link everything into ada300-baremetal.elf
  # ---------------------------------------------------------------------------
  set(_BM_ELF "${_BM_OBJ_DIR}/ada300-baremetal.elf")

  add_custom_command(
    OUTPUT "${_BM_ELF}"
    COMMAND "${_ADA300_CC}"
              ${_ADA300_ARCH_FLAGS}
              -nostdlib
              -fuse-ld=lld
              -Wl,--gc-sections
              -Wl,-Map=${_BM_OBJ_DIR}/ada300-baremetal.map
              "-T${_LD_SCRIPT}"
              $<$<BOOL:${ADA300_NEWLIB_SYSROOT}>:-L${ADA300_NEWLIB_SYSROOT}/lib>
              # Objects: BSP (startup must be first), main, bridge, subgraph
              ${_BSP_OBJS}
              "${_BM_OBJ_DIR}/ada300-baremetal-main.o"
              "${_BM_OBJ_DIR}/rx_ops_bridge_ada300.o"
              "${_BM_OBJ_DIR}/subgraph0_rxops_rv64.o"
              ${_RXOPS_OBJS}
              # Link newlib libm only when a sysroot is available
              $<$<BOOL:${ADA300_NEWLIB_SYSROOT}>:-Wl,--start-group>
              $<$<BOOL:${ADA300_NEWLIB_SYSROOT}>:-lm>
              $<$<BOOL:${ADA300_NEWLIB_SYSROOT}>:-Wl,--end-group>
              -o "${_BM_ELF}"
    DEPENDS
      ${_BSP_OBJS}
      "${_BM_OBJ_DIR}/ada300-baremetal-main.o"
      "${_BM_OBJ_DIR}/rx_ops_bridge_ada300.o"
      "${_BM_OBJ_DIR}/subgraph0_rxops_rv64.o"
      ${_RXOPS_OBJS}
    COMMENT "Ada300 Baremetal Step F: link → ada300-baremetal.elf"
    VERBATIM
  )

  # Post-build disassembly (handy for debugging)
  add_custom_command(
    OUTPUT "${_BM_OBJ_DIR}/ada300-baremetal.asm"
    COMMAND "${_ADA300_OBJDUMP}" -d "${_BM_ELF}"
            > "${_BM_OBJ_DIR}/ada300-baremetal.asm"
    DEPENDS "${_BM_ELF}"
    COMMENT "Ada300 Baremetal: generating disassembly"
    VERBATIM
  )

  # ---------------------------------------------------------------------------
  # Convenience CMake targets
  # ---------------------------------------------------------------------------

  # Build target: just produce the ELF (+ disasm)
  add_custom_target(ada300-baremetal
    DEPENDS "${_BM_ELF}"
            "${_BM_OBJ_DIR}/ada300-baremetal.asm"
    COMMENT "Ada300 Baremetal ELF ready: ${_BM_ELF}"
  )

  # Run target: boot in QEMU ada300s_evk
  add_custom_target(run-ada300-baremetal
    COMMAND "${ADA300_QEMU}"
              -machine ada300s_evk
              -cpu ada300s
              -nographic
              -serial mon:stdio
              -kernel "${_BM_ELF}"
    DEPENDS "${_BM_ELF}"
    USES_TERMINAL
    COMMENT "Booting ada300-baremetal.elf on QEMU ada300s_evk (Ctrl-A x to exit)"
  )

  message(STATUS "Ada300: baremetal target enabled")
  message(STATUS "  ADA300_LLVM_ROOT = ${ADA300_LLVM_ROOT}")
  message(STATUS "  ADA300_SDK_ROOT  = ${ADA300_SDK_ROOT}")
  message(STATUS "  ADA300_QEMU      = ${ADA300_QEMU}")

else()
  message(STATUS "Ada300: baremetal target disabled (cmake -DADA300_BAREMETAL=ON to enable)")
endif() # ADA300_BAREMETAL

