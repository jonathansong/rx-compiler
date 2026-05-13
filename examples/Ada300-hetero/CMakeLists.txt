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
# CMake build rules for the Ada300 heterogeneous example.
#
# This example splits inference across two compute units:
#
#   Op         Backend        Location
#   ────────   ───────────    ────────────────────────────────────────
#   MatMul     RXOPS_ADA300   Ada300 RISC-V SNPU (QEMU, ivshmem)
#   Add        RXOPS_ADA300   Ada300 RISC-V SNPU (QEMU, ivshmem)
#   Exp        RXOPS_C        Host x86 CPU
#   Sqrt       RXOPS_C        Host x86 CPU
#
#   (Controlled by DEVICE_* flags in ada300-import.py — edit there to change routing.)
#
# Lowering path (host runner):
#
#   ada300-import.py annotates Top ops with device = "ada300" for SNPU ops.
#
#   ada300-import.py  (PyTorch → Top dialect MLIR)
#     ↓ python ada300-import.py --output-dir output/
#   subgraph0_top.mlir  (Top dialect — ops carry device attr)
#     ↓ tpuc-opt --convert-top-to-rxops
#         ops with device="ada300"  → llvm.call @rxops_bridge_ada300_<op>_f32
#         ops without device attr   → llvm.call @rxops_bridge_<op>_f32
#     ↓ mlir-opt  (bufferize + lower)
#     ↓ mlir-translate --mlir-to-llvmir
#     ↓ llc -filetype=obj → subgraph0_rxops.o
    #     ↓ + rx_ops_bridge.o              (all ops; ada300_*→ivshmem via #ifndef __riscv)
#     ↓ + host/hetero_shmem_host.o     (shm_open / mmap transport)
#     ↓ + host/ada300-hetero-main.o
#     ↓ link → ada300-hetero-runner
#
# Device firmware (RISC-V, cross-compiled):
#
#   BSP + device/ada300-hetero-main.c + device/hetero_shmem_device.c
#   + rx_ops_bridge_ada300.c (matmul → RXOPS_ADA300)
#   + rx-ops sources (register + ops + ada300 backend)
#   → ada300-hetero-device.elf
#
# Convenience targets:
#
#   ada300-hetero-host    — build the host runner binary
#   ada300-hetero-device  — cross-compile the device ELF (needs HETERO_DEVICE=ON)
#   run-ada300-hetero     — start QEMU + run host runner (needs HETERO_DEVICE=ON)
#
# Usage (standalone):
#   cd examples/ada300-hetero && mkdir -p build && cd build
#   cmake .. -DMLIR_DIR=/usr/local/lib/cmake/mlir
#   ninja ada300-hetero-host
#   cmake .. -DHETERO_DEVICE=ON   && ninja ada300-hetero-device
#   cmake .. -DHETERO_DEVICE=ON   && ninja run-ada300-hetero
#
# ===---------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Standalone vs. subdirectory mode.
# ---------------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(Ada300Hetero CXX C)

  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Ada300Hetero: standalone build, MLIR found at ${MLIR_DIR}")

  get_filename_component(_HETERO_REPO_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

  find_program(TPUC_OPT tpuc-opt
    HINTS "${_HETERO_REPO_ROOT}/install/bin"
          "${_HETERO_REPO_ROOT}/build/bin"
    DOC "Path to tpuc-opt (rx-mlir)"
  )
  if(NOT TPUC_OPT)
    message(FATAL_ERROR "Ada300Hetero: tpuc-opt not found.")
  endif()
  message(STATUS "Ada300Hetero: tpuc-opt = ${TPUC_OPT}")

  set(RXOPS_SRC_DIR    "${_HETERO_REPO_ROOT}/third_party/rx-ops")
  set(HETERO_SRC_DIR   "${CMAKE_CURRENT_SOURCE_DIR}")
else()
  set(TPUC_OPT       ${CMAKE_BINARY_DIR}/bin/tpuc-opt)
  set(RXOPS_SRC_DIR  ${CMAKE_SOURCE_DIR}/third_party/rx-ops)
  set(HETERO_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# ---------------------------------------------------------------------------
# Output / object directories.
# HETERO_DUMP_DIR: generated MLIR / LL / object files (source-tree).
# HETERO_OBJ_DIR:  compiled host and device objects (build-tree).
# ---------------------------------------------------------------------------
set(HETERO_DUMP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")
file(MAKE_DIRECTORY "${HETERO_DUMP_DIR}")

# The pre-built librx_ops.a in rx-ops/build/ is compiled for Ada300 RISC-V.
# The host runner needs an x86 build of the reference C backend.
# build_host/ is configured without CONFIG_BUILD_ADA300_QEMU so it builds
# only the portable reference C sources.
set(RXOPS_LIB_DEVICE "${RXOPS_SRC_DIR}/build/librx_ops.a")
set(RXOPS_LIB_HOST   "${RXOPS_SRC_DIR}/build_host/librx_ops.a")
# For the host runner we always use the x86 build.
set(RXOPS_LIB "${RXOPS_LIB_HOST}")

# ---------------------------------------------------------------------------
# Find a system llc with an x86 backend (same logic as Ada300).
# ---------------------------------------------------------------------------
find_program(LLC_SYSTEM
  # Prefer llc versions that have an X86 backend built in.
  # rx-llvm-main/build/bin/llc only has RISC-V — skip it here.
  NAMES llc-18 llc-17 llc-16 llc-15 llc-14 llc
  HINTS
    /usr/lib/llvm-18/bin
    /usr/lib/llvm-17/bin
    /usr/lib/llvm-16/bin
    /usr/lib/llvm-15/bin
    /usr/lib/llvm-14/bin
    /usr/bin
  DOC "llc with X86 target"
  NO_DEFAULT_PATH
)
if(NOT LLC_SYSTEM)
  find_program(LLC_SYSTEM NAMES llc DOC "System llc with X86 target")
endif()
if(NOT LLC_SYSTEM)
  message(WARNING "Ada300Hetero: Could not find a system llc; compilation of .ll may fail.")
  set(LLC_SYSTEM "${LLVM_TOOLS_BINARY_DIR}/llc")
endif()
message(STATUS "Ada300Hetero: LLC_SYSTEM = ${LLC_SYSTEM}")

# ===========================================================================
# HOST RUNNER TARGET  (ada300-hetero-runner)
# ===========================================================================
option(HETERO_HOST "Build Ada300 heterogeneous host runner" ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
message(STATUS "Ada300Hetero: Python3 = ${Python3_EXECUTABLE}")

if(HETERO_HOST)

  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_HETERO_TPUC_DEP "${TPUC_OPT}")
  else()
    set(_HETERO_TPUC_DEP tpuc-opt)
  endif()

  # -- Step 0: run ada300-import.py → subgraph0_top.mlir + weights -----------
  add_custom_command(
    OUTPUT  "${HETERO_DUMP_DIR}/subgraph0_top.mlir"
            "${HETERO_DUMP_DIR}/subgraph0_top_weight.npz"
    COMMAND "${Python3_EXECUTABLE}"
              "${HETERO_SRC_DIR}/ada300-import.py"
              --output-dir "${HETERO_DUMP_DIR}"
    WORKING_DIRECTORY "${HETERO_SRC_DIR}"
    DEPENDS "${HETERO_SRC_DIR}/ada300-import.py"
            "${HETERO_SRC_DIR}/model.py"
    COMMENT "Hetero Step 0: ada300-import.py → subgraph0_top.mlir"
    VERBATIM
  )

  # -- Step 1: --convert-top-to-rxops ----------------------------------------
  add_custom_command(
    OUTPUT "${HETERO_DUMP_DIR}/subgraph0_rxops.mlir"
    COMMAND "${TPUC_OPT}"
              --convert-top-to-rxops
              "${HETERO_DUMP_DIR}/subgraph0_top.mlir"
              -o "${HETERO_DUMP_DIR}/subgraph0_rxops.mlir"
    DEPENDS "${HETERO_DUMP_DIR}/subgraph0_top.mlir"
            ${_HETERO_TPUC_DEP}
    COMMENT "Hetero Step 1: --convert-top-to-rxops → subgraph0_rxops.mlir"
    VERBATIM
  )

  # -- Step 2: finish lowering to LLVM dialect --------------------------------
  set(_HETERO_BUFFERIZE_OPT
      "-one-shot-bufferize=bufferize-function-boundaries allow-return-allocs")
  add_custom_command(
    OUTPUT "${HETERO_DUMP_DIR}/subgraph0_rxops_finished.mlir"
    COMMAND "${LLVM_TOOLS_BINARY_DIR}/mlir-opt"
              "${HETERO_DUMP_DIR}/subgraph0_rxops.mlir"
              "${_HETERO_BUFFERIZE_OPT}"
              -finalize-memref-to-llvm
              -convert-arith-to-llvm
              -llvm-request-c-wrappers
              -convert-func-to-llvm
              -reconcile-unrealized-casts
              -o "${HETERO_DUMP_DIR}/subgraph0_rxops_finished.mlir"
    DEPENDS "${HETERO_DUMP_DIR}/subgraph0_rxops.mlir"
    COMMENT "Hetero Step 2: finishing passes → subgraph0_rxops_finished.mlir"
    VERBATIM
  )

  # -- Step 3: mlir-translate → LLVM IR (.ll) --------------------------------
  add_custom_command(
    OUTPUT "${HETERO_DUMP_DIR}/subgraph0_rxops.ll"
    COMMAND "${LLVM_TOOLS_BINARY_DIR}/mlir-translate"
              --mlir-to-llvmir
              "${HETERO_DUMP_DIR}/subgraph0_rxops_finished.mlir"
              -o "${HETERO_DUMP_DIR}/subgraph0_rxops.ll"
    DEPENDS "${HETERO_DUMP_DIR}/subgraph0_rxops_finished.mlir"
    COMMENT "Hetero Step 3: mlir-translate → subgraph0_rxops.ll"
    VERBATIM
  )

  # -- Step 4: llvm-as + llc → native object file ----------------------------
  add_custom_command(
    OUTPUT "${HETERO_DUMP_DIR}/subgraph0_rxops.o"
    COMMAND "${LLVM_TOOLS_BINARY_DIR}/llvm-as"
              "${HETERO_DUMP_DIR}/subgraph0_rxops.ll"
              -o "${HETERO_DUMP_DIR}/subgraph0_rxops.bc"
    COMMAND "${LLC_SYSTEM}"
              -opaque-pointers
              -filetype=obj
              -relocation-model=pic
              -O3
              "${HETERO_DUMP_DIR}/subgraph0_rxops.bc"
              -o "${HETERO_DUMP_DIR}/subgraph0_rxops.o"
    DEPENDS "${HETERO_DUMP_DIR}/subgraph0_rxops.ll"
    COMMENT "Hetero Step 4: llvm-as + llc → subgraph0_rxops.o"
    VERBATIM
  )

  # -- Step 5a: compile hetero shmem host transport (C) ----------------------
  add_custom_command(
    OUTPUT "${HETERO_DUMP_DIR}/hetero_shmem_host.o"
    COMMAND "${CMAKE_C_COMPILER}"
              -O2
              -I${HETERO_SRC_DIR}/include
              -c "${HETERO_SRC_DIR}/host/hetero_shmem_host.c"
              -o "${HETERO_DUMP_DIR}/hetero_shmem_host.o"
    DEPENDS "${HETERO_SRC_DIR}/host/hetero_shmem_host.c"
            "${HETERO_SRC_DIR}/include/hetero_shmem.h"
    COMMENT "Hetero Step 5a: compile hetero_shmem_host.c"
    VERBATIM
  )

  # -- Step 5b: compile the ops bridge (C) -----------------------------------
  #    rx_ops_bridge.c uses #ifndef __riscv to auto-select the implementation:
  #      x86 host  → ivshmem dispatch to QEMU RISC-V device
  #      __riscv   → RXOPS_ADA300 hardware backend directly (bare-metal only)
  #    No explicit compile flag needed; the target architecture decides.
  add_custom_command(
    OUTPUT "${HETERO_DUMP_DIR}/rx_ops_bridge.o"
    COMMAND "${CMAKE_C_COMPILER}"
              -O2
              -I${HETERO_SRC_DIR}
              -I${HETERO_SRC_DIR}/include
              -I${RXOPS_SRC_DIR}/include
              -I${RXOPS_SRC_DIR}/include/interface
              -c "${HETERO_SRC_DIR}/rx_ops_bridge.c"
              -o "${HETERO_DUMP_DIR}/rx_ops_bridge.o"
    DEPENDS "${HETERO_SRC_DIR}/rx_ops_bridge.c"
            "${HETERO_SRC_DIR}/rx_ops_bridge.h"
            "${HETERO_SRC_DIR}/include/hetero_shmem.h"
    COMMENT "Hetero Step 5b: compile rx_ops_bridge.c (ivshmem dispatch via __riscv auto-detect)"
    VERBATIM
  )

  set(_HETERO_BRIDGE_OBJS
      "${HETERO_DUMP_DIR}/hetero_shmem_host.o"
      "${HETERO_DUMP_DIR}/rx_ops_bridge.o")
  set(_HETERO_BRIDGE_DEPS
      "${HETERO_DUMP_DIR}/hetero_shmem_host.o"
      "${HETERO_DUMP_DIR}/rx_ops_bridge.o")

  # -- Step 6: compile the host runner (C++) ---------------------------------
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ada300-hetero-main.o"
    COMMAND "${CMAKE_CXX_COMPILER}"
              -O2
              -std=c++17
              -I"${MLIR_INCLUDE_DIRS}"
              -I"${HETERO_SRC_DIR}/include"
              -c "${HETERO_SRC_DIR}/host/ada300-hetero-main.cpp"
              -o "${CMAKE_CURRENT_BINARY_DIR}/ada300-hetero-main.o"
    DEPENDS "${HETERO_SRC_DIR}/host/ada300-hetero-main.cpp"
            "${HETERO_SRC_DIR}/include/hetero_shmem.h"
    COMMENT "Hetero Step 6: compile ada300-hetero-main.cpp"
    VERBATIM
  )

  # -- Step 7: link everything → host runner binary --------------------------
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ada300-hetero-runner"
    COMMAND "${CMAKE_CXX_COMPILER}"
              "${HETERO_DUMP_DIR}/subgraph0_rxops.o"
              ${_HETERO_BRIDGE_OBJS}
              "${CMAKE_CURRENT_BINARY_DIR}/ada300-hetero-main.o"
              -Wl,--whole-archive "${RXOPS_LIB}" -Wl,--no-whole-archive
              -lm -lstdc++ -fopenmp -lrt
              -o "${CMAKE_CURRENT_BINARY_DIR}/ada300-hetero-runner"
    DEPENDS
      "${HETERO_DUMP_DIR}/subgraph0_rxops.o"
      ${_HETERO_BRIDGE_DEPS}
      "${CMAKE_CURRENT_BINARY_DIR}/ada300-hetero-main.o"
      "${RXOPS_LIB}"
    COMMENT "Hetero Step 7: link → ada300-hetero-runner"
    VERBATIM
  )

  add_custom_target(ada300-hetero-host
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/ada300-hetero-runner"
    COMMENT "Ada300 hetero host runner ready → ${CMAKE_CURRENT_BINARY_DIR}/ada300-hetero-runner"
  )

  message(STATUS "Ada300Hetero: host runner target enabled")

else()
  message(STATUS "Ada300Hetero: host runner disabled (cmake -DHETERO_HOST=ON to enable)")
endif() # HETERO_HOST

# ===========================================================================
# DEVICE FIRMWARE TARGET  (ada300-hetero-device.elf)
#
# Cross-compiles the persistent RISC-V firmware.  Requires the Ada300 LLVM
# toolchain and ada300_snpu_sdk (same prerequisites as Ada300 BAREMETAL).
#
# Enable with: cmake -DHETERO_DEVICE=ON
# ===========================================================================
option(HETERO_DEVICE
  "Cross-compile Ada300 heterogeneous device firmware for RISC-V QEMU" OFF)

if(HETERO_DEVICE)

  # ---- Configurable paths --------------------------------------------------
  set(ADA300_LLVM_ROOT "/workspace/rx-llvm-main/build"
      CACHE PATH "Ada300 LLVM toolchain root (provides riscv64-unknown-elf clang)")

  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    get_filename_component(_sdk_default
      "${CMAKE_CURRENT_SOURCE_DIR}/../../../embed-system-main/ada300_snpu_sdk"
      ABSOLUTE)
  else()
    get_filename_component(_sdk_default
      "${CMAKE_SOURCE_DIR}/../embed-system-main/ada300_snpu_sdk"
      ABSOLUTE)
  endif()

  set(ADA300_SDK_ROOT "${_sdk_default}"
      CACHE PATH "Path to ada300_snpu_sdk (BSP sources + linker script)")

  set(ADA300_QEMU "/opt/ada300-qemu/bin/qemu-system-riscv64"
      CACHE FILEPATH "QEMU binary with ada300s_evk machine")

  set(ADA300_NEWLIB_SYSROOT "/usr/lib/picolibc/riscv64-unknown-elf"
      CACHE PATH "Picolibc/newlib sysroot for riscv64-unknown-elf")

  # Validate
  if(NOT EXISTS "${ADA300_LLVM_ROOT}/bin/clang")
    message(FATAL_ERROR
      "Ada300Hetero: LLVM toolchain not found at '${ADA300_LLVM_ROOT}/bin/clang'. "
      "Pass -DADA300_LLVM_ROOT=/path/to/ada300-llvm")
  endif()
  if(NOT EXISTS "${ADA300_SDK_ROOT}/bsp/ld/ada300s_evk.ld")
    message(FATAL_ERROR
      "Ada300Hetero: ada300_snpu_sdk not found at '${ADA300_SDK_ROOT}'. "
      "Pass -DADA300_SDK_ROOT=/path/to/ada300_snpu_sdk")
  endif()

  set(_ADA300_CC      "${ADA300_LLVM_ROOT}/bin/clang")
  set(_ADA300_OBJDUMP "${ADA300_LLVM_ROOT}/bin/llvm-objdump")

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

  # All device object files go into a dedicated build sub-directory.
  set(_DEV_OBJ_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build/device")
  file(MAKE_DIRECTORY "${_DEV_OBJ_DIR}")

  # Compile macro for the Ada300 RISC-V target.
  macro(ada300_hetero_compile _out _src)
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
                -I${HETERO_SRC_DIR}
                -I${HETERO_SRC_DIR}/include
                -I${RXOPS_SRC_DIR}/include
                -I${RXOPS_SRC_DIR}/include/interface
                -I${RXOPS_SRC_DIR}/include/backend
                $<$<BOOL:${ADA300_NEWLIB_SYSROOT}>:-isystem${ADA300_NEWLIB_SYSROOT}/include>
                ${_extra_incs}
                -c "${_src}"
                -o "${_out}"
      DEPENDS "${_src}"
      COMMENT "Ada300Hetero Device: compile ${_src}"
      VERBATIM
    )
  endmacro()

  # -- Step DA: compile device firmware main ---------------------------------
  ada300_hetero_compile(
    "${_DEV_OBJ_DIR}/ada300-hetero-main.o"
    "${HETERO_SRC_DIR}/device/ada300-hetero-main.c")

  # -- Step DB: compile device ivshmem transport ----------------------------
  ada300_hetero_compile(
    "${_DEV_OBJ_DIR}/hetero_shmem_device.o"
    "${HETERO_SRC_DIR}/device/hetero_shmem_device.c")

  # -- Step DC: compile rx-ops bridge for device (rx_ops_bridge.c shared with
  #             host; rxnn_c_init is declared weak so it is safely absent here)
  ada300_hetero_compile(
    "${_DEV_OBJ_DIR}/rx_ops_bridge.o"
    "${HETERO_SRC_DIR}/rx_ops_bridge.c")

  # -- Step DD: compile BSP sources ------------------------------------------
  ada300_hetero_compile(
    "${_DEV_OBJ_DIR}/bm_startup.o"
    "${ADA300_SDK_ROOT}/bsp/src/startup.S")

  ada300_hetero_compile(
    "${_DEV_OBJ_DIR}/bm_bsp_stubs.o"
    "${RXOPS_SRC_DIR}/src/bsp_stubs.c")

  set(_BSP_OBJS "${_DEV_OBJ_DIR}/bm_startup.o" "${_DEV_OBJ_DIR}/bm_bsp_stubs.o")
  foreach(_bsp_src bsp_main.c uart.c mini_printf.c mr_heap.c)
    get_filename_component(_bsp_base "${_bsp_src}" NAME_WE)
    set(_bsp_obj "${_DEV_OBJ_DIR}/bm_${_bsp_base}.o")
    ada300_hetero_compile("${_bsp_obj}" "${ADA300_SDK_ROOT}/bsp/src/${_bsp_src}")
    list(APPEND _BSP_OBJS "${_bsp_obj}")
  endforeach()

  # -- Step DE: compile rx-ops Ada300 backend sources -----------------------
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
    string(MD5 _rhash "${_rsrc}")
    string(SUBSTRING "${_rhash}" 0 6 _rhash)
    set(_robj "${_DEV_OBJ_DIR}/rxops_${_rbase}_${_rhash}.o")
    ada300_hetero_compile("${_robj}" "${_rsrc}"
        -I${RXOPS_SRC_DIR}/include/ada300
        -I${RXOPS_SRC_DIR}/src/backend/ada300)
    list(APPEND _RXOPS_OBJS "${_robj}")
  endforeach()

  foreach(_rsrc ${_RXOPS_ADA300_ASM_SRCS})
    get_filename_component(_rbase "${_rsrc}" NAME_WE)
    string(MD5 _rhash "${_rsrc}")
    string(SUBSTRING "${_rhash}" 0 6 _rhash)
    set(_robj "${_DEV_OBJ_DIR}/rxops_asm_${_rbase}_${_rhash}.o")
    ada300_hetero_compile("${_robj}" "${_rsrc}"
        -I${RXOPS_SRC_DIR}/include/ada300)
    list(APPEND _RXOPS_OBJS "${_robj}")
  endforeach()

  # -- Step DF: link → ada300-hetero-device.elf ------------------------------
  set(_DEV_ELF "${_DEV_OBJ_DIR}/ada300-hetero-device.elf")

  add_custom_command(
    OUTPUT "${_DEV_ELF}"
    COMMAND "${_ADA300_CC}"
              ${_ADA300_ARCH_FLAGS}
              -nostdlib
              -fuse-ld=lld
              -Wl,--gc-sections
              -Wl,-Map=${_DEV_OBJ_DIR}/ada300-hetero-device.map
              -T${_LD_SCRIPT}
              $<$<BOOL:${ADA300_NEWLIB_SYSROOT}>:-L${ADA300_NEWLIB_SYSROOT}/lib>
              # BSP startup must be first
              ${_BSP_OBJS}
              "${_DEV_OBJ_DIR}/ada300-hetero-main.o"
              "${_DEV_OBJ_DIR}/hetero_shmem_device.o"
              "${_DEV_OBJ_DIR}/rx_ops_bridge.o"
              ${_RXOPS_OBJS}
              $<$<BOOL:${ADA300_NEWLIB_SYSROOT}>:-Wl,--start-group>
              $<$<BOOL:${ADA300_NEWLIB_SYSROOT}>:-lm>
              $<$<BOOL:${ADA300_NEWLIB_SYSROOT}>:-Wl,--end-group>
              -o "${_DEV_ELF}"
    DEPENDS
      ${_BSP_OBJS}
      "${_DEV_OBJ_DIR}/ada300-hetero-main.o"
      "${_DEV_OBJ_DIR}/hetero_shmem_device.o"
      "${_DEV_OBJ_DIR}/rx_ops_bridge.o"
      ${_RXOPS_OBJS}
    COMMENT "Ada300Hetero Device Step DF: link → ada300-hetero-device.elf"
    VERBATIM
  )

  add_custom_command(
    OUTPUT "${_DEV_OBJ_DIR}/ada300-hetero-device.asm"
    COMMAND "${_ADA300_OBJDUMP}" -d "${_DEV_ELF}"
              > "${_DEV_OBJ_DIR}/ada300-hetero-device.asm"
    DEPENDS "${_DEV_ELF}"
    COMMENT "Ada300Hetero Device: disassembly"
    VERBATIM
  )

  add_custom_target(ada300-hetero-device
    DEPENDS "${_DEV_ELF}"
            "${_DEV_OBJ_DIR}/ada300-hetero-device.asm"
    COMMENT "Ada300 hetero device ELF ready: ${_DEV_ELF}"
  )

  # ---------------------------------------------------------------------------
  # run-ada300-hetero: launch QEMU (background) + host runner.
  #
  # Steps:
  #   1. Ensure /dev/shm/ada300_bar exists (truncate to 64 MiB).
  #   2. Start QEMU with the ivshmem backend, serial output to UART log file.
  #   3. Wait 3 seconds for firmware to boot and reach its polling loop.
  #   4. Run the host runner (which sends matmul commands via shmem).
  #   5. Kill QEMU.
  # ---------------------------------------------------------------------------
  if(HETERO_HOST)
    set(_RUNNER "${CMAKE_CURRENT_BINARY_DIR}/ada300-hetero-runner")
    set(_SHM_PATH "/dev/shm/ada300_bar")
    set(_QEMU_LOG "${CMAKE_CURRENT_BINARY_DIR}/ada300-hetero-qemu.log")

    # Script run by the run-ada300-hetero target.
    # Steps:
    #   1. Zero the shared memory file (rm+dd avoids stale ctrl data).
    #   2. Launch QEMU in the background.
    #   3. Wait until the firmware prints its "Polling" banner (max 15 s).
    #   4. Run the host inference runner.
    #   5. Show the device UART log.
    #   6. Kill QEMU; propagate the host runner's exit code.
    #
    # configure_file expands @VAR@ at cmake-time; plain ${...} shell syntax
    # is left untouched so the generated .sh works correctly.
    set(_UART_LOG "${CMAKE_CURRENT_BINARY_DIR}/ada300-hetero-uart.log")
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/run-ada300-hetero.sh.in"
      "${CMAKE_CURRENT_BINARY_DIR}/run-ada300-hetero.sh"
      @ONLY
    )

    add_custom_target(run-ada300-hetero
      COMMAND sh "${CMAKE_CURRENT_BINARY_DIR}/run-ada300-hetero.sh"
      DEPENDS "${_DEV_ELF}" "${_RUNNER}"
      USES_TERMINAL
      COMMENT "Running Ada300 hetero demo (QEMU + host runner)"
    )
  endif()

  message(STATUS "Ada300Hetero: device firmware target enabled")
  message(STATUS "  ADA300_LLVM_ROOT = ${ADA300_LLVM_ROOT}")
  message(STATUS "  ADA300_SDK_ROOT  = ${ADA300_SDK_ROOT}")
  message(STATUS "  ADA300_QEMU      = ${ADA300_QEMU}")

else()
  message(STATUS "Ada300Hetero: device firmware disabled (cmake -DHETERO_DEVICE=ON to enable)")
endif() # HETERO_DEVICE
