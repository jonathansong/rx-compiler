#!/usr/bin/env python3
# ===- ada300-import.py --------------------------------------------------------
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
# Ada300 model AOT importer.
#
# This script uses TorchDynamo (DynamoCompiler) to trace the Ada300Model and
# capture its compute graph, then drives the graph through the Buddy MLIR
# pipeline to produce:
#
#   subgraph0.mlir  – the compute subgraph in tensor-level MLIR (linalg +
#                     math dialect ops: matmul, exp, sqrt).
#   forward.mlir    – the top-level dispatch function that calls subgraph0.
#   arg0.data       – flattened float32 parameter weights (for AOT execution).
#
# Op mapping (PyTorch → Buddy graph IR → MLIR dialect):
#   nn.Linear  →  AddMMOp  →  linalg.matmul  (+ arith.addf for bias)
#   torch.exp  →  ExpOp    →  math.exp
#   torch.sqrt →  SqrtOp   →  math.sqrt
#
# The script merges the TOSA and Math ops registries so that linalg-backed
# ops (AddMMOp) are taken from the TOSA registry while transcendental ops
# (ExpOp, SqrtOp) are lowered to the MLIR math dialect via the Math registry.
#
# ===---------------------------------------------------------------------------

import os
import sys
import argparse

import numpy as np
import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy.compiler.ops import math as buddy_math

from model import Ada300Model

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Ada300 model AOT importer")
parser.add_argument(
    "--output-dir",
    type=str,
    default="./",
    help="Directory to save the generated MLIR and parameter files.",
)
parser.add_argument(
    "--in-features",
    type=int,
    default=64,
    help="Number of input features for the Ada300Model (default: 64).",
)
parser.add_argument(
    "--hidden-features",
    type=int,
    default=128,
    help="Width of the hidden layer (default: 128).",
)
parser.add_argument(
    "--out-features",
    type=int,
    default=64,
    help="Number of output features (default: 64).",
)
args = parser.parse_args()

output_dir = os.path.abspath(args.output_dir)
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Build the model
# ---------------------------------------------------------------------------
model = Ada300Model(
    in_features=args.in_features,
    hidden_features=args.hidden_features,
    out_features=args.out_features,
).eval()

# ---------------------------------------------------------------------------
# Build the combined ops registry
#
# - tosa.ops_registry  provides linalg-backed handlers for AddMMOp, etc.
# - buddy_math.ops_registry  provides math-dialect handlers for ExpOp and
#   SqrtOp (math.exp / math.sqrt).
#
# The Math registry is merged last so its entries take precedence for ops
# that appear in both (e.g. ExpOp, RsqrtOp).
# ---------------------------------------------------------------------------
combined_registry = {**tosa.ops_registry, **buddy_math.ops_registry}

# ---------------------------------------------------------------------------
# Initialise DynamoCompiler
# ---------------------------------------------------------------------------
dynamo_compiler = DynamoCompiler(
    primary_registry=combined_registry,
    aot_autograd_decomposition=inductor_decomp,
)

# ---------------------------------------------------------------------------
# Trace and capture the compute graph
# ---------------------------------------------------------------------------
sample_input = torch.randn(1, args.in_features)

with torch.no_grad():
    graphs = dynamo_compiler.importer(model, sample_input)

assert len(graphs) == 1, (
    f"Expected exactly 1 captured graph, got {len(graphs)}. "
    "Check for graph breaks in the model."
)

graph = graphs[0]
params = dynamo_compiler.imported_params[graph]

# ---------------------------------------------------------------------------
# Graph optimisation – simple operator fusion
# ---------------------------------------------------------------------------
graphs[0].fuse_ops([simply_fuse])

# ---------------------------------------------------------------------------
# Lower to top-level MLIR (tensor-level linalg + math dialect)
# ---------------------------------------------------------------------------
driver = GraphDriver(graphs[0])
driver.subgraphs[0].lower_to_top_level_ir()

# ---------------------------------------------------------------------------
# Write output files
# ---------------------------------------------------------------------------
subgraph_path = os.path.join(output_dir, "subgraph0.mlir")
forward_path = os.path.join(output_dir, "forward.mlir")
param_path = os.path.join(output_dir, "arg0.data")

with open(subgraph_path, "w") as f:
    print(driver.subgraphs[0]._imported_module, file=f)

with open(forward_path, "w") as f:
    print(driver.construct_main_graph(True), file=f)

float32_param = np.concatenate(
    [param.detach().numpy().reshape([-1]) for param in params]
)
float32_param.tofile(param_path)

print(f"[ada300-import] subgraph MLIR  → {subgraph_path}")
print(f"[ada300-import] forward MLIR   → {forward_path}")
print(f"[ada300-import] parameters     → {param_path}  "
      f"({float32_param.size} float32 values)")
