# ===- model.py ----------------------------------------------------------------
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
# PyTorch model definition for the Ada300 MLIR dialect example.
#
# The model is a simple feedforward network that explicitly exercises three
# compute primitives of interest for the Ada300 target:
#
#   1. matmul  – two fully-connected (Linear) projections, lowered to
#                linalg.matmul in the generated MLIR.
#   2. exp     – element-wise exponential activation, lowered to math.exp.
#   3. sqrt    – element-wise square root (applied after exp, so inputs are
#                always positive), lowered to math.sqrt.
#
# Architecture:
#   input  ──► fc1 (matmul) ──► exp ──► sqrt ──► fc2 (matmul) ──► output
#
# ===---------------------------------------------------------------------------

import torch
import torch.nn as nn


class Ada300Model(nn.Module):
    """Feedforward model showcasing exp, sqrt, and matmul (Linear) operations.

    Args:
        in_features:     Number of input features.  Default: 64.
        hidden_features: Width of the hidden layer.  Default: 128.
        out_features:    Number of output features.  Default: 64.
    """

    def __init__(
        self,
        in_features: int = 64,
        hidden_features: int = 128,
        out_features: int = 64,
    ) -> None:
        super().__init__()
        # Two learnable linear projections (each internally uses addmm).
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # matmul #1 – input projection
        x = self.fc1(x)
        # exp – exponential activation; output is positive, safe for sqrt
        x = torch.exp(x)
        # sqrt – square-root normalization
        x = torch.sqrt(x)
        # matmul #2 – output projection
        x = self.fc2(x)
        return x
