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
# Ada300 model importer: PyTorch Dynamo → Top dialect MLIR.
#
# Uses torch.export.export() to trace Ada300Model and produces:
#
#   <output-dir>/subgraph0_top.mlir        – compute graph in Top dialect
#   <output-dir>/subgraph0_top_weight.npz  – model weights as float32 arrays
#
# No Buddy Compiler dependency.
#
# Op mapping (PyTorch FX → Top dialect):
#   aten.permute / aten.t   →  (alias, folded into the func arg shape)
#   aten.addmm(bias,A,B)    →  top.MatMul(A, B, %none) + top.Add(bias, result)
#   aten.exp                →  top.Exp
#   aten.sqrt               →  top.Sqrt
#
# Func arg ordering (matches runner in ada300-rxops-main.cpp):
#   %arg0  fc1.weight^T   [in, hidden]  – transposed from storage [hidden, in]
#   %arg1  fc1.bias       [1, hidden]   – unsqueezed from storage [hidden]
#   %arg2  fc2.weight^T   [hidden, out] – transposed from storage [out, hidden]
#   %arg3  fc2.bias       [1, out]      – unsqueezed from storage [out]
#   %arg4  input          [batch, in]   – user input
#
# ===---------------------------------------------------------------------------

import os
import argparse

import numpy as np
import torch

from model import Ada300Model


# ---------------------------------------------------------------------------
# MLIR type helpers
# ---------------------------------------------------------------------------

def tensor_type(shape):
    """Return MLIR tensor type string, e.g. 'tensor<1x64xf32>'."""
    return "tensor<" + "x".join(str(d) for d in shape) + "xf32>"


# ---------------------------------------------------------------------------
# FX graph → Top dialect MLIR emitter
# ---------------------------------------------------------------------------

def emit_top_mlir(ep, func_args, val_map, weight_file):
    """Walk the exported FX graph and emit a Top dialect MLIR module.

    Args:
        ep:          torch.export.ExportedProgram
        func_args:   list of (arg_name, mlir_type, description)
        val_map:     dict {node.name: (mlir_value_str, mlir_type_str)}
                     pre-populated for placeholder nodes
        weight_file: absolute path written into module.weight_file attr

    Returns:
        MLIR module as a string.
    """
    graph = ep.graph
    ops_lines = []
    none_emitted = False
    tmp_idx = [0]

    def none_val():
        nonlocal none_emitted
        if not none_emitted:
            ops_lines.append('    %none = "top.None"() : () -> none')
            none_emitted = True
        return "%none"

    def new_val():
        v = f"%v{tmp_idx[0]}"
        tmp_idx[0] += 1
        return v

    output_val = None
    output_type = None

    for node in graph.nodes:
        if node.op == "placeholder":
            continue

        if node.op == "output":
            ret = node.args[0]
            if isinstance(ret, (list, tuple)):
                ret = ret[0]
            output_val, output_type = val_map[ret.name]
            continue

        target = str(node.target)

        # ------------------------------------------------------------------
        # aten.permute / aten.t  – weight transposition.
        # The source placeholder is already stored with the transposed shape
        # in func_args / val_map, so we alias the permute result to the same
        # MLIR value but with the actual (post-permute) shape from node.meta.
        # ------------------------------------------------------------------
        if "permute" in target or target == "aten.t.default":
            src = node.args[0]
            src_val, _ = val_map[src.name]
            shape = tuple(node.meta["val"].shape)
            val_map[node.name] = (src_val, tensor_type(shape))
            continue

        # ------------------------------------------------------------------
        # aten.addmm(bias, mat1, mat2)  →  top.MatMul + top.Add
        # ------------------------------------------------------------------
        if "addmm" in target:
            bias_node, input_node, weight_node = node.args[:3]
            bias_v,   bias_t   = val_map[bias_node.name]
            input_v,  input_t  = val_map[input_node.name]
            weight_v, weight_t = val_map[weight_node.name]

            out_shape = tuple(node.meta["val"].shape)
            out_t = tensor_type(out_shape)
            mm_v  = new_val()
            add_v = new_val()

            ops_lines.append(
                f'    {mm_v} = "top.MatMul"({input_v}, {weight_v}, {none_val()}) {{'
                f' do_relu = false, hdim_is_batch = false, keep_dims = true,'
                f' left_transpose = false, output_transpose = false,'
                f' relu_limit = -1.000000e+00 : f64, right_transpose = false'
                f' }} : ({input_t}, {weight_t}, none) -> {out_t}'
            )
            ops_lines.append(
                f'    {add_v} = "top.Add"({bias_v}, {mm_v}) {{'
                f' do_relu = false, is_scalar = false,'
                f' relu_limit = -1.000000e+00 : f64'
                f' }} : ({bias_t}, {out_t}) -> {out_t}'
            )
            val_map[node.name] = (add_v, out_t)
            continue

        # ------------------------------------------------------------------
        # aten.exp  →  top.Exp
        # ------------------------------------------------------------------
        if target.endswith("exp.default"):
            src_v, src_t = val_map[node.args[0].name]
            v = new_val()
            ops_lines.append(
                f'    {v} = "top.Exp"({src_v}) : ({src_t}) -> {src_t}'
            )
            val_map[node.name] = (v, src_t)
            continue

        # ------------------------------------------------------------------
        # aten.sqrt  →  top.Sqrt
        # ------------------------------------------------------------------
        if target.endswith("sqrt.default"):
            src_v, src_t = val_map[node.args[0].name]
            v = new_val()
            ops_lines.append(
                f'    {v} = "top.Sqrt"({src_v}) : ({src_t}) -> {src_t}'
            )
            val_map[node.name] = (v, src_t)
            continue

        raise NotImplementedError(f"Unhandled FX op: {target!r} (node: {node.name})")

    # Build the MLIR module text.
    func_arg_str = ",\n      ".join(f"{a}: {t}" for a, t, _ in func_args)
    arg_comments = "".join(
        f"//   {a}: {t}  ({desc})\n" for a, t, desc in func_args
    )
    body = "\n".join(ops_lines)

    lines = [
        "// Ada300 example – Top dialect compute graph",
        "// Auto-generated by ada300-import.py (torch.export.export).",
        "//",
        "// Lower with:",
        "//   tpuc-opt --convert-top-to-rxops output/subgraph0_top.mlir",
        "//",
        "// Func args:",
    ]
    for a, t, d in func_args:
        lines.append(f"//   {a}: {t}  ({d})")
    lines += [
        "",
        f'module attributes {{module.chip = "ALL", module.platform = "ONNX",',
        f'                   module.state = "TOSA_F32", module.top_run_mode = "STATIC",',
        f'                   module.weight_file = "{weight_file}"}} {{',
        f'  func.func @subgraph0(',
        f'      {func_arg_str}) -> {output_type} {{',
        "",
        body,
        f"    return {output_val} : {output_type}",
        "  }",
        "}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ada300Model → Top dialect MLIR importer (rx-mlir, no Buddy)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Directory for generated files (default: output/)",
    )
    parser.add_argument("--in-features",      type=int, default=64)
    parser.add_argument("--hidden-features",  type=int, default=128)
    parser.add_argument("--out-features",     type=int, default=64)
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Build and export the model via torch.export
    # -----------------------------------------------------------------------
    model = Ada300Model(
        in_features=args.in_features,
        hidden_features=args.hidden_features,
        out_features=args.out_features,
    ).eval()

    sample_input = torch.zeros(1, args.in_features)
    with torch.no_grad():
        ep = torch.export.export(model, (sample_input,))

    sig = ep.graph_signature
    node_to_param = sig.inputs_to_parameters   # {node_name: fqn}
    placeholder_nodes = [n for n in ep.graph.nodes if n.op == "placeholder"]

    # -----------------------------------------------------------------------
    # Build func args and val_map.
    #
    # Placeholder ordering from torch.export (for Ada300Model):
    #   parameters: fc1.weight [out,in], fc1.bias [hidden], fc2.weight [out,in], fc2.bias [out]
    #   user inputs: input [batch, in]
    #
    # We transform each parameter for Top dialect:
    #   2D weight  [out, in]  →  transpose  →  [in, out]   (weight^T)
    #   1D bias    [N]        →  unsqueeze  →  [1, N]
    # -----------------------------------------------------------------------
    func_args = []   # [(arg_name, mlir_type, description)]
    val_map   = {}   # {node.name: (mlir_value_str, mlir_type_str)}
    weights   = {}   # {key: np.ndarray}  – written to npz

    for idx, node in enumerate(placeholder_nodes):
        arg_name = f"%arg{idx}"

        if node.name in node_to_param:
            fqn    = node_to_param[node.name]
            tensor = ep.state_dict[fqn].detach().float()

            if tensor.ndim == 2:
                param_np = tensor.t().contiguous().numpy()   # [out,in] → [in,out]
                desc = f"{fqn} (transposed to weight^T)"
            elif tensor.ndim == 1:
                param_np = tensor.unsqueeze(0).numpy()       # [N] → [1, N]
                desc = f"{fqn} (unsqueezed to [1,N])"
            else:
                param_np = tensor.numpy()
                desc = fqn

            weights[fqn] = param_np
            shape = param_np.shape
        else:
            val    = node.meta["val"]
            shape  = tuple(val.shape)
            param_np = None
            desc   = "user input"

        mtype = tensor_type(shape)
        func_args.append((arg_name, mtype, desc))
        val_map[node.name] = (arg_name, mtype)

    # -----------------------------------------------------------------------
    # Emit MLIR and save files
    # -----------------------------------------------------------------------
    mlir_path = os.path.join(output_dir, "subgraph0_top.mlir")
    npz_path  = os.path.join(output_dir, "subgraph0_top_weight.npz")

    mlir_text = emit_top_mlir(ep, func_args, val_map, npz_path)

    with open(mlir_path, "w") as f:
        f.write(mlir_text)

    np.savez(npz_path, **weights)

    print(f"[ada300-import] Top dialect MLIR  → {mlir_path}")
    print(f"[ada300-import] weights (.npz)    → {npz_path}")
    print("[ada300-import] func args:")
    for a, t, d in func_args:
        print(f"                  {a}: {t}  // {d}")


if __name__ == "__main__":
    main()

