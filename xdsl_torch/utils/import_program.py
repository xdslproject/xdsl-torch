from typing import Dict
import torch
from xdsl.dialects import func
from xdsl.dialects.builtin import TensorType
from xdsl.ir import SSAValue
from xdsl.builder import ImplicitBuilder

from xdsl_torch.dialects.torch import XDSL_TORCH_OPS
from xdsl_torch.dialects.types import TORCH_DTYPE_TO_XDSL_TYPE

def import_program(prog: torch.export.ExportedProgram, func_name: str = "main") -> func.FuncOp:
    placeholder_nodes: Dict[str, torch.Node] = {}
    all_producer_nodes: Dict[str, torch.Node] = {}
    for node in prog.graph.nodes:
        if node.op == "placeholder":
            placeholder_nodes[node.name] = node
            all_producer_nodes[node.name] = node
        elif node.op == "call_function":
            all_producer_nodes[node.name] = node
    
    # Generate func signature
    sign = prog.graph_signature
    inp_sign, out_sign = [], []

    for arg in sign.input_specs:
        arg_node = all_producer_nodes[arg.arg.name]
        tensor_meta: torch.fx.passes.TensorMetadata = arg_node.meta['tensor_meta']
        inp_sign.append(TensorType(TORCH_DTYPE_TO_XDSL_TYPE[tensor_meta.dtype], tensor_meta.shape))
    
    for arg in sign.output_specs:
        arg_node = all_producer_nodes[arg.arg.name]
        tensor_meta: torch.fx.passes.TensorMetadata = arg_node.meta['tensor_meta']
        out_sign.append(TensorType(TORCH_DTYPE_TO_XDSL_TYPE[tensor_meta.dtype], tensor_meta.shape))

    # Build a FuncOp
    func_op = func.FuncOp(func_name, (inp_sign, out_sign))

    with ImplicitBuilder(func_op.body) as args:
        for i, original_arg in enumerate(sign.input_specs):
            args[i].name_hint = original_arg.arg.name
        xdsl_nodes : Dict[str, SSAValue] = {arg.name_hint: arg for arg in func_op.args}

        for node in prog.graph.nodes:
            if node.op == "call_function":
                if node.target not in XDSL_TORCH_OPS:
                    raise NotImplementedError(
                        f"FIX ME: Unimplemented call_function: target={node.target}, {node.meta}"
                    )
                tensor_meta = node.meta["tensor_meta"]
                new_op = XDSL_TORCH_OPS[node.target](
                    operands=[xdsl_nodes[arg.name] for arg in node.args],
                    result_types=[TensorType(TORCH_DTYPE_TO_XDSL_TYPE[tensor_meta.dtype], tensor_meta.shape)] # we currently think that everything returns a single tensor
                )
                new_op.result.name_hint = node.name
                xdsl_nodes[node.name] = new_op
        func.ReturnOp(*[xdsl_nodes[out_node.arg.name] for out_node in sign.output_specs])

    return func_op
