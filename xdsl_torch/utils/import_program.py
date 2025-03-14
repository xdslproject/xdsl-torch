from typing import Dict
import torch
from xdsl.dialects import func
from xdsl.dialects.builtin import TensorType
from xdsl.ir import SSAValue
from xdsl.builder import ImplicitBuilder

from xdsl_torch.dialects.torch import XDSL_TORCH_OPS
from xdsl_torch.dialects.types import TORCH_DTYPE_TO_XDSL_TYPE

def import_program(prog: torch.export.ExportedProgram, func_name: str = "main") -> func.FuncOp:
    placeholder_nodes: Dict[str, torch.fx.Node] = {}
    all_producer_nodes: Dict[str, torch.fx.Node] = {}
    for node in prog.graph.nodes:
        if node.op == "placeholder":
            placeholder_nodes[node.name] = node
            all_producer_nodes[node.name] = node
        elif node.op == "call_function":
            all_producer_nodes[node.name] = node
    
    # Generate func signature
    def make_tensor_type(arg: torch.export.graph_signature.InputSpec | torch.export.graph_signature.OutputSpec):
        tensor_meta = all_producer_nodes[arg.arg.name].meta["tensor_meta"]
        return TensorType(TORCH_DTYPE_TO_XDSL_TYPE[tensor_meta.dtype], tensor_meta.shape)

    inp_sign = list(map(make_tensor_type, prog.graph_signature.input_specs))
    out_sign = list(map(make_tensor_type, prog.graph_signature.output_specs))

    # Build a FuncOp
    func_op = func.FuncOp(func_name, (inp_sign, out_sign))

    with ImplicitBuilder(func_op.body) as args:
        xdsl_nodes : Dict[str, SSAValue] = {}
        for i, original_arg in enumerate(prog.graph_signature.input_specs):
            args[i].name_hint = original_arg.arg.name
            xdsl_nodes[original_arg.arg.name] = args[i]

        for node in prog.graph.nodes:
            if node.op == "call_function":
                if node.target not in XDSL_TORCH_OPS:
                    raise NotImplementedError(
                        f"FIX ME: Unimplemented call_function: target={node.target}, {node.meta}"
                    )
                arg_names = [arg.name if type(arg) is torch.fx.Node else "" for arg in node.args]
                assert(all(arg_names))
                new_op = XDSL_TORCH_OPS[node.target](
                    operands=[xdsl_nodes[name] for name in arg_names],
                    result_types=[TensorType(TORCH_DTYPE_TO_XDSL_TYPE[node.meta["tensor_meta"].dtype], node.meta["tensor_meta"].shape)] # we currently think that everything returns a single tensor
                )
                new_op.result.name_hint = node.name
                xdsl_nodes[node.name] = new_op
        func.ReturnOp(*[xdsl_nodes[out_node.arg.name] for out_node in prog.graph_signature.output_specs])

    return func_op
