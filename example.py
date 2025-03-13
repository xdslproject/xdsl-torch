from typing import Dict
import torch
from torch.export import export

from xdsl.context import MLContext
from xdsl.dialects import get_all_dialects, arith, func
from xdsl.passes import PipelinePass, PipelinePassSpec, ModulePass
from xdsl.parser import Parser
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def
from xdsl.dialects.builtin import AnyTensorTypeConstr, ModuleOp, TensorType, DenseIntOrFPElementsAttr, AnyFloatConstr, f32
from xdsl.ir import Dialect, SSAValue
from xdsl.pattern_rewriter import RewritePattern, op_type_rewrite_pattern, PatternRewriter, PatternRewriteWalker
from xdsl.printer import Printer
from xdsl.builder import ImplicitBuilder
import re
from io import StringIO

@irdl_op_definition
class AtenMulTensorOp(IRDLOperation):
    name = "torch.aten.mul.Tensor"

    self = operand_def(AnyTensorTypeConstr)
    other = operand_def(AnyTensorTypeConstr)
    result = result_def(AnyTensorTypeConstr)

    assembly_format = "$self `,` $other attr-dict `:` type($self) `,` type($other) `->` type($result)"

xdsl_torch_ops: Dict[str, IRDLOperation] = {torch.ops.aten.mul.Tensor: AtenMulTensorOp}
reverse_xdsl_torch_ops = {xdsl_op: torch_op for torch_op, xdsl_op in xdsl_torch_ops.items()}
TorchDialect = Dialect("torch", [AtenMulTensorOp], [])


class Mod(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y

example_args = (torch.randn(10, 10), torch.randn(10, 10))

exported_program: torch.export.ExportedProgram = export(
    Mod(), args=example_args
)

print(exported_program.graph)

TORCH_DTYPE_TO_XDSL_TYPE = {
    torch.float32: f32
}

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
                if node.target not in xdsl_torch_ops:
                    raise NotImplementedError(
                        f"FIX ME: Unimplemented call_function: target={node.target}, {node.meta}"
                    )
                tensor_meta = node.meta["tensor_meta"]
                new_op = xdsl_torch_ops[node.target](
                    operands=[xdsl_nodes[arg.name] for arg in node.args],
                    result_types=[TensorType(TORCH_DTYPE_TO_XDSL_TYPE[tensor_meta.dtype], tensor_meta.shape)] # we currently think that everything returns a single tensor
                )
                new_op.result.name_hint = node.name
                xdsl_nodes[node.name] = new_op
        func.ReturnOp(*[xdsl_nodes[out_node.arg.name] for out_node in sign.output_specs])

    return func_op

xdsl_op = import_program(exported_program)
print(xdsl_op)

def export_program(func_op: func.FuncOp) -> torch.fx.Graph:
    # TODO: instead of a Graph object construct a full ExportedProgram
    graph = torch.fx.Graph()
    nodes: Dict[str, torch.fx.Node] = {}
    for arg in func_op.args:
        nodes[arg.name_hint] = graph.create_node("placeholder", arg.name_hint, None, None, arg.name_hint)
    for op in func_op.body.ops:
        if type(op) is func.ReturnOp:
            graph.create_node("output", "", tuple(nodes[arg.name_hint] for arg in op.operands))
        else:
            nodes[op.results[0].name_hint] = graph.create_node(
                "call_function",
                reverse_xdsl_torch_ops[type(op)],
                tuple(nodes[arg.name_hint] for arg in op.operands),
                None,
                op.results[0].name_hint
            )
    return graph

graph = export_program(xdsl_op)
print(graph)
graph_mod = torch.fx.GraphModule(torch.nn.Module(), graph)
print(torch.isclose(graph_mod.forward(example_args[0], example_args[1]), Mod().forward(example_args[0], example_args[1])).all().item())
