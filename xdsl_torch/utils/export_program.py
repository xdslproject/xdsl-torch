import torch
from xdsl.dialects import func
from xdsl.ir import SSAValue

from xdsl_torch.dialects.torch_mapping import REVERSE_XDSL_TORCH_OPS


def export_program(func_op: func.FuncOp) -> torch.fx.Graph:
    # TODO: instead of a Graph object construct a full ExportedProgram
    graph = torch.fx.Graph()

    if any(arg.name_hint is None for arg in func_op.args):
        raise ValueError(f"Args of {func_op.sym_name} must have name hints")
    nodes: dict[SSAValue, torch.fx.Node] = {
        arg: graph.create_node(
            "placeholder", str(arg.name_hint), None, None, str(arg.name_hint)
        )
        for arg in func_op.args
    }

    for op in func_op.body.ops:
        if isinstance(op, func.ReturnOp):
            graph.create_node("output", "", tuple(nodes[arg] for arg in op.operands))
        else:
            nodes[op.results[0]] = graph.create_node(
                "call_function",
                REVERSE_XDSL_TORCH_OPS[type(op)],
                tuple(nodes[arg] for arg in op.operands),
                None,
                op.results[0].name_hint,
            )
    return graph
