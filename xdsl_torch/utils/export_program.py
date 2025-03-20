import torch
from xdsl.dialects import func

from xdsl_torch.dialects.torch_mapping import REVERSE_XDSL_TORCH_OPS


def export_program(func_op: func.FuncOp) -> torch.fx.Graph:
    # TODO: instead of a Graph object construct a full ExportedProgram
    graph = torch.fx.Graph()

    arg_name_hints = tuple(
        arg.name_hint for arg in func_op.args if arg.name_hint is not None
    )
    if len(arg_name_hints) != len(func_op.args):
        raise ValueError(f"Args of {func_op.sym_name} must have name hints")
    nodes = {
        name_hint: graph.create_node("placeholder", name_hint, None, None, name_hint)
        for name_hint in arg_name_hints
    }

    for op in func_op.body.ops:
        if isinstance(op, func.ReturnOp):
            graph.create_node(
                "output",
                "",
                tuple(
                    nodes[arg.name_hint if arg.name_hint else ""] for arg in op.operands
                ),
            )
        else:
            nodes[op.results[0].name_hint if op.results[0].name_hint else ""] = (
                graph.create_node(
                    "call_function",
                    REVERSE_XDSL_TORCH_OPS[type(op)],
                    tuple(
                        nodes[arg.name_hint if arg.name_hint else ""]
                        for arg in op.operands
                    ),
                    None,
                    op.results[0].name_hint,
                )
            )
    return graph
