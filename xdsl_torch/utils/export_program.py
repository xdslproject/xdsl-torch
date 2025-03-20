import torch
from xdsl.dialects import func
from xdsl.utils.exceptions import DiagnosticException

from xdsl_torch.dialects.torch_mapping import REVERSE_XDSL_TORCH_OPS


def export_program(func_op: func.FuncOp) -> torch.fx.Graph:
    # TODO: instead of a Graph object construct a full ExportedProgram
    graph = torch.fx.Graph()
    nodes: dict[str, torch.fx.Node] = {}
    for arg in func_op.args:
        if arg.name_hint is None:
            # TODO: come up with a scheme that works without name hints just in case
            raise DiagnosticException("Name hints are required for the conversion")
        nodes[arg.name_hint] = graph.create_node(
            "placeholder", arg.name_hint, None, None, arg.name_hint
        )
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
