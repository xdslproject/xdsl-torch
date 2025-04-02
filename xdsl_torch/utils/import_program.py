from itertools import zip_longest
from typing import Any

import torch
from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import BoolAttr, FloatAttr, IntegerAttr, TensorType
from xdsl.ir import SSAValue

from xdsl_torch.dialects.torch_mapping import XDSL_TORCH_OPS
from xdsl_torch.utils.type_mapping import TORCH_DTYPE_TO_XDSL_TYPE


def literal_to_ssa(value: Any) -> tuple[str, SSAValue]:
    match value:
        case int():
            attr = IntegerAttr.from_int_and_width(value, 32)
        case float():
            attr = FloatAttr(value, 32)
        case bool():
            attr = BoolAttr.from_bool(value)
        case _:
            raise NotImplementedError(
                f"Default values not implemented for type {type(value)}"
            )

    new_const = arith.ConstantOp(attr, attr.get_type())
    new_name = f"{type(value).__name__}{value}"
    new_const.result.name_hint = new_name

    return new_name, new_const.result


def get_op_operands(
    node: torch.fx.Node, xdsl_nodes: dict[str, SSAValue]
) -> list[SSAValue]:
    arguments: list[tuple[torch.Argument, Any]] = list(
        zip_longest(node.target._schema.arguments, node.args)  # type: ignore
    )
    operands: list[SSAValue] = []

    for arg_spec, arg_value in arguments:
        if type(arg_value) is torch.fx.Node:
            if arg_value.name not in xdsl_nodes:
                raise Exception(
                    f"Node {arg_value.name} should have been processed before"
                )
            operands.append(xdsl_nodes[arg_value.name])

        if arg_value is None:
            if not (
                arg_spec.has_default_value() and arg_spec.default_value is not None
            ):
                raise Exception("A non provided argument must have a default value")
            new_name, new_const = literal_to_ssa(arg_spec.default_value)
            xdsl_nodes[new_name] = new_const
            operands.append(new_const)

    return operands


def import_program(
    prog: torch.export.ExportedProgram, func_name: str = "main"
) -> func.FuncOp:
    placeholder_nodes: dict[str, torch.fx.Node] = {}
    all_producer_nodes: dict[str, torch.fx.Node] = {}
    for node in prog.graph.nodes:
        if node.op == "placeholder":
            placeholder_nodes[node.name] = node
            all_producer_nodes[node.name] = node
        elif node.op == "call_function":
            all_producer_nodes[node.name] = node

    # Generate func signature
    def make_tensor_type(
        arg: torch.export.graph_signature.InputSpec
        | torch.export.graph_signature.OutputSpec,
    ):
        tensor_meta = all_producer_nodes[arg.arg.name].meta["tensor_meta"]
        return TensorType(
            TORCH_DTYPE_TO_XDSL_TYPE[tensor_meta.dtype],
            tensor_meta.shape,
        )

    inp_sign = list(map(make_tensor_type, prog.graph_signature.input_specs))
    out_sign = list(map(make_tensor_type, prog.graph_signature.output_specs))

    # Build a FuncOp
    func_op = func.FuncOp(func_name, (inp_sign, out_sign))

    with ImplicitBuilder(func_op.body) as args:
        xdsl_nodes: dict[str, SSAValue] = {}
        for i, original_arg in enumerate(prog.graph_signature.input_specs):
            args[i].name_hint = original_arg.arg.name
            xdsl_nodes[original_arg.arg.name] = args[i]

        for node in prog.graph.nodes:
            if node.op == "call_function":
                if node.target not in XDSL_TORCH_OPS:
                    raise NotImplementedError(
                        f"Unimplemented: target={node.target}, {node.meta}"
                    )
                operands = get_op_operands(node, xdsl_nodes)
                new_op = XDSL_TORCH_OPS[node.target](
                    operands=operands,
                    result_types=[
                        TensorType(
                            TORCH_DTYPE_TO_XDSL_TYPE[node.meta["tensor_meta"].dtype],
                            node.meta["tensor_meta"].shape,
                        )
                    ],  # we currently think that everything returns a single tensor
                )
                new_op.result.name_hint = node.name
                xdsl_nodes[node.name] = new_op
        func.ReturnOp(
            *[
                xdsl_nodes[out_node.arg.name]
                for out_node in prog.graph_signature.output_specs
            ]
        )

    return func_op
