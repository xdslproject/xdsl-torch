from collections.abc import Iterable
from itertools import zip_longest
from typing import Any

import torch
from xdsl.builder import Builder
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import (
    BoolAttr,
    FloatAttr,
    IntegerAttr,
    NoneType,
    TensorType,
    VectorType,
)
from xdsl.ir import SSAValue
from xdsl.rewriter import InsertPoint

from xdsl_torch.dialects.torch_dialect import (
    Torch_ConstantNoneOp,
    Torch_PrimListConstructOp,
)
from xdsl_torch.dialects.torch_mapping import XDSL_TORCH_OPS
from xdsl_torch.utils.type_mapping import TORCH_DTYPE_TO_XDSL_TYPE


def create_constant_op_with_value(
    value: Any,
) -> tuple[str, arith.ConstantOp | Torch_ConstantNoneOp]:
    """
    Construct a ConstantOp for a scalar value.
    """
    if value is None:
        new_const = Torch_ConstantNoneOp(result_types=[NoneType()])
        new_const.result.name_hint = "none"
        return "none", new_const

    match value:
        case bool():
            attr = BoolAttr.from_bool(value)
        case int():
            attr = IntegerAttr.from_int_and_width(value, 32)
        case float():
            attr = FloatAttr(value, 32)
        case _:
            raise NotImplementedError(
                f"Default values not implemented for type {type(value)}"
            )

    new_const = arith.ConstantOp(attr)
    new_name = f"{type(value).__name__}{value}"
    new_const.result.name_hint = new_name

    return new_name, new_const


def create_op_operands(
    node: torch.fx.Node, xdsl_nodes: dict[str, SSAValue], builder: Builder
) -> list[SSAValue]:
    """
    Construct a list of operands for a target operation defined by `node`.
    Constructs const ops for static arguments.
    """
    arguments: tuple[tuple[torch.Argument, Any], ...] = tuple(
        zip_longest(node.target._schema.arguments, node.args)  # type: ignore
    )
    operands: list[SSAValue] = []

    for arg_spec, arg_value in arguments:
        if type(arg_value) is torch.fx.Node:
            assert (
                arg_value.name in xdsl_nodes
            ), f"Node {arg_value.name} should have been processed before"
            operands.append(xdsl_nodes[arg_value.name])
            continue

        if arg_value is None:
            assert (
                arg_spec.has_default_value()
            ), "A non provided argument must have a default value"
            value = arg_spec.default_value
        else:
            value = arg_value

        if isinstance(value, Iterable):
            elements: list[SSAValue] = []
            for v in value:  # type: ignore
                new_name, new_const = create_constant_op_with_value(v)
                builder.insert(new_const)
                xdsl_nodes[new_name] = new_const.result
                elements.append(new_const.result)
            new_name = "list"
            new_const = Torch_PrimListConstructOp(
                operands=[elements],
                result_types=[VectorType(elements[0].type, [len(elements)])],
            )
        else:
            new_name, new_const = create_constant_op_with_value(value)

        xdsl_nodes[new_name] = new_const.result
        operands.append(new_const.result)
        builder.insert(new_const)

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
    builder = Builder(InsertPoint.at_end(func_op.body.block))

    xdsl_nodes: dict[str, SSAValue] = {}
    for i, original_arg in enumerate(prog.graph_signature.input_specs):
        func_op.args[i].name_hint = original_arg.arg.name
        xdsl_nodes[original_arg.arg.name] = func_op.args[i]

    for node in prog.graph.nodes:
        if node.op == "call_function":
            if node.target not in XDSL_TORCH_OPS:
                raise NotImplementedError(
                    f"Unimplemented: target={node.target}, {node.meta}"
                )
            operands = create_op_operands(node, xdsl_nodes, builder)
            new_op = XDSL_TORCH_OPS[node.target](
                operands=operands,
                result_types=[
                    TensorType(
                        TORCH_DTYPE_TO_XDSL_TYPE[node.meta["tensor_meta"].dtype],
                        node.meta["tensor_meta"].shape,
                    )
                ],  # we currently think that everything returns a single tensor
            )
            builder.insert(new_op)
            new_op.result.name_hint = node.name
            xdsl_nodes[node.name] = new_op
    builder.insert(
        func.ReturnOp(
            *[
                xdsl_nodes[out_node.arg.name]
                for out_node in prog.graph_signature.output_specs
            ]
        )
    )

    return func_op
