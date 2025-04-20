import operator
from itertools import zip_longest
from typing import Any

import torch
from torch._ops import (
    OpOverload as TorchOpOverload,  # type: ignore
)
from torch._subclasses import (
    FakeTensor as TorchFakeTensor,
)
from torch.fx.passes.shape_prop import TensorMetadata
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
from xdsl.ir import Attribute, SSAValue
from xdsl.rewriter import InsertPoint

from xdsl_torch.dialects.torch_dialect import (
    Torch_ConstantNoneOp,
    Torch_PrimListConstructOp,
)
from xdsl_torch.dialects.torch_mapping import XDSL_TORCH_OPS
from xdsl_torch.utils.type_mapping import TORCH_DTYPE_TO_XDSL_TYPE


def get_single(tup: tuple[Any, ...]) -> Any:
    if len(tup) != 1:
        raise ValueError(f"Tuple should have contained single value. Got: {tup}")
    return tup[0]


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
    node: torch.fx.Node, xdsl_nodes: dict[str, tuple[SSAValue, ...]], builder: Builder
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
            operands.append(get_single(xdsl_nodes[arg_value.name]))
            continue

        if arg_value is None:
            assert (
                arg_spec.has_default_value()
            ), "A non provided argument must have a default value"
            value = arg_spec.default_value
        else:
            value = arg_value

        if isinstance(value, list):
            elements: list[SSAValue] = []
            for v in value:  # type: ignore
                new_name, new_const = create_constant_op_with_value(v)
                builder.insert(new_const)
                xdsl_nodes[new_name] = new_const.results
                elements.append(new_const.result)
            new_name = "list"
            new_const = Torch_PrimListConstructOp(
                operands=[elements],
                result_types=[VectorType(elements[0].type, [len(elements)])],
            )
        else:
            new_name, new_const = create_constant_op_with_value(value)

        xdsl_nodes[new_name] = new_const.results
        operands.append(new_const.result)
        builder.insert(new_const)

    return operands


def get_tensor_type(shape: torch.Size, dtype: torch.dtype) -> TensorType:
    return TensorType(
        TORCH_DTYPE_TO_XDSL_TYPE[dtype],
        shape,
    )


def value_info_to_type(
    val: Any, tensor_meta: TensorMetadata | None = None
) -> Attribute:
    if tensor_meta is not None:
        if isinstance(val, list):
            raise NotImplementedError(
                f"List metadata is currently not supported: {tensor_meta}"
            )
        return get_tensor_type(tensor_meta.shape, tensor_meta.dtype)

    elif val is not None:
        if isinstance(val, list):
            raise NotImplementedError(f"List values are currently not supported: {val}")
        if isinstance(val, TorchFakeTensor):
            return get_tensor_type(val.shape, val.dtype)

    raise NotImplementedError("Either value or tensor_meta should be defined")


def node_val_to_type(node: torch.fx.Node) -> Attribute:
    try:
        tensor_meta = node.meta.get("tensor_meta")
        val = node.meta.get("val")
    except KeyError as e:
        raise RuntimeError(
            f"FIXME: Illegal access to torch.fx.Node.meta: {e} ({node.meta})"
        )
    return value_info_to_type(val, tensor_meta)


def unpack_node_result_types(node: torch.fx.Node) -> list[Attribute]:
    return_count = len(node.target._schema.returns)  # type: ignore
    result_types: list[Attribute] = []

    if return_count == 0:
        result_types = []
    elif return_count == 1:
        result_types = [node_val_to_type(node)]
    else:
        result_types = [value_info_to_type(v) for v in node.meta["val"]]

    return result_types


def import_torch_op_overload(
    node: torch.fx.Node, xdsl_nodes: dict[str, tuple[SSAValue, ...]], builder: Builder
):
    if node.target not in XDSL_TORCH_OPS:
        raise NotImplementedError(f"Unimplemented: target={node.target}, {node.meta}")

    operands = create_op_operands(node, xdsl_nodes, builder)
    result_types = unpack_node_result_types(node)
    new_op = XDSL_TORCH_OPS[node.target](
        operands=operands,
        result_types=result_types,
    )
    builder.insert(new_op)
    xdsl_nodes[node.name] = new_op.results

    if len(result_types) == 1:
        new_op.result.name_hint = node.name


def import_getitem(node: torch.fx.Node, xdsl_nodes: dict[str, tuple[SSAValue, ...]]):
    ref_node, index = node.args
    assert isinstance(
        ref_node, torch.fx.Node
    ), f"Unexpected getitem arguments: {node.args}"
    assert isinstance(index, int), f"Unexpected getitem arguments: {node.args}"

    if len(xdsl_nodes[ref_node.name]) > 1:
        xdsl_nodes[node.name] = (xdsl_nodes[ref_node.name][index],)
    else:
        raise NotImplementedError(
            "getitem is currently only supported for multi output ops"
        )


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
    inp_sign = [
        node_val_to_type(all_producer_nodes[arg.arg.name])
        for arg in prog.graph_signature.input_specs
    ]
    out_sign = [
        node_val_to_type(all_producer_nodes[arg.arg.name])
        for arg in prog.graph_signature.output_specs
    ]

    # Build a FuncOp
    func_op = func.FuncOp(func_name, (inp_sign, out_sign))
    builder = Builder(InsertPoint.at_end(func_op.body.block))

    xdsl_nodes: dict[str, tuple[SSAValue, ...]] = {}
    for i, original_arg in enumerate(prog.graph_signature.input_specs):
        func_op.args[i].name_hint = original_arg.arg.name
        xdsl_nodes[original_arg.arg.name] = (func_op.args[i],)

    for node in prog.graph.nodes:
        if node.op == "call_function":
            if isinstance(node.target, TorchOpOverload):
                import_torch_op_overload(node, xdsl_nodes, builder)
            elif node.target == operator.getitem:
                import_getitem(node, xdsl_nodes)
            else:
                raise NotImplementedError(
                    f"Target {node.target} is not supported in call_function"
                )
        elif node.op == "placeholder" or node.op == "output":
            # we have already parsed this
            continue
        else:
            raise NotImplementedError(f"Op {node.op} is not implemented")

    builder.insert(
        func.ReturnOp(
            *[
                get_single(xdsl_nodes[out_node.arg.name])
                for out_node in prog.graph_signature.output_specs
            ]
        )
    )

    return func_op
