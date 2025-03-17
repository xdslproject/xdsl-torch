from typing import Any

import torch
from xdsl.dialects.builtin import AnyTensorTypeConstr
from xdsl.irdl import (
    IRDLOperation,
    Operation,
    irdl_op_definition,
    operand_def,
    result_def,
)

XDSL_TORCH_OPS: dict[Any, type[Operation]] = {}


@irdl_op_definition
class AtenMulTensorOp(IRDLOperation):
    name = "torch.aten.mul.Tensor"

    self = operand_def(AnyTensorTypeConstr)
    other = operand_def(AnyTensorTypeConstr)
    result = result_def(AnyTensorTypeConstr)

    assembly_format = (
        "$self `,` $other attr-dict `:` type($self) `,` type($other) `->` type($result)"
    )


XDSL_TORCH_OPS[torch.ops.aten.mul.Tensor] = AtenMulTensorOp  # type: ignore


@irdl_op_definition
class AtenSinOp(IRDLOperation):
    name = "torch.aten.sin"

    self = operand_def(AnyTensorTypeConstr)
    result = result_def(AnyTensorTypeConstr)

    assembly_format = "$self attr-dict `:` type($self) `->` type($result)"


XDSL_TORCH_OPS[torch.ops.aten.sin.default] = AtenSinOp  # type: ignore


@irdl_op_definition
class AtenCosOp(IRDLOperation):
    name = "torch.aten.cos"

    self = operand_def(AnyTensorTypeConstr)
    result = result_def(AnyTensorTypeConstr)

    assembly_format = "$self attr-dict `:` type($self) `->` type($result)"


XDSL_TORCH_OPS[torch.ops.aten.cos.default] = AtenCosOp  # type: ignore
