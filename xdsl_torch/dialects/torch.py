from typing import Any

import torch
from xdsl.dialects.builtin import AnyTensorTypeConstr
from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def


@irdl_op_definition
class AtenMulTensorOp(IRDLOperation):
    name = "torch.aten.mul.Tensor"

    self = operand_def(AnyTensorTypeConstr)
    other = operand_def(AnyTensorTypeConstr)
    result = result_def(AnyTensorTypeConstr)

    assembly_format = (
        "$self `,` $other attr-dict `:` type($self) `,` type($other) `->` type($result)"
    )


@irdl_op_definition
class AtenSinOp(IRDLOperation):
    name = "torch.aten.sin"

    self = operand_def(AnyTensorTypeConstr)
    result = result_def(AnyTensorTypeConstr)

    assembly_format = "$self attr-dict `:` type($self) `->` type($result)"


@irdl_op_definition
class AtenCosOp(IRDLOperation):
    name = "torch.aten.cos"

    self = operand_def(AnyTensorTypeConstr)
    result = result_def(AnyTensorTypeConstr)

    assembly_format = "$self attr-dict `:` type($self) `->` type($result)"


XDSL_TORCH_OPS: dict[Any, type] = {
    torch.ops.aten.mul.Tensor: AtenMulTensorOp,  # type: ignore
    torch.ops.aten.sin.default: AtenSinOp,  # type: ignore
    torch.ops.aten.cos.default: AtenCosOp,  # type: ignore
}
REVERSE_XDSL_TORCH_OPS = {
    xdsl_op: torch_op for torch_op, xdsl_op in XDSL_TORCH_OPS.items()
}
TorchDialect = Dialect("torch", [AtenMulTensorOp, AtenSinOp, AtenCosOp], [])
