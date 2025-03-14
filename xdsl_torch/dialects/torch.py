from typing import Dict, Any
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def
from xdsl.dialects.builtin import AnyTensorTypeConstr
from xdsl.ir import Dialect
import torch

@irdl_op_definition
class AtenMulTensorOp(IRDLOperation):
    name = "torch.aten.mul.Tensor"

    self = operand_def(AnyTensorTypeConstr)
    other = operand_def(AnyTensorTypeConstr)
    result = result_def(AnyTensorTypeConstr)

    assembly_format = "$self `,` $other attr-dict `:` type($self) `,` type($other) `->` type($result)"

XDSL_TORCH_OPS: Dict[Any, type] = {torch.ops.aten.mul.Tensor: AtenMulTensorOp} # type: ignore
REVERSE_XDSL_TORCH_OPS = {xdsl_op: torch_op for torch_op, xdsl_op in XDSL_TORCH_OPS.items()}
TorchDialect = Dialect("torch", [AtenMulTensorOp], [])
