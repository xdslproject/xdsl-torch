from xdsl.dialects.builtin import NoneType
from xdsl.ir import Dialect
from xdsl.irdl import (
    EqAttrConstraint,
    IRDLOperation,
    irdl_op_definition,
    result_def,
    traits_def,
)
from xdsl.traits import (
    ConstantLike,
    Pure,
)

from xdsl_torch.dialects.torch_generated_ops import TorchDialect


@irdl_op_definition
class Torch_ConstantNoneOp(IRDLOperation):
    name = "torch.constant.none"
    result = result_def(EqAttrConstraint(NoneType()))
    traits = traits_def(ConstantLike(), Pure())
    assembly_format = "attr-dict"


ops = list(TorchDialect.operations)
ops.append(Torch_ConstantNoneOp)

TorchDialect = Dialect("torch", ops, list(TorchDialect.attributes))
