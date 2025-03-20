import torch
from xdsl.dialects.builtin import f64

from xdsl_torch.dialects.torch_mapping import REVERSE_XDSL_TORCH_OPS, XDSL_TORCH_OPS
from xdsl_torch.dialects.torch_ops import AtenMulTensorOp
from xdsl_torch.utils.type_mapping import TORCH_DTYPE_TO_XDSL_TYPE


def test_mappings():
    assert XDSL_TORCH_OPS[torch.ops.aten.mul.Tensor] is AtenMulTensorOp  # type: ignore
    assert REVERSE_XDSL_TORCH_OPS[AtenMulTensorOp] is torch.ops.aten.mul.Tensor  # type: ignore
    assert TORCH_DTYPE_TO_XDSL_TYPE[torch.float64] is f64
