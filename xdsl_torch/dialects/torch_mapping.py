from typing import Any

import torch

from xdsl_torch.dialects.torch_ops import *

XDSL_TORCH_OPS: dict[Any, type] = {
    torch.ops.aten.mul.Tensor: AtenMulTensorOp,
    torch.ops.aten.sin.default: AtenSinOp,
    torch.ops.aten.cos.default: AtenCosOp,
}

REVERSE_XDSL_TORCH_OPS = {
    xdsl_op: torch_op for torch_op, xdsl_op in XDSL_TORCH_OPS.items()
}
