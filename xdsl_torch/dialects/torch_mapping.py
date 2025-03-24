from typing import Any

import torch

from xdsl_torch.dialects.torch_dialect import *

XDSL_TORCH_OPS: dict[Any, type] = {
    torch.ops.aten.mul.Tensor: Torch_AtenMulTensorOp,
    torch.ops.aten.sin.default: Torch_AtenSinOp,
    torch.ops.aten.cos.default: Torch_AtenCosOp,
}

REVERSE_XDSL_TORCH_OPS = {
    xdsl_op: torch_op for torch_op, xdsl_op in XDSL_TORCH_OPS.items()
}
