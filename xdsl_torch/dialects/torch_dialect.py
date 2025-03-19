from xdsl.ir import Dialect

from xdsl_torch.dialects.torch_ops import AtenCosOp, AtenMulTensorOp, AtenSinOp

TorchDialect = Dialect("torch", [AtenMulTensorOp, AtenSinOp, AtenCosOp], [])
