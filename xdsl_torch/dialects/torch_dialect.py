from xdsl.ir import Dialect

from xdsl_torch.dialects.torch_mapping import XDSL_TORCH_OPS

TorchDialect = Dialect("torch", list(XDSL_TORCH_OPS.values()), [])
