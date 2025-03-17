from torch_ops import XDSL_TORCH_OPS
from xdsl.ir import Dialect

REVERSE_XDSL_TORCH_OPS = {
    xdsl_op: torch_op for torch_op, xdsl_op in XDSL_TORCH_OPS.items()
}
TorchDialect = Dialect("torch", list(XDSL_TORCH_OPS.values()), [])
