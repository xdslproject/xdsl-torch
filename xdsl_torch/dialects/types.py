import torch
from xdsl.dialects.builtin import f32


TORCH_DTYPE_TO_XDSL_TYPE = {
    torch.float32: f32
}