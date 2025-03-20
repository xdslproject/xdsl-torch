import torch
from xdsl.dialects.builtin import f16, f32, f64

TORCH_DTYPE_TO_XDSL_TYPE = {torch.float16: f16, torch.float32: f32, torch.float64: f64}
