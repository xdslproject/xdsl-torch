import torch
import torchvision.models as models
from torch.export import export

from xdsl_torch.utils.import_program import import_program

module = import_program(export(models.mobilenet_v2(), (torch.randn(1, 3, 224, 224),)))

print(str(module))
