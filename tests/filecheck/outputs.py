# RUN: python %s | filecheck %s

import torch
from torch.export import export

from xdsl_torch.utils.import_program import import_program


class MaxPool(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.nn.functional.max_pool2d_with_indices(x, 1, 1)  # type: ignore


exported_program: torch.export.ExportedProgram = export(
    MaxPool(), args=(torch.randn(1, 100, 100),)
)

# CHECK:       %4, %5 = torch.aten.max_pool2d_with_indices %x, %0, %1, %2, %3, %boolFalse : tensor<1x100x100xf32>, vector<2xi32>, vector<2xi32>, vector<2xi32>, vector<2xi32>, i1 -> tensor<1x100x100xf32>, tensor<1x100x100xi64>
# CHECK-NEXT:  func.return %4, %5 : tensor<1x100x100xf32>, tensor<1x100x100xi64>
print(import_program(exported_program))
