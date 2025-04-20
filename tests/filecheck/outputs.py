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

# CHECK:       %{{\S+}}, %{{\S+}} = torch.aten.max_pool2d_with_indices %x, %{{\S+}}, %{{\S+}}, %{{\S+}}, %{{\S+}}, %boolFalse : tensor<1x100x100xf32>, vector<2xi32>, vector<2xi32>, vector<2xi32>, vector<2xi32>, i1 -> tensor<1x100x100xf32>, tensor<1x100x100xi64>
# CHECK-NEXT:  func.return %{{\S+}}, %{{\S+}} : tensor<1x100x100xf32>, tensor<1x100x100xi64>
print(import_program(exported_program))
