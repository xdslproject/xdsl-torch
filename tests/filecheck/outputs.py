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

print(exported_program.graph)

# CHECK:        %int0 = arith.constant 0 : i32
# CHECK-NEXT:   %int0_1 = arith.constant 0 : i32
# CHECK-NEXT:   %int1 = arith.constant 1 : i32
# CHECK-NEXT:   %diagonal = torch.aten.diagonal %x, %int0, %int0_1, %int1 : tensor<10x10xf32>, i32, i32, i32 -> tensor<10xf32>
print(import_program(exported_program))
