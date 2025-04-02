# RUN: python %s | filecheck %s

import torch
from torch.export import export

from xdsl_torch.utils.import_program import import_program


class Diag(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.diagonal(x)


exported_program: torch.export.ExportedProgram = export(
    Diag(), args=(torch.randn(10, 10),)
)

# CHECK:        %int0 = arith.constant 0 : i32
# CHECK-NEXT:   %int0_1 = arith.constant 0 : i32
# CHECK-NEXT:   %int1 = arith.constant 1 : i32
# CHECK-NEXT:   %diagonal = torch.aten.diagonal %x, %int0, %int0_1, %int1 : tensor<10x10xf32>, i32, i32, i32 -> tensor<10xf32>
print(import_program(exported_program))


class Diag(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.diagonal(x, 2, 1, 0)


# CHECK:        %int2 = arith.constant 2 : i32
# CHECK-NEXT:   %int1 = arith.constant 1 : i32
# CHECK-NEXT:   %int0 = arith.constant 0 : i32
# CHECK-NEXT:   %diagonal = torch.aten.diagonal %x, %int2, %int1, %int0 : tensor<10x10xf32>, i32, i32, i32 -> tensor<8xf32>
exported_program: torch.export.ExportedProgram = export(
    Diag(), args=(torch.randn(10, 10),)
)

print(import_program(exported_program))
