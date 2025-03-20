# RUN: python %s | filecheck %s

import torch
from torch.export import export

from xdsl_torch.utils.import_program import import_program


class SimpleMult(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y


args = (torch.randn(10, 10), torch.randn(10, 10))

exported_program: torch.export.ExportedProgram = export(SimpleMult(), args=args)
xdsl_op = import_program(exported_program)
# CHECK:       func.func @main(%x : tensor<10x10xf32>, %y : tensor<10x10xf32>) -> tensor<10x10xf32> {
# CHECK-NEXT:    %mul = torch.aten.mul.Tensor %x, %y : tensor<10x10xf32>, tensor<10x10xf32> -> tensor<10x10xf32>
# CHECK-NEXT:    func.return %mul : tensor<10x10xf32>
# CHECK-NEXT:  }
print(xdsl_op)


class SinCosMult(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.cos(x) * torch.sin(y)


exported_program: torch.export.ExportedProgram = export(SinCosMult(), args=args)
xdsl_op = import_program(exported_program)
# CHECK:       func.func @main(%x : tensor<10x10xf32>, %y : tensor<10x10xf32>) -> tensor<10x10xf32> {
# CHECK-NEXT:    %cos = torch.aten.cos %x : tensor<10x10xf32> -> tensor<10x10xf32>
# CHECK-NEXT:    %sin = torch.aten.sin %y : tensor<10x10xf32> -> tensor<10x10xf32>
# CHECK-NEXT:    %mul = torch.aten.mul.Tensor %cos, %sin : tensor<10x10xf32>, tensor<10x10xf32> -> tensor<10x10xf32>
# CHECK-NEXT:    func.return %mul : tensor<10x10xf32>
# CHECK-NEXT:  }
print(xdsl_op)
