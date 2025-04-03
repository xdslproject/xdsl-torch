# RUN: python %s | filecheck %s

import torch
from torch.export import export

from xdsl_torch.utils.import_program import import_program


class DiagDef(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.diagonal(x)


exported_program: torch.export.ExportedProgram = export(
    DiagDef(), args=(torch.randn(10, 10),)
)

# CHECK:        %int0 = arith.constant 0 : i32
# CHECK-NEXT:   %int0_1 = arith.constant 0 : i32
# CHECK-NEXT:   %int1 = arith.constant 1 : i32
# CHECK-NEXT:   %diagonal = torch.aten.diagonal %x, %int0, %int0_1, %int1 : tensor<10x10xf32>, i32, i32, i32 -> tensor<10xf32>
print(import_program(exported_program))


class Diag(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.diagonal(x, 2, 1, 0)


exported_program: torch.export.ExportedProgram = export(
    Diag(), args=(torch.randn(10, 10),)
)

# CHECK:        %int2 = arith.constant 2 : i32
# CHECK-NEXT:   %int1 = arith.constant 1 : i32
# CHECK-NEXT:   %int0 = arith.constant 0 : i32
# CHECK-NEXT:   %diagonal = torch.aten.diagonal %x, %int2, %int1, %int0 : tensor<10x10xf32>, i32, i32, i32 -> tensor<8xf32>
print(import_program(exported_program))


class Dist(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pairwise_distance(x, y, 3, 1e-1, True)


exported_program: torch.export.ExportedProgram = export(
    Dist(), args=(torch.randn(10), torch.randn(10))
)

# CHECK:        %float3.0 = arith.constant 3.000000e+00 : f32
# CHECK-NEXT:   %float0.1 = arith.constant 1.000000e-01 : f32
# CHECK-NEXT:   %boolTrue = arith.constant true
# CHECK-NEXT:   %pairwise_distance = torch.aten.pairwise_distance %x, %y, %float3.0, %float0.1, %boolTrue : tensor<10xf32>, tensor<10xf32>, f32, f32, i1 -> tensor<1xf32>
print(import_program(exported_program))
