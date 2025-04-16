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


class Add(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 2.5


exported_program: torch.export.ExportedProgram = export(Add(), args=(torch.randn(10),))

# CHECK:       %float2.5 = arith.constant 2.500000e+00 : f32
# CHECK-NEXT:  %int1 = arith.constant 1 : i32
# CHECK-NEXT:  %add = torch.aten.add.Tensor %x, %float2.5, %int1 : tensor<10xf32>, f32, i32 -> tensor<10xf32>
print(import_program(exported_program))


class ArgMin(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmin(x, keepdim=True)


exported_program: torch.export.ExportedProgram = export(
    ArgMin(), args=(torch.randn(10),)
)

# CHECK:       %none = torch.constant.none
# CHECK-NEXT:  %boolTrue = arith.constant true
# CHECK-NEXT:  %argmin = torch.aten.argmin %x, %none, %boolTrue : tensor<10xf32>, none, i1 -> tensor<1xi64>
print(import_program(exported_program))


class AvgPool(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.AvgPool2d(3)(x)


exported_program: torch.export.ExportedProgram = export(
    AvgPool(), args=(torch.randn(1, 16, 50, 100),)
)

# CHECK:        %int3 = arith.constant 3 : i32
# CHECK-NEXT:   %int3_1 = arith.constant 3 : i32
# CHECK-NEXT:   %0 = torch.prim.ListConstruct %int3, %int3_1 : (i32, i32) -> vector<2xi32>
# CHECK-NEXT:   %int3_2 = arith.constant 3 : i32
# CHECK-NEXT:   %int3_3 = arith.constant 3 : i32
# CHECK-NEXT:   %1 = torch.prim.ListConstruct %int3_2, %int3_3 : (i32, i32) -> vector<2xi32>
# CHECK-NEXT:   %int0 = arith.constant 0 : i32
# CHECK-NEXT:   %int0_1 = arith.constant 0 : i32
# CHECK-NEXT:   %2 = torch.prim.ListConstruct %int0, %int0_1 : (i32, i32) -> vector<2xi32>
# CHECK-NEXT:   %boolFalse = arith.constant false
# CHECK-NEXT:   %boolTrue = arith.constant true
# CHECK-NEXT:   %none = torch.constant.none
# CHECK-NEXT:   %avg_pool2d = torch.aten.avg_pool2d %x, %0, %1, %2, %boolFalse, %boolTrue, %none : tensor<1x16x50x100xf32>, vector<2xi32>, vector<2xi32>, vector<2xi32>, i1, i1, none -> tensor<1x16x16x33xf32>
print(import_program(exported_program))
