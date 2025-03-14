# RUN: python %s | filecheck %s

import torch
from torch.export import export

from xdsl_torch.utils.import_program import import_program
from xdsl_torch.utils.export_program import export_program

class SimpleMult(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y
    
example_args = (torch.randn(10, 10), torch.randn(10, 10))

exported_program: torch.export.ExportedProgram = export(
    SimpleMult(), args=example_args
)

xdsl_op = import_program(exported_program)
# CHECK:       func.func @main(%x : tensor<10x10xf32>, %y : tensor<10x10xf32>) -> tensor<10x10xf32> {
# CHECK-NEXT:    %mul = torch.aten.mul.Tensor %x, %y : tensor<10x10xf32>, tensor<10x10xf32> -> tensor<10x10xf32>
# CHECK-NEXT:    func.return %mul : tensor<10x10xf32>
# CHECK-NEXT:  }
print(xdsl_op)

graph = export_program(xdsl_op)
graph_mod = torch.fx.GraphModule(torch.nn.Module(), graph)
assert(torch.isclose(graph_mod.forward(example_args[0], example_args[1]), SimpleMult().forward(example_args[0], example_args[1])).all().item())
