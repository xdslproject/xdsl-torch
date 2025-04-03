// RUN: XDSL_TORCH_ROUNDTRIP

%x, %y = "test.op"() : () -> (tensor<10x10xf32>, tensor<10x10xf32>)

// CHECK: builtin.module {
// CHECK-NEXT: %x, %y = "test.op"() : () -> (tensor<10x10xf32>, tensor<10x10xf32>)

// CHECK-NEXT: %0 = torch.aten.cos %x : tensor<10x10xf32> -> tensor<10x10xf32>
%0 = torch.aten.cos %x : tensor<10x10xf32> -> tensor<10x10xf32>
// CHECK-NEXT: %1 = torch.aten.sin %x : tensor<10x10xf32> -> tensor<10x10xf32>
%1 = torch.aten.sin %x : tensor<10x10xf32> -> tensor<10x10xf32>
// CHECK-NEXT: %2 = torch.aten.mul.Tensor %x, %y : tensor<10x10xf32>, tensor<10x10xf32> -> tensor<10x10xf32>
%2 = torch.aten.mul.Tensor %x, %y : tensor<10x10xf32>, tensor<10x10xf32> -> tensor<10x10xf32>
// CHECK-NEXT: %3 = torch.constant.none
%3 = torch.constant.none
// CHECK-NEXT: %el1, %el2, %el3 = "test.op"() : () -> (f32, f32, f32)
// CHECK-NEXT: %list = torch.prim.ListConstruct %el1, %el2, %el3 : (f32, f32, f32) -> vector<f32>
%el1, %el2, %el3 = "test.op"() : () -> (f32, f32, f32)
%list = torch.prim.ListConstruct %el1, %el2, %el3 : (f32, f32, f32) -> vector<f32>
