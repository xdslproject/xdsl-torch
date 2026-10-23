// RUN: XDSL_TORCH_ROUNDTRIP
// RUN: TORCH_MLIR_ROUNDTRIP
// RUN: TORCH_MLIR_GENERIC_ROUNDTRIP
// RUN: torch-mlir-opt %s --allow-unregistered-dialect | filecheck %s

// Both tools canonicalize the generic dialect spelling to !torch.int.
"test.op"() {generic = !torch<int>, pretty = !torch.int} : () -> ()
%value = "test.op"() : () -> !torch.int
"test.op"(%value) {types = [!torch.int, tuple<!torch.int, i64>]} : (!torch.int) -> ()

// CHECK: module {
// CHECK-NEXT: "test.op"() {generic = !torch.int, pretty = !torch.int} : () -> ()
// CHECK-NEXT: %[[VALUE:.*]] = "test.op"() : () -> !torch.int
// CHECK-NEXT: "test.op"(%[[VALUE]]) {types = [!torch.int, tuple<!torch.int, i64>]} : (!torch.int) -> ()
// CHECK-NEXT: }
