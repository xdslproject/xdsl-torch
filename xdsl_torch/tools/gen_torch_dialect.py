import subprocess
from typing import Any

import torch
from xdsl.dialects.builtin import (
    AnyTensorTypeConstr,
    ContainerOf,
    Float64Type,
    IntegerType,
    NoneType,
    Signedness,
)
from xdsl.irdl import (
    AnyAttr,
    Attribute,
    BaseAttr,
    EqAttrConstraint,
    GenericAttrConstraint,
    OpDef,
    OperandDef,
    ResultDef,
    VarOperandDef,
    traits_def,
)
from xdsl.traits import (
    ConstantLike,
    Pure,
)
from xdsl.utils.dialect_codegen import dump_dialect_pyfile

TORCH_TYPE_TO_ODS_TYPE: dict[str, GenericAttrConstraint[Attribute]] = {
    "Tensor": AnyTensorTypeConstr,
    "List[Tensor]": ContainerOf(AnyTensorTypeConstr),
    "int": BaseAttr(IntegerType),
    "List[int]": ContainerOf(IntegerType),
    "float": BaseAttr(Float64Type),
    "List[float]": ContainerOf(BaseAttr(Float64Type)),
    "bool": EqAttrConstraint(IntegerType(1, Signedness.UNSIGNED)),
    "List[bool]": ContainerOf(EqAttrConstraint(IntegerType(1, Signedness.UNSIGNED))),
    "number": BaseAttr(IntegerType) | BaseAttr(Float64Type),
    "List[number]": ContainerOf(BaseAttr(IntegerType) | BaseAttr(Float64Type)),
}

#### Non aten ops
custom_ops = [
    (
        "Torch_ConstantNoneOp",
        OpDef(
            name="torch.constant.none",
            results=[("result", ResultDef(EqAttrConstraint(NoneType())))],
            traits=traits_def(ConstantLike(), Pure()),
            assembly_format="attr-dict",
        ),
    ),
    (
        "Torch_PrimListConstructOp",
        OpDef(
            name="torch.prim.ListConstruct",
            operands=[("elements", VarOperandDef(AnyAttr()))],
            results=[("result", ResultDef(ContainerOf(AnyAttr())))],
            traits=traits_def(Pure()),
            assembly_format="$elements attr-dict `:` functional-type($elements, $result)",  # noqa: E501
        ),
    ),
]
####

preamble = """###
# This dialect is automatically generated by xdsl_torch/tools/gen_torch_dialect.py
# Please don't edit it manually!
###
"""


def format_name(name: str):
    """
    Format operation name consistent with torch-mlir.
    Example: _some_processor_class_ -> _SomeProcessorClass_
    """
    if not name:
        return name

    new_name = name.title().replace("_", "")
    new_name = (
        ("_" if name[0] == "_" else "") + new_name + ("_" if name[-1] == "_" else "")
    )
    return new_name


def get_base_type(type_str: str) -> str:
    if "Optional" in type_str:
        type_str = type_str[type_str.find("[") + 1 : type_str.rfind("]")]
    return type_str


def get_operand_def(type_str: str) -> OperandDef:
    xdsl_type = TORCH_TYPE_TO_ODS_TYPE[get_base_type(type_str)]
    if "Optional" in type_str:
        xdsl_type |= EqAttrConstraint(NoneType())
    return OperandDef(xdsl_type)


def gen_irdl_op(ns: str, op_name: str, overload_name: str, schema: Any):
    full_op_name = f"torch.{ns}.{op_name}{'.' if overload_name else ''}{overload_name}"
    if overload_name == "out":
        # These are ops that store their results in a given argument-buffer
        # Should be delt with separately
        return None, None
    if full_op_name in [
        "torch.aten._linalg_slogdet.sign",
        "torch.aten._linalg_det.result",
        "torch.aten.kthvalue.values",
        "torch.aten.linalg_cholesky_ex.L",
        "torch.aten.linalg_inv_ex.inverse",
        "torch.aten.nanmedian.dim_values",
        "torch.aten.median.dim_values",
        "torch.aten.mode.values",
        "torch.aten.sort.values",
        "torch.aten.svd.U",
        "torch.aten.topk.values",
        "torch.aten._linalg_solve_ex.result",
        "torch.aten.sort.values_stable",
        "torch.aten.frexp.Tensor_out",
    ]:
        # Ops have argument and return named the same way => we get an error
        return None, None

    class_name = (
        "Torch_" + ns.title() + format_name(op_name) + format_name(overload_name) + "Op"
    )

    # Parse arguments
    if any(
        get_base_type(str(arg.type)) not in TORCH_TYPE_TO_ODS_TYPE
        for arg in schema.arguments
    ):
        return None, None

    operands = [(arg.name, get_operand_def(str(arg.type))) for arg in schema.arguments]

    # Parse results
    if any(str(out.type) not in TORCH_TYPE_TO_ODS_TYPE for out in schema.returns):
        return None, None

    results_names = [out.name if out.name else "result" for out in schema.returns]
    if results_names.count("result") > 1:
        unnamed_idx = 0
        for i, name in enumerate(results_names):
            if name == "result":
                results_names[i] = f"result{unnamed_idx}"
                unnamed_idx += 1

    results = [
        (name, ResultDef(TORCH_TYPE_TO_ODS_TYPE[str(out.type)]))
        for name, out in zip(results_names, schema.returns)
    ]

    # Asm format
    args_asm = " `,` ".join([f"${arg.name}" for arg in schema.arguments])
    args_types_asm = " `,` ".join([f"type(${arg.name})" for arg in schema.arguments])
    outs_types_asm = " `,` ".join([f"type(${out_name})" for out_name in results_names])
    asm = (
        args_asm
        + " attr-dict `:` "
        + args_types_asm
        + (" `->` " + outs_types_asm if results_names else "")
    )

    op_def = OpDef(
        name=full_op_name,
        operands=operands,
        results=results,
        assembly_format=asm,
    )

    return class_name, op_def


def get_core_op_list() -> list[tuple[str, str, str, Any]]:
    core_ops: list[tuple[str, str, str, Any]] = []
    for ns in map(str, torch.ops):
        for op_name in getattr(torch.ops, ns):
            opoverloadpacket = getattr(getattr(torch.ops, ns), op_name)
            for overload_name in opoverloadpacket._dir:
                op = getattr(opoverloadpacket, overload_name)
                if torch.Tag.core in op._tags:
                    overload_name = "" if overload_name == "default" else overload_name
                    core_ops.append((ns, op_name, overload_name, op._schema))
    return core_ops


def generate_ops() -> tuple[list[tuple[str, OpDef]], dict[str, str]]:
    op_class_mapping: dict[str, str] = {}
    ops: list[tuple[str, OpDef]] = []

    for ns, op_name, overload_name, schema in get_core_op_list():
        class_name, opdef = gen_irdl_op(ns, op_name, overload_name, schema)
        full_name = f"torch.ops.{ns}.{op_name}"
        full_name += f".{overload_name if overload_name else "default"}"

        if not opdef or not class_name:
            print(f"Couldn't generate {full_name}")
            continue

        ops.append((class_name, opdef))
        op_class_mapping[full_name] = class_name

    return ops, op_class_mapping


def create_op_class_mapping_file(op_class_mapping: dict[str, str]) -> str:
    imports = """
from typing import Dict, Any
import torch
from xdsl_torch.dialects.torch_dialect import *
    """
    dict_strings = [
        f"{torch_class}: {xdsl_class},  # type: ignore"
        for torch_class, xdsl_class in op_class_mapping.items()
    ]
    dict_strings.sort()

    mapping = "XDSL_TORCH_OPS: Dict[Any, type] = {" + "\n\t".join(dict_strings) + "\n}"
    reverse_mapping = """
REVERSE_XDSL_TORCH_OPS = {
    xdsl_op: torch_op for torch_op, xdsl_op in XDSL_TORCH_OPS.items()
}"""
    content = "\n".join([imports, mapping, reverse_mapping])

    # Format output
    output = subprocess.run(
        [
            "ruff",
            "format",
            "--stdin-filename",
            "mapping.py",
        ],
        input=content,
        capture_output=True,
        text=True,
    )

    return output.stdout


## Running everything

ops, op_class_mapping = generate_ops()
ops.sort(key=lambda x: x[0])

with open("xdsl_torch/dialects/torch_dialect.py", "w+") as f:
    print(preamble, file=f)
    dump_dialect_pyfile("torch", ops + custom_ops, out=f)  # type: ignore

with open("xdsl_torch/dialects/torch_mapping.py", "w+") as f:
    print(preamble, file=f)
    print(create_op_class_mapping_file(op_class_mapping), file=f)
