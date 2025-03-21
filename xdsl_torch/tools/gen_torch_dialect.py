import subprocess
from typing import Any

import torch
from xdsl.dialects.builtin import AnyTensorTypeConstr
from xdsl.ir import Attribute
from xdsl.irdl import GenericRangeConstraint, OpDef, OperandDef, ResultDef, SingleOf

##### This stuff will be in core xdsl

constraint_mapping: dict[GenericRangeConstraint[Attribute], str] = {
    SingleOf(AnyTensorTypeConstr): "AnyTensorTypeConstr",  # plus other constraints
}


def opdef_to_class_string(class_name: str, op: OpDef) -> str:
    operand_str = "\n\t".join(
        [
            f"{name} = operand_def({constraint_mapping[oper.constr]})"
            for name, oper in op.operands
        ]
    )
    results_str = "\n\t".join(
        [
            f"{name} = result_def({constraint_mapping[oper.constr]})"
            for name, oper in op.results
        ]
    )
    return f"""
@irdl_op_definition
class {class_name}(IRDLOperation):
\tname = "{op.name}"
\t{operand_str}
\t{results_str}
\t{f'assembly_format = "{op.assembly_format}"' if op.assembly_format else ""}
    """


def dump_dialect_pyfile(
    dialect_name: str,
    ops: list[tuple[str, OpDef]],
    file_name: str,
    generator_script: str,
    preambule: str = "",
    dialect_obj_name: str = "",
):
    if not dialect_obj_name:
        dialect_obj_name = dialect_name.capitalize() + "Dialect"

    imports = """
from xdsl.dialects.builtin import *
from xdsl.ir import *
from xdsl.irdl import *

# ruff: noqa: F403, F405
    """
    if not preambule:
        preambule = f"""
###
# This dialect is automatically generated by {generator_script}
# Please don't edit it manually!
###
        """

    op_class_defs = "\n".join(
        [opdef_to_class_string(class_name, op) for class_name, op in ops]
    )

    op_list = ",".join([name for name, _ in ops])
    dialect_def = f'{dialect_obj_name} = Dialect("{dialect_name}", [{op_list}], [])'

    content = "\n".join([imports, preambule, op_class_defs, dialect_def])

    # Format output
    output = subprocess.run(
        [
            "ruff",
            "format",
            "--stdin-filename",
            f"{dialect_name}.py",
        ],
        input=content,
        capture_output=True,
        text=True,
    )

    with open(file_name, "w+") as f:
        print(output.stdout, file=f, end="")


#######

TORCH_TYPE_TO_ODS_TYPE = {
    "Tensor": AnyTensorTypeConstr,
}

preambule = """
###
# This dialect is automatically generated by tools/gen_torch_dialect.py
# Please don't edit it manually!
###
"""

imports = """
from xdsl.dialects.builtin import AnyTensorTypeConstr
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def
"""


def format_name(name: str):
    # Format in a way consistent with torch-mlir
    if not name:
        return name

    new_name = name.title().replace("_", "")
    new_name = (
        ("_" if name[0] == "_" else "") + new_name + ("_" if name[-1] == "_" else "")
    )
    return new_name


def gen_irdl_op(ns: str, op_name: str, overload_name: str, schema: Any):
    full_op_name = f"torch.{ns}.{op_name}{'.' if overload_name else ''}{overload_name}"
    if "out" in full_op_name:
        # These are ops that store their results in a given argument-buffer
        # Should be delt with separately
        return None, None
    if full_op_name in [
        "torch.aten._linalg_slogdet.sign",
        "torch.aten._linalg_det.result",
    ]:
        # Ops have argument and return named the same way => we get an error
        return None, None

    class_name = (
        "Torch_" + ns.title() + format_name(op_name) + format_name(overload_name) + "Op"
    )

    # Parse arguments
    if any(str(arg.type) not in TORCH_TYPE_TO_ODS_TYPE for arg in schema.arguments):
        return None, None
    operands = [
        (arg.name, OperandDef(TORCH_TYPE_TO_ODS_TYPE[str(arg.type)]))
        for arg in schema.arguments
    ]

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
        name=full_op_name, operands=operands, results=results, assembly_format=asm
    )

    return class_name, op_def


def generate_ops() -> tuple[list[tuple[str, OpDef]], dict[str, str]]:
    op_class_mapping: dict[str, str] = {}
    ops: list[tuple[str, OpDef]] = []
    for ns in map(str, torch.ops):
        for op_name in getattr(torch.ops, ns):
            opoverloadpacket = getattr(getattr(torch.ops, ns), op_name)
            for overload_name, schema in opoverloadpacket._schemas.items():
                class_name, opdef = gen_irdl_op(ns, op_name, overload_name, schema)
                if opdef and class_name:
                    ops.append((class_name, opdef))
                    full_name = f"torch.ops.{ns}.{op_name}"
                    full_name += f".{overload_name if overload_name else "default"}"
                    op_class_mapping[full_name] = class_name
    return ops, op_class_mapping


def create_op_class_mapping_file(op_class_mapping: dict[str, str]):
    imports = """
from typing import Dict, Any
import torch
from xdsl_torch.dialects.torch_generated import *
    """
    mapping = (
        "XDSL_TORCH_OPS: Dict[Any, type] = {"
        + "\n\t".join(
            [
                f"{torch_class}: {xdsl_class},  # type: ignore"
                for torch_class, xdsl_class in op_class_mapping.items()
            ]
        )
        + "\n}"
    )
    reverse_mapping = """
REVERSE_XDSL_TORCH_OPS = {
    xdsl_op: torch_op for torch_op, xdsl_op in XDSL_TORCH_OPS.items()
}
    """
    content = "\n".join([preambule, imports, mapping, reverse_mapping])

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

dump_dialect_pyfile(
    "torch", ops, "xdsl_torch/dialects/torch_generated.py", "tools/gen_torch_dialect.py"
)

with open("xdsl_torch/dialects/torch_mapping_generated.py", "w+") as f:
    f.writelines(create_op_class_mapping_file(op_class_mapping))
