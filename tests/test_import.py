import pytest

from xdsl_torch.utils.import_program import create_constant_op_with_value


def test_import_bad_value():
    with pytest.raises(NotImplementedError):
        create_constant_op_with_value("string")
