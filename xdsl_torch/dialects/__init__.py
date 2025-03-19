from collections.abc import Callable

from xdsl.ir import Dialect


def get_all_dialects() -> dict[str, Callable[[], Dialect]]:
    """Returns all available dialects."""

    def get_torch():
        from xdsl_torch.dialects.torch_dialect import TorchDialect

        return TorchDialect

    def get_builtin():
        from xdsl.dialects.builtin import Builtin

        return Builtin

    def get_test():
        from xdsl.dialects.test import Test

        return Test

    return {"torch": get_torch, "test": get_test, "builtin": get_builtin}
