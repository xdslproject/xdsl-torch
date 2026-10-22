# xdsl-torch

Use xDSL for custom compiler rewrites in PyTorch.

## Installation

```sh
pip install 'xdsl-torch[torch]'
```

The optional `torch` extra installs pinned PyTorch and
[torch-mlir](https://github.com/llvm/torch-mlir/releases/tag/v20261001) versions for
compatibility checking. `uv sync` includes this extra for development, and
`make tests` runs the compatibility tests alongside the rest of the suite.

## Generating files

To generate the dialect files, run:

```sh
uv run python -m xdsl_torch.tools.gen_torch_dialect
```
