MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables

# make tasks run all commands in a single shell
.ONESHELL:

# set up all precommit hooks
.PHONY: precommit-install
precommit-install:
	pre-commit install

# run all precommit hooks and apply them
.PHONY: precommit
precommit:
	pre-commit run --all

# run filecheck tests
.PHONY: filecheck
filecheck:
	uv run lit -vv tests/filecheck --order=smart --timeout=20

# run pytest tests
.PHONY: pytest
pytest:
	uv run pytest tests -W error -vv

# run all tests
.PHONY: tests-functional
tests-functional: pytest filecheck
	@echo All functional tests done.

# run all tests
.PHONY: tests
tests: tests-functional pyright
	@echo All tests done.

# run pyright on all files in the current git commit
.PHONY: pyright
pyright:
	uv run pyright $(shell git diff --staged --name-only  -- '*.py')
