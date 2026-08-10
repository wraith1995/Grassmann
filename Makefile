# grassmann — local dev tasks.

PYTHON ?= python

.PHONY: test lint lint-docs

test:
	$(PYTHON) -m pytest tests -q

# Docstring + annotation gate (config in pyproject.toml): ruff's D/ANN rules
# under the numpy convention, then numpydoc's consistency checks. Both must be
# clean before committing. The per-file-ignores allowlist in pyproject.toml
# names the modules not yet swept; it only shrinks.
lint-docs:
	$(PYTHON) -m ruff check --select D,ANN src
	$(PYTHON) -m numpydoc lint $(shell find src -name '*.py' -not -path '*__pycache__*')

lint: lint-docs
	$(PYTHON) -m mypy
