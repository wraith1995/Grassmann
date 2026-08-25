# grassmann — local dev tasks.

PYTHON ?= python

.PHONY: test lint lint-docs audit

test:
	$(PYTHON) -m pytest tests -q

# Exhaustive-dispatch / lowering audit: fails on any unapproved ERROR over the closed sum
# types (Space/Term/Basis/Dim). Pure stdlib, no install needed. See audit/README.md.
audit:
	$(PYTHON) audit/run.py --min-severity ERROR

# Docstring + annotation gate (config in pyproject.toml): ruff's D/ANN rules
# under the numpy convention, then numpydoc's consistency checks. Both must be
# clean before committing. The per-file-ignores allowlist in pyproject.toml
# names the modules not yet swept; it only shrinks.
# The ignore list is declared in pyproject, but a CLI `--select` takes precedence over the
# config's ignore, so it has to be repeated here. Without it the gate silently re-enables
# rules the project deliberately turns off — D105 (magic-method docstrings) alone accounts
# for 183 findings — and can never go green.
DOC_IGNORES ?= D105,D107,D203,D213

lint-docs:
	$(PYTHON) -m ruff check --select D,ANN --ignore $(DOC_IGNORES) src
	$(PYTHON) -m numpydoc lint $(shell find src -name '*.py' -not -path '*__pycache__*')

lint: lint-docs
	$(PYTHON) -m mypy
