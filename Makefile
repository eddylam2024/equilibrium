.PHONY: install dev test test-live lint typecheck check demo uninstall

install:        ## Install the `equilibrium` / `eq` commands globally (editable)
	uv tool install --force --editable .

uninstall:
	uv tool uninstall equilibrium

dev:            ## Create the development environment
	uv sync

test:           ## Offline test suite (synthetic data)
	uv run pytest

test-live:      ## Live Yahoo Finance integration tests (network)
	uv run pytest -m live

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src

check: lint typecheck test

demo:           ## Offline demo on synthetic data
	uv run equilibrium portfolio AAA BBB CCC DDD --synthetic
