.DEFAULT_GOAL := help

UV ?= uv
HOST ?= 127.0.0.1
PORT ?= 8000
MCP_PATH ?= /mcp/
GEMINI_MODEL ?= gemini-2.5-flash

.PHONY: help server-stdio server-http format format-check lint lint-fix typecheck test build check

help:
	@printf '%s\n' \
		'Available targets:' \
		'  make server-stdio                 Run the MCP server over stdio' \
		'  make server-http HOST=0.0.0.0     Run the MCP server over HTTP' \
		'  make format                       Run ruff format' \
		'  make format-check                 Check ruff formatting' \
		'  make lint                         Run ruff check' \
		'  make lint-fix                     Run ruff check --fix' \
		'  make typecheck                    Run ty check' \
		'  make test                         Run pytest' \
		'  make build                        Build wheel + sdist with uv' \
		'  make check                        Run lint, typecheck, and tests'

server-stdio:
	$(UV) run fastmcp-pr-review --transport stdio --gemini-model $(GEMINI_MODEL)

server-http:
	$(UV) run fastmcp-pr-review --transport http --host $(HOST) --port $(PORT) --path $(MCP_PATH) --gemini-model $(GEMINI_MODEL)

format:
	$(UV) run ruff format src/ tests/

format-check:
	$(UV) run ruff format --check src/ tests/

lint:
	$(UV) run ruff check src/ tests/

lint-fix:
	$(UV) run ruff check src/ tests/ --fix

typecheck:
	$(UV) run ty check

test:
	$(UV) run pytest tests/ -q

build:
	$(UV) build

check: format-check lint typecheck test
