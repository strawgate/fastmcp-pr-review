# FastMCP PR Review

A GitHub PR review server built with [FastMCP](https://github.com/jlowin/fastmcp), with two review modes: **fast** and **thorough**.

Each mode is **self-contained** — read either file top-to-bottom to understand the complete pattern.

## Review Modes

| Tool | File | What it does |
|------|------|-------------|
| `review_pr_fast` | `fast.py` | One LLM call, structured output, no tools |
| `review_pr_thorough` | `thorough.py` | Multi-pass pipeline: filter → review → verify |

### Fast

One `ctx.sample()` call with `result_type=PRReviewResult`. Sends the full diff, gets back a structured review. Best for small PRs or quick checks.

### Thorough

Multi-pass pipeline with batched file filtering, per-file review with a verification protocol, and agentic verification where the LLM explores the repo to confirm or disprove findings. Configurable intensity (conservative/balanced/aggressive). Only confirmed findings survive.

## Quick Start

```bash
uv sync
cp .env.example .env
# edit .env with your GitHub and Gemini credentials

uv run fastmcp-pr-review
# or override runtime settings explicitly
uv run fastmcp-pr-review --gemini-model gemini-2.5-pro
make server-stdio
make server-http HOST=0.0.0.0 PORT=8000
```

The server auto-loads `.env` when started from the repo directory, and the Click launcher also reads `GITHUB_TOKEN` and `GEMINI_API_KEY` directly from the process environment. That means hosted environments like Horizon can inject those vars without extra CLI flags.

The server also exposes data tools: `get_pr_info`, `get_pr_diff`, `get_pr_files`.

## Deploy to Horizon

For Horizon-style deployments that import a module and look for an `mcp` object, point it at the repo-root `run_horizon_server.py`. That file loads env vars and exposes:

```python
mcp = create_server()
```

Set `GITHUB_TOKEN` and `GEMINI_API_KEY` in Horizon, and it can import `run_horizon_server.py` directly.

## Add to Claude

Add the server to your Claude MCP config with a stdio entry that runs this repo through `uv`:

```json
{
  "mcpServers": {
    "fastmcp-pr-review": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/fastmcp-pr-review",
        "run",
        "fastmcp-pr-review",
        "--gemini-model",
        "gemini-2.5-pro"
      ],
      "env": {}
    }
  }
}
```

Keep `GITHUB_TOKEN` and `GEMINI_API_KEY` in `/absolute/path/to/fastmcp-pr-review/.env` or inject them via the host environment so Claude does not need inline secrets in its MCP config.
Use CLI flags for non-secret runtime settings like the fallback Gemini model.

## Makefile

The repo includes a Makefile for the common local workflows:

```bash
make server-stdio
make server-http HOST=0.0.0.0 PORT=8000
make format
make lint
make lint-fix
make typecheck
make test
make build
```

## Development

See [DEVELOPING.md](DEVELOPING.md) for testing and sampling patterns, and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for architecture and design decisions.

## License

MIT
