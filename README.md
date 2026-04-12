# FastMCP PR Review

A GitHub PR review server built with [FastMCP](https://github.com/jlowin/fastmcp), demonstrating three levels of structured sampling with LLMs.

Each implementation is **self-contained** — read any single file top-to-bottom to understand the complete pattern.

## The Three Implementations

| Tool | File | What it does |
|------|------|-------------|
| `review_pr_simple` | `v1_simple.py` | One LLM call, structured output, no tools |
| `review_pr` | `v2_per_file.py` | Per-file loop with tool calling |
| `review_pr_deep` | `v3_production.py` | Multi-pass pipeline: filter → review → verify |

### v1: Simple (111 lines)

One `ctx.sample()` call with `result_type=PRReviewResult`. Sends the full diff, gets back a structured review. Best for small PRs or quick checks.

### v2: Per-File (251 lines)

Loops over each changed file, calling `ctx.sample()` with tools the LLM can use to explore the codebase (read files, look up diffs, list changed files). Good balance of depth and speed.

### v3: Production (783 lines)

Four-pass pipeline with batched file filtering, per-file review with a verification protocol, and agentic verification where the LLM explores the repo to confirm or disprove findings. Configurable intensity (conservative/balanced/aggressive). Only confirmed findings survive.

## Quick Start

```bash
uv sync

export GITHUB_TOKEN="ghp_..."
export GEMINI_API_KEY="AIza..."

uv run fastmcp-pr-review
```

The server also exposes data tools: `get_pr_info`, `get_pr_diff`, `get_pr_files`.

## Development

See [DEVELOPING.md](DEVELOPING.md) for architecture, testing, and how the sampling patterns work.

## License

MIT
