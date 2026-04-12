# Agent Instructions

## Project

FastMCP PR review server with three implementations of increasing complexity. Each `v*.py` is a self-contained example of a FastMCP sampling pattern.

## Key Files

- `src/fastmcp_pr_review/v1_simple.py` — Structured output only
- `src/fastmcp_pr_review/v2_per_file.py` — Structured output + tool calling
- `src/fastmcp_pr_review/v3_production.py` — Multi-pass pipeline with agentic verification
- `src/fastmcp_pr_review/models.py` — Shared types (PRReviewResult, ReviewComment)
- `src/fastmcp_pr_review/server.py` — MCP tool definitions
- `src/fastmcp_pr_review/github_client.py` — GitHub API wrapper

## Rules

1. **Each v\*.py must be self-contained.** Prompts inline, stage models inline. A reader should understand the complete pattern from one file.
2. **v1/v2/v3 must not import from each other.** They share only `models.py` types.
3. **Prompts must use general principles, not specific pattern exclusions.** Don't hard-code "don't flag inline imports" — instead say "don't flag intentional style choices."
4. **All functions use `async def`.** The GitHub client and sampling are async throughout.
5. **Tests mock `ctx.sample()` and `GitHubPRClient`.** No real API calls in unit tests.

## Commands

See [DEVELOPING.md](DEVELOPING.md) for build, test, and lint commands.

## Environment

- Python 3.13+, uv, ruff, ty
- `GITHUB_TOKEN` and `GEMINI_API_KEY` required at runtime
