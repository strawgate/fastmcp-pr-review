# Agent Instructions

## Project

FastMCP PR review server with two modes. Each mode file is a self-contained FastMCP sampling pattern.

## Key Files

- `src/fastmcp_pr_review/fast.py` — Structured output only
- `src/fastmcp_pr_review/thorough.py` — Multi-pass pipeline with agentic verification
- `src/fastmcp_pr_review/models.py` — Shared types (PRReviewResult, ReviewComment)
- `src/fastmcp_pr_review/server.py` — MCP tool definitions
- `src/fastmcp_pr_review/github_client.py` — GitHub API wrapper

## Rules

1. **Each mode file must be self-contained.** Prompts inline, stage models inline. A reader should understand the complete pattern from one file.
2. **`fast.py` and `thorough.py` must not import from each other.** They share only `models.py` types.
3. **Prompts must use general principles, not specific pattern exclusions.** Don't hard-code "don't flag inline imports" — instead say "don't flag intentional style choices."
4. **All functions use `async def`.** The GitHub client and sampling are async throughout.
5. **Tests mock `ctx.sample()` and `GitHubPRClient`.** No real API calls in unit tests.

## Commands

See [DEVELOPING.md](DEVELOPING.md) for build, test, and lint commands.

## Environment

- Python 3.13+, uv, ruff, ty
- `GITHUB_TOKEN` and `GEMINI_API_KEY` required at runtime
