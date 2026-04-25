# Agent Instructions

## Project

FastMCP PR review server with pipeline classes. Each mode file is a self-contained FastMCP sampling pattern implemented as a class with overridable step methods.

## Key Files

- `src/fastmcp_pr_review/fast.py` — `FastReview` class: single-shot structured review
- `src/fastmcp_pr_review/thorough.py` — `ThoroughReview` class: multi-pass pipeline (filter → review → verify)
- `src/fastmcp_pr_review/models.py` — Shared types: `PRReviewResult`, `ReviewComment`, `ReviewInput`, `FileReader`
- `src/fastmcp_pr_review/server.py` — MCP tool definitions, data wiring (`ReviewInput` construction)
- `src/fastmcp_pr_review/context.py` — Project context gathering (README, AGENTS.md, linked issues)
- `src/fastmcp_pr_review/github_client.py` — GitHub API wrapper

## Architecture

Data source and review strategy are decoupled:
- **`ReviewInput`** (frozen dataclass) bundles diff, metadata, and context — pipeline classes never fetch data themselves
- **`FileReader`** (Protocol) provides full file contents — different implementations for PRs (GitHub API), local repos, or raw diffs
- **`server.py`** constructs `ReviewInput` + `FileReader` from the data source and passes them to the pipeline

Extend by subclassing: override prompt class attributes and/or step methods (e.g. `SecurityReview(ThoroughReview)`).

## Rules

1. **Each mode file must be self-contained.** Prompts inline, stage models inline. A reader should understand the complete pattern from one file.
2. **`fast.py` and `thorough.py` must not import from each other.** They share only `models.py` types.
3. **Prompts must use general principles, not specific pattern exclusions.** Don't hard-code "don't flag inline imports" — instead say "don't flag intentional style choices."
4. **Pipeline entry points and MCP tools use `async def`.** Internal helpers may be sync where appropriate.
5. **Tests mock `ctx.sample()` and `GitHubPRClient`.** No real API calls in unit tests.

## Commands

See [DEVELOPING.md](DEVELOPING.md) for build, test, and lint commands.

## Environment

- Python 3.13+, uv, ruff, ty
- `GITHUB_TOKEN` and `GEMINI_API_KEY` required at runtime
