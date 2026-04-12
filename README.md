# FastMCP PR Review

A GitHub PR review server built with [FastMCP](https://github.com/jlowin/fastmcp), demonstrating three levels of complexity for structured sampling with LLMs.

Each implementation is **self-contained** — you can read any single file top-to-bottom and understand the complete pattern without opening anything else.

## The Three Implementations

### v1: Simple (`v1_simple.py` — 111 lines)

**One `ctx.sample()` call. Structured output. No tools.**

The "built it in 15 minutes" version. Sends the full PR diff to the LLM and gets back a structured `PRReviewResult` validated against a Pydantic schema.

```
PRFile[] → ctx.sample(result_type=PRReviewResult) → done
```

**Demonstrates:** `result_type` parameter, how FastMCP creates a hidden `final_response` tool from the Pydantic schema, automatic validation and retry.

### v2: Per-File (`v2_per_file.py` — 251 lines)

**Loop over files. `ctx.sample()` per file with tool calling.**

The "spent a couple days on it" version. Reviews each file individually and gives the LLM tools to explore the codebase — read other files, look up diffs, list changed files.

```
for file in files:
    ctx.sample(result_type=FileReview, tools=[get_file_contents, lookup_file_diff, list_pr_files])
```

**Demonstrates:** `tools=[...]` parameter, the LLM calling Python functions mid-sampling, async tools, per-file structured output aggregated into a final result.

### v3: Production (`v3_production.py` — 783 lines)

**Multi-pass pipeline. Batched filtering. Agentic verification.**

The "production system" version with four passes:

```
Pass 1 — Context:  Gather PR metadata, prior reviews, existing threads
Pass 2 — Filter:   Batch-classify files by interest (structured output)
Pass 3 — Review:   Per-file deep review with tools + verification protocol
Pass 4 — Verify:   Agentic exploration — LLM calls confirm_finding()/dismiss_finding()
```

**Demonstrates:** Batched sampling (10 files per call), configurable intensity (conservative/balanced/aggressive), prior review awareness, existing thread dedup, a verification protocol (4-point self-check), tool-based result collection via closures, bounded concurrency.

## Quick Start

```bash
# Install
uv sync

# Set credentials
export GITHUB_TOKEN="ghp_..."
export GEMINI_API_KEY="AIza..."

# Run the server
uv run fastmcp-pr-review
```

The server exposes six MCP tools:

| Tool | Description |
|------|-------------|
| `get_pr_info` | PR timeline (body, comments, reviews, commits) |
| `get_pr_diff` | Raw unified diff |
| `get_pr_files` | Changed files with per-file patches |
| `review_pr_simple` | v1 — single-shot review |
| `review_pr` | v2 — per-file review with tools |
| `review_pr_deep` | v3 — production multi-pass pipeline |

## Architecture

```
src/fastmcp_pr_review/
    models.py          Shared Pydantic models (PRReviewResult, ReviewComment, etc.)
    github_client.py   Async GitHub API wrapper using githubkit
    server.py          FastMCP server with tool definitions
    v1_simple.py       Single-shot review (structured output only)
    v2_per_file.py     Per-file review (structured output + tools)
    v3_production.py   Production pipeline (filter + review + verify)
```

Each `v*.py` file defines its own stage-specific models inline so the data flow is visible in one place. The shared `models.py` contains only types used by all three: `PRReviewResult`, `ReviewComment`, `Severity`, `CommentCategory`, and scoring helpers.

## How Sampling Works

FastMCP's `ctx.sample()` sends a request to the connected LLM (or a fallback handler like Gemini) and returns a structured response:

```python
# Structured output — the LLM must match the Pydantic schema
result = await ctx.sample(
    messages="Review this code...",
    system_prompt="You are an expert reviewer...",
    result_type=PRReviewResult,  # Pydantic model → JSON schema → validated response
    temperature=0.2,
    max_tokens=16384,
)
review = result.result  # A PRReviewResult instance

# Tool calling — the LLM can call Python functions during sampling
result = await ctx.sample(
    messages="Review this file...",
    result_type=FileReview,
    tools=[get_file_contents, lookup_file_diff],  # LLM can call these
)

# Agentic tool loop — the LLM calls tools repeatedly, accumulating state
confirmed = []
def confirm_finding(title, evidence, severity, ...):
    confirmed.append(ReviewComment(...))
    return f"Confirmed '{title}'"

await ctx.sample(
    messages="Verify these findings...",
    result_type=VerifyComplete,  # Trivial schema — real results via tool calls
    tools=[confirm_finding, dismiss_finding, get_file_contents, ...],
)
# confirmed list is populated by tool calls during the loop
```

## Sampling Fallback

The server uses `GoogleGenaiSamplingHandler` with Gemini 2.5 Flash as a fallback when the MCP client doesn't support sampling natively:

```python
mcp = FastMCP(
    sampling_handler=GoogleGenaiSamplingHandler(default_model="gemini-2.5-flash"),
    sampling_handler_behavior="fallback",
)
```

## Development

```bash
# Install with dev dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Lint + format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check
uv run ty check
```

## License

MIT
