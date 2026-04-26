# Developing

## Project Structure

```
src/fastmcp_pr_review/
    models.py          Shared types: PRReviewResult, ReviewComment, ReviewInput, FileReader
    github_client.py   Async GitHub API wrapper using githubkit
    server.py          FastMCP server — tool definitions, timeline formatting, data wiring
    fast.py            FastReview class — single-shot structured review
    thorough.py        ThoroughReview class — multi-pass pipeline (filter → review → verify)
    context.py         Project context gathering (README, AGENTS.md, linked issues)

tests/
    conftest.py          Shared fixtures (PR data models)
    test_models.py       Model validation, scoring helpers
    test_github_client.py  GitHub API wrapper (mocked)
    test_server.py       Tool registration, timeline formatting, diff parser
    test_fast.py         FastReview class
    test_thorough.py     ThoroughReview pipeline stages
```

### Architecture: Pipeline Classes

The review system separates two orthogonal axes:

- **Data source** — where the diff/files/context come from → `ReviewInput` + `FileReader` (in `models.py`)
- **Review strategy** — how the review is performed → pipeline class with step methods

`ReviewInput` is a frozen dataclass bundling everything the review logic needs (files, title, description, project context, etc.). `FileReader` is a Protocol for reading full file contents — different implementations for PRs (GitHub API), local repos (filesystem), or raw diffs (no-op).

**FastReview** (`fast.py`): Single `ctx.sample()` call with structured output. Override `SYSTEM_PROMPT` or `build_prompt()` for variants.

**ThoroughReview** (`thorough.py`): Multi-pass pipeline with overridable step methods:
1. `prefilter()` — skip binary/generated files
2. `filter_files()` — LLM classifies files as skip/review
3. `review_files()` — batched review with tool calls
4. `verify_findings()` — agentic verification with file exploration
5. `aggregate()` — score and produce final result

Each mode file is self-contained: prompts inline, stage models inline. They share only `models.py` types.

### Extension Example

```python
class SecurityReview(ThoroughReview):
    SYSTEM_PROMPT = "You are a security auditor..."
    intensity = "aggressive"

    async def filter_files(self, ctx, chunks, inp):
        return chunks  # review everything
```

## Commands

```bash
uv sync                           # Install dependencies
make format                       # Ruff format
make lint                         # Ruff check
make lint-fix                     # Ruff check --fix
make typecheck                    # Type check
make test                         # Run tests
make server-stdio                 # Run stdio server
make server-http                  # Run HTTP server at http://127.0.0.1:8000/mcp/
uv run pytest tests/ -v           # Run tests
uv run ruff check src/ tests/     # Lint
uv run ruff format src/ tests/    # Format
uv run ty check                   # Type check
uv build                          # Build sdist + wheel with uv
uv run fastmcp-pr-review          # Run the server
uv run python scripts/run_review.py owner/repo 123 fast  # Dev smoke test
```

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
```

### Tool Calling

Pass `tools=[...]` and the LLM can call Python functions during sampling:

```python
result = await ctx.sample(
    messages="Review this file...",
    result_type=FileReview,
    tools=[get_file_contents, lookup_file_diff],  # LLM can call these
)
```

FastMCP runs the tool, feeds the result back to the LLM, and the LLM continues until it produces the structured output.

### Agentic Tool Loop (thorough pattern)

Instead of asking the LLM to produce a complex nested schema, thorough mode uses tool calls to accumulate results:

```python
confirmed = []

def confirm_finding(title, evidence, severity, ...):
    confirmed.append(ReviewComment(...))
    return f"Confirmed '{title}'"

await ctx.sample(
    messages="Verify these findings...",
    result_type=VerifyComplete,  # Trivial schema — real results via tool calls
    tools=[confirm_finding, dismiss_finding, get_file_contents, ...],
)
# confirmed list populated by tool calls during the loop
```

### Sampling Fallback

The server uses `GoogleGenaiSamplingHandler` as a fallback when the MCP client doesn't support sampling:

```python
mcp = FastMCP(
    sampling_handler=GoogleGenaiSamplingHandler(default_model="gemini-2.5-flash"),
    sampling_handler_behavior="fallback",  # only used when client can't sample
)
```

## Adding a New Review Mode

1. Create `src/fastmcp_pr_review/whatever.py`
2. Define a pipeline class (e.g. `WhateverReview`) — or subclass `FastReview`/`ThoroughReview`
3. Define stage models and prompts inline in the file
4. The class's `run()` method takes `(ctx, ReviewInput)` or `(ctx, ReviewInput, FileReader)` and returns `PRReviewResult`
5. Register it as a tool in `server.py` — use `_build_review_input()` to construct input from PR data
6. Add `tests/test_whatever.py`

## Dependencies

- **fastmcp** — MCP server framework
- **githubkit** — Async GitHub API client
- **google-genai** — Google Gemini SDK (for sampling fallback handler)
- **pydantic** — Data validation and JSON schema generation
