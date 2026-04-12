# Developing

## Project Structure

```
src/fastmcp_pr_review/
    models.py          Shared Pydantic models (PRReviewResult, ReviewComment, etc.)
    github_client.py   Async GitHub API wrapper using githubkit
    server.py          FastMCP server — tool definitions, timeline formatting
    v1_simple.py       Single-shot review (structured output only)
    v2_per_file.py     Per-file review (structured output + tools)
    v3_production.py   Production pipeline (filter + review + verify)

tests/
    conftest.py          Shared fixtures (PR data models)
    test_models.py       Model validation, scoring helpers
    test_github_client.py  GitHub API wrapper (mocked)
    test_server.py       Tool registration, timeline formatting
    test_v1_simple.py    v1 review function
    test_v2_per_file.py  v2 per-file loop + aggregation
    test_v3_production.py  v3 pipeline stages
```

Each `v*.py` file defines its own stage-specific Pydantic models **inline** so the data flow is visible in one file. The shared `models.py` contains only types used by all three implementations: `PRReviewResult`, `ReviewComment`, `Severity`, `CommentCategory`, and scoring helpers (`compute_verdict`, `compute_scores`).

## Commands

```bash
uv sync                           # Install dependencies
uv run pytest tests/ -v           # Run tests
uv run ruff check src/ tests/     # Lint
uv run ruff format src/ tests/    # Format
uv run ty check                   # Type check
uv run fastmcp-pr-review          # Run the server
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

### Agentic Tool Loop (v3 pattern)

Instead of asking the LLM to produce a complex nested schema, v3 uses tool calls to accumulate results:

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

## Adding a New Review Implementation

1. Create `src/fastmcp_pr_review/v4_whatever.py`
2. Define stage models inline at the top of the file
3. Write the prompts inline (don't import from a shared prompts module)
4. Export a top-level async function: `async def whatever_review(gh, ctx, repo, pr_number, ...) -> PRReviewResult`
5. Register it as a tool in `server.py`
6. Add `tests/test_v4_whatever.py`

## Dependencies

- **fastmcp** — MCP server framework (currently pinned to a fork with sampling handler fixes)
- **githubkit** — Async GitHub API client
- **google-genai** — Google Gemini SDK (for sampling fallback handler)
- **pydantic** — Data validation and JSON schema generation
