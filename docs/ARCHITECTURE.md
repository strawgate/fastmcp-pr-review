# Architecture

## Overview

A GitHub PR review server with two review pipelines — **fast** (single LLM call) and **thorough** (multi-pass with verification). Both pipelines receive a [`ReviewInput`] and produce a [`PRReviewResult`][]. They never call the GitHub API directly.

## Data Flow

```
GitHub API (githubkit)
    │
    ▼
GitHubPRClient.get_timeline()   ← one call, fetches details + files + commits + comments + reviews
    │
    ▼
ReviewInput + FileReader         ← frozen dataclass bundle, source-agnostic
    │
    ▼
FastReview.run()  ──────────────►  ctx.sample(structured output)
    │                              ─────────────────────────────
ThoroughReview.run()
    │
    ├──► prefilter()             ← sync, pattern-based skip
    ├──► filter_files()          ← LLM classifies files skip/review (structured output)
    ├──► review_files()          ← batched, tool-based finding collection
    │                              ctx.sample(tools=[add_finding, get_file_contents, ...])
    ├──► verify_findings()        ← agentic, confirms/disproves findings
    │                              ctx.sample(tools=[confirm_finding, dismiss_finding, ...])
    └──► aggregate()              ← score + verdict
    │
    ▼
PRReviewResult
```

## Key Abstractions

### ReviewInput

A [`frozen dataclass`][] bundling everything a pipeline needs: files, title, description, author, PR metadata, project context, linked issues, focus areas, and — for thorough mode — existing review threads, prior reviews, and commits.

```python
@dataclass(frozen=True)
class ReviewInput:
    files: list[PRFile]
    title: str
    description: str
    author: str
    # ... more fields
    existing_threads: dict[str, list[PRReviewComment]]  # thorough only
    prior_reviews: list[str]                             # thorough only
    commits: list[PRCommit]                              # thorough only
```

Pipelines receive `ReviewInput` and never call the GitHub API. This means the same pipeline works for PR reviews, raw diffs, or local worktree diffs — only the `ReviewInput` construction differs.

### FileReader

A [`Protocol`][] (structural subtype) for reading full file contents. Different implementations for different sources:

```python
class FileReader(Protocol):
    async def __call__(self, filepath: str) -> str: ...
```

- **PR reviews**: `gh.get_file_contents(repo, path, head_sha)` — fetches file at PR head
- **Local diffs**: reads from working tree
- **Raw diffs**: reads from default branch

The [`_make_pr_file_reader()`][] function in `server.py` creates a closure that wraps `gh.get_file_contents()` with the repo and SHA already bound.

### Tool-Based Finding Collection

Thorough mode avoids a deeply nested Pydantic schema for review results. Instead:

1. **Review pass** uses `add_finding()` as a tool — each call appends to a closure-scoped list
2. **Verify pass** uses `confirm_finding()` / `dismiss_finding()` — each call updates the confirmed list

The final `VerifyComplete` schema is trivial; the real data flows through tool calls. This pattern keeps each LLM call's output schema simple while enabling complex multi-stage reasoning.

## FastReview vs ThoroughReview

| | FastReview | ThoroughReview |
|---|---|---|
| **LLM calls** | 1 | 4+ (filter, review batches, verify, aggregate) |
| **Tools** | None | Yes (file reader, finding accumulation) |
| **Context** | Diff + project context + commits | All of fast + prior reviews + existing threads |
| **Output** | Direct structured output | Tool-accumulated, then verified |
| **Best for** | Small PRs, quick checks | Large PRs, nuanced reviews |
| **Verdict** | LLM decides directly | Algorithm from verified findings |

### Why FastReview doesn't use prior context

`FastReview.build_prompt()` intentionally omits `existing_threads` and `prior_reviews` from `ReviewInput`. This is by design:

- Fast mode is meant for quick, lightweight checks where one LLM call covers everything
- Prior review awareness requires a separate LLM pass (as in thorough mode) to process and reason over existing comments
- Including raw prior review text in a single-shot prompt would bloat the context and dilute focus

Thorough mode handles this correctly: it reads existing threads, acknowledges prior reviews in its instructions, and explicitly tells the LLM not to repeat specific style nitpicks already mentioned.

## Design Decisions

### Why not GraphQL for timeline?

GraphQL's per-item cost model would blow through rate limits on PRs with many commits or comments. The REST approach with `asyncio.gather()` fetches everything concurrently in one round trip, then assembles the timeline in-memory.

### Why tool-based accumulation instead of nested schemas?

A single-pass LLM call producing a complex nested `list[list[ReviewComment]]` (one list per file) is unreliable — LLMs struggle with deep structural constraints in structured output. Tool-based accumulation sidesteps this: each tool call appends to a list, and the LLM focuses on one finding at a time.

### Why a separate verify pass?

The review pass produces *potential* findings with a `verification_needs` field. The verify pass is agentic — the LLM explores the repo with `get_file_contents` to confirm or disprove each finding before it becomes a final comment. This significantly reduces false positives.

### Why Semaphore in review batches?

`asyncio.Semaphore(self.concurrency)` bounds how many review batches run in parallel. Without it, a PR with 20 batchable files would trigger 20 concurrent LLM calls, risking rate limits and noisy logs. The default concurrency of 3 is conservative; adjust via constructor.

### Why GitHubKit auto-retry is sufficient

GitHubKit's [`RETRY_DEFAULT`][] policy (imported from `githubkit.retry`) handles:
- **Rate limit (429)**: 1 retry, respects `Retry-After` header
- **Server errors (5xx)**: 3 retries with exponential backoff (`2s`, `4s`, `8s`)

No additional retry logic is needed. The `GitHub` client is initialized with `auto_retry=True` (default).

## File Map

```
src/fastmcp_pr_review/
├── models.py          # Shared types: ReviewInput, FileReader, PRReviewResult, scoring helpers
├── github_client.py   # GitHubREST API wrapper (githubkit)
├── server.py          # FastMCP tool registration + ReviewInput construction + CLI
├── fast.py            # FastReview: single-shot, structured output
├── thorough.py        # ThoroughReview: multi-pass pipeline + stage models
└── context.py         # Project context (README, AGENTS.md) + linked issue extraction

tests/
├── conftest.py        # Fixtures for all model types
├── test_fast.py       # FastReview.build_prompt() and run()
├── test_thorough.py   # ThoroughReview stage methods + aggregation
├── test_server.py     # Tool registration, timeline formatting, ReviewInput construction
└── test_github_client.py  # GitHubREST API wrapper (mocked)
```

## Extending the Pipeline

### New Review Mode (e.g., SecurityReview)

```python
# src/fastmcp_pr_review/security.py
class SecurityReview(ThoroughReview):
    """Thorough review focused on security concerns."""

    SYSTEM_PROMPT = "You are a security auditor..."
    intensity = "aggressive"  # Lower thresholds → more findings

    async def filter_files(self, ctx, chunks, inp):
        return chunks  # Review everything; don't skip any files
```

### New Data Source

Implement a new `FileReader` and construct `ReviewInput` from the source:

```python
# Local worktree diff
reader = lambda path: Path(path).read_text()
inp = ReviewInput(files=parse_diff(local_diff), title="Local change", ...)

# Raw unified diff (no PR context)
reader = lambda path: ""  # Not available
inp = ReviewInput(files=parse_diff(raw_diff), ...)
```

[`ReviewInput`]: ../src/fastmcp_pr_review/models.py
[`PRReviewResult`]: ../src/fastmcp_pr_review/models.py
[`frozen dataclass`]: https://docs.python.org/3/library/dataclasses.html#frozen-instances
[`Protocol`]: https://docs.python.org/3/library/typing.html#typing.Protocol
[`_make_pr_file_reader()`]: ../src/fastmcp_pr_review/server.py
[`RETRY_DEFAULT`]: ../.venv/lib/python3.13/site-packages/githubkit/retry.py
