"""v1: Simple single-shot PR review.

Demonstrates the simplest FastMCP sampling pattern:
  - One ctx.sample() call
  - Structured output via result_type (Pydantic model)
  - No tool calling

The LLM receives the full diff in one prompt and returns a structured
PRReviewResult with verdict, comments, and scores — all validated
against the Pydantic schema automatically by FastMCP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastmcp_pr_review.models import (
    PRReviewResult,
)

if TYPE_CHECKING:
    from fastmcp import Context

    from fastmcp_pr_review.github_client import GitHubPRClient

# ---------------------------------------------------------------------------
# Prompt — inlined so you can read the full example in one file
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert code reviewer. Analyze the pull request diff and context.

Produce a thorough but concise review. Focus on:
1. Security vulnerabilities (injection, auth bypass, secrets exposure)
2. Correctness bugs (logic errors, edge cases, race conditions)
3. Performance issues (N+1 queries, memory leaks, blocking I/O)
4. Error handling gaps (unhandled exceptions, missing validation)
5. Maintainability (unclear naming, dead code, missing abstractions)

For each issue found, assign:
- severity: critical, high, medium, low, or nitpick
- category: bug, security, performance, style, logic, error_handling, testing, maintainability
- confidence: 0-100 score

Be specific. Reference file paths and line numbers.
Finding no issues is a valid outcome — do not invent problems."""


# ---------------------------------------------------------------------------
# The review function — this is the whole thing
# ---------------------------------------------------------------------------


async def simple_review(
    gh: GitHubPRClient,
    ctx: Context,
    repo: str,
    pr_number: int,
    *,
    focus_areas: str | None = None,
) -> PRReviewResult:
    """Review a PR with a single LLM call. No tools, just structured output.

    This is the simplest possible FastMCP sampling pattern:
    1. Build a prompt with the full diff
    2. Call ctx.sample() with result_type=PRReviewResult
    3. FastMCP ensures the response matches the Pydantic schema
    """
    # Fetch PR data
    timeline = await gh.get_timeline(repo, pr_number)
    diff = await gh.get_diff(repo, pr_number)

    # Build the prompt — just markdown with the diff inlined
    pr = timeline.pr
    patches = "\n\n".join(
        f"### {f.filename} ({f.status})\n```diff\n{f.patch}\n```" for f in timeline.files if f.patch
    )
    diff_context = patches or (diff[:100_000] if len(diff) > 100_000 else diff)

    user_prompt = (
        f"## Pull Request: {pr.title} (#{pr.number})\n"
        f"Author: @{pr.author.login} | {pr.head_ref} -> {pr.base_ref}\n"
        f"Stats: +{pr.additions} -{pr.deletions} across {pr.changed_files} files\n\n"
        f"### Description\n{pr.body or '(no description)'}\n\n"
        f"### Diff\n{diff_context}\n\n"
        "Review this PR and provide your structured assessment."
    )

    if focus_areas:
        user_prompt += f"\n\nFocus especially on: {focus_areas}"

    # -----------------------------------------------------------------------
    # THE INTERESTING PART: one ctx.sample() call with structured output
    # -----------------------------------------------------------------------
    # result_type=PRReviewResult tells FastMCP to:
    #   1. Create a hidden "final_response" tool from the Pydantic schema
    #   2. Have the LLM call that tool with structured JSON
    #   3. Validate the response against the model
    #   4. Return it as result.result (a PRReviewResult instance)
    result = await ctx.sample(
        messages=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        result_type=PRReviewResult,
        temperature=0.2,
        max_tokens=4096,
    )

    return result.result
