"""v1: Simple single-shot PR review.

Demonstrates the simplest FastMCP sampling pattern:
  - One ctx.sample() call
  - Structured output via result_type (Pydantic model)
  - No tool calling

The LLM receives the full diff in one prompt and returns a structured
PRReviewResult with verdict, comments, and scores -- all validated
against the Pydantic schema automatically by FastMCP.
"""

import logging

from fastmcp import Context

from fastmcp_pr_review.github_client import GitHubPRClient
from fastmcp_pr_review.models import PRReviewResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt -- inlined so you can read the full example in one file
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert code reviewer. Analyze the pull request diff and context.

Focus on these categories, in priority order:
1. Security vulnerabilities (injection, XSS, auth bypass, secrets exposure)
2. Logic bugs that could cause runtime failures or incorrect behavior
3. Data integrity issues (race conditions, missing transactions)
4. Performance bottlenecks (N+1 queries, memory leaks, blocking I/O)
5. Error handling gaps (unhandled exceptions, missing validation)
6. Breaking changes to public APIs without migration path
7. Missing or incorrect test coverage for critical paths

For each issue, assign:
- severity: critical, high, medium, low, or nitpick
- category: bug, security, performance, style, logic, \
error_handling, testing, maintainability
- confidence: 0-100

Determine severity AFTER investigating the issue, not before.

Verdict rules:
- CHANGES_REQUESTED: only for critical or 2+ high-severity issues
- COMMENTED: for 1 high or 3+ medium issues
- APPROVED: everything else (low, nitpick, or no issues)

Silence is better than noise. A false positive wastes the author's time \
and erodes trust in every future review. Only report findings you could \
defend in code review -- avoid hedging like "might" or "could possibly."

Do NOT flag:
- Input sanitized upstream, by framework, or via parameterized queries
- Null/undefined guarded by type system, assertion, or schema validation
- Error handling delegated to caller, middleware, or framework
- Performance concerns where N is demonstrably small
- Missing tests for trivial getters/setters or auto-generated code
- Style/naming unless it violates the project's documented guidelines
- Any issue where you cannot describe a concrete failure scenario

Be specific. Reference file paths and line numbers.
Explain *why* each issue matters and suggest a fix when possible.
Finding no issues is a valid outcome -- do not invent problems."""


# ---------------------------------------------------------------------------
# Review function
# ---------------------------------------------------------------------------


async def simple_review(
    gh: GitHubPRClient,
    ctx: Context,
    repo: str,
    pr_number: int,
    *,
    focus_areas: str | None = None,
    project_context: str = "",
    linked_issues: list[str] | None = None,
) -> PRReviewResult:
    """Review a PR with a single LLM call. No tools, just structured output.

    This is the simplest possible FastMCP sampling pattern:
    1. Build a prompt with the full diff
    2. Call ctx.sample() with result_type=PRReviewResult
    3. FastMCP ensures the response matches the Pydantic schema

    Args:
        project_context: Pre-fetched project docs (README, AGENTS.md, etc.)
        linked_issues: Pre-fetched linked issue summaries
    """
    logger.info("v1: reviewing %s#%d", repo, pr_number)

    timeline = await gh.get_timeline(repo, pr_number)
    pr = timeline.pr

    # Build the prompt
    patches = "\n\n".join(
        f"### {f.filename} ({f.status})\n```diff\n{f.patch}\n```" for f in timeline.files if f.patch
    )
    diff_context = patches or "(no patches available)"

    # Inject project context and linked issues into the prompt
    context_section = ""
    if project_context:
        context_section = f"\n### Project Context\n{project_context}\n"
    if linked_issues:
        issues_text = "\n\n".join(linked_issues)
        context_section += f"\n### Linked Issues\n{issues_text}\n"

    user_prompt = (
        f"## Pull Request: {pr.title} (#{pr.number})\n"
        f"Author: @{pr.author.login} | {pr.head_ref} -> {pr.base_ref}\n"
        f"Stats: +{pr.additions} -{pr.deletions} "
        f"across {pr.changed_files} files\n\n"
        f"### Description\n{pr.body or '(no description)'}\n"
        f"{context_section}\n"
        f"### Diff\n{diff_context}\n\n"
        "Review this PR and provide your structured assessment."
    )

    if focus_areas:
        user_prompt += f"\n\nFocus especially on: {focus_areas}"

    n_files = len([f for f in timeline.files if f.patch])
    logger.info(
        "v1: sampling — %d files, %d chars prompt",
        n_files,
        len(user_prompt),
    )

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
        max_tokens=16384,
    )

    review = result.result
    logger.info(
        "v1: done — %s, %d comments",
        review.verdict,
        len(review.comments),
    )
    return review
