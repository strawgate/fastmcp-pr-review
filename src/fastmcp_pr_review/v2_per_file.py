"""v2: Per-file review with sampling tools.

Demonstrates the natural next step beyond v1:
  - Loop over changed files, calling ctx.sample() for each one
  - Give the LLM tools it can call during review (tool calling)
  - Structured output per file, aggregated into a final result

New concepts beyond v1:
  - tools=[...] parameter -- the LLM can call Python functions mid-review
  - Per-file iteration -- each file gets focused attention
  - Async tool -- get_file_contents reads from the repo via GitHub API

This is what you'd build after spending a couple days iterating on v1:
"What if I gave the LLM tools to explore the codebase while reviewing?"
"""

from __future__ import annotations

from collections import Counter

from fastmcp import Context  # noqa: TC002
from pydantic import BaseModel, Field

from fastmcp_pr_review.github_client import GitHubPRClient  # noqa: TC001
from fastmcp_pr_review.models import (
    CommentCategory,
    PRReviewResult,
    ReviewComment,
    ReviewStats,
    Severity,
    compute_scores,
    compute_verdict,
)

# ---------------------------------------------------------------------------
# Output model for each file -- the LLM fills this in via structured output
# ---------------------------------------------------------------------------


class Finding(BaseModel):
    """A single issue found during review."""

    path: str = Field(description="File path relative to the repo root")
    line: int | None = Field(default=None, description="Line number where the issue starts")
    end_line: int | None = Field(default=None, description="End line for multi-line issues")
    severity: Severity = Field(description="How serious: critical, high, medium, low, nitpick")
    category: CommentCategory = Field(
        description="What kind of issue: bug, security, performance, style, etc."
    )
    title: str = Field(description="Short (<80 char) title")
    body: str = Field(description="What's wrong and what could go wrong")
    why: str = Field(description="Why this matters")
    suggested_code: str | None = Field(
        default=None, description="Replacement code if you have a concrete fix"
    )
    confidence: int = Field(ge=0, le=100, description="How sure you are (0-100)")


class FileReview(BaseModel):
    """result_type for per-file ctx.sample() -- what the LLM returns."""

    filename: str = Field(description="Path of the file being reviewed")
    findings: list[Finding] = Field(default_factory=list, description="Issues found in this file")
    summary: str = Field(description="1-2 sentence summary of this file's changes")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert code reviewer analyzing a single file from a pull request.

Review the diff and produce structured findings. For each issue:
- Explain what's wrong (body) and why it matters (why)
- Assign a severity: critical, high, medium, low, or nitpick
- Assign a category: bug, security, performance, style, logic, \
error_handling, testing, or maintainability
- Rate your confidence 0-100

You have tools to explore the codebase:
- get_file_contents(path) -- read any file in the repo
- lookup_file_diff(path) -- see the diff for another changed file
- list_pr_files() -- list all files changed in this PR

Use these tools when you need context beyond the current file's diff.
For example, check if a function you're concerned about is tested,
or if input validation happens in a caller. USE the tools before
flagging an issue -- verify your concern is real.

Only flag issues that could cause bugs, security problems, or breakage.
Do not flag style preferences or code organization choices that are
clearly intentional. If you cannot describe a concrete failure scenario
for a finding, do not include it.

Finding no issues is a valid outcome -- do not invent problems."""


# ---------------------------------------------------------------------------
# The review function
# ---------------------------------------------------------------------------


async def per_file_review(
    gh: GitHubPRClient,
    ctx: Context,
    repo: str,
    pr_number: int,
    *,
    focus_areas: str | None = None,
    min_confidence: int = 50,
) -> PRReviewResult:
    """Review a PR file-by-file, giving the LLM tools to explore the repo.

    For each changed file:
    1. Build a prompt with the file's diff + PR context
    2. Call ctx.sample() with tools the LLM can use
    3. Collect the structured findings
    4. Aggregate into a final PRReviewResult
    """
    timeline = await gh.get_timeline(repo, pr_number)
    pr = timeline.pr

    # Skip files with no patch (binary files, renames without content change)
    reviewable_files = [f for f in timeline.files if f.patch]

    # -- Define tools the LLM can call during review --
    # These are defined outside the per-file loop because they close over
    # `timeline` (shared PR state). Each ctx.sample() call gets the same
    # three tools; only the prompt changes per file.

    async def get_file_contents(filepath: str) -> str:
        """Read the current contents of any file in the repository."""
        return await gh.get_file_contents(repo, filepath, pr.head_sha)

    def lookup_file_diff(filename: str) -> str:
        """See the diff for another file changed in this PR."""
        for f in timeline.files:
            if f.filename == filename:
                return f.patch or "(no patch available)"
        return f"File '{filename}' not found in this PR"

    def list_pr_files() -> str:
        """List all files changed in this PR."""
        return "\n".join(
            f"  {f.status:>10} {f.filename} (+{f.additions} -{f.deletions})" for f in timeline.files
        )

    # -- Review each file --

    file_reviews: list[FileReview] = []

    for file in reviewable_files:
        user_prompt = (
            f"<context>\n"
            f"PR #{pr.number}: {pr.title}\n"
            f"Author: @{pr.author.login} | {pr.head_ref} -> {pr.base_ref}\n"
            f"Description: {pr.body or '(none)'}\n"
            f"</context>\n\n"
            f"<file_diff>\n"
            f"File: {file.filename} ({file.status}, "
            f"+{file.additions} -{file.deletions})\n"
            f"```diff\n{file.patch}\n```\n"
            f"</file_diff>\n\n"
            f"Review this file."
        )

        if focus_areas:
            user_prompt += f"\n\nFocus especially on: {focus_areas}"

        # ---------------------------------------------------------------
        # ctx.sample() with structured output + TOOL CALLING
        # ---------------------------------------------------------------
        # The LLM can call get_file_contents(), lookup_file_diff(), or
        # list_pr_files() at any point during its response. FastMCP
        # executes the tool, feeds the result back, and the LLM continues.
        # When done exploring, it returns a FileReview via structured output.
        result = await ctx.sample(
            messages=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            result_type=FileReview,
            tools=[get_file_contents, lookup_file_diff, list_pr_files],
            temperature=0.2,
            max_tokens=16384,
        )

        file_reviews.append(result.result)

    # -- Aggregate all file reviews into a single PRReviewResult --
    return _aggregate(file_reviews, len(timeline.files), min_confidence)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(
    file_reviews: list[FileReview],
    total_files: int,
    min_confidence: int,
) -> PRReviewResult:
    """Combine per-file reviews into a final result."""
    comments: list[ReviewComment] = []
    for fr in file_reviews:
        for f in fr.findings:
            if f.confidence < min_confidence:
                continue
            comments.append(
                ReviewComment(
                    path=f.path,
                    line=f.line,
                    end_line=f.end_line,
                    severity=f.severity,
                    category=f.category,
                    title=f.title,
                    body=f.body,
                    why=f.why,
                    suggested_code=f.suggested_code,
                    confidence=f.confidence,
                )
            )

    # Compute severity and category distributions once
    by_severity = Counter(c.severity.value for c in comments)
    by_category = Counter(c.category.value for c in comments)

    n_crit = by_severity.get("critical", 0)
    n_high = by_severity.get("high", 0)
    risk, health = compute_scores(dict(by_severity))

    summaries = [f"- **{fr.filename}**: {fr.summary}" for fr in file_reviews if fr.summary]

    return PRReviewResult(
        verdict=compute_verdict(n_crit, n_high, by_severity.get("medium", 0)),
        summary=(
            f"Reviewed {len(file_reviews)} of {total_files} files "
            f"({len(comments)} comments, "
            f"{n_crit} critical, {n_high} high).\n" + "\n".join(summaries)
        ),
        comments=comments,
        risk_score=risk,
        health_score=health,
        files_reviewed=len(file_reviews),
        files_skipped=total_files - len(file_reviews),
        stats=ReviewStats(
            total_comments=len(comments),
            by_severity=dict(by_severity),
            by_category=dict(by_category),
        ),
    )
