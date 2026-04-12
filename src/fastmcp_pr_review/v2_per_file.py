"""v2: Batched file review with sampling tools.

Demonstrates the natural next step beyond v1:
  - Group files into batches by size, calling ctx.sample() per batch
  - Give the LLM tools it can call during review (tool calling)
  - Structured output per batch, aggregated into a final result

New concepts beyond v1:
  - tools=[...] parameter -- the LLM can call Python functions mid-review
  - Size-based batching -- small files reviewed together, large files alone
  - Async tool -- get_file_contents reads from the repo via GitHub API

This is what you'd build after spending a couple days iterating on v1:
"What if I gave the LLM tools to explore the codebase while reviewing?"
"""

from __future__ import annotations

import logging
from collections import Counter

from fastmcp import Context  # noqa: TC002
from pydantic import BaseModel, Field

from fastmcp_pr_review.github_client import GitHubPRClient  # noqa: TC001
from fastmcp_pr_review.models import (
    CommentCategory,
    PRFile,
    PRReviewResult,
    ReviewComment,
    ReviewStats,
    Severity,
    compute_scores,
    compute_verdict,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output models -- the LLM fills these in via structured output
# ---------------------------------------------------------------------------


class Finding(BaseModel):
    """A single issue found during review."""

    path: str = Field(description="File path")
    line: int | None = Field(default=None, description="Line number where the issue starts")
    end_line: int | None = Field(default=None, description="End line for multi-line issues")
    severity: Severity = Field(description="How serious: critical, high, medium, low, nitpick")
    category: CommentCategory = Field(
        description="What kind of issue: bug, security, performance, etc."
    )
    title: str = Field(description="Short (<80 char) title")
    body: str = Field(description="What's wrong and what could go wrong")
    why: str = Field(description="Why this matters")
    suggested_code: str | None = Field(
        default=None, description="Replacement code if you have a fix"
    )
    confidence: int = Field(ge=0, le=100, description="How sure you are")


class FileReview(BaseModel):
    """Review results for a single file within a batch."""

    filename: str = Field(description="Path of the file being reviewed")
    findings: list[Finding] = Field(default_factory=list, description="Issues found in this file")
    summary: str = Field(description="1-2 sentence summary of this file's changes")


class BatchReview(BaseModel):
    """result_type for batched ctx.sample() -- reviews for multiple files."""

    files: list[FileReview] = Field(description="One review per file in the batch")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert code reviewer analyzing files from a pull request.

Review each file's diff and produce structured findings. For each issue:
- Assign severity AFTER investigation: critical, high, medium, low, nitpick
- Assign a category: bug, security, performance, style, logic, \
error_handling, testing, or maintainability
- Rate your confidence 0-100

You have tools to explore the codebase:
- get_file_contents(path) -- read any file in the repo
- lookup_file_diff(path) -- see the diff for another changed file
- list_pr_files() -- list all files changed in this PR

USE the tools before flagging an issue. Verify your concern is real:
1. What specific code pattern triggers this concern?
2. Is it handled elsewhere? (Read the caller, check for middleware)
3. Can you construct a concrete failure scenario? If not, STOP.
4. Would a senior engineer agree this is a real issue? If unsure, STOP.

Silence is better than noise. A false positive wastes the author's time \
and erodes trust. Only report findings defensible in code review -- \
avoid hedging like "might" or "could possibly."

Do NOT flag:
- Input sanitized upstream, by framework, or via parameterized queries
- Null/undefined guarded by type system, assertion, or schema validation
- Error handling delegated to caller, middleware, or framework
- Performance concerns where N is demonstrably small
- Missing tests for trivial getters/setters or auto-generated code
- Style/naming unless it violates documented project guidelines

Produce one FileReview per file. Finding no issues is valid."""


# ---------------------------------------------------------------------------
# Batching -- group files by total patch size
# ---------------------------------------------------------------------------

MAX_BATCH_FILES = 10
MAX_BATCH_BYTES = 10_000


def _make_batches(files: list[PRFile]) -> list[list[PRFile]]:
    """Group files into batches by size.

    Small files get reviewed together. Large files get their own batch.
    Each batch stays under MAX_BATCH_FILES files and MAX_BATCH_BYTES of
    total patch content.
    """
    batches: list[list[PRFile]] = []
    current_batch: list[PRFile] = []
    current_size = 0

    for f in files:
        patch_size = len(f.patch or "")

        # If this single file exceeds the limit, give it its own batch
        if patch_size > MAX_BATCH_BYTES:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_size = 0
            batches.append([f])
            continue

        # Would adding this file exceed the batch limits?
        if current_size + patch_size > MAX_BATCH_BYTES or len(current_batch) >= MAX_BATCH_FILES:
            batches.append(current_batch)
            current_batch = []
            current_size = 0

        current_batch.append(f)
        current_size += patch_size

    if current_batch:
        batches.append(current_batch)

    return batches


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
    """Review a PR in batches, giving the LLM tools to explore the repo.

    Files are grouped into batches by size (max 10 files or 10KB of
    patch content per batch). Each batch gets one ctx.sample() call
    with tools the LLM can use for cross-file context.
    """
    logger.info("v2: reviewing %s#%d", repo, pr_number)

    timeline = await gh.get_timeline(repo, pr_number)
    pr = timeline.pr

    # Skip files with no patch (binary files, renames without content)
    reviewable = [f for f in timeline.files if f.patch]
    batches = _make_batches(reviewable)
    logger.info(
        "v2: %d files → %d batches (skipped %d binary)",
        len(reviewable),
        len(batches),
        len(timeline.files) - len(reviewable),
    )

    # -- Tools the LLM can call during review --
    # Defined once, shared across all batch calls.

    async def get_file_contents(filepath: str) -> str:
        """Read the current contents of any file in the repository."""
        logger.debug("v2: tool call get_file_contents(%s)", filepath)
        return await gh.get_file_contents(repo, filepath, pr.head_sha)

    def lookup_file_diff(filename: str) -> str:
        """See the diff for another file changed in this PR."""
        logger.debug("v2: tool call lookup_file_diff(%s)", filename)
        for f in timeline.files:
            if f.filename == filename:
                return f.patch or "(no patch available)"
        return f"File '{filename}' not found in this PR"

    def list_pr_files() -> str:
        """List all files changed in this PR."""
        logger.debug("v2: tool call list_pr_files()")
        return "\n".join(
            f"  {f.status:>10} {f.filename} (+{f.additions} -{f.deletions})" for f in timeline.files
        )

    # -- Review each batch --

    all_file_reviews: list[FileReview] = []

    for batch_idx, batch in enumerate(batches):
        batch_files = [f.filename for f in batch]
        logger.info(
            "v2: batch %d/%d — %d files: %s",
            batch_idx + 1,
            len(batches),
            len(batch),
            batch_files,
        )
        # Build one prompt with all files in the batch
        file_sections = []
        for file in batch:
            file_sections.append(
                f"<file_diff>\n"
                f"File: {file.filename} ({file.status}, "
                f"+{file.additions} -{file.deletions})\n"
                f"```diff\n{file.patch}\n```\n"
                f"</file_diff>"
            )

        user_prompt = (
            f"<context>\n"
            f"PR #{pr.number}: {pr.title}\n"
            f"Author: @{pr.author.login} | "
            f"{pr.head_ref} -> {pr.base_ref}\n"
            f"Description: {pr.body or '(none)'}\n"
            f"</context>\n\n" + "\n\n".join(file_sections) + "\n\nReview each file above."
        )

        if focus_areas:
            user_prompt += f"\n\nFocus especially on: {focus_areas}"

        # ---------------------------------------------------------------
        # ctx.sample() with structured output + TOOL CALLING
        # ---------------------------------------------------------------
        # The LLM reviews all files in the batch at once. It can call
        # tools at any point to explore the codebase. When done, it
        # returns a BatchReview with one FileReview per file.
        result = await ctx.sample(
            messages=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            result_type=BatchReview,
            tools=[get_file_contents, lookup_file_diff, list_pr_files],
            temperature=0.2,
            max_tokens=16384,
        )

        batch_findings = sum(len(fr.findings) for fr in result.result.files)
        logger.info("v2: batch %d done — %d findings", batch_idx + 1, batch_findings)
        all_file_reviews.extend(result.result.files)

    # -- Aggregate all reviews into a single PRReviewResult --
    review = _aggregate(all_file_reviews, len(timeline.files), min_confidence)
    logger.info("v2: done — %s, %d comments", review.verdict, len(review.comments))
    return review


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
