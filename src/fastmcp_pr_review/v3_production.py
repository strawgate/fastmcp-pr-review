"""v3: Production PR review pipeline.

A full multi-pass review system demonstrating advanced FastMCP patterns:

  Pass 1 — Context: Gather PR metadata, prior reviews, existing threads
  Pass 2 — Filter: Batch-classify files by interest (structured output)
  Pass 3 — Review: Per-file deep review with tools + verification protocol
  Pass 4 — Verify: Agentic exploration to confirm/disprove findings

Production features beyond v2:
  - Prior review awareness (don't repeat what's already been said)
  - Existing thread dedup (don't re-flag resolved issues)
  - Batched filtering (50 files classified in 5 parallel sample() calls)
  - Verification protocol (4-point self-check before each finding)
  - Agentic verification (LLM explores repo to confirm/disprove)
  - Confidence gating (only high-confidence findings survive)
  - Configurable intensity (conservative → balanced → aggressive)
  - Bounded concurrency for parallel stages
"""

from __future__ import annotations

import asyncio
import fnmatch
from collections import Counter
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from fastmcp_pr_review.models import (
    CommentCategory,
    PRReviewResult,
    ReviewComment,
    ReviewStats,
    Severity,
    compute_scores,
    compute_verdict,
)

if TYPE_CHECKING:
    from fastmcp import Context

    from fastmcp_pr_review.github_client import GitHubPRClient
    from fastmcp_pr_review.models import PRFile, PRReviewComment, PRTimeline


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

SKIP_PATTERNS = [
    "*.lock", "*.min.js", "*.min.css", "*.map", "*.generated.*",
    "vendor/*", "node_modules/*", "dist/*", "build/*",
    "__pycache__/*", "*.pyc", "*.png", "*.jpg", "*.svg", "*.ico",
]  # fmt: skip

INTENSITY_GUIDELINES = {
    "conservative": (
        "Only flag issues you're highly confident about (confidence >= 80). "
        "If you can construct a counterargument, do not comment. "
        "Approval with zero comments is the expected outcome."
    ),
    "balanced": (
        "Flag issues at medium confidence or higher. "
        "Lean toward not commenting when ambiguous. "
        "A clean file with no comments is a valid outcome."
    ),
    "aggressive": (
        "Flag everything you notice, including style and improvements. "
        "Still require evidence — no speculative concerns."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Stage models — each stage's output feeds into the next
# ═══════════════════════════════════════════════════════════════════════════

# -- Pass 2: Filter --


class DiffChunk(BaseModel):
    """A file's diff, ready for triage."""

    index: int = Field(description="Stable index for batch correlation")
    filename: str
    status: str
    additions: int
    deletions: int
    patch: str


class FilteredChunk(BaseModel):
    """Triage result for one file."""

    index: int = Field(description="Matches DiffChunk.index")
    interest: str = Field(description="high, medium, low, or skip")
    rationale: str
    focus_hints: list[str] = Field(
        default_factory=list,
        description="Specific review focus areas",
    )


class FilterBatchResult(BaseModel):
    """result_type for filter batch — structured output, no tools."""

    chunks: list[FilteredChunk]


# -- Pass 3: Review --


class PotentialFinding(BaseModel):
    """A suspected issue, pre-verification."""

    path: str
    line: int | None = Field(default=None)
    end_line: int | None = Field(default=None)
    severity: Severity
    category: CommentCategory
    title: str = Field(description="Short title (<80 chars)")
    body: str = Field(description="Detailed explanation")
    why: str = Field(description="Why this matters / failure scenario")
    suggested_code: str | None = Field(default=None)
    confidence: int = Field(ge=0, le=100)
    verification_needs: str = Field(
        description=(
            "What to check to confirm or disprove this finding "
            "(e.g. 'check if input is sanitized in the caller')"
        )
    )


class FileFindings(BaseModel):
    """result_type for per-file review — structured output + tools."""

    filename: str
    findings: list[PotentialFinding] = Field(default_factory=list)
    file_summary: str = Field(description="1-2 sentence summary")


# -- Pass 4: Verify --


class VerifiedFinding(BaseModel):
    """A finding after agentic verification."""

    original_title: str
    status: str = Field(description="confirmed, disproved, or inconclusive")
    evidence: str = Field(description="Concrete evidence: file paths, line numbers, code")
    comment: ReviewComment | None = Field(
        default=None,
        description="Final comment — only when confirmed",
    )


class ExploreResult(BaseModel):
    """result_type for verification — structured output + multiple tools."""

    filename: str
    verified: list[VerifiedFinding] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════

FILTER_SYSTEM_PROMPT = """\
<task>
You are a senior engineer triaging pull request diffs. For each file,
classify how interesting it is for code review. Be a quality gate.
</task>

<classification>
- high: Logic changes, new features, security-relevant, API changes.
- medium: Configs with consequences, tests that could mask bugs, refactors.
- low: Renaming, formatting, comment updates, dep bumps.
- skip: Generated code, pure dead-code deletion, no review value.
</classification>

<focus_hints>
For high/medium files, provide 1-3 *specific* focus hints like:
- "Check that the new retry logic handles timeout correctly"
- "Verify SQL parameterization in the new query builder"
Do NOT write generic hints like "review carefully".</focus_hints>"""


def _review_system_prompt(intensity: str) -> str:
    guideline = INTENSITY_GUIDELINES.get(intensity, INTENSITY_GUIDELINES["balanced"])
    return f"""\
<task>
You are an expert code reviewer. Analyze this file's diff and produce
structured findings. Be concise, specific, and actionable.
Finding no issues is a valid and valuable outcome.
</task>

<signal_noise>
{guideline}
</signal_noise>

<constraints>
- ONLY comment on code in the diff (added or modified lines)
- Do NOT re-flag issues that have existing review threads
- Every finding MUST include a concrete failure scenario
- Describe what verification would confirm or disprove each finding
</constraints>

<severity_classification>
Assign severity AFTER investigation, not before:
- critical: Must fix. Security vulns, data corruption, prod-breaking.
- high: Should fix. Logic errors, missing validation, perf regression.
- medium: Non-blocking. Error handling gaps, suboptimal patterns.
- low: Author discretion. Minor improvements, style.
- nitpick: Optional. Cosmetic preferences.
</severity_classification>

<false_positives>
Do NOT flag:
- Input sanitized upstream or by framework
- Null/undefined guarded by types or prior assertion
- Error handling delegated to caller or middleware
- Performance concern where N is demonstrably small
- Style not violating documented project guidelines
</false_positives>

<verification_protocol>
Before including each finding, verify:
1. What specific code pattern triggers this concern?
2. Is it handled elsewhere (caller, framework, tests)?
3. Can you construct a concrete failure scenario? If not, STOP.
4. Would a senior engineer request changes? If unsure, STOP.
</verification_protocol>"""


EXPLORE_SYSTEM_PROMPT = """\
<task>
You are verifying suspected code review findings. Use the available tools
to explore the repository and confirm or disprove each finding.
</task>

<protocol>
For each finding:
1. Read its verification_needs.
2. Use tools to gather evidence (read files, check callers, imports).
3. Decide: confirmed, disproved, or inconclusive.
4. Provide specific evidence (file paths, line numbers, code snippets).
5. If confirmed → produce a ReviewComment with updated confidence.
   If disproved/inconclusive → explain why, do NOT produce a comment.
</protocol>

<guidelines>
- Be thorough but efficient.
- Start with what verification_needs suggests.
- Do NOT invent new findings — only verify what was found.
- Raise confidence when evidence strongly supports the finding.
</guidelines>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════


async def production_review(
    gh: GitHubPRClient,
    ctx: Context,
    repo: str,
    pr_number: int,
    *,
    focus_areas: str | None = None,
    intensity: str = "balanced",
    max_files: int = 50,
    filter_batch_size: int = 10,
    concurrency: int = 3,
    min_confidence: int = 50,
) -> PRReviewResult:
    """Production PR review: Filter → Review → Verify.

    Pass 1: Gather full context (timeline, prior reviews, existing threads)
    Pass 2: Batch-filter files by interest level
    Pass 3: Deep per-file review with tools + verification protocol
    Pass 4: Agentic verification of findings with repo exploration
    """

    # ═══════════════════════════════════════════════════════════════════
    # PASS 1: Context — gather everything we need
    # ═══════════════════════════════════════════════════════════════════
    # Fetch timeline, existing review threads, and prior review bodies
    # in parallel. Prior reviews tell us what's already been said so we
    # don't repeat it. Existing threads tell us what's been flagged.

    timeline, comments_by_file, prior_reviews = await asyncio.gather(
        gh.get_timeline(repo, pr_number),
        gh.get_review_comments_by_file(repo, pr_number),
        gh.get_prior_review_bodies(repo, pr_number),
    )

    total_files = len(timeline.files)

    # Pre-filter: skip binary/generated files (no LLM needed)
    chunks = _prefilter(timeline.files, max_files)
    files_prefiltered = total_files - len(chunks)

    # ═══════════════════════════════════════════════════════════════════
    # PASS 2: Filter — batched ctx.sample(), structured output, NO tools
    # ═══════════════════════════════════════════════════════════════════
    # Classify files in batches of 10. All batches run concurrently.
    # This is cheap — just classification, not full review.

    reviewables = await _filter_files(ctx, chunks, timeline, filter_batch_size)
    files_filtered = len(chunks) - len(reviewables)

    # ═══════════════════════════════════════════════════════════════════
    # PASS 3: Review — per-file ctx.sample() + tools + verification
    # ═══════════════════════════════════════════════════════════════════
    # Each file gets a focused review with:
    # - The filter stage's focus hints guiding attention
    # - Existing threads injected to prevent re-flagging
    # - Prior review context to avoid repeating points
    # - Tools to explore related files
    # - A verification protocol the LLM must follow

    file_findings = await _review_files(
        gh,
        ctx,
        reviewables,
        timeline,
        comments_by_file,
        prior_reviews,
        focus_areas,
        intensity,
        concurrency,
        repo,
    )

    # ═══════════════════════════════════════════════════════════════════
    # PASS 4: Verify — agentic ctx.sample() + MULTIPLE tools
    # ═══════════════════════════════════════════════════════════════════
    # For each file with findings, the LLM explores the repo to confirm
    # or disprove them. It can read any file, check callers, look at
    # imports. Only confirmed findings survive to the final output.

    explore_results = await _verify_findings(gh, ctx, file_findings, timeline, repo, concurrency)

    # Aggregate — only confirmed, verified findings
    return _aggregate(
        file_findings,
        explore_results,
        total_files,
        files_prefiltered + files_filtered,
        min_confidence,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Pass 2: Filter
# ═══════════════════════════════════════════════════════════════════════════


def _prefilter(files: list[PRFile], max_files: int) -> list[DiffChunk]:
    """Drop binary/generated files before wasting LLM calls."""

    def should_skip(f: PRFile) -> bool:
        if f.patch is None:
            return True
        return any(fnmatch.fnmatch(f.filename, p) for p in SKIP_PATTERNS)

    chunks = [
        DiffChunk(
            index=i,
            filename=f.filename,
            status=f.status,
            additions=f.additions,
            deletions=f.deletions,
            patch=f.patch or "",
        )
        for i, f in enumerate(files)
        if not should_skip(f)
    ]
    chunks.sort(key=lambda c: c.additions + c.deletions, reverse=True)
    return chunks[:max_files]


async def _filter_files(
    ctx: Context,
    chunks: list[DiffChunk],
    timeline: PRTimeline,
    batch_size: int,
) -> list[tuple[DiffChunk, FilteredChunk]]:
    """Batch-classify files. Returns (chunk, classification) pairs."""
    if not chunks:
        return []

    pr = timeline.pr
    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]

    async def filter_batch(batch: list[DiffChunk]) -> FilterBatchResult:
        chunk_texts = []
        for c in batch:
            preview = c.patch[:1500]
            if len(c.patch) > 1500:
                preview += f"\n... ({len(c.patch) - 1500} chars truncated)"
            chunk_texts.append(
                f'<chunk index="{c.index}">\n'
                f"File: {c.filename} ({c.status}, +{c.additions} -{c.deletions})\n"
                f"```diff\n{preview}\n```\n</chunk>"
            )

        prompt = (
            f"<context>\nPR #{pr.number}: {pr.title}\n"
            f"Author: @{pr.author.login} | {pr.head_ref} -> {pr.base_ref}\n"
            f"Description: {pr.body or '(none)'}\n</context>\n\n"
            f"Classify these {len(batch)} files:\n\n" + "\n\n".join(chunk_texts)
        )

        r = await ctx.sample(
            messages=prompt,
            system_prompt=FILTER_SYSTEM_PROMPT,
            result_type=FilterBatchResult,
            temperature=0.1,
            max_tokens=2048,
        )
        return r.result

    # All batches run concurrently
    results = await asyncio.gather(*(filter_batch(b) for b in batches))

    # Join back and sort by interest
    chunk_by_idx = {c.index: c for c in chunks}
    pairs: list[tuple[DiffChunk, FilteredChunk]] = []
    for br in results:
        for fc in br.chunks:
            if fc.interest == "skip":
                continue
            chunk = chunk_by_idx.get(fc.index)
            if chunk:
                pairs.append((chunk, fc))

    priority = {"high": 0, "medium": 1, "low": 2}
    pairs.sort(key=lambda p: priority.get(p[1].interest, 99))
    return pairs


# ═══════════════════════════════════════════════════════════════════════════
# Pass 3: Review (per-file, tools + verification protocol)
# ═══════════════════════════════════════════════════════════════════════════


async def _review_files(
    gh: GitHubPRClient,
    ctx: Context,
    reviewables: list[tuple[DiffChunk, FilteredChunk]],
    timeline: PRTimeline,
    existing_threads: dict[str, list[PRReviewComment]],
    prior_reviews: list[str],
    focus_areas: str | None,
    intensity: str,
    concurrency: int,
    repo: str,
) -> list[FileFindings]:
    """Review files with bounded concurrency."""
    if not reviewables:
        return []

    sem = asyncio.Semaphore(concurrency)

    async def review(chunk: DiffChunk, fc: FilteredChunk) -> FileFindings:
        async with sem:
            return await _review_one(
                gh,
                ctx,
                chunk,
                fc,
                timeline,
                existing_threads.get(chunk.filename, []),
                prior_reviews,
                focus_areas,
                intensity,
                repo,
            )

    return list(await asyncio.gather(*(review(c, fc) for c, fc in reviewables)))


async def _review_one(
    gh: GitHubPRClient,
    ctx: Context,
    chunk: DiffChunk,
    classification: FilteredChunk,
    timeline: PRTimeline,
    existing_threads: list[PRReviewComment],
    prior_reviews: list[str],
    focus_areas: str | None,
    intensity: str,
    repo: str,
) -> FileFindings:
    """Review a single file with full context."""
    pr = timeline.pr

    # Format existing threads so LLM knows not to re-flag
    threads_section = "(none)"
    if existing_threads:
        threads_section = "\n".join(
            f"- @{t.author.login} on line {t.line}: {t.body[:150]}" for t in existing_threads
        )

    # Format prior review bodies
    prior_section = "(none)"
    if prior_reviews:
        prior_section = "\n---\n".join(body[:300] for body in prior_reviews[:3])

    # Format focus hints from filter stage
    hints_section = ""
    if classification.focus_hints:
        hints = "\n".join(f"- {h}" for h in classification.focus_hints)
        hints_section = (
            f"\n<focus_hints>\nThe triage pass identified these areas:\n{hints}\n</focus_hints>\n"
        )

    prompt = (
        f"<context>\n"
        f"PR #{pr.number}: {pr.title}\n"
        f"Author: @{pr.author.login} | {pr.head_ref} -> {pr.base_ref}\n"
        f"Triage: {classification.interest} — {classification.rationale}\n"
        f"</context>\n\n"
        f"<pr_description>\n{pr.body or '(none)'}\n</pr_description>\n"
        f"{hints_section}\n"
        f"<file_diff>\n"
        f"File: {chunk.filename} ({chunk.status})\n"
        f"```diff\n{chunk.patch}\n```\n"
        f"</file_diff>\n\n"
        f"<existing_threads>\n"
        f"Already flagged on this file — do NOT re-flag:\n"
        f"{threads_section}\n"
        f"</existing_threads>\n\n"
        f"<prior_reviews>\n"
        f"Points already made in prior reviews — do NOT repeat:\n"
        f"{prior_section}\n"
        f"</prior_reviews>\n\n"
        f"Review this file."
    )
    if focus_areas:
        prompt += f"\n\nFocus especially on: {focus_areas}"

    # Tools for cross-file exploration
    all_files = timeline.files

    async def get_file_contents(filepath: str) -> str:
        """Read any file in the repo at the PR's head ref."""
        return await gh.get_file_contents(repo, filepath, pr.head_sha)

    def lookup_file_diff(filename: str) -> str:
        """See another file's diff from this PR."""
        for f in all_files:
            if f.filename == filename:
                return f.patch or "(no patch available)"
        return f"File '{filename}' not in this PR"

    def list_pr_files() -> str:
        """List all files changed in this PR."""
        return "\n".join(
            f"  {f.status:>10} {f.filename} (+{f.additions} -{f.deletions})" for f in all_files
        )

    # ctx.sample() with tools + verification protocol in the system prompt
    result = await ctx.sample(
        messages=prompt,
        system_prompt=_review_system_prompt(intensity),
        result_type=FileFindings,
        tools=[get_file_contents, lookup_file_diff, list_pr_files],
        temperature=0.2,
        max_tokens=4096,
    )
    return result.result


# ═══════════════════════════════════════════════════════════════════════════
# Pass 4: Verify (agentic exploration)
# ═══════════════════════════════════════════════════════════════════════════


async def _verify_findings(
    gh: GitHubPRClient,
    ctx: Context,
    all_findings: list[FileFindings],
    timeline: PRTimeline,
    repo: str,
    concurrency: int,
) -> list[ExploreResult]:
    """Verify findings agentically. Only files with findings are explored."""
    with_findings = [f for f in all_findings if f.findings]
    if not with_findings:
        return []

    sem = asyncio.Semaphore(concurrency)

    async def verify(ff: FileFindings) -> ExploreResult:
        async with sem:
            return await _verify_one_file(gh, ctx, ff, timeline, repo)

    return list(await asyncio.gather(*(verify(ff) for ff in with_findings)))


async def _verify_one_file(
    gh: GitHubPRClient,
    ctx: Context,
    findings: FileFindings,
    timeline: PRTimeline,
    repo: str,
) -> ExploreResult:
    """Verify one file's findings via agentic tool loop.

    The LLM can call tools repeatedly — read files, check callers,
    look at imports — until it has enough evidence to confirm or
    disprove each finding.
    """
    pr = timeline.pr
    all_files = timeline.files

    # Format findings for the prompt
    findings_text = []
    for i, f in enumerate(findings.findings):
        extra = f"\nSuggested fix: {f.suggested_code}" if f.suggested_code else ""
        findings_text.append(
            f'<finding index="{i}">\n'
            f"Title: {f.title}\n"
            f"Severity: {f.severity} | Category: {f.category}\n"
            f"Path: {f.path}:{f.line or '?'}\n"
            f"Body: {f.body}\n"
            f"Why: {f.why}\n"
            f"Confidence: {f.confidence}\n"
            f"Verification needs: {f.verification_needs}"
            f"{extra}\n"
            f"</finding>"
        )

    prompt = (
        f"<context>\n"
        f"PR #{pr.number}: {pr.title} | @{pr.author.login}\n"
        f"File: {findings.filename}\n"
        f"Summary: {findings.file_summary}\n"
        f"</context>\n\n"
        f"Verify these {len(findings.findings)} findings:\n\n" + "\n\n".join(findings_text)
    )

    # Three tools for repo exploration
    async def get_file_contents(filepath: str) -> str:
        """Read any file in the repo at the PR's head ref."""
        return await gh.get_file_contents(repo, filepath, pr.head_sha)

    def lookup_file_diff(filename: str) -> str:
        """See another file's diff from this PR."""
        for f in all_files:
            if f.filename == filename:
                return f.patch or "(no patch available)"
        return f"File '{filename}' not in this PR"

    def list_changed_files() -> str:
        """List all files changed in this PR."""
        return "\n".join(
            f"  {f.status:>10} {f.filename} (+{f.additions} -{f.deletions})" for f in all_files
        )

    # Agentic sampling: LLM calls tools in a loop until done
    result = await ctx.sample(
        messages=prompt,
        system_prompt=EXPLORE_SYSTEM_PROMPT,
        result_type=ExploreResult,
        tools=[get_file_contents, lookup_file_diff, list_changed_files],
        temperature=0.2,
        max_tokens=8192,
    )
    return result.result


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation — only confirmed findings survive
# ═══════════════════════════════════════════════════════════════════════════


def _aggregate(
    file_findings: list[FileFindings],
    explore_results: list[ExploreResult],
    total_files: int,
    files_skipped: int,
    min_confidence: int,
) -> PRReviewResult:
    """Build the final result from verified findings only."""
    comments: list[ReviewComment] = []
    for er in explore_results:
        for vf in er.verified:
            if (
                vf.status == "confirmed"
                and vf.comment is not None
                and vf.comment.confidence >= min_confidence
            ):
                comments.append(vf.comment)

    sev = Counter(c.severity.value for c in comments)
    cat = Counter(c.category.value for c in comments)
    n_crit, n_high = sev.get("critical", 0), sev.get("high", 0)
    risk, health = compute_scores(dict(sev))

    summaries = [
        f"- **{ff.filename}**: {ff.file_summary}" for ff in file_findings if ff.file_summary
    ]

    return PRReviewResult(
        verdict=compute_verdict(n_crit, n_high, sev.get("medium", 0)),
        summary=(
            f"Reviewed {len(file_findings)} files, verified findings "
            f"({len(comments)} confirmed, "
            f"{n_crit} critical, {n_high} high).\n" + "\n".join(summaries)
        ),
        comments=comments,
        risk_score=risk,
        health_score=health,
        files_reviewed=len(file_findings),
        files_skipped=files_skipped,
        stats=ReviewStats(
            total_comments=len(comments),
            by_severity=dict(sev),
            by_category=dict(cat),
        ),
    )
