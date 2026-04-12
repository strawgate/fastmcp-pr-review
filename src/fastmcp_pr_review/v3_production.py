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
import logging
from collections import Counter
from dataclasses import dataclass, field
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
    from collections.abc import Callable

    from fastmcp import Context

    from fastmcp_pr_review.github_client import GitHubPRClient
    from fastmcp_pr_review.models import PRFile, PRReviewComment, PRTimeline

logger = logging.getLogger(__name__)


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
# Review context — bundles shared state passed through the pipeline
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class _ReviewCtx:
    """Shared context threaded through review and verify passes.

    Bundles the parameters that every pass needs so individual functions
    take 2-3 arguments instead of 10+ positional params.
    """

    gh: GitHubPRClient
    ctx: Context
    repo: str
    timeline: PRTimeline
    existing_threads: dict[str, list[PRReviewComment]] = field(default_factory=dict)
    prior_reviews: list[str] = field(default_factory=list)
    focus_areas: str | None = None
    intensity: str = "balanced"
    concurrency: int = 3


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
# Instead of asking the LLM to produce a complex nested schema as structured
# output, the verify pass uses TOOL CALLS to collect results. The LLM calls
# confirm_finding() or dismiss_finding() as it explores, and the confirmed
# findings accumulate in a list via closure. The result_type is trivial.


class VerifyComplete(BaseModel):
    """Trivial result_type for the verify pass — just signals completion.

    The real results come from confirm_finding/dismiss_finding tool calls
    that accumulate ReviewComments in a closure during the agentic loop.
    """

    summary: str = Field(description="Brief summary of verification results")


# ═══════════════════════════════════════════════════════════════════════════
# Shared tool factory — used by both Pass 3 (Review) and Pass 4 (Verify)
# ═══════════════════════════════════════════════════════════════════════════


def _make_tools(
    gh: GitHubPRClient,
    timeline: PRTimeline,
    repo: str,
) -> list[Callable[..., object]]:
    """Build the three exploration tools used by review and verify passes.

    Returns a consistent set of tool closures so both passes expose
    identical capabilities with identical names.
    """
    pr = timeline.pr
    all_files = timeline.files

    async def get_file_contents(filepath: str) -> str:
        """Read any file in the repo at the PR's head ref."""
        logger.debug("v3: tool get_file_contents(%s)", filepath)
        return await gh.get_file_contents(repo, filepath, pr.head_sha)

    def lookup_file_diff(filename: str) -> str:
        """See another file's diff from this PR."""
        logger.debug("v3: tool lookup_file_diff(%s)", filename)
        for f in all_files:
            if f.filename == filename:
                return f.patch or "(no patch available)"
        return f"File '{filename}' not in this PR"

    def list_changed_files() -> str:
        """List all files changed in this PR."""
        logger.debug("v3: tool list_changed_files()")
        return "\n".join(
            f"  {f.status:>10} {f.filename} (+{f.additions} -{f.deletions})" for f in all_files
        )

    return [get_file_contents, lookup_file_diff, list_changed_files]


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


REVIEW_SYSTEM_PROMPT = """\
<task>
You are an expert code reviewer. Analyze this file's diff and produce
structured findings. Be concise, specific, and actionable.
Finding no issues is a valid and valuable outcome.
</task>

<constraints>
- ONLY comment on code in the diff (added or modified lines)
- Do NOT re-flag issues that have existing review threads
- Every finding MUST include a concrete failure scenario
- Describe what verification would confirm or disprove each finding
- Check completeness: if the PR fixes a pattern across files, verify all instances
- Use tools to check related files for consistency with the change
</constraints>

<severity_classification>
Determine severity AFTER investigating the issue, not before. First
identify the problem and trace through the code, then assign severity.
- critical: Must fix before merge. Security vulns, data corruption, prod-breaking.
- high: Should fix before merge. Logic errors, missing validation, perf regression.
- medium: Address soon, non-blocking. Error handling gaps, suboptimal patterns.
- low: Author discretion, non-blocking. Minor improvements, documentation.
- nitpick: Truly optional. Stylistic preferences, alternative approaches.
</severity_classification>

<false_positives>
Do NOT flag:
- Input sanitized upstream, by framework, or via parameterized queries
- Null/undefined guarded by type system, assertion, schema validation, or upstream check
- Error handling delegated to caller, middleware, or framework error boundary
- Performance concerns where N is demonstrably small in context
- Missing validation handled at another layer (API gateway, schema, middleware)
- Missing tests for trivial getters/setters, auto-generated code, or simple delegation
- Style/naming unless it violates the project's documented coding guidelines
- Any issue where you cannot describe a concrete failure scenario
</false_positives>

<verification_protocol>
Before including each finding, verify:
1. What specific code pattern or change triggers this concern?
2. Read surrounding context -- is it handled elsewhere in the file, caller, or framework?
3. Construct a concrete failure scenario. What input or state causes the bug? If you cannot, STOP.
4. Challenge your finding -- would a senior engineer agree this is real? If unsure, STOP.
</verification_protocol>

<calibration_examples>
Use these to calibrate your judgment. Each pair shows a real issue and a
similar-looking pattern that is NOT an issue.

Null access:
  FLAG: `user = await db.find(id); res.json(user.name)` -- find() can return
  null, accessing .name throws. No upstream guard.
  SKIP: `settings = user.getSettings()` -- user is typed non-null and
  guaranteed by auth middleware.

SQL injection:
  FLAG: `cursor.execute(f"SELECT * WHERE id = '{user_id}'")` -- string
  interpolation with user input, no parameterization.
  SKIP: `cursor.execute(f"SELECT * WHERE status = '{Status.ACTIVE.value}'")`
  -- interpolated value is a hardcoded enum, not user input.

Performance:
  SKIP: Nested loop over items and tags -- without evidence that N is large
  in practice, this is speculative. Do not flag theoretical concerns.
</calibration_examples>

<rigor>
Silence is better than noise. A false positive wastes the author's time
and erodes trust in every future review. Only report findings you could
defend in code review. Avoid hedging language like "might," "could," or
"possibly." If you are not confident, do not include the finding.
Finding no issues is better than findings that waste time.
</rigor>"""


EXPLORE_SYSTEM_PROMPT = """\
<task>
You are verifying suspected code review findings. Use the available tools
to explore the repository and confirm or disprove each finding.
</task>

<protocol>
For each finding:
1. Read its verification_needs to understand what to check.
2. Use exploration tools to gather evidence:
   - get_file_contents(path) to read source files
   - lookup_file_diff(path) to see other files' diffs
   - list_changed_files() to see all changed files
3. Based on evidence, call ONE of:
   - confirm_finding(...) if the issue is real
   - dismiss_finding(...) if the issue is not real or inconclusive
4. Move on to the next finding.
</protocol>

<guidelines>
- Be thorough but efficient. Start with what verification_needs suggests.
- Do NOT invent new findings — only verify what was already found.
- Every finding MUST be either confirmed or dismissed EXACTLY ONCE.
  Do not call confirm_finding or dismiss_finding more than once per finding.
- When confirming, provide an updated confidence score based on evidence.
- You can call multiple tools per turn (5-10 is fine). For example, read
  several files at once, or confirm/dismiss multiple findings in one turn.
</guidelines>

<dismiss_criteria>
DISMISS a finding if ANY of these apply:
- The issue is a style preference, not a bug or security concern
- The concern is handled elsewhere (caller, framework, middleware, types)
- You cannot construct a concrete failure scenario with specific input/state
- A senior engineer would close this as "not a real issue"
- The finding uses hedging language ("might," "could," "possibly") because
  even after investigation you are not confident it is real
Silence is better than noise. Dismissing a weak finding is always better
than confirming it and wasting the author's time.
</dismiss_criteria>"""


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
    """Production PR review: Filter -> Review -> Verify.

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

    logger.info("v3: reviewing %s#%d (intensity=%s)", repo, pr_number, intensity)

    timeline, comments_by_file, prior_reviews = await asyncio.gather(
        gh.get_timeline(repo, pr_number),
        gh.get_review_comments_by_file(repo, pr_number),
        gh.get_prior_review_bodies(repo, pr_number),
    )

    total_files = len(timeline.files)
    logger.info(
        "v3: pass 1 context — %d files, %d existing threads, %d prior reviews",
        total_files,
        sum(len(v) for v in comments_by_file.values()),
        len(prior_reviews),
    )

    # Pre-filter: skip binary/generated files (no LLM needed)
    chunks = _prefilter(timeline.files, max_files)
    files_prefiltered = total_files - len(chunks)

    # ═══════════════════════════════════════════════════════════════════
    # PASS 2: Filter — batched ctx.sample(), structured output, NO tools
    # ═══════════════════════════════════════════════════════════════════
    # Classify files in batches of 10. All batches run concurrently.
    # This is cheap — just classification, not full review.

    logger.info("v3: pass 2 filter — %d chunks to classify", len(chunks))
    reviewables = await _filter_files(ctx, chunks, timeline, filter_batch_size)
    files_filtered = len(chunks) - len(reviewables)
    interest_counts = {}
    for _, fc in reviewables:
        interest_counts[fc.interest] = interest_counts.get(fc.interest, 0) + 1
    logger.info(
        "v3: pass 2 done — %d reviewable (%s), %d filtered",
        len(reviewables),
        interest_counts,
        files_filtered,
    )

    # ═══════════════════════════════════════════════════════════════════
    # PASS 3: Review — per-file ctx.sample() + tools + verification
    # ═══════════════════════════════════════════════════════════════════
    # Each file gets a focused review with:
    # - The filter stage's focus hints guiding attention
    # - Existing threads injected to prevent re-flagging
    # - Prior review context to avoid repeating points
    # - Tools to explore related files
    # - A verification protocol the LLM must follow

    rctx = _ReviewCtx(
        gh=gh,
        ctx=ctx,
        repo=repo,
        timeline=timeline,
        existing_threads=comments_by_file,
        prior_reviews=prior_reviews,
        focus_areas=focus_areas,
        intensity=intensity,
        concurrency=concurrency,
    )

    logger.info("v3: pass 3 review — %d files (concurrency=%d)", len(reviewables), concurrency)
    file_findings = await _review_files(rctx, reviewables=reviewables)
    total_findings = sum(len(ff.findings) for ff in file_findings)
    logger.info(
        "v3: pass 3 done — %d potential findings across %d files",
        total_findings,
        len(file_findings),
    )

    # ═══════════════════════════════════════════════════════════════════
    # PASS 4: Verify — agentic ctx.sample() + tool-based collection
    # ═══════════════════════════════════════════════════════════════════
    # For each file with findings, the LLM explores the repo and calls
    # confirm_finding() or dismiss_finding() tools. Only confirmed
    # findings survive. No complex structured output needed.

    logger.info("v3: pass 4 verify — %d findings to verify", total_findings)
    confirmed_comments = await _verify_findings(rctx, all_findings=file_findings)
    logger.info("v3: pass 4 done — %d confirmed", len(confirmed_comments))

    # Aggregate
    result = _aggregate(
        file_findings,
        confirmed_comments,
        total_files,
        files_prefiltered + files_filtered,
        min_confidence,
    )
    logger.info("v3: done — %s, %d comments", result.verdict, len(result.comments))
    return result


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
            max_tokens=16384,
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
    rctx: _ReviewCtx,
    *,
    reviewables: list[tuple[DiffChunk, FilteredChunk]],
) -> list[FileFindings]:
    """Review files with bounded concurrency."""
    if not reviewables:
        return []

    sem = asyncio.Semaphore(rctx.concurrency)

    async def review(chunk: DiffChunk, fc: FilteredChunk) -> FileFindings:
        async with sem:
            return await _review_one(rctx, chunk=chunk, classification=fc)

    return list(await asyncio.gather(*(review(c, fc) for c, fc in reviewables)))


async def _review_one(
    rctx: _ReviewCtx,
    *,
    chunk: DiffChunk,
    classification: FilteredChunk,
) -> FileFindings:
    """Review a single file with full context."""
    pr = rctx.timeline.pr
    existing_threads = rctx.existing_threads.get(chunk.filename, [])

    # Format existing threads so LLM knows not to re-flag
    threads_section = "(none)"
    if existing_threads:
        threads_section = "\n".join(
            f"- @{t.author.login} on line {t.line}: {t.body[:150]}" for t in existing_threads
        )

    # Format prior review bodies
    prior_section = "(none)"
    if rctx.prior_reviews:
        prior_section = "\n---\n".join(body[:300] for body in rctx.prior_reviews[:3])

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
        f"Already flagged on this file. Rules:\n"
        f"- Resolved with reviewer reply: reviewer's decision is final, do NOT re-flag\n"
        f"- Resolved without reply: author likely fixed it, do NOT re-raise\n"
        f"- Unresolved: already flagged, do NOT duplicate\n"
        f"Threads:\n{threads_section}\n"
        f"</existing_threads>\n\n"
        f"<prior_reviews>\n"
        f"Points already made — do NOT repeat. Only include new observations:\n"
        f"{prior_section}\n"
        f"</prior_reviews>\n\n"
        f"Review this file."
    )
    if rctx.focus_areas:
        prompt += f"\n\nFocus especially on: {rctx.focus_areas}"

    tools = _make_tools(rctx.gh, rctx.timeline, rctx.repo)

    # Add intensity guideline to the user message (not system prompt)
    # so the system prompt stays identical across all per-file calls
    guideline = INTENSITY_GUIDELINES.get(rctx.intensity, INTENSITY_GUIDELINES["balanced"])
    prompt += f"\n\n<signal_noise>\n{guideline}\n</signal_noise>"

    result = await rctx.ctx.sample(
        messages=prompt,
        system_prompt=REVIEW_SYSTEM_PROMPT,
        result_type=FileFindings,
        tools=tools,
        temperature=0.2,
        max_tokens=16384,
    )
    return result.result


# ═══════════════════════════════════════════════════════════════════════════
# Pass 4: Verify (agentic exploration)
# ═══════════════════════════════════════════════════════════════════════════


MAX_VERIFY_BATCH_FINDINGS = 10
MAX_VERIFY_BATCH_BYTES = 5000


def _batch_findings_for_verify(
    all_findings: list[FileFindings],
) -> list[list[FileFindings]]:
    """Group files with findings into batches for verification.

    Small finding sets get verified together. Files with many or large
    findings get their own batch. Empty files are dropped.
    """
    with_findings = [f for f in all_findings if f.findings]
    if not with_findings:
        return []

    batches: list[list[FileFindings]] = []
    current: list[FileFindings] = []
    current_count = 0
    current_bytes = 0

    for ff in with_findings:
        n = len(ff.findings)
        size = sum(len(f.body) + len(f.why) for f in ff.findings)

        if size > MAX_VERIFY_BATCH_BYTES or n > MAX_VERIFY_BATCH_FINDINGS:
            if current:
                batches.append(current)
                current, current_count, current_bytes = [], 0, 0
            batches.append([ff])
            continue

        if (
            current_count + n > MAX_VERIFY_BATCH_FINDINGS
            or current_bytes + size > MAX_VERIFY_BATCH_BYTES
        ):
            batches.append(current)
            current, current_count, current_bytes = [], 0, 0

        current.append(ff)
        current_count += n
        current_bytes += size

    if current:
        batches.append(current)
    return batches


async def _verify_findings(
    rctx: _ReviewCtx,
    *,
    all_findings: list[FileFindings],
) -> list[ReviewComment]:
    """Verify findings agentically. Returns only confirmed comments.

    Findings from multiple files are batched together for efficiency.
    The LLM calls confirm_finding() / dismiss_finding() tools as it
    explores. Confirmed findings accumulate in a list via closure.
    """
    batches = _batch_findings_for_verify(all_findings)
    if not batches:
        return []

    sem = asyncio.Semaphore(rctx.concurrency)

    async def verify(batch: list[FileFindings]) -> list[ReviewComment]:
        async with sem:
            return await _verify_batch(rctx, batch=batch)

    results = await asyncio.gather(*(verify(b) for b in batches))
    return [c for batch in results for c in batch]


async def _verify_batch(
    rctx: _ReviewCtx,
    *,
    batch: list[FileFindings],
) -> list[ReviewComment]:
    """Verify a batch of findings across one or more files.

    The LLM calls confirm_finding() or dismiss_finding() for each
    suspected issue. Exploration tools let it read files, check
    callers, and look at imports. Confirmed findings are collected
    via closure — no complex structured output needed.
    """
    pr = rctx.timeline.pr

    # --- State that accumulates across the tool loop ---
    confirmed: list[ReviewComment] = []

    # --- Finding management tools ---

    def confirm_finding(
        title: str,
        evidence: str,
        path: str,
        line: int | None = None,
        severity: str = "medium",
        category: str = "bug",
        body: str = "",
        why: str = "",
        suggested_code: str | None = None,
        confidence: int = 80,
    ) -> str:
        """Confirm a finding is real and add it to the review.

        Call this when your investigation confirms the issue exists.
        Provide the evidence you found and an updated confidence score.
        """
        logger.info("v3: CONFIRMED [%s] %s:%s — %s", severity, path, line, title)
        confirmed.append(
            ReviewComment(
                path=path,
                line=line,
                severity=Severity(severity),
                category=CommentCategory(category),
                title=title,
                body=body or f"Confirmed: {title}",
                why=why or evidence,
                suggested_code=suggested_code,
                confidence=confidence,
            )
        )
        already = [c.title for c in confirmed]
        return (
            f"Confirmed '{title}' (confidence={confidence}). "
            f"Already confirmed: {already}. "
            f"Move on to the next unprocessed finding."
        )

    def dismiss_finding(title: str, reason: str) -> str:
        """Dismiss a finding — it's not a real issue.

        Call this when your investigation shows the issue doesn't exist,
        is handled elsewhere, or is inconclusive.
        """
        logger.info("v3: DISMISSED %s — %s", title, reason[:80])
        return f"Dismissed '{title}'. Reason: {reason}. Move on to the next unprocessed finding."

    # --- Build prompt with all findings across files ---
    all_findings_text = []
    finding_idx = 0
    for ff in batch:
        all_findings_text.append(f"### File: {ff.filename}\n{ff.file_summary}")
        for f in ff.findings:
            extra = f"\nSuggested fix: {f.suggested_code}" if f.suggested_code else ""
            all_findings_text.append(
                f'<finding index="{finding_idx}">\n'
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
            finding_idx += 1

    total = finding_idx
    files = ", ".join(ff.filename for ff in batch)
    prompt = (
        f"<context>\n"
        f"PR #{pr.number}: {pr.title} | @{pr.author.login}\n"
        f"Files: {files}\n"
        f"</context>\n\n"
        f"Verify these {total} findings across {len(batch)} file(s).\n"
        f"For each, explore the repo then call "
        f"confirm_finding() or dismiss_finding().\n\n" + "\n\n".join(all_findings_text)
    )

    exploration_tools = _make_tools(rctx.gh, rctx.timeline, rctx.repo)

    # --- Agentic sampling with tool-based result collection ---
    await rctx.ctx.sample(
        messages=prompt,
        system_prompt=EXPLORE_SYSTEM_PROMPT,
        result_type=VerifyComplete,
        tools=[confirm_finding, dismiss_finding, *exploration_tools],
        temperature=0.2,
        max_tokens=16384,
    )

    return confirmed


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation — only confirmed findings survive
# ═══════════════════════════════════════════════════════════════════════════


def _aggregate(
    file_findings: list[FileFindings],
    confirmed_comments: list[ReviewComment],
    total_files: int,
    files_skipped: int,
    min_confidence: int,
) -> PRReviewResult:
    """Build the final result from verified findings only."""
    comments = [c for c in confirmed_comments if c.confidence >= min_confidence]

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
