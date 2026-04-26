"""Thorough class-based review pipeline.

A multi-pass review system:

  Pass 1 — Prefilter: Skip binary/generated files (sync, pattern-based)
  Pass 2 — Filter:    Batch-classify files as skip/review (structured output)
  Pass 3 — Review:    Batched review with tool-based finding collection
  Pass 4 — Verify:    Single agentic call to confirm/disprove findings

The ``ThoroughReview`` class decouples the pipeline from data sources.
Callers build a ``ReviewInput`` and supply a ``FileReader`` — the same
pipeline works for PR data, raw diffs, or local worktree diffs.

Key patterns in thorough mode:
  - Prior review awareness (don't repeat what's already been said)
  - Existing thread dedup (don't re-flag resolved issues)
  - Tool-based result collection (add_finding/confirm_finding/dismiss_finding)
  - Agentic verification (LLM explores repo to confirm/disprove)
  - Configurable intensity (conservative / balanced / aggressive)
  - Bounded concurrency for parallel stages
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
from collections import Counter
from typing import TYPE_CHECKING

import logfire
from pydantic import BaseModel, Field

from fastmcp_pr_review.models import (
    CommentCategory,
    FileReader,
    PRReviewComment,
    PRReviewResult,
    ReviewComment,
    ReviewInput,
    ReviewStats,
    Severity,
    compute_scores,
    compute_verdict,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastmcp import Context

    from fastmcp_pr_review.models import PRFile

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

SKIP_PATTERNS = [
    "*.lock", "*.min.js", "*.min.css", "*.map", "*.generated.*",
    "vendor/*", "node_modules/*", "dist/*", "build/*",
    "__pycache__/*", "*.pyc", "*.png", "*.jpg", "*.svg", "*.ico",
]  # fmt: skip


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
    """Triage result for one file. Minimal output to save tokens."""

    index: int = Field(description="Matches DiffChunk.index")
    skip: bool = Field(description="True to skip, false to review")
    reason: str = Field(
        default="",
        description="Why this file was skipped (only when skip=true)",
    )


class FilterBatchResult(BaseModel):
    """result_type for filter batch — structured output, no tools."""

    chunks: list[FilteredChunk]


# -- Pass 3: Review (collected via add_finding tool calls) --


class PotentialFinding(BaseModel):
    """A suspected issue found during review, not yet verified."""

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


# -- Pass 4: Verify (collected via confirm_finding/dismiss_finding tool calls) --


class VerifyComplete(BaseModel):
    """Signals the verify pass is done. The real results come from
    confirm_finding/dismiss_finding tool calls during the agentic loop.
    """

    summary: str = Field(description="Brief summary of verification results")


# ═══════════════════════════════════════════════════════════════════════════
# Pass 3: Review — batch model and batching helper
# ═══════════════════════════════════════════════════════════════════════════


class ReviewDone(BaseModel):
    """Trivial result_type for review — signals the batch is complete.

    Findings are collected via add_finding() tool calls during the loop.
    """

    summary: str = Field(description="Brief summary of review results")


# ---------------------------------------------------------------------------
# Batching — group files by total patch size
# ---------------------------------------------------------------------------

MAX_BATCH_ITEMS = 10
MAX_BATCH_BYTES = 10_000


def _make_batches(
    items: list[DiffChunk],
    *,
    max_items: int = MAX_BATCH_ITEMS,
    max_bytes: int = MAX_BATCH_BYTES,
) -> list[list[DiffChunk]]:
    """Group files into batches by size for review.

    Small files get batched together. Large files get their own batch.
    Delegates to ThoroughReview._make_batches for the implementation.
    """
    return ThoroughReview._make_batches(items, max_items=max_items, max_bytes=max_bytes)


# ═══════════════════════════════════════════════════════════════════════════
# ThoroughReview pipeline
# ═══════════════════════════════════════════════════════════════════════════


class ThoroughReview:
    """Multi-pass review pipeline: filter → review → verify.

    Override any step method or prompt for custom behavior.
    """

    # -- Prompts (class attributes — override in subclasses) ----------------

    SYSTEM_PROMPT = """\
You are an expert code reviewer. You think like a senior engineer doing a \
real code review: you read the diff, understand the intent, trace data flow, \
and only speak up when something is genuinely wrong.

Your north star: would a senior engineer on this team agree this is worth \
commenting on? If not, stay silent. Silence is the default. Zero findings \
is the most common and best outcome.

== Severity (assign AFTER investigating, not before) ==
  critical: Must fix before merge — security vuln, data corruption, prod outage
  high: Should fix before merge — logic error, missing validation, perf regression
  medium: Non-blocking, address soon — error handling gap, suboptimal pattern
  low: Author's discretion — minor improvement, documentation
  nitpick: Truly optional — stylistic preference, alternative approach

== What NOT to flag ==
These are the most common false positives. Do not flag ANY of these:
- Input already sanitized: by framework, parameterized query, upstream validation
- Null safety: guarded by type system, assertion, schema validation, or upstream check
- Error handling: delegated to caller, middleware, framework boundary, or error type
- Performance: N is small in practice, or concern is theoretical without evidence
- Missing tests: for trivial code, auto-generated code, or simple delegation
- Missing validation: handled at another layer (API gateway, schema, middleware)
- Style/naming: unless it violates the project's own documented guidelines
- Anything where you cannot construct a concrete failure scenario with specific input

== Calibration examples ==
These pairs train your threshold. The first is a real issue. The second looks \
similar but is NOT an issue — learn why.

Null access:
  FLAG: `user = await db.find(id); return user.name` — find() returns nullable, \
  .name throws on null. No upstream guard, no type narrowing.
  SKIP: `settings = user.getSettings()` — user is typed non-null, guaranteed by \
  auth middleware. The type system prevents this from being null.

SQL injection:
  FLAG: `cursor.execute(f"SELECT * WHERE id = '{user_id}'")` — user_id comes \
  from request params, string interpolation bypasses parameterization.
  SKIP: `cursor.execute(f"SELECT * WHERE status = '{Status.ACTIVE.value}'")` — \
  interpolated value is a hardcoded enum member, not user input.

Missing error handling:
  FLAG: `data = json.loads(response.text); return data["key"]` — response could \
  be non-JSON (HTML error page), json.loads throws, no try/except.
  SKIP: `data = await client.get_json("/api/data")` — client.get_json() handles \
  parsing and raises a typed error, caller catches at the boundary.

Race condition:
  FLAG: `if not cache.has(key): cache.set(key, compute())` — TOCTOU between \
  has() and set(), concurrent requests duplicate expensive compute().
  SKIP: `cache.get_or_set(key, compute)` — atomic operation, framework handles it.

Performance:
  SKIP: Nested loop over items and tags — without evidence that N is large in \
  practice, this is speculative. Do not flag theoretical perf concerns.

== Rigor ==
Silence is better than noise. A false positive wastes the author's time and \
erodes trust in every future review. Never hedge — no "might", "could", \
"possibly". If you aren't confident after investigation, do not include it. \
Finding zero issues is better than findings that waste time.

== Tool use ==
Call MULTIPLE tools in a single turn. Batch 3-5 tool calls per turn. \
When done, include final_response in the same turn as your last tool call."""

    FILTER_INSTRUCTIONS = """\
MODE: FILTER — Triage files for code review.

For each file, decide: skip or review. Set skip=true ONLY for files with \
zero code review value:
- Pure formatting/whitespace changes with no logic change
- Auto-generated files (lock files, snapshots, compiled output, .min.js)
- Pure deletions of dead/unused code with no replacement logic
- Dependency version bumps with no accompanying code changes

Everything else gets reviewed. Most files should NOT be skipped. \
Output minimal JSON — keep skip reasons under 10 words."""

    REVIEW_INSTRUCTIONS = """\
MODE: REVIEW — Find real bugs in pull request diffs.

Call add_finding(filename, ...) for each real issue you find. Use exploration \
tools (get_file_contents, lookup_file_diff, list_changed_files) to verify \
your concerns before flagging them.

Verification protocol — complete ALL steps before calling add_finding:
1. What specific code pattern or change triggers this concern?
2. Is it handled elsewhere? Read the caller, check for middleware, look at types.
3. Construct a concrete failure scenario with specific input/state.
4. Would a senior engineer on this team agree this is worth flagging?

== Guard clauses that claim guarantees ==
When code has an early-return guard that promises to preserve behavior (e.g.,
"suppress_X", "if not enabled: return early"), the guard itself is not the
complete picture. Read the skipped path and ask: does the promised behavior
actually happen in the suppressed code? A contradiction between a guard's
promise and what the skipped path does is a bug.

== Stopping rule ==
If you cannot complete step 3 (construct a failure scenario) from the
visible code, do NOT stop — read the surrounding and called code before
giving up. Architectural issues often require tracing through call chains.

== Prior review awareness ==
Do not repeat specific style nitpicks from existing review threads.
Do not re-report architectural concerns from prior reviews without independent
verification. However: independently analyze such concerns using the protocol
above — confirm it properly or find that the prior report was wrong. Do not
blindly suppress architectural concerns just because a prior review mentioned them.

== Behavioral changes ==
Flag changes that alter the semantics of an API or behavior, even if the new
behavior appears more correct. Callers who relied on the old behavior may be
silently broken. This includes changes to how special values (None, 0, empty)
are interpreted, or changes to what a guard clause actually guarantees.

No findings is valid and expected for most batches.

Intensity levels (the message will specify which):
  conservative: only flag confidence >= 80, zero comments is the expected outcome
  balanced: flag medium+ confidence, lean toward silence when ambiguous
  aggressive: flag everything noticed, still require evidence for each"""

    VERIFY_INSTRUCTIONS = """\
MODE: VERIFY — Confirm or disprove suspected code review findings.

You will receive a list of findings from the review pass. For each one:
1. Read the verification_needs field to understand what to check
2. Use get_file_contents() to read the relevant source code
3. Trace the data flow, check callers, look for guards and handlers
4. Call confirm_finding() if the issue is real, dismiss_finding() if not

Rules:
- Every finding MUST be confirmed or dismissed EXACTLY ONCE
- Do NOT invent new findings — only verify what was found
- DISMISS if: style preference, handled elsewhere (caller/framework/middleware/types), \
  no concrete failure scenario constructible, or inconclusive after investigation
- When confirming, provide evidence and an updated confidence score
- Dismissing a weak finding is always better than confirming it"""

    # -- Constructor --------------------------------------------------------

    def __init__(
        self,
        *,
        intensity: str = "balanced",
        concurrency: int = 3,
        max_files: int = 50,
        filter_batch_size: int = 10,
        min_confidence: int = 50,
    ) -> None:
        self.intensity = intensity
        self.concurrency = concurrency
        self.max_files = max_files
        self.filter_batch_size = filter_batch_size
        self.min_confidence = min_confidence

    # -- Batching -----------------------------------------------------------

    @staticmethod
    def _make_batches(
        items: list[DiffChunk],
        *,
        max_items: int = MAX_BATCH_ITEMS,
        max_bytes: int = MAX_BATCH_BYTES,
    ) -> list[list[DiffChunk]]:
        """Group files into batches by size for review.

        Small files get batched together. Large files get their own batch.
        """
        batches: list[list[DiffChunk]] = []
        current: list[DiffChunk] = []
        current_size = 0

        for c in items:
            patch_size = len(c.patch)

            if patch_size > max_bytes:
                if current:
                    batches.append(current)
                    current, current_size = [], 0
                batches.append([c])
                continue

            if current_size + patch_size > max_bytes or len(current) >= max_items:
                batches.append(current)
                current, current_size = [], 0

            current.append(c)
            current_size += patch_size

        if current:
            batches.append(current)
        return batches

    # -- Prompt building helpers ---------------------------------------------

    @staticmethod
    def _format_file_section(
        chunk: DiffChunk,
        existing_threads: dict[str, list[PRReviewComment]],
    ) -> str:
        """Build a single <file_diff> section with existing thread context."""
        existing = existing_threads.get(chunk.filename, [])
        threads_text = "(none)"
        if existing:
            threads_text = "\n".join(
                f"  - @{t.author.login} on L{t.line}: {t.body[:120]}" for t in existing
            )
        return (
            f"<file_diff>\n"
            f"File: {chunk.filename} ({chunk.status}, "
            f"+{chunk.additions} -{chunk.deletions})\n"
            f"```diff\n{chunk.patch}\n```\n"
            f"Existing threads: {threads_text}\n"
            f"</file_diff>"
        )

    @staticmethod
    def _build_review_message(
        batch: list[DiffChunk],
        inp: ReviewInput,
        file_sections: list[str],
        intensity: str,
    ) -> str:
        """Build the user message content for a review batch.

        Pure function — fully testable without mocks.
        Static content ordering for KV cache efficiency: instructions come first,
        dynamic content (diffs, prior reviews) comes last.
        """
        prior_section = "(none)"
        if inp.prior_reviews:
            prior_section = "\n---\n".join(body[:300] for body in inp.prior_reviews[:3])

        project_section = ""
        if inp.project_context:
            project_section = f"\n<project_context>\n{inp.project_context}\n</project_context>\n"

        issues_section = ""
        if inp.linked_issues:
            issues_text = "\n\n".join(inp.linked_issues)
            issues_section = (
                f"\n<linked_issues>\n"
                f"Review against these requirements:\n"
                f"{issues_text}\n"
                f"</linked_issues>\n"
            )

        data = f"Intensity: {intensity}\n"
        data += f"{inp.title} | @{inp.author} | {inp.head_ref} -> {inp.base_ref}\n"
        if inp.description:
            data += f"Description: {inp.description}\n"
        if inp.focus_areas:
            data += f"Focus: {inp.focus_areas}\n"
        data += f"{project_section}{issues_section}\n"

        if inp.commits:
            msgs = [c.message.split("\n")[0] for c in inp.commits[:10]]
            commits_section = f"<commits>\n{inp.title} has {len(inp.commits)} commits:\n"
            for msg in msgs:
                commits_section += f"  - {msg}\n"
            if len(inp.commits) > 10:
                commits_section += f"  ... and {len(inp.commits) - 10} more\n"
            commits_section += "</commits>\n"
            data += commits_section

        data += "\n\n".join(file_sections)
        if prior_section != "(none)":
            data += f"\n\nPrior reviews (do NOT repeat):\n{prior_section}"

        return data

    @staticmethod
    def _build_verify_message(
        inp: ReviewInput,
        findings: list[PotentialFinding],
    ) -> str:
        """Build the user message content for the verify pass.

        Pure function — fully testable without mocks.
        """
        finding_lines = []
        for i, f in enumerate(findings):
            finding_lines.append(
                f'<finding index="{i}">\n'
                f"{f.path}:{f.line or '?'} -- {f.title}\n"
                f"{f.body}\n"
                f"Verify: {f.verification_needs}\n"
                f"</finding>"
            )

        n = len(findings)
        return f"{inp.title} | @{inp.author}\nVerify {n} findings:\n\n" + "\n\n".join(finding_lines)

    # -- Exploration tools --------------------------------------------------

    def make_exploration_tools(
        self,
        inp: ReviewInput,
        file_reader: FileReader,
    ) -> list[Callable[..., object]]:
        """Build exploration tools shared by the review and verify passes.

        These let the LLM read files, check diffs, and list changed files
        during its tool-calling loop.
        """

        async def get_file_contents(filepath: str) -> str:
            """Read any file in the repo at the PR's head ref."""
            logger.debug("thorough: tool get_file_contents(%s)", filepath)
            return await file_reader(filepath)

        async def lookup_file_diff(filename: str) -> str:
            """See another file's diff from this PR."""
            logger.debug("thorough: tool lookup_file_diff(%s)", filename)
            for f in inp.files:
                if f.filename == filename:
                    return f.patch or "(no patch available)"
            return f"File '{filename}' not in this PR"

        async def list_changed_files() -> str:
            """List all files changed in this PR."""
            logger.debug("thorough: tool list_changed_files()")
            return "\n".join(
                f"  {f.status:>10} {f.filename} (+{f.additions} -{f.deletions})" for f in inp.files
            )

        return [get_file_contents, lookup_file_diff, list_changed_files]

    # -- Pipeline steps -----------------------------------------------------

    def prefilter(self, files: list[PRFile]) -> list[DiffChunk]:
        """Drop binary/generated files and convert to DiffChunks for the pipeline.

        Skips files with no patch (binary) or matching SKIP_PATTERNS (generated).
        Returns at most max_files chunks, sorted largest-first.
        """

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
        return chunks[: self.max_files]

    async def filter_files(
        self,
        ctx: Context,
        chunks: list[DiffChunk],
        inp: ReviewInput,
    ) -> list[DiffChunk]:
        """Batch-classify files. Returns chunks that should be reviewed."""
        if not chunks:
            return []

        batches = [
            chunks[i : i + self.filter_batch_size]
            for i in range(0, len(chunks), self.filter_batch_size)
        ]

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

            data = (
                f"{inp.title} | @{inp.author} | "
                f"{inp.head_ref} -> {inp.base_ref}\n"
                f"Classify {len(batch)} files:\n\n" + "\n\n".join(chunk_texts)
            )

            try:
                r = await ctx.sample(
                    messages=[self.FILTER_INSTRUCTIONS, data],
                    system_prompt=self.SYSTEM_PROMPT,
                    result_type=FilterBatchResult,
                    temperature=0.1,
                    max_tokens=8192,
                )
                return r.result
            except Exception as exc:
                logger.warning("thorough: filter batch failed: %s", exc)
                # On failure, don't skip any files — let them through to review
                return FilterBatchResult(
                    chunks=[FilteredChunk(index=c.index, skip=False) for c in batch]
                )

        # All batches run concurrently
        results = await asyncio.gather(*(filter_batch(b) for b in batches))

        # Keep only chunks not marked for skipping
        chunk_by_idx = {c.index: c for c in chunks}
        reviewable: list[DiffChunk] = []
        for br in results:
            for fc in br.chunks:
                if fc.skip:
                    chunk = chunk_by_idx.get(fc.index)
                    name = chunk.filename if chunk else f"index={fc.index}"
                    logger.debug("thorough: filter skipped %s — %s", name, fc.reason)
                    continue
                chunk = chunk_by_idx.get(fc.index)
                if chunk:
                    reviewable.append(chunk)

        return reviewable

    async def review_files(
        self,
        ctx: Context,
        chunks: list[DiffChunk],
        inp: ReviewInput,
        file_reader: FileReader,
    ) -> list[PotentialFinding]:
        """Review files in batches. Returns flat list of findings."""
        if not chunks:
            return []

        batches = _make_batches(chunks)
        logger.info("thorough: %d files -> %d review batches", len(chunks), len(batches))

        sem = asyncio.Semaphore(self.concurrency)

        async def review(batch: list[DiffChunk]) -> list[PotentialFinding]:
            async with sem:
                return await self._review_batch(ctx, batch=batch, inp=inp, file_reader=file_reader)

        results = await asyncio.gather(*(review(b) for b in batches))
        return [f for batch_results in results for f in batch_results]

    async def _review_batch(
        self,
        ctx: Context,
        *,
        batch: list[DiffChunk],
        inp: ReviewInput,
        file_reader: FileReader,
    ) -> list[PotentialFinding]:
        """Review a batch of files. Findings collected via add_finding() tool calls."""
        file_sections = [
            self._format_file_section(chunk, inp.existing_threads) for chunk in batch
        ]
        data = self._build_review_message(batch, inp, file_sections, self.intensity)

        # --- State that accumulates via tool calls ---
        findings: list[PotentialFinding] = []

        def add_finding(
            filename: str,
            title: str,
            body: str,
            why: str,
            verification_needs: str,
            severity: str = "medium",
            category: str = "bug",
            line: int | None = None,
            end_line: int | None = None,
            suggested_code: str | None = None,
            confidence: int = 70,
        ) -> str:
            """Report a code review finding.

            Call this for each issue you find. Provide all required fields.
            Only call this for issues you've verified through the protocol.
            """
            findings.append(
                PotentialFinding(
                    path=filename,
                    line=line,
                    end_line=end_line,
                    severity=Severity(severity),
                    category=CommentCategory(category),
                    title=title,
                    body=body,
                    why=why,
                    suggested_code=suggested_code,
                    confidence=confidence,
                    verification_needs=verification_needs,
                )
            )
            return f"Finding recorded: '{title}'. Continue reviewing or complete."

        exploration_tools = self.make_exploration_tools(inp, file_reader)

        # Static-first message ordering for KV cache efficiency:
        # - system_prompt: identical every call → always cached
        # - REVIEW_INSTRUCTIONS: identical every call → cached as second message
        # - data (diffs, prior reviews, commits): varies per batch → cache break
        try:
            await ctx.sample(
                messages=[self.REVIEW_INSTRUCTIONS, data],
                system_prompt=self.SYSTEM_PROMPT,
                result_type=ReviewDone,
                tools=[add_finding, *exploration_tools],
                temperature=0.2,
                max_tokens=8192,
            )
        except Exception as exc:
            logger.warning("thorough: review batch failed: %s", exc)

        for f in findings:
            logger.info("thorough: finding [%s] %s:%s — %s", f.severity, f.path, f.line, f.title)
        return findings

    async def verify_findings(
        self,
        ctx: Context,
        findings: list[PotentialFinding],
        inp: ReviewInput,
        file_reader: FileReader,
    ) -> list[ReviewComment]:
        """Verify all findings in one agentic ctx.sample() call.

        The LLM gets a compact list (file:line + reason) and exploration
        tools. It calls confirm_finding() or dismiss_finding() for each.
        One call, minimal prompt, tools do the heavy lifting.
        """
        if not findings:
            return []

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
            logger.info("thorough: CONFIRMED [%s] %s:%s — %s", severity, path, line, title)
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
            logger.info("thorough: DISMISSED %s — %s", title, reason[:80])
            return (
                f"Dismissed '{title}'. Reason: {reason}. Move on to the next unprocessed finding."
            )

        data = self._build_verify_message(inp, findings)

        exploration_tools = self.make_exploration_tools(inp, file_reader)

        # --- Agentic sampling with tool-based result collection ---
        try:
            await ctx.sample(
                messages=[self.VERIFY_INSTRUCTIONS, data],
                system_prompt=self.SYSTEM_PROMPT,
                result_type=VerifyComplete,
                tools=[confirm_finding, dismiss_finding, *exploration_tools],
                temperature=0.2,
                max_tokens=8192,
            )
        except Exception as exc:
            logger.warning("thorough: verify failed: %s", exc)

        return confirmed

    def aggregate(
        self,
        confirmed_comments: list[ReviewComment],
        total_files: int,
        files_reviewed: int,
        files_skipped: int,
    ) -> PRReviewResult:
        """Build the final result from verified findings only."""
        comments = [c for c in confirmed_comments if c.confidence >= self.min_confidence]

        sev = Counter(c.severity.value for c in comments)
        cat = Counter(c.category.value for c in comments)
        n_crit, n_high = sev.get("critical", 0), sev.get("high", 0)
        risk, health = compute_scores(dict(sev))

        return PRReviewResult(
            verdict=compute_verdict(n_crit, n_high, sev.get("medium", 0)),
            summary=(
                f"Reviewed {files_reviewed} of {total_files} files "
                f"({len(comments)} confirmed, "
                f"{n_crit} critical, {n_high} high)."
            ),
            comments=comments,
            risk_score=risk,
            health_score=health,
            files_reviewed=files_reviewed,
            files_skipped=files_skipped,
            stats=ReviewStats(
                total_comments=len(comments),
                by_severity=dict(sev),
                by_category=dict(cat),
            ),
        )

    # -- Main entry point ---------------------------------------------------

    async def run(
        self,
        ctx: Context,
        inp: ReviewInput,
        file_reader: FileReader,
    ) -> PRReviewResult:
        """Full pipeline: prefilter → filter → review → verify → aggregate."""
        with logfire.span(
            "thorough_review",
            intensity=self.intensity,
        ):
            return await self._run_inner(ctx, inp, file_reader)

    async def _run_inner(
        self,
        ctx: Context,
        inp: ReviewInput,
        file_reader: FileReader,
    ) -> PRReviewResult:
        total_files = len(inp.files)

        # Pass 1: Prefilter (sync — skip binary/generated files)
        chunks = self.prefilter(inp.files)
        files_prefiltered = total_files - len(chunks)

        # Pass 2: Filter (LLM classifies files as skip/review)
        with logfire.span("pass 2: filter", files=len(chunks)):
            reviewables = await self.filter_files(ctx, chunks, inp)
            files_filtered = len(chunks) - len(reviewables)

        # Pass 3: Review (batched review with tool-based finding collection)
        with logfire.span("pass 3: review", files=len(reviewables)):
            all_findings = await self.review_files(ctx, reviewables, inp, file_reader)

        # Pass 4: Verify (agentic exploration to confirm/disprove)
        with logfire.span("pass 4: verify", findings=len(all_findings)):
            confirmed = await self.verify_findings(ctx, all_findings, inp, file_reader)

        return self.aggregate(
            confirmed,
            total_files,
            len(reviewables),
            files_prefiltered + files_filtered,
        )
