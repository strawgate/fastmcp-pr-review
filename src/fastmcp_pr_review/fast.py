"""Fast single-shot review — one ctx.sample() call, structured output, no tools.

The LLM receives the full diff in one prompt and returns a structured
PRReviewResult with verdict, comments, and scores — all validated
against the Pydantic schema automatically by FastMCP.

To customize, subclass ``FastReview`` and override ``SYSTEM_PROMPT``
and/or ``build_prompt()``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastmcp import Context

from fastmcp_pr_review.models import PRReviewResult, ReviewInput

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# FastReview pipeline
# ═══════════════════════════════════════════════════════════════════════════


class FastReview:
    """Single-shot review pipeline — one LLM call, structured output.

    Fast mode is intentionally lightweight: it receives the full diff and
    project context, but does NOT include ``existing_threads`` or
    ``prior_reviews`` from ``ReviewInput``. Prior review awareness requires
    a separate LLM pass to reason over existing comments (as in thorough
    mode). Including raw prior review text in a single-shot prompt would
    bloat context and dilute focus.

    Override ``SYSTEM_PROMPT`` for domain-specific review focus.
    Override ``build_prompt()`` for custom prompt formatting.
    Override ``run()`` for a completely different flow.
    """

    # -- Prompt (class attribute — override in subclasses) ------------------

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

    # -- Methods -----------------------------------------------------------

    def build_prompt(self, inp: ReviewInput) -> str:
        """Build the user prompt from a ReviewInput. Override for custom format."""
        patches = "\n\n".join(
            f"### {f.filename} ({f.status})\n```diff\n{f.patch}\n```" for f in inp.files if f.patch
        )
        diff_context = patches or "(no patches available)"

        context_section = ""
        if inp.project_context:
            context_section = f"\n### Project Context\n{inp.project_context}\n"
        if inp.linked_issues:
            issues_text = "\n\n".join(inp.linked_issues)
            context_section += f"\n### Linked Issues\n{issues_text}\n"

        commits_section = ""
        if inp.commits:
            commit_msgs = [c.message.split("\n")[0] for c in inp.commits[:10]]
            commits_section = f"\n### Commits ({len(inp.commits)} total)\n" + "\n".join(
                f"- {msg}" for msg in commit_msgs
            )
            if len(inp.commits) > 10:
                commits_section += f"\n... and {len(inp.commits) - 10} more"

        header_parts = [f"## Review: {inp.title}"]
        if inp.pr_number is not None:
            header_parts.append(f"PR #{inp.pr_number}")
        if inp.author:
            header_parts.append(f"Author: @{inp.author}")
        if inp.head_ref and inp.base_ref:
            header_parts.append(f"{inp.head_ref} -> {inp.base_ref}")
        elif inp.head_ref:
            header_parts.append(inp.head_ref)
        if inp.additions or inp.deletions or inp.changed_files:
            header_parts.append(
                f"Stats: +{inp.additions} -{inp.deletions} across {inp.changed_files} files"
            )

        prompt = (
            f"{header_parts[0]}\n"
            + " | ".join(header_parts[1:])
            + f"\n\n### Description\n{inp.description or '(no description)'}\n"
            f"{context_section}"
            f"{commits_section}\n"
            f"### Diff\n{diff_context}\n\n"
            "Review this code and provide your structured assessment."
        )

        if inp.focus_areas:
            prompt += f"\n\nFocus especially on: {inp.focus_areas}"

        return prompt

    async def run(self, ctx: Context, inp: ReviewInput) -> PRReviewResult:
        """Execute the review — one ctx.sample() call with structured output.

        Override for a completely custom flow while keeping the class
        interface consistent.
        """
        prompt = self.build_prompt(inp)

        n_files = len([f for f in inp.files if f.patch])
        logger.info("fast: sampling — %d files, %d chars prompt", n_files, len(prompt))

        result = await ctx.sample(
            messages=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            result_type=PRReviewResult,
            temperature=0.2,
            max_tokens=16384,
        )

        review = result.result
        # The LLM can't reliably count files — set these from the input data.
        review.files_reviewed = n_files
        review.files_skipped = len(inp.files) - n_files

        logger.info(
            "fast: done — %s, %d comments",
            review.verdict,
            len(review.comments),
        )
        return review
