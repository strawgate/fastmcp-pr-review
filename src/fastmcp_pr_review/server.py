"""FastMCP server exposing three PR review tools of increasing depth."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import logfire
from fastmcp import Context, FastMCP

from fastmcp_pr_review.context import extract_linked_issues, gather_project_context
from fastmcp_pr_review.github_client import GitHubPRClient
from fastmcp_pr_review.models import (
    PRFile,
    PRReviewResult,
    PRTimeline,
    TimelineEvent,
    TimelineEventType,
)

if TYPE_CHECKING:
    from fastmcp.client.sampling import SamplingHandler


def _format_timeline_event(event: TimelineEvent) -> str:
    ts = event.timestamp.isoformat() if event.timestamp else "unknown"
    prefix = f"[{ts}] @{event.author.login}"

    match event.type:
        case TimelineEventType.PR_OPENED:
            return f"{prefix} opened the PR\n{event.body}"
        case TimelineEventType.COMMIT:
            return f"{prefix} pushed: {event.body}"
        case TimelineEventType.COMMENT:
            return f"{prefix} commented:\n{event.body}"
        case TimelineEventType.REVIEW:
            state = event.review_state or "COMMENTED"
            return f"{prefix} reviewed ({state}):\n{event.body}"
        case TimelineEventType.REVIEW_COMMENT:
            loc = f"{event.path}:{event.line}" if event.path else "unknown"
            return f"{prefix} commented on {loc}:\n{event.body}"
        case _:
            return f"{prefix} {event.type}: {event.body}"


def _format_timeline(timeline: PRTimeline) -> str:
    pr = timeline.pr
    header = (
        f"# PR #{pr.number}: {pr.title}\n"
        f"State: {pr.state} | {pr.head_ref} -> {pr.base_ref}\n"
        f"Author: @{pr.author.login}\n"
        f"+{pr.additions} -{pr.deletions} across {pr.changed_files} files\n"
    )
    events_text = "\n---\n".join(_format_timeline_event(e) for e in timeline.events)
    files_text = "\n".join(
        f"  {f.status:>10} {f.filename} (+{f.additions} -{f.deletions})" for f in timeline.files
    )
    return f"{header}\n## Timeline\n{events_text}\n\n## Changed Files\n{files_text}"


def _make_gemini_handler(model: str) -> SamplingHandler:
    from fastmcp.client.sampling.handlers.google_genai import GoogleGenaiSamplingHandler

    return GoogleGenaiSamplingHandler(default_model=model)


def create_server(
    *,
    github_token: str | None = None,
    gemini_model: str = "gemini-2.5-flash",
    sampling_handler: SamplingHandler | None = None,
) -> FastMCP:
    """Create and configure the FastMCP PR review server."""
    logfire.configure()
    logfire.instrument_mcp()

    token = github_token or os.environ.get("GITHUB_TOKEN", "")
    if not token:
        msg = "GITHUB_TOKEN must be set via argument or environment variable"
        raise ValueError(msg)

    gh = GitHubPRClient(token)

    handler = sampling_handler or _make_gemini_handler(gemini_model)
    mcp = FastMCP(
        name="pr-review",
        instructions=(
            "GitHub PR review server with three tools of increasing depth:\n"
            "- review_pr_simple: Quick single-shot review (one LLM call)\n"
            "- review_pr: Per-file review with tools (LLM explores the repo)\n"
            "- review_pr_deep: Production pipeline (filter + review + verify)"
        ),
        sampling_handler=handler,
        sampling_handler_behavior="fallback",
    )

    # ── Data tools ───────────────────────────────────────────────────────

    @mcp.tool
    async def get_pr_info(repo: str, pr_number: int) -> str:
        """Get pull request info as a chronological timeline.

        Args:
            repo: Repository in 'owner/repo' format
            pr_number: The pull request number
        """
        timeline = await gh.get_timeline(repo, pr_number)
        return _format_timeline(timeline)

    @mcp.tool
    async def get_pr_diff(repo: str, pr_number: int) -> str:
        """Get the raw unified diff for a pull request.

        Args:
            repo: Repository in 'owner/repo' format
            pr_number: The pull request number
        """
        return await gh.get_diff(repo, pr_number)

    @mcp.tool
    async def get_pr_files(repo: str, pr_number: int) -> list[PRFile]:
        """Get changed files in a pull request with per-file diffs.

        Args:
            repo: Repository in 'owner/repo' format
            pr_number: The pull request number
        """
        return await gh.get_files(repo, pr_number)

    # ── Shared context gathering ────────────────────────────────────────

    async def _gather_context(repo: str, pr_number: int) -> tuple[str, list[str]]:
        """Fetch project docs + linked issues for any review tool."""
        import asyncio

        timeline = await gh.get_timeline(repo, pr_number)
        pr = timeline.pr
        project_ctx, issues = await asyncio.gather(
            gather_project_context(gh, repo, pr.head_sha),
            extract_linked_issues(gh, repo, pr.body, pr.head_ref),
        )
        return project_ctx, issues

    # ── v1: Simple (one sample call, structured output, no tools) ────────

    @mcp.tool
    async def review_pr_simple(
        repo: str,
        pr_number: int,
        focus_areas: str | None = None,
        ctx: Context | None = None,
    ) -> PRReviewResult:
        """Fast single-shot PR review using structured output only.

        One LLM call — sends the full diff and gets back a structured
        review result. No tool calling. Best for small PRs or quick checks.

        Args:
            repo: Repository in 'owner/repo' format
            pr_number: The pull request number
            focus_areas: Optional areas to focus on (e.g. 'security')
        """
        assert ctx is not None
        from fastmcp_pr_review.v1_simple import simple_review

        project_ctx, issues = await _gather_context(repo, pr_number)
        return await simple_review(
            gh,
            ctx,
            repo,
            pr_number,
            focus_areas=focus_areas,
            project_context=project_ctx,
            linked_issues=issues,
        )

    # ── v2: Per-file review with tools ──────────────────────────────────

    @mcp.tool
    async def review_pr(
        repo: str,
        pr_number: int,
        focus_areas: str | None = None,
        ctx: Context | None = None,
    ) -> PRReviewResult:
        """Per-file PR review with tool calling.

        Loops over each changed file and reviews it individually.
        The LLM can call tools to read other files, look up diffs,
        and explore the codebase during review.

        Args:
            repo: Repository in 'owner/repo' format
            pr_number: The pull request number
            focus_areas: Optional areas to focus on (e.g. 'security')
        """
        assert ctx is not None
        from fastmcp_pr_review.v2_per_file import per_file_review

        project_ctx, issues = await _gather_context(repo, pr_number)
        return await per_file_review(
            gh,
            ctx,
            repo,
            pr_number,
            focus_areas=focus_areas,
            project_context=project_ctx,
            linked_issues=issues,
        )

    # ── v3: Production pipeline ──────────────────────────────────────────

    @mcp.tool
    async def review_pr_deep(
        repo: str,
        pr_number: int,
        focus_areas: str | None = None,
        intensity: str = "balanced",
        ctx: Context | None = None,
    ) -> PRReviewResult:
        """Production PR review: filter + review + agentic verification.

        Multi-pass pipeline with prior review awareness, intelligent
        file filtering, per-file review with verification protocol,
        and agentic exploration to confirm findings. Configurable
        intensity: conservative, balanced, or aggressive.

        Args:
            repo: Repository in 'owner/repo' format
            pr_number: The pull request number
            focus_areas: Optional areas to focus on (e.g. 'security')
            intensity: Review depth — conservative, balanced, aggressive
        """
        assert ctx is not None
        from fastmcp_pr_review.v3_production import production_review

        project_ctx, issues = await _gather_context(repo, pr_number)
        return await production_review(
            gh,
            ctx,
            repo,
            pr_number,
            project_context=project_ctx,
            linked_issues=issues,
            focus_areas=focus_areas,
            intensity=intensity,
        )

    return mcp


def main() -> None:
    """Entry point for the MCP server."""
    server = create_server()
    server.run()


if __name__ == "__main__":
    main()
