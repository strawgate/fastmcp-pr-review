"""FastMCP server exposing fast and thorough PR review modes."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import click
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

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8000
DEFAULT_HTTP_PATH = "/mcp/"


def _require_sampling_context(ctx: Context | None) -> Context:
    """Ensure a review tool received the sampling context required to run."""
    if ctx is None:
        msg = "Sampling context is required to run review tools."
        raise ValueError(msg)
    return ctx


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
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    sampling_handler: SamplingHandler | None = None,
) -> FastMCP:
    """Create and configure the FastMCP PR review server."""
    logfire.configure()
    logfire.instrument_mcp()
    logfire.instrument_google_genai()

    token = github_token or os.environ.get("GITHUB_TOKEN", "")
    if not token:
        msg = "GITHUB_TOKEN must be set via argument or environment variable"
        raise ValueError(msg)

    gh = GitHubPRClient(token)

    handler = sampling_handler or _make_gemini_handler(gemini_model)
    mcp = FastMCP(
        name="pr-review",
        instructions=(
            "GitHub PR review server with two review modes:\n"
            "- review_pr_fast: Quick single-shot review (one LLM call)\n"
            "- review_pr_thorough: Multi-pass pipeline (filter + review + verify)"
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

    # ── Fast mode: one sample call, structured output, no tools ──────────

    @mcp.tool
    async def review_pr_fast(
        repo: str,
        pr_number: int,
        focus_areas: str | None = None,
        ctx: Context | None = None,
    ) -> PRReviewResult:
        """Fast PR review using a single structured sampling call.

        One LLM call — sends the full diff and gets back a structured
        review result. No tool calling. Best for small PRs or quick checks.

        Args:
            repo: Repository in 'owner/repo' format
            pr_number: The pull request number
            focus_areas: Optional areas to focus on (e.g. 'security')
        """
        ctx = _require_sampling_context(ctx)
        from fastmcp_pr_review.fast import fast_review

        project_ctx, issues = await _gather_context(repo, pr_number)
        return await fast_review(
            gh,
            ctx,
            repo,
            pr_number,
            focus_areas=focus_areas,
            project_context=project_ctx,
            linked_issues=issues,
        )

    # ── Thorough mode: multi-pass pipeline ────────────────────────────────

    @mcp.tool
    async def review_pr_thorough(
        repo: str,
        pr_number: int,
        focus_areas: str | None = None,
        intensity: str = "balanced",
        ctx: Context | None = None,
    ) -> PRReviewResult:
        """Thorough PR review: filter + review + agentic verification.

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
        ctx = _require_sampling_context(ctx)
        from fastmcp_pr_review.thorough import thorough_review

        project_ctx, issues = await _gather_context(repo, pr_number)
        return await thorough_review(
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


def _load_env_file() -> None:
    from dotenv import load_dotenv

    load_dotenv()


def _apply_runtime_env(*, gemini_api_key: str | None) -> None:
    if gemini_api_key is not None:
        os.environ["GEMINI_API_KEY"] = gemini_api_key


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--github-token",
    envvar="GITHUB_TOKEN",
    show_envvar=True,
    help="Override GITHUB_TOKEN for this server process.",
)
@click.option(
    "--gemini-api-key",
    envvar="GEMINI_API_KEY",
    show_envvar=True,
    help="Override GEMINI_API_KEY for this server process.",
)
@click.option(
    "--gemini-model",
    default=DEFAULT_GEMINI_MODEL,
    show_default=True,
    help="Fallback Gemini model when the MCP client cannot provide sampling.",
)
@click.option(
    "--transport",
    type=click.Choice(["stdio", "http"], case_sensitive=False),
    default="stdio",
    show_default=True,
    help="Server transport to run.",
)
@click.option(
    "--host",
    default=DEFAULT_HTTP_HOST,
    show_default=True,
    help="Host to bind when running with HTTP transport.",
)
@click.option(
    "--port",
    default=DEFAULT_HTTP_PORT,
    show_default=True,
    type=int,
    help="Port to bind when running with HTTP transport.",
)
@click.option(
    "--path",
    "http_path",
    default=DEFAULT_HTTP_PATH,
    show_default=True,
    help="HTTP MCP path when running with HTTP transport.",
)
def cli(
    github_token: str | None,
    gemini_api_key: str | None,
    gemini_model: str,
    transport: str,
    host: str,
    port: int,
    http_path: str,
) -> None:
    """Run the FastMCP PR review MCP server."""
    _load_env_file()
    _apply_runtime_env(gemini_api_key=gemini_api_key)
    server = create_server(github_token=github_token, gemini_model=gemini_model)
    if transport == "http":
        server.run(transport="http", host=host, port=port, path=http_path)
        return
    server.run(transport="stdio")


def main() -> None:
    """Entry point for the MCP server."""
    cli.main(standalone_mode=False)


if __name__ == "__main__":
    main()
