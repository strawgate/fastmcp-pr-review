"""FastMCP server exposing fast and thorough PR review modes."""

from __future__ import annotations

import asyncio
import os
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import click
import logfire
from fastmcp import Context, FastMCP

from fastmcp_pr_review.context import extract_linked_issues, gather_project_context
from fastmcp_pr_review.fast import FastReview
from fastmcp_pr_review.github_client import GitHubPRClient
from fastmcp_pr_review.models import (
    FileReader,
    PRFile,
    PRReviewComment,
    PRReviewResult,
    PRTimeline,
    ReviewInput,
    TimelineEvent,
    TimelineEventType,
)
from fastmcp_pr_review.thorough import ThoroughReview


def _build_review_context_from_events(
    events: list[TimelineEvent],
) -> tuple[dict[str, list[PRReviewComment]], list[str]]:
    """Extract existing threads and prior review bodies from timeline events.

    This is a pure function — easily tested without mocking the GitHub client.
    """
    existing_threads: dict[str, list[PRReviewComment]] = {}
    prior_reviews: list[str] = []
    for event in events:
        if event.type == TimelineEventType.REVIEW_COMMENT and event.path:
            existing_threads.setdefault(event.path, []).append(
                PRReviewComment(
                    id=0,
                    author=event.author,
                    body=event.body,
                    path=event.path,
                    diff_hunk=event.diff_hunk or "",
                    line=event.line,
                    created_at=event.timestamp or datetime.now(UTC),
                )
            )
        elif event.type == TimelineEventType.REVIEW and event.body:
            prior_reviews.append(event.body)
    return existing_threads, prior_reviews


if TYPE_CHECKING:
    from fastmcp.client.sampling import SamplingHandler

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8000
DEFAULT_HTTP_PATH = "/mcp/"


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


def _parse_diff_to_files(diff: str) -> list[PRFile]:
    """Parse a unified diff into PRFile objects.

    Simple parser — handles standard ``diff --git`` and ``---/+++`` headers.
    Each file gets the raw patch text; line counts are best-effort.
    """
    files: list[PRFile] = []
    # Split on diff headers
    parts = re.split(r"^diff --git ", diff, flags=re.MULTILINE)

    for part in parts[1:]:  # skip anything before first diff header
        lines = part.split("\n")
        # Extract filename from "a/path b/path"
        header = lines[0]
        match = re.match(r"a/(.+?) b/(.+)", header)
        if not match:
            continue
        filename = match.group(2)

        # Determine status from the diff metadata
        status = "modified"
        patch_lines = []
        for line in lines[1:]:
            if line.startswith("new file"):
                status = "added"
            elif line.startswith("deleted file"):
                status = "removed"
            elif line.startswith("rename from"):
                status = "renamed"
            # Collect actual diff content (from first @@ onwards)
            if line.startswith("@@") or (patch_lines and not line.startswith("diff --git")):
                patch_lines.append(line)

        patch = "\n".join(patch_lines) if patch_lines else None
        additions = sum(1 for ln in patch_lines if ln.startswith("+") and not ln.startswith("+++"))
        deletions = sum(1 for ln in patch_lines if ln.startswith("-") and not ln.startswith("---"))

        files.append(
            PRFile(
                filename=filename,
                status=status,
                additions=additions,
                deletions=deletions,
                changes=additions + deletions,
                patch=patch,
            )
        )

    return files


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
            "GitHub code review server.\n\n"
            "Review tools:\n"
            "- review_pr_fast: Quick single-shot PR review (one LLM call)\n"
            "- review_pr_thorough: Multi-pass PR pipeline (filter + review + verify)\n"
            "- review_diff_fast: Review a raw unified diff (not tied to a PR)\n\n"
            "Data tools:\n"
            "- get_pr_info: PR timeline (events, comments, reviews)\n"
            "- get_pr_diff: Raw diff text\n"
            "- get_pr_files: List of changed files with stats"
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

    async def _fetch_pr_context(
        repo: str,
        pr_number: int,
    ) -> tuple[PRTimeline, str, list[str]]:
        """Fetch PR timeline, project docs, and linked issues.

        Returns (timeline, project_context, linked_issues).
        """
        timeline = await gh.get_timeline(repo, pr_number)
        pr = timeline.pr
        project_ctx, issues = await asyncio.gather(
            gather_project_context(gh, repo, pr.head_sha),
            extract_linked_issues(gh, repo, pr.body, pr.head_ref),
        )
        return timeline, project_ctx, issues

    async def _build_review_input(
        repo: str,
        pr_number: int,
        *,
        focus_areas: str | None = None,
    ) -> ReviewInput:
        """Build a ReviewInput from PR data — fetches timeline, project docs, linked issues."""
        timeline, project_ctx, issues = await _fetch_pr_context(
            repo,
            pr_number,
        )
        pr = timeline.pr
        return ReviewInput(
            files=timeline.files,
            title=pr.title,
            description=pr.body or "",
            author=pr.author.login,
            pr_number=pr_number,
            head_ref=pr.head_ref,
            base_ref=pr.base_ref,
            additions=pr.additions,
            deletions=pr.deletions,
            changed_files=pr.changed_files,
            project_context=project_ctx,
            linked_issues=issues,
            focus_areas=focus_areas,
        )

    async def _build_thorough_review_input(
        repo: str,
        pr_number: int,
        *,
        focus_areas: str | None = None,
    ) -> tuple[ReviewInput, str]:
        """Build ReviewInput with thorough-mode extras (threads, prior reviews, commits).

        Derives existing_threads and prior_reviews from the timeline instead of
        re-fetching — get_timeline() already has all the data.
        """
        timeline = await gh.get_timeline(repo, pr_number)
        pr = timeline.pr
        project_ctx, issues = await asyncio.gather(
            gather_project_context(gh, repo, pr.head_sha),
            extract_linked_issues(gh, repo, pr.body, pr.head_ref),
        )

        existing_threads, prior_reviews = _build_review_context_from_events(timeline.events)

        inp = ReviewInput(
            files=timeline.files,
            title=pr.title,
            description=pr.body or "",
            author=pr.author.login,
            pr_number=pr_number,
            head_ref=pr.head_ref,
            base_ref=pr.base_ref,
            additions=pr.additions,
            deletions=pr.deletions,
            changed_files=pr.changed_files,
            project_context=project_ctx,
            linked_issues=issues,
            focus_areas=focus_areas,
            existing_threads=existing_threads,
            prior_reviews=prior_reviews,
            commits=timeline.commits,
        )
        return inp, pr.head_sha

    def _make_pr_file_reader(repo: str, head_sha: str) -> FileReader:
        """Create a FileReader that reads files from a PR's head ref."""

        async def file_reader(filepath: str) -> str:
            return await gh.get_file_contents(repo, filepath, head_sha) or ""

        return file_reader

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
        if ctx is None:
            raise ValueError("Context is required for review tools")
        inp = await _build_review_input(repo, pr_number, focus_areas=focus_areas)
        return await FastReview().run(ctx, inp)

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
        if ctx is None:
            raise ValueError("Context is required for review tools")
        inp, head_sha = await _build_thorough_review_input(
            repo,
            pr_number,
            focus_areas=focus_areas,
        )
        file_reader = _make_pr_file_reader(repo, head_sha)
        pipeline = ThoroughReview(intensity=intensity)
        return await pipeline.run(ctx, inp, file_reader)

    # ── Diff review: review a raw unified diff ────────────────────────────

    @mcp.tool
    async def review_diff_fast(
        diff: str,
        repo: str | None = None,
        title: str = "Diff review",
        description: str = "",
        focus_areas: str | None = None,
        ctx: Context | None = None,
    ) -> PRReviewResult:
        """Review a raw unified diff (not tied to a PR).

        Accepts a unified diff string and reviews it using the fast
        single-shot pipeline. Optionally provide a repo for project
        context (README, AGENTS.md, etc.).

        Args:
            diff: Raw unified diff text
            repo: Optional 'owner/repo' for project context
            title: Label for the review (default: 'Diff review')
            description: Description or context for the diff
            focus_areas: Optional areas to focus on (e.g. 'security')
        """
        if ctx is None:
            raise ValueError("Context is required for review tools")
        files = _parse_diff_to_files(diff)

        project_ctx = ""
        if repo:
            project_ctx = await gather_project_context(gh, repo)

        inp = ReviewInput(
            files=files,
            title=title,
            description=description,
            project_context=project_ctx,
            focus_areas=focus_areas,
        )
        return await FastReview().run(ctx, inp)

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
