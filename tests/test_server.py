"""Tests for the MCP server tool registration and data tools."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner
from fastmcp import Client

from fastmcp_pr_review.models import (
    PRAuthor,
    PRDetails,
    PRState,
    PRTimeline,
    ReviewState,
    TimelineEvent,
    TimelineEventType,
)
from fastmcp_pr_review.server import (
    _build_review_context_from_events,
    _build_review_input,
    _build_thorough_review_input,
    _fetch_pr_context,
    _format_timeline,
    _format_timeline_event,
    _make_pr_file_reader,
    _parse_diff_to_files,
    cli,
    create_server,
)


class TestFormatTimelineEvent:
    def test_pr_opened(self, pr_author: PRAuthor) -> None:
        event = TimelineEvent(
            type=TimelineEventType.PR_OPENED,
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
            author=pr_author,
            body="PR body",
        )
        result = _format_timeline_event(event)
        assert "opened the PR" in result
        assert "@octocat" in result

    def test_commit(self) -> None:
        event = TimelineEvent(
            type=TimelineEventType.COMMIT,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            author=PRAuthor(login="dev"),
            body="feat: add",
        )
        assert "pushed:" in _format_timeline_event(event)

    def test_none_timestamp(self, pr_author: PRAuthor) -> None:
        event = TimelineEvent(
            type=TimelineEventType.COMMENT,
            timestamp=None,
            author=pr_author,
            body="No timestamp",
        )
        assert "[unknown]" in _format_timeline_event(event)


class TestFormatTimeline:
    def test_full_timeline(self, pr_timeline: PRTimeline) -> None:
        result = _format_timeline(pr_timeline)
        assert "PR #42: Add widget feature" in result
        assert "## Timeline" in result


class TestCreateServer:
    def test_requires_token(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "GITHUB_TOKEN"}
        with (
            patch.dict(os.environ, env, clear=True),
            pytest.raises(ValueError, match="GITHUB_TOKEN"),
        ):
            create_server()

    def test_creates_with_token(self) -> None:
        server = create_server(github_token="fake", sampling_handler=MagicMock())
        assert server is not None

    def test_cli_loads_dotenv_and_runs(self) -> None:
        from fastmcp_pr_review import server as server_module

        mock_server = MagicMock()
        runner = CliRunner()

        with (
            patch.object(server_module, "_load_env_file") as load_env_file,
            patch.object(server_module, "_apply_runtime_env") as apply_runtime_env,
            patch.object(server_module, "create_server", return_value=mock_server) as create_server,
        ):
            result = runner.invoke(
                cli,
                [
                    "--github-token",
                    "cli-token",
                    "--gemini-api-key",
                    "cli-gemini-key",
                    "--gemini-model",
                    "gemini-2.5-pro",
                ],
            )

        assert result.exit_code == 0
        load_env_file.assert_called_once_with()
        apply_runtime_env.assert_called_once_with(gemini_api_key="cli-gemini-key")
        create_server.assert_called_once_with(
            github_token="cli-token",
            gemini_model="gemini-2.5-pro",
        )
        mock_server.run.assert_called_once_with(transport="stdio")

    def test_cli_runs_http_transport(self) -> None:
        from fastmcp_pr_review import server as server_module

        mock_server = MagicMock()
        runner = CliRunner()

        with (
            patch.object(server_module, "_load_env_file"),
            patch.object(server_module, "_apply_runtime_env"),
            patch.object(server_module, "create_server", return_value=mock_server) as create_server,
        ):
            result = runner.invoke(
                cli,
                [
                    "--github-token",
                    "cli-token",
                    "--transport",
                    "http",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "9000",
                    "--path",
                    "/api/mcp/",
                ],
            )

        assert result.exit_code == 0
        create_server.assert_called_once_with(
            github_token="cli-token",
            gemini_model="gemini-2.5-flash",
        )
        mock_server.run.assert_called_once_with(
            transport="http",
            host="0.0.0.0",
            port=9000,
            path="/api/mcp/",
        )

    def test_cli_reads_hosted_env_vars(self) -> None:
        from fastmcp_pr_review import server as server_module

        mock_server = MagicMock()
        runner = CliRunner()

        with (
            patch.object(server_module, "_load_env_file"),
            patch.object(server_module, "_apply_runtime_env") as apply_runtime_env,
            patch.object(server_module, "create_server", return_value=mock_server) as create_server,
        ):
            result = runner.invoke(
                cli,
                [],
                env={
                    "GITHUB_TOKEN": "horizon-github-token",
                    "GEMINI_API_KEY": "horizon-gemini-key",
                },
            )

        assert result.exit_code == 0
        apply_runtime_env.assert_called_once_with(gemini_api_key="horizon-gemini-key")
        create_server.assert_called_once_with(
            github_token="horizon-github-token",
            gemini_model="gemini-2.5-flash",
        )
        mock_server.run.assert_called_once_with(transport="stdio")

    def test_main_dispatches_to_click(self) -> None:
        from fastmcp_pr_review import server as server_module

        with patch.object(server_module.cli, "main") as cli_main:
            server_module.main()

        cli_main.assert_called_once_with(standalone_mode=False)


class TestParseDiffToFiles:
    def test_single_modified_file(self) -> None:
        diff = (
            "diff --git a/src/main.py b/src/main.py\n"
            "index abc..def 100644\n"
            "--- a/src/main.py\n"
            "+++ b/src/main.py\n"
            "@@ -1,3 +1,4 @@\n"
            " line1\n"
            "-old\n"
            "+new\n"
            "+added\n"
            " line3\n"
        )
        files = _parse_diff_to_files(diff)
        assert len(files) == 1
        assert files[0].filename == "src/main.py"
        assert files[0].status == "modified"
        assert files[0].additions == 2
        assert files[0].deletions == 1

    def test_new_file(self) -> None:
        diff = (
            "diff --git a/new.py b/new.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/new.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+hello\n"
            "+world\n"
        )
        files = _parse_diff_to_files(diff)
        assert len(files) == 1
        assert files[0].status == "added"
        assert files[0].additions == 2

    def test_deleted_file(self) -> None:
        diff = (
            "diff --git a/old.py b/old.py\n"
            "deleted file mode 100644\n"
            "--- a/old.py\n"
            "+++ /dev/null\n"
            "@@ -1,2 +0,0 @@\n"
            "-bye\n"
            "-world\n"
        )
        files = _parse_diff_to_files(diff)
        assert len(files) == 1
        assert files[0].status == "removed"
        assert files[0].deletions == 2

    def test_multiple_files(self) -> None:
        diff = (
            "diff --git a/a.py b/a.py\n"
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
            "diff --git a/b.py b/b.py\n"
            "--- a/b.py\n"
            "+++ b/b.py\n"
            "@@ -1 +1 @@\n"
            "-x\n"
            "+y\n"
        )
        files = _parse_diff_to_files(diff)
        assert len(files) == 2
        assert files[0].filename == "a.py"
        assert files[1].filename == "b.py"

    def test_empty_diff(self) -> None:
        assert _parse_diff_to_files("") == []

    def test_renamed_file(self) -> None:
        diff = (
            "diff --git a/old_name.py b/new_name.py\n"
            "rename from old_name.py\n"
            "rename to new_name.py\n"
        )
        files = _parse_diff_to_files(diff)
        assert len(files) == 1
        assert files[0].status == "renamed"
        assert files[0].filename == "new_name.py"


class TestToolRegistration:
    @pytest.fixture
    def server(self):
        return create_server(github_token="fake", sampling_handler=MagicMock())

    @pytest.mark.asyncio
    async def test_all_tools_registered(self, server) -> None:
        async with Client(server) as client:
            names = {t.name for t in await client.list_tools()}
            assert "get_pr_info" in names
            assert "get_pr_diff" in names
            assert "get_pr_files" in names
            assert "review_pr_fast" in names
            assert "review_pr_thorough" in names
            assert "review_diff_fast" in names

    @pytest.mark.asyncio
    async def test_get_pr_info(self, server) -> None:
        mock_timeline = PRTimeline(
            pr=PRDetails(
                number=1,
                title="Test",
                state=PRState.OPEN,
                author=PRAuthor(login="user"),
                head_ref="feat",
                base_ref="main",
                head_sha="abc",
                created_at=datetime(2025, 1, 1, tzinfo=UTC),
                updated_at=datetime(2025, 1, 1, tzinfo=UTC),
            ),
            events=[
                TimelineEvent(
                    type=TimelineEventType.PR_OPENED,
                    timestamp=datetime(2025, 1, 1, tzinfo=UTC),
                    author=PRAuthor(login="user"),
                    body="Test PR",
                )
            ],
            files=[],
        )
        with patch(
            "fastmcp_pr_review.server.GitHubPRClient.get_timeline",
            new_callable=AsyncMock,
            return_value=mock_timeline,
        ):
            async with Client(server) as client:
                result = await client.call_tool("get_pr_info", {"repo": "o/r", "pr_number": 1})
                assert "PR #1: Test" in result.content[0].text


class TestBuildReviewContextFromEvents:
    """Tests for the pure function that extracts threads and reviews from timeline events."""

    def test_empty_events(self) -> None:
        threads, reviews = _build_review_context_from_events([])
        assert threads == {}
        assert reviews == []

    def test_review_comment_creates_thread(self) -> None:
        events = [
            TimelineEvent(
                type=TimelineEventType.REVIEW_COMMENT,
                timestamp=datetime(2025, 1, 1, tzinfo=UTC),
                author=PRAuthor(login="reviewer"),
                body="Consider this.",
                path="src/main.py",
                diff_hunk="@@ -1,3 +1,3 @@",
                line=10,
            )
        ]
        threads, reviews = _build_review_context_from_events(events)
        assert "src/main.py" in threads
        assert len(threads["src/main.py"]) == 1
        assert threads["src/main.py"][0].body == "Consider this."
        assert reviews == []

    def test_review_creates_prior_review(self) -> None:
        events = [
            TimelineEvent(
                type=TimelineEventType.REVIEW,
                timestamp=datetime(2025, 1, 1, tzinfo=UTC),
                author=PRAuthor(login="reviewer"),
                body="LGTM with nits.",
                review_state=ReviewState.COMMENTED,
            )
        ]
        threads, reviews = _build_review_context_from_events(events)
        assert threads == {}
        assert reviews == ["LGTM with nits."]

    def test_skips_pr_opened_events(self) -> None:
        events = [
            TimelineEvent(
                type=TimelineEventType.PR_OPENED,
                timestamp=datetime(2025, 1, 1, tzinfo=UTC),
                author=PRAuthor(login="author"),
                body="Initial PR",
            )
        ]
        threads, reviews = _build_review_context_from_events(events)
        assert threads == {}
        assert reviews == []

    def test_multiple_files_and_reviews(self) -> None:
        events = [
            TimelineEvent(
                type=TimelineEventType.REVIEW_COMMENT,
                timestamp=datetime(2025, 1, 1, tzinfo=UTC),
                author=PRAuthor(login="r1"),
                body="Fix A.",
                path="a.py",
                line=1,
            ),
            TimelineEvent(
                type=TimelineEventType.REVIEW_COMMENT,
                timestamp=datetime(2025, 1, 2, tzinfo=UTC),
                author=PRAuthor(login="r2"),
                body="Fix B.",
                path="b.py",
                line=5,
            ),
            TimelineEvent(
                type=TimelineEventType.REVIEW,
                timestamp=datetime(2025, 1, 3, tzinfo=UTC),
                author=PRAuthor(login="r1"),
                body="First pass.",
            ),
            TimelineEvent(
                type=TimelineEventType.REVIEW,
                timestamp=datetime(2025, 1, 4, tzinfo=UTC),
                author=PRAuthor(login="r2"),
                body="Second pass.",
            ),
        ]
        threads, reviews = _build_review_context_from_events(events)
        assert list(threads.keys()) == ["a.py", "b.py"]
        assert reviews == ["First pass.", "Second pass."]


class TestFetchPrContext:
    """Tests for _fetch_pr_context (now module-level, gh-injected)."""

    @pytest.mark.asyncio
    async def test_returns_timeline_project_context_and_issues(
        self,
        pr_timeline: PRTimeline,
    ) -> None:
        gh = MagicMock()
        gh.get_timeline = AsyncMock(return_value=pr_timeline)

        with patch("fastmcp_pr_review.server.gather_project_context") as mock_gpc, patch(
            "fastmcp_pr_review.server.extract_linked_issues"
        ) as mock_eli:
            mock_gpc.return_value = "Project context text."
            mock_eli.return_value = ["Fixes #123"]

            timeline, project_ctx, issues = await _fetch_pr_context(gh, "owner/repo", 42)

        assert timeline is pr_timeline
        assert project_ctx == "Project context text."
        assert issues == ["Fixes #123"]
        gh.get_timeline.assert_awaited_once_with("owner/repo", 42)

    @pytest.mark.asyncio
    async def test_concurrent_fetches(self, pr_timeline: PRTimeline) -> None:
        gh = MagicMock()
        gh.get_timeline = AsyncMock(return_value=pr_timeline)

        with patch("fastmcp_pr_review.server.gather_project_context") as mock_gpc, patch(
            "fastmcp_pr_review.server.extract_linked_issues"
        ) as mock_eli:
            mock_gpc.return_value = ""
            mock_eli.return_value = []

            await _fetch_pr_context(gh, "owner/repo", 42)

            mock_gpc.assert_called_once()
            mock_eli.assert_called_once()


class TestBuildReviewInput:
    """Tests for _build_review_input (now module-level, gh-injected)."""

    @pytest.mark.asyncio
    async def test_builds_input_from_timeline(
        self,
        pr_details: PRDetails,
        pr_timeline: PRTimeline,
    ) -> None:
        gh = MagicMock()
        gh.get_timeline = AsyncMock(return_value=pr_timeline)

        with patch("fastmcp_pr_review.server.gather_project_context") as mock_gpc, patch(
            "fastmcp_pr_review.server.extract_linked_issues"
        ) as mock_eli:
            mock_gpc.return_value = "Project context."
            mock_eli.return_value = []

            inp = await _build_review_input(gh, "owner/repo", 42)

        assert inp.title == pr_details.title
        assert inp.author == pr_details.author.login
        assert inp.pr_number == 42
        assert inp.head_ref == pr_details.head_ref
        assert inp.base_ref == pr_details.base_ref
        assert inp.files == pr_timeline.files
        assert inp.project_context == "Project context."
        assert inp.focus_areas is None

    @pytest.mark.asyncio
    async def test_passes_focus_areas(self, pr_timeline: PRTimeline) -> None:
        gh = MagicMock()
        gh.get_timeline = AsyncMock(return_value=pr_timeline)

        with patch("fastmcp_pr_review.server.gather_project_context"), patch(
            "fastmcp_pr_review.server.extract_linked_issues"
        ):
            inp = await _build_review_input(
                gh, "owner/repo", 42, focus_areas="security"
            )

        assert inp.focus_areas == "security"


class TestBuildThoroughReviewInput:
    """Tests for _build_thorough_review_input (now module-level, gh-injected)."""

    @pytest.mark.asyncio
    async def test_builds_input_with_threads_and_commits(
        self,
        pr_timeline: PRTimeline,
    ) -> None:
        gh = MagicMock()
        gh.get_timeline = AsyncMock(return_value=pr_timeline)

        with patch("fastmcp_pr_review.server.gather_project_context") as mock_gpc, patch(
            "fastmcp_pr_review.server.extract_linked_issues"
        ) as mock_eli:
            mock_gpc.return_value = ""
            mock_eli.return_value = []

            inp, head_sha = await _build_thorough_review_input(
                gh, "owner/repo", 42
            )

        assert inp.commits == pr_timeline.commits
        assert "src/widget.py" in inp.existing_threads
        assert head_sha == pr_timeline.pr.head_sha

    @pytest.mark.asyncio
    async def test_prior_reviews_extracted_from_events(
        self,
        pr_author: PRAuthor,
    ) -> None:
        events = [
            TimelineEvent(
                type=TimelineEventType.PR_OPENED,
                timestamp=datetime(2025, 1, 1, tzinfo=UTC),
                author=pr_author,
                body="PR body",
            ),
            TimelineEvent(
                type=TimelineEventType.REVIEW,
                timestamp=datetime(2025, 1, 2, tzinfo=UTC),
                author=pr_author,
                body="LGTM with nits.",
                review_state=ReviewState.APPROVED,
            ),
        ]
        timeline = PRTimeline(
            pr=PRDetails(
                number=1,
                title="T",
                state=PRState.OPEN,
                author=pr_author,
                head_ref="f",
                base_ref="main",
                head_sha="abc",
                created_at=datetime(2025, 1, 1, tzinfo=UTC),
                updated_at=datetime(2025, 1, 1, tzinfo=UTC),
            ),
            events=events,
            files=[],
            commits=[],
        )
        gh = MagicMock()
        gh.get_timeline = AsyncMock(return_value=timeline)

        with patch("fastmcp_pr_review.server.gather_project_context"), patch(
            "fastmcp_pr_review.server.extract_linked_issues"
        ):
            inp, _ = await _build_thorough_review_input(gh, "owner/repo", 1)

        assert inp.prior_reviews == ["LGTM with nits."]


class TestMakePrFileReader:
    """Tests for _make_pr_file_reader (now module-level, gh-injected)."""

    @pytest.mark.asyncio
    async def test_returns_file_reader_that_delegates_to_gh(self) -> None:
        gh = MagicMock()
        gh.get_file_contents = AsyncMock(return_value="file contents")

        reader = _make_pr_file_reader(gh, "owner/repo", "abc123")
        result = await reader("src/main.py")

        assert result == "file contents"
        gh.get_file_contents.assert_awaited_once_with(
            "owner/repo", "src/main.py", "abc123"
        )

    @pytest.mark.asyncio
    async def test_returns_empty_string_on_none(self) -> None:
        gh = MagicMock()
        gh.get_file_contents = AsyncMock(return_value=None)

        reader = _make_pr_file_reader(gh, "owner/repo", "abc123")
        result = await reader("missing.py")

        assert result == ""
