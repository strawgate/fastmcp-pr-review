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
    TimelineEvent,
    TimelineEventType,
)
from fastmcp_pr_review.server import (
    _format_timeline,
    _format_timeline_event,
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
