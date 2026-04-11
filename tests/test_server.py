"""Tests for the MCP server tool registration and data tools."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
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
            assert "review_pr_simple" in names
            assert "review_pr" in names
            assert "review_pr_deep" in names

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
