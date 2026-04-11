"""Tests for v1 simple single-shot review."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from fastmcp_pr_review.models import (
    PRAuthor,
    PRDetails,
    PRFile,
    PRReviewResult,
    PRState,
    PRTimeline,
    ReviewState,
)
from fastmcp_pr_review.v1_simple import simple_review


def _make_timeline() -> PRTimeline:
    return PRTimeline(
        pr=PRDetails(
            number=1,
            title="Test",
            state=PRState.OPEN,
            author=PRAuthor(login="dev"),
            head_ref="feat",
            base_ref="main",
            head_sha="abc",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            updated_at=datetime(2025, 1, 1, tzinfo=UTC),
            html_url="https://github.com/owner/repo/pull/1",
        ),
        events=[],
        files=[
            PRFile(
                filename="src/main.py",
                status="modified",
                additions=10,
                deletions=5,
                changes=15,
                patch="+new\n-old",
            )
        ],
    )


def _make_result() -> PRReviewResult:
    return PRReviewResult(
        verdict=ReviewState.APPROVED,
        summary="Looks good",
        risk_score=1,
        health_score=100,
    )


class TestSimpleReview:
    @pytest.mark.asyncio
    async def test_single_sample_call_no_tools(self) -> None:
        """v1 makes exactly one ctx.sample() call with no tools."""
        ctx = MagicMock()
        ctx.sample = AsyncMock(return_value=MagicMock(result=_make_result()))
        gh = MagicMock()
        gh.get_timeline = AsyncMock(return_value=_make_timeline())
        gh.get_diff = AsyncMock(return_value="diff content")

        result = await simple_review(gh, ctx, "owner/repo", 1)

        assert result.verdict == ReviewState.APPROVED
        ctx.sample.assert_awaited_once()
        call_kwargs = ctx.sample.call_args.kwargs
        assert call_kwargs["result_type"] is PRReviewResult
        assert "tools" not in call_kwargs

    @pytest.mark.asyncio
    async def test_includes_focus_areas(self) -> None:
        ctx = MagicMock()
        ctx.sample = AsyncMock(return_value=MagicMock(result=_make_result()))
        gh = MagicMock()
        gh.get_timeline = AsyncMock(return_value=_make_timeline())
        gh.get_diff = AsyncMock(return_value="diff")

        await simple_review(gh, ctx, "owner/repo", 1, focus_areas="security")

        messages = ctx.sample.call_args.kwargs["messages"]
        assert "security" in messages

    @pytest.mark.asyncio
    async def test_includes_diff_in_prompt(self) -> None:
        ctx = MagicMock()
        ctx.sample = AsyncMock(return_value=MagicMock(result=_make_result()))
        gh = MagicMock()
        gh.get_timeline = AsyncMock(return_value=_make_timeline())
        gh.get_diff = AsyncMock(return_value="unique_diff_marker")

        await simple_review(gh, ctx, "owner/repo", 1)

        messages = ctx.sample.call_args.kwargs["messages"]
        assert "src/main.py" in messages
