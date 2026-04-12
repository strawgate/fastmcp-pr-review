"""Tests for v2 per-file review with tools."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from fastmcp_pr_review.models import (
    CommentCategory,
    PRAuthor,
    PRDetails,
    PRFile,
    PRState,
    PRTimeline,
    ReviewState,
    Severity,
)
from fastmcp_pr_review.v2_per_file import (
    FileReview,
    Finding,
    _aggregate,
    per_file_review,
)


def _make_timeline(files: list[PRFile] | None = None) -> PRTimeline:
    return PRTimeline(
        pr=PRDetails(
            number=1,
            title="Test PR",
            body="Body",
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
        files=files
        or [
            PRFile(
                filename="src/main.py",
                status="modified",
                additions=10,
                deletions=5,
                changes=15,
                patch="+new\n-old",
            ),
        ],
    )


def _make_file_review(filename: str = "src/main.py") -> FileReview:
    return FileReview(
        filename=filename,
        findings=[
            Finding(
                path=filename,
                line=10,
                severity=Severity.MEDIUM,
                category=CommentCategory.BUG,
                title="Off by one",
                body="Loop bound wrong",
                why="Skips last element",
                confidence=75,
            ),
        ],
        summary="Adds main logic",
    )


class TestPerFileReview:
    @pytest.mark.asyncio
    async def test_reviews_each_file(self) -> None:
        """Should call ctx.sample() once per reviewable file."""
        files = [
            PRFile(
                filename="a.py",
                status="modified",
                additions=5,
                deletions=0,
                changes=5,
                patch="+code",
            ),
            PRFile(
                filename="b.py",
                status="added",
                additions=10,
                deletions=0,
                changes=10,
                patch="+new",
            ),
            PRFile(
                filename="c.bin",
                status="added",
                additions=0,
                deletions=0,
                changes=0,
                patch=None,
            ),  # binary, skipped
        ]
        timeline = _make_timeline(files)

        ctx = MagicMock()
        ctx.sample = AsyncMock(return_value=MagicMock(result=_make_file_review()))
        gh = MagicMock()
        gh.get_timeline = AsyncMock(return_value=timeline)
        gh.get_file_contents = AsyncMock(return_value="contents")

        result = await per_file_review(gh, ctx, "owner/repo", 1)

        # 2 files with patches, 1 binary skipped
        assert ctx.sample.await_count == 2
        assert result.files_reviewed == 2
        assert result.files_skipped == 1

    @pytest.mark.asyncio
    async def test_passes_tools(self) -> None:
        """ctx.sample() should receive 3 tools."""
        ctx = MagicMock()
        ctx.sample = AsyncMock(return_value=MagicMock(result=_make_file_review()))
        gh = MagicMock()
        gh.get_timeline = AsyncMock(return_value=_make_timeline())
        gh.get_file_contents = AsyncMock(return_value="contents")

        await per_file_review(gh, ctx, "owner/repo", 1)

        call_kwargs = ctx.sample.call_args.kwargs
        assert call_kwargs["result_type"] is FileReview
        assert len(call_kwargs["tools"]) == 3

    @pytest.mark.asyncio
    async def test_filters_low_confidence(self) -> None:
        low_conf = FileReview(
            filename="a.py",
            findings=[
                Finding(
                    path="a.py",
                    line=1,
                    severity=Severity.LOW,
                    category=CommentCategory.STYLE,
                    title="Meh",
                    body="b",
                    why="w",
                    confidence=30,
                )
            ],
            summary="Changes",
        )
        ctx = MagicMock()
        ctx.sample = AsyncMock(return_value=MagicMock(result=low_conf))
        gh = MagicMock()
        gh.get_timeline = AsyncMock(return_value=_make_timeline())
        gh.get_file_contents = AsyncMock(return_value="")

        result = await per_file_review(gh, ctx, "o/r", 1, min_confidence=50)
        assert len(result.comments) == 0
        assert result.verdict == ReviewState.APPROVED

    @pytest.mark.asyncio
    async def test_includes_focus_areas(self) -> None:
        ctx = MagicMock()
        ctx.sample = AsyncMock(return_value=MagicMock(result=_make_file_review()))
        gh = MagicMock()
        gh.get_timeline = AsyncMock(return_value=_make_timeline())
        gh.get_file_contents = AsyncMock(return_value="")

        await per_file_review(gh, ctx, "o/r", 1, focus_areas="security")
        messages = ctx.sample.call_args.kwargs["messages"]
        assert "security" in messages


class TestAggregate:
    def test_empty_reviews(self) -> None:
        """No file reviews should produce an approved result with zero comments."""
        result = _aggregate([], total_files=0, min_confidence=50)
        assert result.verdict == ReviewState.APPROVED
        assert len(result.comments) == 0
        assert result.stats.total_comments == 0

    def test_filters_below_min_confidence(self) -> None:
        """Findings below min_confidence should be excluded."""
        review = FileReview(
            filename="x.py",
            findings=[
                Finding(
                    path="x.py",
                    line=1,
                    severity=Severity.HIGH,
                    category=CommentCategory.BUG,
                    title="Low conf",
                    body="b",
                    why="w",
                    confidence=20,
                ),
                Finding(
                    path="x.py",
                    line=2,
                    severity=Severity.MEDIUM,
                    category=CommentCategory.LOGIC,
                    title="High conf",
                    body="b",
                    why="w",
                    confidence=80,
                ),
            ],
            summary="Mixed",
        )
        result = _aggregate([review], total_files=1, min_confidence=50)
        assert len(result.comments) == 1
        assert result.comments[0].title == "High conf"

    def test_severity_and_category_counts(self) -> None:
        """Stats should correctly tally severity and category distributions."""
        review = FileReview(
            filename="a.py",
            findings=[
                Finding(
                    path="a.py",
                    line=1,
                    severity=Severity.HIGH,
                    category=CommentCategory.BUG,
                    title="Bug 1",
                    body="b",
                    why="w",
                    confidence=90,
                ),
                Finding(
                    path="a.py",
                    line=2,
                    severity=Severity.HIGH,
                    category=CommentCategory.SECURITY,
                    title="Sec 1",
                    body="b",
                    why="w",
                    confidence=90,
                ),
            ],
            summary="Two highs",
        )
        result = _aggregate([review], total_files=1, min_confidence=50)
        assert result.stats.by_severity == {"high": 2}
        assert result.stats.by_category == {"bug": 1, "security": 1}
        assert result.verdict == ReviewState.CHANGES_REQUESTED
