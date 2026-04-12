"""Tests for v2 batched file review with tools."""

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
    BatchReview,
    FileReview,
    Finding,
    _aggregate,
    _make_batches,
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


class TestMakeBatches:
    def test_small_files_grouped(self) -> None:
        files = [
            PRFile(
                filename=f"f{i}.py",
                status="modified",
                additions=1,
                deletions=0,
                changes=1,
                patch="+x",
            )
            for i in range(5)
        ]
        batches = _make_batches(files)
        assert len(batches) == 1
        assert len(batches[0]) == 5

    def test_large_file_gets_own_batch(self) -> None:
        small = PRFile(
            filename="small.py",
            status="modified",
            additions=1,
            deletions=0,
            changes=1,
            patch="+x",
        )
        large = PRFile(
            filename="large.py",
            status="modified",
            additions=500,
            deletions=0,
            changes=500,
            patch="+" * 15000,
        )
        batches = _make_batches([small, large])
        assert len(batches) == 2
        assert batches[0][0].filename == "small.py"
        assert batches[1][0].filename == "large.py"

    def test_respects_max_batch_bytes(self) -> None:
        files = [
            PRFile(
                filename=f"f{i}.py",
                status="modified",
                additions=1,
                deletions=0,
                changes=1,
                patch="+" * 4000,
            )
            for i in range(5)
        ]
        batches = _make_batches(files)
        assert len(batches) >= 2

    def test_empty_files(self) -> None:
        assert _make_batches([]) == []


class TestPerFileReview:
    @pytest.mark.asyncio
    async def test_reviews_files(self) -> None:
        ctx = MagicMock()
        ctx.sample = AsyncMock(
            return_value=MagicMock(result=BatchReview(files=[_make_file_review()]))
        )
        gh = MagicMock()
        gh.get_timeline = AsyncMock(return_value=_make_timeline())
        gh.get_file_contents = AsyncMock(return_value="contents")

        result = await per_file_review(gh, ctx, "owner/repo", 1)

        ctx.sample.assert_awaited_once()
        call_kwargs = ctx.sample.call_args.kwargs
        assert call_kwargs["result_type"] is BatchReview
        assert len(call_kwargs["tools"]) == 3
        assert result.files_reviewed == 1

    @pytest.mark.asyncio
    async def test_skips_binary_files(self) -> None:
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
                filename="b.bin",
                status="added",
                additions=0,
                deletions=0,
                changes=0,
                patch=None,
            ),
        ]
        ctx = MagicMock()
        ctx.sample = AsyncMock(
            return_value=MagicMock(result=BatchReview(files=[_make_file_review("a.py")]))
        )
        gh = MagicMock()
        gh.get_timeline = AsyncMock(return_value=_make_timeline(files))
        gh.get_file_contents = AsyncMock(return_value="")

        result = await per_file_review(gh, ctx, "owner/repo", 1)
        assert result.files_reviewed == 1
        assert result.files_skipped == 1

    @pytest.mark.asyncio
    async def test_includes_focus_areas(self) -> None:
        ctx = MagicMock()
        ctx.sample = AsyncMock(
            return_value=MagicMock(result=BatchReview(files=[_make_file_review()]))
        )
        gh = MagicMock()
        gh.get_timeline = AsyncMock(return_value=_make_timeline())
        gh.get_file_contents = AsyncMock(return_value="")

        await per_file_review(gh, ctx, "o/r", 1, focus_areas="security")
        messages = ctx.sample.call_args.kwargs["messages"]
        assert "security" in messages


class TestAggregate:
    def test_empty_reviews(self) -> None:
        result = _aggregate([], total_files=0, min_confidence=50)
        assert result.verdict == ReviewState.APPROVED

    def test_filters_below_min_confidence(self) -> None:
        fr = FileReview(
            filename="a.py",
            findings=[
                Finding(
                    path="a.py",
                    line=1,
                    severity=Severity.HIGH,
                    category=CommentCategory.BUG,
                    title="Bug",
                    body="b",
                    why="w",
                    confidence=30,
                )
            ],
            summary="Changes",
        )
        result = _aggregate([fr], total_files=1, min_confidence=50)
        assert len(result.comments) == 0

    def test_counts_severity_and_category(self) -> None:
        fr = FileReview(
            filename="a.py",
            findings=[
                Finding(
                    path="a.py",
                    line=1,
                    severity=Severity.HIGH,
                    category=CommentCategory.SECURITY,
                    title="XSS",
                    body="b",
                    why="w",
                    confidence=90,
                ),
                Finding(
                    path="a.py",
                    line=5,
                    severity=Severity.MEDIUM,
                    category=CommentCategory.BUG,
                    title="Bug",
                    body="b",
                    why="w",
                    confidence=80,
                ),
            ],
            summary="Changes",
        )
        result = _aggregate([fr], total_files=1, min_confidence=50)
        assert result.stats.by_severity == {"high": 1, "medium": 1}
        assert result.stats.by_category == {"security": 1, "bug": 1}
