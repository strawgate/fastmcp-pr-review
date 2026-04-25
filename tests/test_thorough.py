"""Tests for the thorough review pipeline."""

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
    ReviewComment,
    ReviewInput,
    ReviewState,
    Severity,
)
from fastmcp_pr_review.thorough import (
    FilterBatchResult,
    FilteredChunk,
    PotentialFinding,
    ReviewDone,
    ThoroughReview,
    VerifyComplete,
)


def _make_timeline() -> PRTimeline:
    return PRTimeline(
        pr=PRDetails(
            number=1,
            title="Test",
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
        files=[
            PRFile(
                filename="src/main.py",
                status="modified",
                additions=20,
                deletions=5,
                changes=25,
                patch="+code",
            )
        ],
    )


def _make_inp(timeline: PRTimeline | None = None) -> ReviewInput:
    tl = timeline or _make_timeline()
    pr = tl.pr
    return ReviewInput(
        files=tl.files,
        title=pr.title,
        description=pr.body or "",
        author=pr.author.login,
        head_ref=pr.head_ref,
        base_ref=pr.base_ref,
        additions=pr.additions,
        deletions=pr.deletions,
        changed_files=pr.changed_files,
    )


def _make_finding(confidence: int = 80) -> PotentialFinding:
    return PotentialFinding(
        path="src/main.py",
        line=10,
        severity=Severity.HIGH,
        category=CommentCategory.SECURITY,
        title="SQL injection",
        body="User input in query",
        why="Could execute SQL",
        confidence=confidence,
        verification_needs="Check sanitization",
    )


def _make_comment(confidence: int = 90) -> ReviewComment:
    return ReviewComment(
        path="src/main.py",
        line=10,
        severity=Severity.HIGH,
        category=CommentCategory.SECURITY,
        title="SQL injection confirmed",
        body="Verified: unsanitized",
        why="Arbitrary SQL",
        confidence=confidence,
    )


async def _noop_file_reader(filepath: str) -> str:
    return "contents"


class TestPrefilter:
    def test_skips_binary(self) -> None:
        files = [
            PRFile(
                filename="a.py", status="modified", additions=1, deletions=0, changes=1, patch="+x"
            ),
            PRFile(
                filename="b.bin", status="added", additions=0, deletions=0, changes=0, patch=None
            ),
        ]
        pipeline = ThoroughReview(max_files=50)
        assert len(pipeline.prefilter(files)) == 1

    def test_skips_patterns(self) -> None:
        files = [
            PRFile(
                filename="src/main.py",
                status="modified",
                additions=1,
                deletions=0,
                changes=1,
                patch="+x",
            ),
            PRFile(
                filename="node_modules/x.js",
                status="added",
                additions=1,
                deletions=0,
                changes=1,
                patch="+x",
            ),
        ]
        pipeline = ThoroughReview(max_files=50)
        chunks = pipeline.prefilter(files)
        assert [c.filename for c in chunks] == ["src/main.py"]


class TestVerifyFindings:
    @pytest.mark.asyncio
    async def test_calls_sample_with_finding_tools(self) -> None:
        """Verify pass should provide confirm/dismiss + exploration tools."""
        ctx = MagicMock()
        ctx.sample = AsyncMock(
            return_value=MagicMock(result=VerifyComplete(summary="Done"))
        )
        pipeline = ThoroughReview(concurrency=1)
        inp = _make_inp()

        await pipeline.verify_findings(ctx, [_make_finding()], inp, _noop_file_reader)

        call_kwargs = ctx.sample.call_args.kwargs
        tool_names = [t.__name__ for t in call_kwargs["tools"]]
        assert "confirm_finding" in tool_names
        assert "dismiss_finding" in tool_names
        assert "get_file_contents" in tool_names
        assert call_kwargs["result_type"] is VerifyComplete

    @pytest.mark.asyncio
    async def test_skips_empty_findings(self) -> None:
        ctx = MagicMock()
        ctx.sample = AsyncMock()
        pipeline = ThoroughReview(concurrency=1)
        inp = _make_inp()

        result = await pipeline.verify_findings(ctx, [], inp, _noop_file_reader)
        assert result == []
        ctx.sample.assert_not_awaited()


class TestAggregate:
    def test_with_confirmed_comments(self) -> None:
        pipeline = ThoroughReview(min_confidence=50)
        result = pipeline.aggregate(
            [_make_comment()],
            total_files=1,
            files_reviewed=1,
            files_skipped=0,
        )
        assert len(result.comments) == 1

    def test_empty_comments(self) -> None:
        pipeline = ThoroughReview(min_confidence=50)
        result = pipeline.aggregate(
            [],
            total_files=1,
            files_reviewed=1,
            files_skipped=0,
        )
        assert result.verdict == ReviewState.APPROVED
        assert len(result.comments) == 0

    def test_confidence_filter(self) -> None:
        low = _make_comment(confidence=30)
        pipeline = ThoroughReview(min_confidence=50)
        result = pipeline.aggregate(
            [low],
            total_files=1,
            files_reviewed=1,
            files_skipped=0,
        )
        assert len(result.comments) == 0


class TestReviewFiles:
    @pytest.mark.asyncio
    async def test_provides_add_finding_tool(self) -> None:
        """Review pass should provide add_finding + exploration tools."""
        from fastmcp_pr_review.thorough import DiffChunk

        ctx = MagicMock()
        ctx.sample = AsyncMock(
            return_value=MagicMock(result=ReviewDone(summary="Clean"))
        )
        pipeline = ThoroughReview(concurrency=1)
        inp = _make_inp()

        chunks = [
            DiffChunk(
                index=0,
                filename="a.py",
                status="modified",
                additions=5,
                deletions=0,
                patch="+x",
            )
        ]
        results = await pipeline.review_files(ctx, chunks, inp, _noop_file_reader)

        call_kwargs = ctx.sample.call_args.kwargs
        tool_names = [t.__name__ for t in call_kwargs["tools"]]
        assert "add_finding" in tool_names
        assert "get_file_contents" in tool_names
        assert call_kwargs["result_type"] is ReviewDone
        assert results == []  # no findings from mock

    @pytest.mark.asyncio
    async def test_clean_batch_no_findings(self) -> None:
        """A clean batch should return no findings."""
        from fastmcp_pr_review.thorough import DiffChunk

        ctx = MagicMock()
        ctx.sample = AsyncMock(
            return_value=MagicMock(result=ReviewDone(summary="All clean"))
        )
        pipeline = ThoroughReview(concurrency=1)
        inp = _make_inp()

        chunks = [
            DiffChunk(
                index=0,
                filename="b.py",
                status="modified",
                additions=2,
                deletions=1,
                patch="+y",
            )
        ]
        results = await pipeline.review_files(ctx, chunks, inp, _noop_file_reader)
        assert results == []


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_filter_and_review_passes(self) -> None:
        """filter + review = 2 sample calls when review finds nothing."""
        filter_result = FilterBatchResult(chunks=[FilteredChunk(index=0, skip=False)])
        review_done = ReviewDone(summary="All clean")

        ctx = MagicMock()
        ctx.sample = AsyncMock(
            side_effect=[
                MagicMock(result=filter_result),
                MagicMock(result=review_done),
            ]
        )

        inp = _make_inp()
        pipeline = ThoroughReview()
        result = await pipeline.run(ctx, inp, _noop_file_reader)

        # filter (1 batch) + review (1 batch) = 2 calls
        # verify skipped because review found no findings
        assert ctx.sample.await_count == 2
        assert result.verdict == ReviewState.APPROVED
        assert result.files_reviewed == 1

    @pytest.mark.asyncio
    async def test_filter_skip_reduces_review_calls(self) -> None:
        """Files marked skip=True should not get a review sample call."""
        filter_result = FilterBatchResult(
            chunks=[FilteredChunk(index=0, skip=True, reason="Auto-generated")]
        )

        ctx = MagicMock()
        ctx.sample = AsyncMock(
            return_value=MagicMock(result=filter_result),
        )

        inp = _make_inp()
        pipeline = ThoroughReview()
        result = await pipeline.run(ctx, inp, _noop_file_reader)

        # Only 1 call: the filter. No review or verify calls needed.
        assert ctx.sample.await_count == 1
        assert result.verdict == ReviewState.APPROVED
        assert result.files_reviewed == 0
