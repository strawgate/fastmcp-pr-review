"""Tests for v3 production pipeline."""

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
    ReviewState,
    Severity,
)
from fastmcp_pr_review.v3_production import (
    FilterBatchResult,
    FilteredChunk,
    PotentialFinding,
    ReviewDone,
    VerifyComplete,
    _aggregate,
    _prefilter,
    _review_files,
    _ReviewCtx,
    _verify_findings,
    production_review,
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


def _make_rctx(
    gh: object | None = None,
    ctx: object | None = None,
    concurrency: int = 1,
) -> _ReviewCtx:
    if gh is None:
        gh = MagicMock()
        gh.get_file_contents = AsyncMock(return_value="contents")
    if ctx is None:
        ctx = MagicMock()
        ctx.sample = AsyncMock()
    return _ReviewCtx(
        gh=gh,  # ty: ignore[invalid-argument-type]
        ctx=ctx,  # ty: ignore[invalid-argument-type]
        repo="o/r",
        timeline=_make_timeline(),
        concurrency=concurrency,
    )


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
        assert len(_prefilter(files, max_files=50)) == 1

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
        chunks = _prefilter(files, max_files=50)
        assert [c.filename for c in chunks] == ["src/main.py"]


class TestVerifyFindings:
    @pytest.mark.asyncio
    async def test_calls_sample_with_finding_tools(self) -> None:
        """Verify pass should provide confirm/dismiss + exploration tools."""
        rctx = _make_rctx()
        rctx.ctx.sample = AsyncMock(  # ty: ignore[invalid-assignment]
            return_value=MagicMock(result=VerifyComplete(summary="Done"))
        )

        await _verify_findings(rctx, findings=[_make_finding()])

        call_kwargs = rctx.ctx.sample.call_args.kwargs  # ty: ignore[unresolved-attribute]
        tool_names = [t.__name__ for t in call_kwargs["tools"]]
        assert "confirm_finding" in tool_names
        assert "dismiss_finding" in tool_names
        assert "get_file_contents" in tool_names
        assert call_kwargs["result_type"] is VerifyComplete

    @pytest.mark.asyncio
    async def test_skips_empty_findings(self) -> None:
        rctx = _make_rctx()
        result = await _verify_findings(rctx, findings=[])
        assert result == []
        rctx.ctx.sample.assert_not_awaited()  # ty: ignore[unresolved-attribute]


class TestAggregate:
    def test_with_confirmed_comments(self) -> None:
        result = _aggregate(
            [_make_comment()], total_files=1,
            files_reviewed=1, files_skipped=0, min_confidence=50,
        )
        assert len(result.comments) == 1

    def test_empty_comments(self) -> None:
        result = _aggregate(
            [], total_files=1,
            files_reviewed=1, files_skipped=0, min_confidence=50,
        )
        assert result.verdict == ReviewState.APPROVED
        assert len(result.comments) == 0

    def test_confidence_filter(self) -> None:
        low = _make_comment(confidence=30)
        result = _aggregate(
            [low], total_files=1,
            files_reviewed=1, files_skipped=0, min_confidence=50,
        )
        assert len(result.comments) == 0


class TestReviewFiles:
    @pytest.mark.asyncio
    async def test_provides_add_finding_tool(self) -> None:
        """Review pass should provide add_finding + exploration tools."""
        from fastmcp_pr_review.v3_production import DiffChunk

        rctx = _make_rctx()
        rctx.ctx.sample = AsyncMock(  # ty: ignore[invalid-assignment]
            return_value=MagicMock(result=ReviewDone(summary="Clean"))
        )

        chunks = [
            DiffChunk(
                index=0, filename="a.py", status="modified",
                additions=5, deletions=0, patch="+x",
            )
        ]
        results = await _review_files(rctx, chunks=chunks)

        call_kwargs = rctx.ctx.sample.call_args.kwargs  # ty: ignore[unresolved-attribute]
        tool_names = [t.__name__ for t in call_kwargs["tools"]]
        assert "add_finding" in tool_names
        assert "get_file_contents" in tool_names
        assert call_kwargs["result_type"] is ReviewDone
        assert results == []  # no findings from mock

    @pytest.mark.asyncio
    async def test_clean_batch_no_findings(self) -> None:
        """A clean batch should return no findings."""
        from fastmcp_pr_review.v3_production import DiffChunk

        rctx = _make_rctx()
        rctx.ctx.sample = AsyncMock(  # ty: ignore[invalid-assignment]
            return_value=MagicMock(result=ReviewDone(summary="All clean"))
        )

        chunks = [
            DiffChunk(
                index=0, filename="b.py", status="modified",
                additions=2, deletions=1, patch="+y",
            )
        ]
        results = await _review_files(rctx, chunks=chunks)
        assert results == []


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_filter_and_review_passes(self) -> None:
        """filter + review = 2 sample calls when review finds nothing."""
        filter_result = FilterBatchResult(
            chunks=[FilteredChunk(index=0, skip=False)]
        )
        review_done = ReviewDone(summary="All clean")

        ctx = MagicMock()
        ctx.sample = AsyncMock(
            side_effect=[
                MagicMock(result=filter_result),
                MagicMock(result=review_done),
            ]
        )

        gh = MagicMock()
        gh.get_timeline = AsyncMock(return_value=_make_timeline())
        gh.get_review_comments_by_file = AsyncMock(return_value={})
        gh.get_prior_review_bodies = AsyncMock(return_value=[])
        gh.get_file_contents = AsyncMock(return_value="contents")

        result = await production_review(gh, ctx, "o/r", 1)

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

        gh = MagicMock()
        gh.get_timeline = AsyncMock(return_value=_make_timeline())
        gh.get_review_comments_by_file = AsyncMock(return_value={})
        gh.get_prior_review_bodies = AsyncMock(return_value=[])

        result = await production_review(gh, ctx, "o/r", 1)

        # Only 1 call: the filter. No review or verify calls needed.
        assert ctx.sample.await_count == 1
        assert result.verdict == ReviewState.APPROVED
        assert result.files_reviewed == 0
