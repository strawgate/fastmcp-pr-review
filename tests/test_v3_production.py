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
    ExploreResult,
    FileFindings,
    FilterBatchResult,
    FilteredChunk,
    PotentialFinding,
    VerifiedFinding,
    _aggregate,
    _prefilter,
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
    async def test_uses_three_tools(self) -> None:
        explore_result = ExploreResult(
            filename="src/main.py",
            verified=[
                VerifiedFinding(
                    original_title="SQL injection",
                    status="confirmed",
                    evidence="No sanitization",
                    comment=_make_comment(),
                )
            ],
        )
        ctx = MagicMock()
        ctx.sample = AsyncMock(return_value=MagicMock(result=explore_result))
        gh = MagicMock()
        gh.get_file_contents = AsyncMock(return_value="contents")

        findings = [
            FileFindings(
                filename="src/main.py",
                findings=[_make_finding()],
                file_summary="Query",
            )
        ]
        results = await _verify_findings(gh, ctx, findings, _make_timeline(), "o/r", concurrency=1)

        assert len(results) == 1
        assert len(ctx.sample.call_args.kwargs["tools"]) == 3
        assert ctx.sample.call_args.kwargs["max_tokens"] == 8192

    @pytest.mark.asyncio
    async def test_skips_clean_files(self) -> None:
        ctx = MagicMock()
        ctx.sample = AsyncMock()
        gh = MagicMock()

        findings = [FileFindings(filename="a.py", findings=[], file_summary="Clean")]
        results = await _verify_findings(gh, ctx, findings, _make_timeline(), "o/r", concurrency=1)
        assert results == []
        ctx.sample.assert_not_awaited()


class TestAggregate:
    def test_only_confirmed(self) -> None:
        findings = [
            FileFindings(
                filename="a.py",
                findings=[_make_finding()],
                file_summary="Changes",
            )
        ]
        explore = [
            ExploreResult(
                filename="a.py",
                verified=[
                    VerifiedFinding(
                        original_title="SQL injection",
                        status="confirmed",
                        evidence="Confirmed",
                        comment=_make_comment(),
                    ),
                    VerifiedFinding(
                        original_title="Other",
                        status="disproved",
                        evidence="Handled",
                        comment=None,
                    ),
                ],
            )
        ]
        result = _aggregate(findings, explore, 1, 0, 50)
        assert len(result.comments) == 1

    def test_all_disproved(self) -> None:
        explore = [
            ExploreResult(
                filename="a.py",
                verified=[
                    VerifiedFinding(
                        original_title="False alarm",
                        status="disproved",
                        evidence="OK",
                        comment=None,
                    )
                ],
            )
        ]
        result = _aggregate([], explore, 1, 0, 50)
        assert result.verdict == ReviewState.APPROVED
        assert len(result.comments) == 0


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_four_passes(self) -> None:
        """filter + review + verify = 3 sample calls for 1 file."""
        filter_result = FilterBatchResult(
            chunks=[FilteredChunk(index=0, interest="high", rationale="Logic")]
        )
        file_findings = FileFindings(
            filename="src/main.py",
            findings=[_make_finding()],
            file_summary="Query",
        )
        explore_result = ExploreResult(
            filename="src/main.py",
            verified=[
                VerifiedFinding(
                    original_title="SQL injection",
                    status="confirmed",
                    evidence="Confirmed",
                    comment=_make_comment(),
                )
            ],
        )

        ctx = MagicMock()
        ctx.sample = AsyncMock(
            side_effect=[
                MagicMock(result=filter_result),
                MagicMock(result=file_findings),
                MagicMock(result=explore_result),
            ]
        )

        gh = MagicMock()
        gh.get_timeline = AsyncMock(return_value=_make_timeline())
        gh.get_review_comments_by_file = AsyncMock(return_value={})
        gh.get_prior_review_bodies = AsyncMock(return_value=[])
        gh.get_file_contents = AsyncMock(return_value="contents")

        result = await production_review(gh, ctx, "o/r", 1)

        assert ctx.sample.await_count == 3
        assert len(result.comments) == 1
        assert "verified" in result.summary
