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
    DiffChunk,
    FilterBatchResult,
    FilteredChunk,
    PotentialFinding,
    ReviewDone,
    ThoroughReview,
    VerifyComplete,
    _make_batches,
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
        ctx.sample = AsyncMock(return_value=MagicMock(result=VerifyComplete(summary="Done")))
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
        ctx.sample = AsyncMock(return_value=MagicMock(result=ReviewDone(summary="Clean")))
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
        ctx.sample = AsyncMock(return_value=MagicMock(result=ReviewDone(summary="All clean")))
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


class TestBuildReviewMessage:
    """Tests for the pure _build_review_message static method."""

    def test_includes_title_and_refs(self) -> None:
        inp = ReviewInput(
            files=[],
            title="Fix bug",
            description="Fixes a bug",
            author="dev",
            pr_number=1,
            head_ref="feat",
            base_ref="main",
            additions=10,
            deletions=2,
            changed_files=1,
            commits=[],
        )
        batch = [
            DiffChunk(
                index=0,
                filename="a.py",
                status="modified",
                additions=10,
                deletions=2,
                patch="+x",
            )
        ]
        sections = [ThoroughReview._format_file_section(batch[0], {})]
        msg = ThoroughReview._build_review_message(batch, inp, sections, "balanced")
        assert "Fix bug" in msg
        assert "feat -> main" in msg
        assert "Intensity: balanced" in msg

    def test_includes_description(self) -> None:
        inp = ReviewInput(
            files=[],
            title="T",
            description="This is the PR body",
            author="dev",
            pr_number=1,
            head_ref="f",
            base_ref="m",
            additions=1,
            deletions=0,
            changed_files=1,
            commits=[],
        )
        msg = ThoroughReview._build_review_message([], inp, [], "balanced")
        assert "Description: This is the PR body" in msg

    def test_prior_reviews_section_absent_when_empty(self) -> None:
        inp = ReviewInput(
            files=[],
            title="T",
            description="",
            author="dev",
            pr_number=1,
            head_ref="f",
            base_ref="m",
            additions=1,
            deletions=0,
            changed_files=1,
            commits=[],
        )
        msg = ThoroughReview._build_review_message([], inp, [], "balanced")
        assert "Prior reviews" not in msg

    def test_prior_reviews_section_present(self) -> None:
        inp = ReviewInput(
            files=[],
            title="T",
            description="",
            author="dev",
            pr_number=1,
            head_ref="f",
            base_ref="m",
            additions=1,
            deletions=0,
            changed_files=1,
            prior_reviews=["LGTM", "Nit: typo"],
            commits=[],
        )
        msg = ThoroughReview._build_review_message([], inp, [], "balanced")
        assert "Prior reviews" in msg
        assert "LGTM" in msg
        assert "Nit: typo" in msg

    def test_truncates_prior_reviews_at_300_chars(self) -> None:
        long_body = "x" * 400
        inp = ReviewInput(
            files=[],
            title="T",
            description="",
            author="dev",
            pr_number=1,
            head_ref="f",
            base_ref="m",
            additions=1,
            deletions=0,
            changed_files=1,
            prior_reviews=[long_body],
            commits=[],
        )
        msg = ThoroughReview._build_review_message([], inp, [], "balanced")
        assert "Prior reviews" in msg
        assert "x" * 300 in msg
        assert "x" * 400 not in msg

    def test_commits_section_absent_when_empty(self) -> None:
        inp = ReviewInput(
            files=[],
            title="T",
            description="",
            author="dev",
            pr_number=1,
            head_ref="f",
            base_ref="m",
            additions=1,
            deletions=0,
            changed_files=1,
            commits=[],
        )
        msg = ThoroughReview._build_review_message([], inp, [], "balanced")
        assert "commits" not in msg

    def test_commits_section_with_messages(self) -> None:
        from fastmcp_pr_review.models import PRCommit

        inp = ReviewInput(
            files=[],
            title="T",
            description="",
            author="dev",
            pr_number=1,
            head_ref="f",
            base_ref="m",
            additions=1,
            deletions=0,
            changed_files=1,
            commits=[
                PRCommit(
                    sha="abc",
                    message="feat: add feature",
                    author_name="dev",
                    author_date=datetime(2025, 1, 1, tzinfo=UTC),
                ),
                PRCommit(
                    sha="def",
                    message="fix: broken thing",
                    author_name="dev",
                    author_date=datetime(2025, 1, 2, tzinfo=UTC),
                ),
            ],
        )
        msg = ThoroughReview._build_review_message([], inp, [], "balanced")
        assert "commits" in msg
        assert "feat: add feature" in msg
        assert "fix: broken thing" in msg

    def test_project_context_tag_present(self) -> None:
        inp = ReviewInput(
            files=[],
            title="T",
            description="",
            author="dev",
            pr_number=1,
            head_ref="f",
            base_ref="m",
            additions=1,
            deletions=0,
            changed_files=1,
            project_context="README says use X pattern",
            commits=[],
        )
        msg = ThoroughReview._build_review_message([], inp, [], "balanced")
        assert "<project_context>" in msg
        assert "README says use X pattern" in msg

    def test_linked_issues_tag_present(self) -> None:
        inp = ReviewInput(
            files=[],
            title="T",
            description="",
            author="dev",
            pr_number=1,
            head_ref="f",
            base_ref="m",
            additions=1,
            deletions=0,
            changed_files=1,
            linked_issues=["Fixes #123"],
            commits=[],
        )
        msg = ThoroughReview._build_review_message([], inp, [], "balanced")
        assert "<linked_issues>" in msg
        assert "Fixes #123" in msg

    def test_file_sections_appended(self) -> None:
        inp = ReviewInput(
            files=[],
            title="T",
            description="",
            author="dev",
            pr_number=1,
            head_ref="f",
            base_ref="m",
            additions=1,
            deletions=0,
            changed_files=1,
            commits=[],
        )
        batch = [
            DiffChunk(
                index=0,
                filename="a.py",
                status="modified",
                additions=3,
                deletions=1,
                patch="+a\n-b",
            )
        ]
        sections = [ThoroughReview._format_file_section(batch[0], {})]
        msg = ThoroughReview._build_review_message(batch, inp, sections, "balanced")
        assert "a.py" in msg
        assert "<file_diff>" in msg

    def test_focus_areas_line(self) -> None:
        inp = ReviewInput(
            files=[],
            title="T",
            description="",
            author="dev",
            pr_number=1,
            head_ref="f",
            base_ref="m",
            additions=1,
            deletions=0,
            changed_files=1,
            focus_areas="security",
            commits=[],
        )
        msg = ThoroughReview._build_review_message([], inp, [], "balanced")
        assert "Focus: security" in msg


class TestBuildVerifyMessage:
    """Tests for the pure _build_verify_message static method."""

    def test_empty_findings(self) -> None:
        inp = ReviewInput(
            files=[],
            title="T",
            description="",
            author="dev",
            pr_number=1,
            head_ref="f",
            base_ref="m",
            additions=1,
            deletions=0,
            changed_files=1,
            commits=[],
        )
        msg = ThoroughReview._build_verify_message(inp, [])
        assert "T" in msg
        assert "dev" in msg

    def test_findings_serialized(self) -> None:
        inp = ReviewInput(
            files=[],
            title="T",
            description="",
            author="dev",
            pr_number=1,
            head_ref="f",
            base_ref="m",
            additions=1,
            deletions=0,
            changed_files=1,
            commits=[],
        )
        finding = PotentialFinding(
            path="src/main.py",
            line=10,
            severity=Severity.HIGH,
            category=CommentCategory.BUG,
            title="SQL injection",
            body="User input in query",
            why="Could execute SQL",
            confidence=80,
            verification_needs="Check sanitization",
        )
        msg = ThoroughReview._build_verify_message(inp, [finding])
        assert "src/main.py:10" in msg
        assert "SQL injection" in msg
        assert "Verify: Check sanitization" in msg
        assert 'index="0"' in msg

    def test_multiple_findings_indexed(self) -> None:
        inp = ReviewInput(
            files=[],
            title="T",
            description="",
            author="dev",
            pr_number=1,
            head_ref="f",
            base_ref="m",
            additions=1,
            deletions=0,
            changed_files=1,
            commits=[],
        )
        f1 = PotentialFinding(
            path="a.py",
            line=1,
            severity=Severity.HIGH,
            category=CommentCategory.BUG,
            title="Bug1",
            body="",
            why="",
            confidence=80,
            verification_needs="v1",
        )
        f2 = PotentialFinding(
            path="b.py",
            line=2,
            severity=Severity.MEDIUM,
            category=CommentCategory.BUG,
            title="Bug2",
            body="",
            why="",
            confidence=75,
            verification_needs="v2",
        )
        msg = ThoroughReview._build_verify_message(inp, [f1, f2])
        assert 'index="0"' in msg
        assert 'index="1"' in msg
        assert "Bug1" in msg
        assert "Bug2" in msg


class TestMakeBatches:
    """Tests for the _make_batches batching helper."""

    def _chunk(self, index: int, patch_size: int = 10) -> DiffChunk:
        return DiffChunk(
            index=index,
            filename=f"f{index}.py",
            status="modified",
            additions=1,
            deletions=0,
            patch="x" * patch_size,
        )

    def test_empty_list(self) -> None:
        assert _make_batches([]) == []

    def test_single_small_item(self) -> None:
        batches = _make_batches([self._chunk(0)])
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_multiple_items_under_limits(self) -> None:
        items = [self._chunk(i, patch_size=10) for i in range(3)]
        batches = _make_batches(items, max_items=10, max_bytes=1000)
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_exceeding_max_items(self) -> None:
        items = [self._chunk(i, patch_size=5) for i in range(5)]
        batches = _make_batches(items, max_items=2, max_bytes=10_000)
        assert len(batches) == 3  # [0,1], [2,3], [4]
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    def test_single_oversized_item(self) -> None:
        big = self._chunk(0, patch_size=500)
        batches = _make_batches([big], max_items=10, max_bytes=100)
        assert len(batches) == 1
        assert len(batches[0]) == 1
        assert batches[0][0].index == 0

    def test_mix_normal_and_oversized(self) -> None:
        items = [
            self._chunk(0, patch_size=10),
            self._chunk(1, patch_size=500),  # oversized
            self._chunk(2, patch_size=10),
        ]
        batches = _make_batches(items, max_items=10, max_bytes=100)
        # Batch 1: [chunk0], flushed before oversized
        # Batch 2: [chunk1] (oversized, own batch)
        # Batch 3: [chunk2]
        assert len(batches) == 3
        assert batches[0][0].index == 0
        assert batches[1][0].index == 1
        assert batches[2][0].index == 2

    def test_boundary_exactly_at_max_items(self) -> None:
        items = [self._chunk(i, patch_size=5) for i in range(3)]
        batches = _make_batches(items, max_items=3, max_bytes=10_000)
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_one_more_than_max_items(self) -> None:
        items = [self._chunk(i, patch_size=5) for i in range(4)]
        batches = _make_batches(items, max_items=3, max_bytes=10_000)
        assert len(batches) == 2
        assert len(batches[0]) == 3
        assert len(batches[1]) == 1


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
