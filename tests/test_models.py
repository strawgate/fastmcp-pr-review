"""Tests for shared Pydantic models and scoring helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from fastmcp_pr_review.models import (
    CommentCategory,
    PRAuthor,
    PRDetails,
    PRFile,
    PRReviewResult,
    PRState,
    PRTimeline,
    ReviewComment,
    ReviewState,
    Severity,
    TimelineEvent,
    TimelineEventType,
    compute_scores,
    compute_verdict,
)


class TestPRState:
    def test_values(self) -> None:
        assert PRState.OPEN == "open"
        assert PRState.CLOSED == "closed"
        assert PRState.MERGED == "merged"


class TestReviewState:
    def test_values(self) -> None:
        assert ReviewState.APPROVED == "APPROVED"
        assert ReviewState.CHANGES_REQUESTED == "CHANGES_REQUESTED"


class TestPRDetails:
    def test_minimal(self) -> None:
        pr = PRDetails(
            number=1,
            title="Test",
            state=PRState.OPEN,
            author=PRAuthor(login="user"),
            head_ref="feature",
            base_ref="main",
            head_sha="abc",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            updated_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        assert pr.number == 1
        assert pr.body is None
        assert pr.additions == 0

    def test_full(self, pr_details: PRDetails) -> None:
        assert pr_details.number == 42
        assert pr_details.additions == 50


class TestPRFile:
    def test_with_patch(self, pr_file: PRFile) -> None:
        assert pr_file.filename == "src/widget.py"
        assert pr_file.patch is not None

    def test_without_patch(self) -> None:
        f = PRFile(filename="big.bin", status="added", additions=0, deletions=0, changes=0)
        assert f.patch is None


class TestTimelineEvent:
    def test_pr_opened(self) -> None:
        event = TimelineEvent(
            type=TimelineEventType.PR_OPENED,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            author=PRAuthor(login="user"),
        )
        assert event.review_state is None

    def test_review_event(self) -> None:
        event = TimelineEvent(
            type=TimelineEventType.REVIEW,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            author=PRAuthor(login="reviewer"),
            review_state=ReviewState.APPROVED,
        )
        assert event.review_state == ReviewState.APPROVED


class TestPRTimeline:
    def test_timeline(self, pr_timeline: PRTimeline) -> None:
        assert pr_timeline.pr.number == 42
        assert len(pr_timeline.events) == 5

    def test_empty_events(self, pr_details: PRDetails) -> None:
        timeline = PRTimeline(pr=pr_details, events=[], files=[])
        assert len(timeline.events) == 0


class TestSeverity:
    def test_five_levels(self) -> None:
        assert len(Severity) == 5
        assert Severity.CRITICAL == "critical"
        assert Severity.NITPICK == "nitpick"


class TestCommentCategory:
    def test_all_categories(self) -> None:
        assert CommentCategory.BUG == "bug"
        assert len(CommentCategory) == 9


class TestReviewComment:
    def test_full(self) -> None:
        c = ReviewComment(
            path="src/db.py",
            line=42,
            end_line=48,
            severity=Severity.CRITICAL,
            category=CommentCategory.SECURITY,
            title="SQL injection",
            body="User input in query",
            why="Attacker could execute SQL",
            confidence=95,
        )
        assert c.confidence == 95

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ReviewComment(
                path="x.py",
                severity=Severity.LOW,
                category=CommentCategory.STYLE,
                title="t",
                body="b",
                why="w",
                confidence=-1,
            )
        with pytest.raises(ValidationError):
            ReviewComment(
                path="x.py",
                severity=Severity.LOW,
                category=CommentCategory.STYLE,
                title="t",
                body="b",
                why="w",
                confidence=101,
            )


class TestPRReviewResult:
    def test_approved(self) -> None:
        r = PRReviewResult(
            verdict=ReviewState.APPROVED, summary="Clean", risk_score=1, health_score=100
        )
        assert r.comments == []
        assert r.files_skipped == 0

    def test_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            PRReviewResult(verdict=ReviewState.APPROVED, summary="", risk_score=0, health_score=100)
        with pytest.raises(ValidationError):
            PRReviewResult(verdict=ReviewState.APPROVED, summary="", risk_score=1, health_score=0)


class TestComputeVerdict:
    def test_critical_requests_changes(self) -> None:
        assert compute_verdict(1, 0, 0) == ReviewState.CHANGES_REQUESTED

    def test_two_high_requests_changes(self) -> None:
        assert compute_verdict(0, 2, 0) == ReviewState.CHANGES_REQUESTED

    def test_one_high_comments(self) -> None:
        assert compute_verdict(0, 1, 0) == ReviewState.COMMENTED

    def test_three_medium_comments(self) -> None:
        assert compute_verdict(0, 0, 3) == ReviewState.COMMENTED

    def test_low_only_approves(self) -> None:
        assert compute_verdict(0, 0, 2) == ReviewState.APPROVED

    def test_no_comments_approves(self) -> None:
        assert compute_verdict(0, 0, 0) == ReviewState.APPROVED


class TestComputeScores:
    def test_no_issues(self) -> None:
        risk, health = compute_scores({})
        assert risk == 1
        assert health == 100

    def test_one_critical(self) -> None:
        risk, health = compute_scores({"critical": 1})
        assert risk == 4
        assert health == 75

    def test_mixed(self) -> None:
        risk, health = compute_scores({"critical": 1, "high": 1, "medium": 1})
        assert risk == 7  # 4 + 2 + 1
        assert health == 62  # 100 - 25 - 10 - 3
