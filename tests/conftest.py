"""Shared fixtures for tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fastmcp_pr_review.models import (
    PRAuthor,
    PRComment,
    PRCommit,
    PRDetails,
    PRFile,
    PRReview,
    PRReviewComment,
    PRState,
    PRTimeline,
    ReviewState,
    TimelineEvent,
    TimelineEventType,
)


@pytest.fixture
def pr_author() -> PRAuthor:
    return PRAuthor(login="octocat", avatar_url="https://example.com/avatar.png")


@pytest.fixture
def pr_details(pr_author: PRAuthor) -> PRDetails:
    return PRDetails(
        number=42,
        title="Add widget feature",
        body="This PR adds a new widget feature.",
        state=PRState.OPEN,
        author=pr_author,
        head_ref="feature/widget",
        base_ref="main",
        head_sha="abc123",
        created_at=datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
        updated_at=datetime(2025, 1, 2, 12, 0, 0, tzinfo=UTC),
        additions=50,
        deletions=10,
        changed_files=3,
        html_url="https://github.com/owner/repo/pull/42",
    )


@pytest.fixture
def pr_comment(pr_author: PRAuthor) -> PRComment:
    return PRComment(
        id=1,
        author=pr_author,
        body="Looks good!",
        created_at=datetime(2025, 1, 1, 13, 0, 0, tzinfo=UTC),
        html_url="https://github.com/owner/repo/pull/42#issuecomment-1",
    )


@pytest.fixture
def pr_review(pr_author: PRAuthor) -> PRReview:
    return PRReview(
        id=100,
        author=pr_author,
        state=ReviewState.APPROVED,
        body="LGTM",
        submitted_at=datetime(2025, 1, 1, 14, 0, 0, tzinfo=UTC),
        commit_id="abc123",
    )


@pytest.fixture
def pr_review_comment(pr_author: PRAuthor) -> PRReviewComment:
    return PRReviewComment(
        id=200,
        review_id=100,
        author=pr_author,
        body="Consider using a constant here.",
        path="src/widget.py",
        diff_hunk="@@ -10,3 +10,5 @@\n+magic_number = 42",
        line=12,
        side="RIGHT",
        created_at=datetime(2025, 1, 1, 14, 30, 0, tzinfo=UTC),
        html_url="https://github.com/owner/repo/pull/42#discussion_r200",
    )


@pytest.fixture
def pr_commit() -> PRCommit:
    return PRCommit(
        sha="abc123",
        message="feat: add widget",
        author_name="octocat",
        author_date=datetime(2025, 1, 1, 11, 0, 0, tzinfo=UTC),
    )


@pytest.fixture
def pr_file() -> PRFile:
    return PRFile(
        filename="src/widget.py",
        status="added",
        additions=50,
        deletions=0,
        changes=50,
        patch="@@ -0,0 +1,50 @@\n+class Widget:\n+    pass",
    )


@pytest.fixture
def pr_timeline(
    pr_details: PRDetails,
    pr_author: PRAuthor,
    pr_commit: PRCommit,
    pr_comment: PRComment,
    pr_review: PRReview,
    pr_review_comment: PRReviewComment,
    pr_file: PRFile,
) -> PRTimeline:
    events = [
        TimelineEvent(
            type=TimelineEventType.PR_OPENED,
            timestamp=pr_details.created_at,
            author=pr_author,
            body=pr_details.body or "",
        ),
        TimelineEvent(
            type=TimelineEventType.COMMIT,
            timestamp=pr_commit.author_date,
            author=PRAuthor(login=pr_commit.author_name),
            body=pr_commit.message,
        ),
        TimelineEvent(
            type=TimelineEventType.COMMENT,
            timestamp=pr_comment.created_at,
            author=pr_author,
            body=pr_comment.body,
        ),
        TimelineEvent(
            type=TimelineEventType.REVIEW,
            timestamp=pr_review.submitted_at,
            author=pr_author,
            body=pr_review.body,
            review_state=pr_review.state,
        ),
        TimelineEvent(
            type=TimelineEventType.REVIEW_COMMENT,
            timestamp=pr_review_comment.created_at,
            author=pr_author,
            body=pr_review_comment.body,
            path=pr_review_comment.path,
            diff_hunk=pr_review_comment.diff_hunk,
            line=pr_review_comment.line,
        ),
    ]
    return PRTimeline(pr=pr_details, events=events, files=[pr_file])
