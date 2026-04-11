"""Shared Pydantic models for PR data, review output, and scoring.

The v1/v2/v3 pipeline modules define their own stage-specific models
(DiffChunk, FilterBatchResult, FileFindings, ExploreResult, etc.)
inline so each example is self-contained. This module contains only
the types shared across all three implementations.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from enum import StrEnum

from pydantic import BaseModel, Field

# ── PR Data Models (from GitHub API) ───────────────────────────────────────


class PRState(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"


class ReviewState(StrEnum):
    APPROVED = "APPROVED"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    COMMENTED = "COMMENTED"
    DISMISSED = "DISMISSED"
    PENDING = "PENDING"


class PRAuthor(BaseModel):
    login: str
    avatar_url: str = ""


class PRDetails(BaseModel):
    number: int
    title: str
    body: str | None = None
    state: PRState
    author: PRAuthor
    head_ref: str
    base_ref: str
    head_sha: str
    created_at: datetime
    updated_at: datetime
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0
    html_url: str = ""


class PRComment(BaseModel):
    id: int
    author: PRAuthor
    body: str
    created_at: datetime
    html_url: str = ""


class PRReview(BaseModel):
    id: int
    author: PRAuthor
    state: ReviewState
    body: str = ""
    submitted_at: datetime | None = None
    commit_id: str | None = None


class PRReviewComment(BaseModel):
    id: int
    review_id: int | None = None
    author: PRAuthor
    body: str
    path: str
    diff_hunk: str
    line: int | None = None
    side: str | None = None
    start_line: int | None = None
    in_reply_to_id: int | None = None
    created_at: datetime
    html_url: str = ""


class PRCommit(BaseModel):
    sha: str
    message: str
    author_name: str
    author_date: datetime | None = None


class PRFile(BaseModel):
    filename: str
    status: str
    additions: int
    deletions: int
    changes: int
    patch: str | None = None


# ── Timeline ───────────────────────────────────────────────────────────────


class TimelineEventType(StrEnum):
    PR_OPENED = "pr_opened"
    COMMIT = "commit"
    COMMENT = "comment"
    REVIEW = "review"
    REVIEW_COMMENT = "review_comment"


class TimelineEvent(BaseModel):
    type: TimelineEventType
    timestamp: datetime | None = None
    author: PRAuthor
    body: str = ""
    review_state: ReviewState | None = None
    path: str | None = None
    diff_hunk: str | None = None
    line: int | None = None


class PRTimeline(BaseModel):
    pr: PRDetails
    events: list[TimelineEvent]
    files: list[PRFile]


# ── Review Output (shared by all three v* implementations) ─────────────────


class Severity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NITPICK = "nitpick"


class CommentCategory(StrEnum):
    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    LOGIC = "logic"
    ERROR_HANDLING = "error_handling"
    TESTING = "testing"
    MAINTAINABILITY = "maintainability"


class ReviewComment(BaseModel):
    """An individual review comment on a specific file/line."""

    path: str = Field(description="File path relative to repo root")
    line: int | None = Field(default=None, description="Line number in the new file")
    end_line: int | None = Field(default=None, description="End line for multi-line comments")
    severity: Severity
    category: CommentCategory
    title: str = Field(description="Short (<80 char) title for the comment")
    body: str = Field(description="Detailed explanation of the issue")
    why: str = Field(description="Why this matters / what could go wrong")
    suggested_code: str | None = Field(default=None, description="Code suggestion if applicable")
    confidence: int = Field(ge=0, le=100, description="Confidence score 0-100")


class ReviewStats(BaseModel):
    total_comments: int = 0
    by_severity: dict[str, int] = Field(default_factory=dict)
    by_category: dict[str, int] = Field(default_factory=dict)


class PRReviewResult(BaseModel):
    """Final output of any review pipeline. All three versions produce this."""

    verdict: ReviewState = Field(
        description="Overall verdict: APPROVED, CHANGES_REQUESTED, or COMMENTED"
    )
    summary: str = Field(description="High-level summary of the review findings")
    comments: list[ReviewComment] = Field(default_factory=list)
    risk_score: int = Field(ge=1, le=10, description="Risk score from 1 (trivial) to 10 (critical)")
    health_score: int = Field(
        ge=1, le=100, description="Code health score from 1 (critical) to 100 (healthy)"
    )
    files_reviewed: int = 0
    files_skipped: int = 0
    stats: ReviewStats = Field(default_factory=ReviewStats)


# ── Scoring helpers ────────────────────────────────────────────────────────

SEVERITY_WEIGHTS: dict[Severity, float] = {
    Severity.CRITICAL: 4.0,
    Severity.HIGH: 2.0,
    Severity.MEDIUM: 1.0,
    Severity.LOW: 0.25,
    Severity.NITPICK: 0.0,
}

HEALTH_DEDUCTIONS: dict[Severity, int] = {
    Severity.CRITICAL: 25,
    Severity.HIGH: 10,
    Severity.MEDIUM: 3,
    Severity.LOW: 0,
    Severity.NITPICK: 0,
}


def compute_verdict(n_critical: int, n_high: int, n_medium: int) -> ReviewState:
    """Compute review verdict from severity counts."""
    if n_critical > 0 or n_high >= 2:
        return ReviewState.CHANGES_REQUESTED
    if n_high >= 1 or n_medium >= 3:
        return ReviewState.COMMENTED
    return ReviewState.APPROVED


def compute_scores(severity_counts: dict[str, int]) -> tuple[int, int]:
    """Compute (risk_score 1-10, health_score 1-100) from severity counts."""
    raw_risk = sum(severity_counts.get(s.value, 0) * w for s, w in SEVERITY_WEIGHTS.items())
    risk_score = max(1, min(10, round(raw_risk)))

    deduction = sum(severity_counts.get(s.value, 0) * d for s, d in HEALTH_DEDUCTIONS.items())
    health_score = max(1, 100 - deduction)

    return risk_score, health_score
