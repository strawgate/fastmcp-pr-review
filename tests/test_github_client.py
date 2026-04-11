"""Tests for the GitHub client wrapper."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from fastmcp_pr_review.github_client import GitHubPRClient, _parse_owner_repo
from fastmcp_pr_review.models import PRState, ReviewState


class TestParseOwnerRepo:
    def test_valid(self) -> None:
        assert _parse_owner_repo("owner/repo") == ("owner", "repo")

    def test_invalid_no_slash(self) -> None:
        with pytest.raises(ValueError, match="Invalid repo format"):
            _parse_owner_repo("noslash")

    def test_invalid_too_many_slashes(self) -> None:
        with pytest.raises(ValueError, match="Invalid repo format"):
            _parse_owner_repo("a/b/c")


def _make_pr_response() -> SimpleNamespace:
    """Create a mock PR response matching githubkit's structure."""
    return SimpleNamespace(
        parsed_data=SimpleNamespace(
            number=42,
            title="Test PR",
            body="PR body",
            state="open",
            merged=False,
            user=SimpleNamespace(login="octocat", avatar_url="https://example.com/avatar.png"),
            head=SimpleNamespace(ref="feature", sha="abc123", repo=None),
            base=SimpleNamespace(ref="main", sha="def456", repo=None),
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            updated_at=datetime(2025, 1, 2, tzinfo=UTC),
            additions=10,
            deletions=5,
            changed_files=2,
            html_url="https://github.com/owner/repo/pull/42",
        )
    )


def _make_merged_pr_response() -> SimpleNamespace:
    """Create a mock merged PR response."""
    resp = _make_pr_response()
    resp.parsed_data.state = "closed"
    resp.parsed_data.merged = True
    return resp


def _make_comments_response() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            id=1,
            user=SimpleNamespace(login="reviewer", avatar_url=""),
            body="Nice work!",
            created_at=datetime(2025, 1, 1, 13, 0, tzinfo=UTC),
            html_url="https://github.com/owner/repo/pull/42#issuecomment-1",
        )
    ]


def _make_reviews_response() -> SimpleNamespace:
    return SimpleNamespace(
        parsed_data=[
            SimpleNamespace(
                id=100,
                user=SimpleNamespace(login="reviewer", avatar_url=""),
                state="APPROVED",
                body="LGTM",
                submitted_at=datetime(2025, 1, 1, 14, 0, tzinfo=UTC),
                commit_id="abc123",
            )
        ]
    )


def _make_review_comments_response() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            id=200,
            pull_request_review_id=100,
            user=SimpleNamespace(login="reviewer", avatar_url=""),
            body="Use a constant",
            path="src/main.py",
            diff_hunk="@@ -1,3 +1,5 @@",
            line=10,
            side="RIGHT",
            start_line=None,
            in_reply_to_id=None,
            created_at=datetime(2025, 1, 1, 14, 30, tzinfo=UTC),
            html_url="https://github.com/owner/repo/pull/42#discussion_r200",
        )
    ]


def _make_commits_response() -> SimpleNamespace:
    return SimpleNamespace(
        parsed_data=[
            SimpleNamespace(
                sha="abc123",
                commit=SimpleNamespace(
                    message="feat: add feature",
                    author=SimpleNamespace(
                        name="octocat",
                        date=datetime(2025, 1, 1, 11, 0, tzinfo=UTC),
                    ),
                ),
                author=SimpleNamespace(login="octocat"),
                html_url="https://github.com/owner/repo/commit/abc123",
            )
        ]
    )


def _make_files_response() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            filename="src/main.py",
            status="modified",
            additions=10,
            deletions=5,
            changes=15,
            patch="@@ -1,3 +1,5 @@\n+new_line",
        )
    ]


def _make_diff_response() -> SimpleNamespace:
    return SimpleNamespace(text="diff --git a/src/main.py b/src/main.py\n+new_line")


class TestGetPRDetails:
    @pytest.mark.asyncio
    async def test_open_pr(self) -> None:
        client = GitHubPRClient("fake-token")
        with patch.object(
            client._github.rest.pulls, "async_get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = _make_pr_response()
            pr = await client.get_pr_details("owner/repo", 42)

        assert pr.number == 42
        assert pr.title == "Test PR"
        assert pr.state == PRState.OPEN
        assert pr.author.login == "octocat"
        mock_get.assert_awaited_once_with("owner", "repo", 42)

    @pytest.mark.asyncio
    async def test_merged_pr(self) -> None:
        client = GitHubPRClient("fake-token")
        with patch.object(
            client._github.rest.pulls, "async_get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = _make_merged_pr_response()
            pr = await client.get_pr_details("owner/repo", 42)

        assert pr.state == PRState.MERGED


class TestGetComments:
    @pytest.mark.asyncio
    async def test_paginated_comments(self) -> None:
        client = GitHubPRClient("fake-token")

        async def fake_paginate(method, *, owner, repo, issue_number, per_page):
            for item in _make_comments_response():
                yield item

        with patch.object(client._github.rest, "paginate", side_effect=fake_paginate):
            comments = await client.get_comments("owner/repo", 42)

        assert len(comments) == 1
        assert comments[0].body == "Nice work!"
        assert comments[0].author.login == "reviewer"


class TestGetReviews:
    @pytest.mark.asyncio
    async def test_list_reviews(self) -> None:
        client = GitHubPRClient("fake-token")
        with patch.object(
            client._github.rest.pulls, "async_list_reviews", new_callable=AsyncMock
        ) as mock_list:
            mock_list.return_value = _make_reviews_response()
            reviews = await client.get_reviews("owner/repo", 42)

        assert len(reviews) == 1
        assert reviews[0].state == ReviewState.APPROVED
        assert reviews[0].body == "LGTM"


class TestGetReviewComments:
    @pytest.mark.asyncio
    async def test_paginated_review_comments(self) -> None:
        client = GitHubPRClient("fake-token")

        async def fake_paginate(method, *, owner, repo, pull_number, per_page):
            for item in _make_review_comments_response():
                yield item

        with patch.object(client._github.rest, "paginate", side_effect=fake_paginate):
            review_comments = await client.get_review_comments("owner/repo", 42)

        assert len(review_comments) == 1
        assert review_comments[0].path == "src/main.py"
        assert review_comments[0].line == 10


class TestGetCommits:
    @pytest.mark.asyncio
    async def test_list_commits(self) -> None:
        client = GitHubPRClient("fake-token")
        with patch.object(
            client._github.rest.pulls, "async_list_commits", new_callable=AsyncMock
        ) as mock_list:
            mock_list.return_value = _make_commits_response()
            commits = await client.get_commits("owner/repo", 42)

        assert len(commits) == 1
        assert commits[0].sha == "abc123"
        assert commits[0].message == "feat: add feature"


class TestGetFiles:
    @pytest.mark.asyncio
    async def test_paginated_files(self) -> None:
        client = GitHubPRClient("fake-token")

        async def fake_paginate(method, *, owner, repo, pull_number, per_page):
            for item in _make_files_response():
                yield item

        with patch.object(client._github.rest, "paginate", side_effect=fake_paginate):
            files = await client.get_files("owner/repo", 42)

        assert len(files) == 1
        assert files[0].filename == "src/main.py"
        assert files[0].patch is not None


class TestGetDiff:
    @pytest.mark.asyncio
    async def test_raw_diff(self) -> None:
        client = GitHubPRClient("fake-token")
        with patch.object(
            client._github.rest.pulls, "async_get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = _make_diff_response()
            diff = await client.get_diff("owner/repo", 42)

        assert "diff --git" in diff
        mock_get.assert_awaited_once_with(
            "owner",
            "repo",
            42,
            headers={"Accept": "application/vnd.github.diff"},
        )


class TestGetTimeline:
    @pytest.mark.asyncio
    async def test_assembles_timeline(self) -> None:
        client = GitHubPRClient("fake-token")

        with (
            patch.object(
                client._github.rest.pulls, "async_get", new_callable=AsyncMock
            ) as mock_pr_get,
            patch.object(
                client._github.rest.pulls, "async_list_reviews", new_callable=AsyncMock
            ) as mock_reviews,
            patch.object(
                client._github.rest.pulls, "async_list_commits", new_callable=AsyncMock
            ) as mock_commits,
        ):
            mock_pr_get.return_value = _make_pr_response()
            mock_reviews.return_value = _make_reviews_response()
            mock_commits.return_value = _make_commits_response()

            async def fake_paginate_comments(method, *, owner, repo, issue_number, per_page):
                for item in _make_comments_response():
                    yield item

            async def fake_paginate_review_comments(method, *, owner, repo, pull_number, per_page):
                for item in _make_review_comments_response():
                    yield item

            async def fake_paginate_files(method, *, owner, repo, pull_number, per_page):
                for item in _make_files_response():
                    yield item

            async def route_paginate(method, **kwargs):
                if "issue_number" in kwargs:
                    async for item in fake_paginate_comments(method, **kwargs):
                        yield item
                elif method == client._github.rest.pulls.async_list_review_comments:
                    async for item in fake_paginate_review_comments(method, **kwargs):
                        yield item
                else:
                    async for item in fake_paginate_files(method, **kwargs):
                        yield item

            with patch.object(client._github.rest, "paginate", side_effect=route_paginate):
                timeline = await client.get_timeline("owner/repo", 42)

        assert timeline.pr.number == 42
        assert len(timeline.events) == 5  # opened + commit + comment + review + review_comment
        assert len(timeline.files) == 1

        # Events should be sorted by timestamp
        types = [e.type for e in timeline.events]
        assert types[0] == "pr_opened"  # 00:00
        assert types[1] == "commit"  # 11:00
        assert types[2] == "comment"  # 13:00
        assert types[3] == "review"  # 14:00
        assert types[4] == "review_comment"  # 14:30


class TestGetReviewCommentsByFile:
    @pytest.mark.asyncio
    async def test_groups_by_path(self) -> None:
        client = GitHubPRClient("fake-token")

        async def fake_paginate(method, *, owner, repo, pull_number, per_page):
            items = [
                SimpleNamespace(
                    id=1,
                    pull_request_review_id=100,
                    user=SimpleNamespace(login="r", avatar_url=""),
                    body="Fix A",
                    path="src/a.py",
                    diff_hunk="@@",
                    line=10,
                    side="RIGHT",
                    start_line=None,
                    in_reply_to_id=None,
                    created_at=datetime(2025, 1, 1, tzinfo=UTC),
                    html_url="",
                ),
                SimpleNamespace(
                    id=2,
                    pull_request_review_id=100,
                    user=SimpleNamespace(login="r", avatar_url=""),
                    body="Fix B",
                    path="src/b.py",
                    diff_hunk="@@",
                    line=20,
                    side="RIGHT",
                    start_line=None,
                    in_reply_to_id=None,
                    created_at=datetime(2025, 1, 1, tzinfo=UTC),
                    html_url="",
                ),
                SimpleNamespace(
                    id=3,
                    pull_request_review_id=100,
                    user=SimpleNamespace(login="r", avatar_url=""),
                    body="Fix A2",
                    path="src/a.py",
                    diff_hunk="@@",
                    line=30,
                    side="RIGHT",
                    start_line=None,
                    in_reply_to_id=None,
                    created_at=datetime(2025, 1, 1, tzinfo=UTC),
                    html_url="",
                ),
            ]
            for item in items:
                yield item

        with patch.object(client._github.rest, "paginate", side_effect=fake_paginate):
            by_file = await client.get_review_comments_by_file("owner/repo", 42)

        assert len(by_file) == 2
        assert len(by_file["src/a.py"]) == 2
        assert len(by_file["src/b.py"]) == 1


class TestGetPriorReviewBodies:
    @pytest.mark.asyncio
    async def test_filters_empty_bodies(self) -> None:
        client = GitHubPRClient("fake-token")
        with patch.object(
            client._github.rest.pulls, "async_list_reviews", new_callable=AsyncMock
        ) as mock_list:
            mock_list.return_value = SimpleNamespace(
                parsed_data=[
                    SimpleNamespace(
                        id=1,
                        user=SimpleNamespace(login="r", avatar_url=""),
                        state="APPROVED",
                        body="LGTM",
                        submitted_at=datetime(2025, 1, 1, tzinfo=UTC),
                        commit_id="abc",
                    ),
                    SimpleNamespace(
                        id=2,
                        user=SimpleNamespace(login="r", avatar_url=""),
                        state="COMMENTED",
                        body="",
                        submitted_at=datetime(2025, 1, 1, tzinfo=UTC),
                        commit_id="abc",
                    ),
                    SimpleNamespace(
                        id=3,
                        user=SimpleNamespace(login="r", avatar_url=""),
                        state="CHANGES_REQUESTED",
                        body="Fix the bug",
                        submitted_at=datetime(2025, 1, 1, tzinfo=UTC),
                        commit_id="abc",
                    ),
                ]
            )
            bodies = await client.get_prior_review_bodies("owner/repo", 42)

        assert bodies == ["LGTM", "Fix the bug"]
