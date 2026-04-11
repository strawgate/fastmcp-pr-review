"""GitHub client wrapper using githubkit for PR data retrieval."""

from __future__ import annotations

import asyncio

from githubkit import GitHub

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


def _parse_owner_repo(repo: str) -> tuple[str, str]:
    """Parse 'owner/repo' string into (owner, repo) tuple."""
    parts = repo.split("/")
    if len(parts) != 2:
        msg = f"Invalid repo format '{repo}', expected 'owner/repo'"
        raise ValueError(msg)
    return parts[0], parts[1]


class GitHubPRClient:
    """Async client for fetching GitHub PR data."""

    def __init__(self, token: str) -> None:
        self._github = GitHub(token)

    async def get_pr_details(self, repo: str, pr_number: int) -> PRDetails:
        owner, repo_name = _parse_owner_repo(repo)
        resp = await self._github.rest.pulls.async_get(owner, repo_name, pr_number)
        pr = resp.parsed_data

        state = PRState.MERGED if pr.merged else PRState(pr.state)

        return PRDetails(
            number=pr.number,
            title=pr.title,
            body=pr.body,
            state=state,
            author=PRAuthor(
                login=pr.user.login if pr.user else "unknown",
                avatar_url=pr.user.avatar_url if pr.user else "",
            ),
            head_ref=pr.head.ref,
            base_ref=pr.base.ref,
            head_sha=pr.head.sha,
            created_at=pr.created_at,
            updated_at=pr.updated_at,
            additions=pr.additions,
            deletions=pr.deletions,
            changed_files=pr.changed_files,
            html_url=pr.html_url,
        )

    async def get_comments(self, repo: str, pr_number: int) -> list[PRComment]:
        owner, repo_name = _parse_owner_repo(repo)
        comments: list[PRComment] = []
        async for comment in self._github.rest.paginate(
            self._github.rest.issues.async_list_comments,
            owner=owner,
            repo=repo_name,
            issue_number=pr_number,
            per_page=100,
        ):
            comments.append(
                PRComment(
                    id=comment.id,
                    author=PRAuthor(
                        login=comment.user.login if comment.user else "unknown",
                        avatar_url=comment.user.avatar_url if comment.user else "",
                    ),
                    body=comment.body or "",
                    created_at=comment.created_at,
                    html_url=comment.html_url,
                )
            )
        return comments

    async def get_reviews(self, repo: str, pr_number: int) -> list[PRReview]:
        owner, repo_name = _parse_owner_repo(repo)
        resp = await self._github.rest.pulls.async_list_reviews(owner, repo_name, pr_number)
        return [
            PRReview(
                id=review.id,
                author=PRAuthor(
                    login=review.user.login if review.user else "unknown",
                    avatar_url=review.user.avatar_url if review.user else "",
                ),
                state=ReviewState(review.state),
                body=review.body or "",
                submitted_at=review.submitted_at,
                commit_id=review.commit_id,
            )
            for review in resp.parsed_data
        ]

    async def get_review_comments(self, repo: str, pr_number: int) -> list[PRReviewComment]:
        owner, repo_name = _parse_owner_repo(repo)
        review_comments: list[PRReviewComment] = []
        async for rc in self._github.rest.paginate(
            self._github.rest.pulls.async_list_review_comments,
            owner=owner,
            repo=repo_name,
            pull_number=pr_number,
            per_page=100,
        ):
            review_comments.append(
                PRReviewComment(
                    id=rc.id,
                    review_id=rc.pull_request_review_id,
                    author=PRAuthor(
                        login=rc.user.login if rc.user else "unknown",
                        avatar_url=rc.user.avatar_url if rc.user else "",
                    ),
                    body=rc.body,
                    path=rc.path,
                    diff_hunk=rc.diff_hunk,
                    line=rc.line,
                    side=rc.side,
                    start_line=rc.start_line,
                    in_reply_to_id=rc.in_reply_to_id,
                    created_at=rc.created_at,
                    html_url=rc.html_url,
                )
            )
        return review_comments

    async def get_commits(self, repo: str, pr_number: int) -> list[PRCommit]:
        owner, repo_name = _parse_owner_repo(repo)
        resp = await self._github.rest.pulls.async_list_commits(
            owner, repo_name, pr_number, per_page=100
        )
        return [
            PRCommit(
                sha=commit.sha,
                message=commit.commit.message,
                author_name=(commit.commit.author.name if commit.commit.author else "unknown"),
                author_date=(commit.commit.author.date if commit.commit.author else None),
            )
            for commit in resp.parsed_data
        ]

    async def get_files(self, repo: str, pr_number: int) -> list[PRFile]:
        owner, repo_name = _parse_owner_repo(repo)
        files: list[PRFile] = []
        async for f in self._github.rest.paginate(
            self._github.rest.pulls.async_list_files,
            owner=owner,
            repo=repo_name,
            pull_number=pr_number,
            per_page=100,
        ):
            files.append(
                PRFile(
                    filename=f.filename,
                    status=f.status,
                    additions=f.additions,
                    deletions=f.deletions,
                    changes=f.changes,
                    patch=f.patch if hasattr(f, "patch") else None,
                )
            )
        return files

    async def get_diff(self, repo: str, pr_number: int) -> str:
        owner, repo_name = _parse_owner_repo(repo)
        resp = await self._github.rest.pulls.async_get(
            owner,
            repo_name,
            pr_number,
            headers={"Accept": "application/vnd.github.diff"},
        )
        return resp.text

    async def get_timeline(self, repo: str, pr_number: int) -> PRTimeline:
        """Fetch all PR data and assemble into a chronological timeline."""
        pr, comments, reviews, review_comments, commits, files = await asyncio.gather(
            self.get_pr_details(repo, pr_number),
            self.get_comments(repo, pr_number),
            self.get_reviews(repo, pr_number),
            self.get_review_comments(repo, pr_number),
            self.get_commits(repo, pr_number),
            self.get_files(repo, pr_number),
        )

        events: list[TimelineEvent] = []

        # PR opened event
        events.append(
            TimelineEvent(
                type=TimelineEventType.PR_OPENED,
                timestamp=pr.created_at,
                author=pr.author,
                body=pr.body or "",
            )
        )

        # Commits
        for commit in commits:
            events.append(
                TimelineEvent(
                    type=TimelineEventType.COMMIT,
                    timestamp=commit.author_date,
                    author=PRAuthor(login=commit.author_name),
                    body=commit.message,
                )
            )

        # Comments
        for comment in comments:
            events.append(
                TimelineEvent(
                    type=TimelineEventType.COMMENT,
                    timestamp=comment.created_at,
                    author=comment.author,
                    body=comment.body,
                )
            )

        # Reviews
        for review in reviews:
            events.append(
                TimelineEvent(
                    type=TimelineEventType.REVIEW,
                    timestamp=review.submitted_at,
                    author=review.author,
                    body=review.body,
                    review_state=review.state,
                )
            )

        # Review comments (inline)
        for rc in review_comments:
            events.append(
                TimelineEvent(
                    type=TimelineEventType.REVIEW_COMMENT,
                    timestamp=rc.created_at,
                    author=rc.author,
                    body=rc.body,
                    path=rc.path,
                    diff_hunk=rc.diff_hunk,
                    line=rc.line,
                )
            )

        # Sort by timestamp (None sorts first)
        events.sort(key=lambda e: (e.timestamp is None, e.timestamp))

        return PRTimeline(pr=pr, events=events, files=files)

    async def get_review_comments_by_file(
        self, repo: str, pr_number: int
    ) -> dict[str, list[PRReviewComment]]:
        """Get existing review comments grouped by file path."""
        comments = await self.get_review_comments(repo, pr_number)
        by_file: dict[str, list[PRReviewComment]] = {}
        for comment in comments:
            by_file.setdefault(comment.path, []).append(comment)
        return by_file

    async def get_prior_review_bodies(self, repo: str, pr_number: int) -> list[str]:
        """Get non-empty body text from prior reviews."""
        reviews = await self.get_reviews(repo, pr_number)
        return [r.body for r in reviews if r.body]

    async def get_file_contents(self, repo: str, filepath: str, ref: str) -> str:
        """Get file contents at a specific git ref."""
        import base64

        owner, repo_name = _parse_owner_repo(repo)
        try:
            resp = await self._github.rest.repos.async_get_content(
                owner, repo_name, filepath, ref=ref
            )
            data = resp.parsed_data
            content = getattr(data, "content", None)
            if isinstance(content, str):
                return base64.b64decode(content).decode("utf-8")
            return ""
        except Exception:
            return f"(unable to read {filepath})"
