"""Tests for project context gathering and linked issue extraction."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fastmcp_pr_review.context import (
    _MAX_DOC_CHARS,
    PROJECT_DOC_PATHS,
    extract_linked_issues,
    gather_project_context,
)


def _make_gh(**overrides: object) -> MagicMock:
    gh = MagicMock()
    gh.get_file_contents = AsyncMock(return_value=None)
    gh.get_issue = AsyncMock(return_value=None)
    for k, v in overrides.items():
        setattr(gh, k, v)
    return gh


# ── gather_project_context ─────────────────────────────────────────────────


class TestGatherProjectContext:
    @pytest.mark.asyncio
    async def test_reads_available_docs(self) -> None:
        async def fake_contents(repo: str, path: str, ref: str | None = None) -> str | None:
            if path == "README.md":
                return "# My Project"
            if path == "AGENTS.md":
                return "# Agents"
            return None

        gh = _make_gh(get_file_contents=AsyncMock(side_effect=fake_contents))
        result = await gather_project_context(gh, "o/r", "abc123")

        assert "### README.md" in result
        assert "# My Project" in result
        assert "### AGENTS.md" in result
        assert "# Agents" in result
        # Missing files should not appear
        assert "CONTRIBUTING.md" not in result

    @pytest.mark.asyncio
    async def test_skips_none_results(self) -> None:
        gh = _make_gh(get_file_contents=AsyncMock(return_value=None))
        result = await gather_project_context(gh, "o/r")
        assert result == ""

    @pytest.mark.asyncio
    async def test_skips_exceptions(self) -> None:
        gh = _make_gh(get_file_contents=AsyncMock(side_effect=RuntimeError("boom")))
        result = await gather_project_context(gh, "o/r")
        assert result == ""

    @pytest.mark.asyncio
    async def test_truncates_long_files(self) -> None:
        long_content = "x" * (_MAX_DOC_CHARS + 500)

        gh = _make_gh(get_file_contents=AsyncMock(return_value=long_content))
        result = await gather_project_context(gh, "o/r")

        assert "... (truncated)" in result
        # Each doc section should not exceed _MAX_DOC_CHARS + truncation marker
        for section in result.split("\n\n"):
            if section.startswith("### "):
                body = section.split("\n", 1)[1]
                assert len(body) <= _MAX_DOC_CHARS + len("\n... (truncated)")

    @pytest.mark.asyncio
    async def test_respects_total_cap(self) -> None:
        """Stop adding docs when total exceeds _MAX_PROJECT_CONTEXT."""
        # Each file returns content just under the per-file cap
        content = "a" * (_MAX_DOC_CHARS - 10)

        gh = _make_gh(get_file_contents=AsyncMock(return_value=content))
        result = await gather_project_context(gh, "o/r")

        # With 6 doc paths each ~2000 chars, total cap of 8000 should limit to ~4 files
        sections = [s for s in result.split("\n\n") if s.startswith("### ")]
        assert len(sections) < len(PROJECT_DOC_PATHS)

    @pytest.mark.asyncio
    async def test_ref_passed_to_client(self) -> None:
        gh = _make_gh(get_file_contents=AsyncMock(return_value=None))
        await gather_project_context(gh, "o/r", "my-ref")

        # Every call should pass the ref
        for call in gh.get_file_contents.call_args_list:
            assert call.args == ("o/r",) or len(call.args) >= 2
            # ref is the third positional arg
            if len(call.args) >= 3:
                assert call.args[2] == "my-ref"


# ── extract_linked_issues ──────────────────────────────────────────────────


class TestExtractLinkedIssues:
    @pytest.mark.asyncio
    async def test_parses_issue_refs_from_body(self) -> None:
        gh = _make_gh(get_issue=AsyncMock(return_value=("Bug title", "open", "Bug body")))
        result = await extract_linked_issues(gh, "o/r", "Fixes #42", "main")

        assert len(result) == 1
        assert "**#42: Bug title**" in result[0]
        assert "(open)" in result[0]

    @pytest.mark.asyncio
    async def test_parses_multiple_refs(self) -> None:
        gh = _make_gh(get_issue=AsyncMock(return_value=("Title", "open", "")))
        result = await extract_linked_issues(gh, "o/r", "Closes #1 and fixes #2", "main")

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_parses_from_branch_name(self) -> None:
        gh = _make_gh(get_issue=AsyncMock(return_value=("Title", "open", "")))
        result = await extract_linked_issues(gh, "o/r", None, "fix/123")

        assert len(result) == 1
        gh.get_issue.assert_awaited_once_with("o/r", 123)

    @pytest.mark.asyncio
    async def test_deduplicates_refs(self) -> None:
        """Same issue referenced in body and branch should only fetch once."""
        gh = _make_gh(get_issue=AsyncMock(return_value=("Title", "open", "")))
        result = await extract_linked_issues(gh, "o/r", "Fixes #42", "fix/42")

        assert len(result) == 1
        assert gh.get_issue.await_count == 1

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_refs(self) -> None:
        gh = _make_gh()
        result = await extract_linked_issues(gh, "o/r", "No issues here", "feature-branch")

        assert result == []
        gh.get_issue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_failed_fetches(self) -> None:
        gh = _make_gh(get_issue=AsyncMock(return_value=None))
        result = await extract_linked_issues(gh, "o/r", "#99", "main")

        assert result == []

    @pytest.mark.asyncio
    async def test_handles_none_body(self) -> None:
        gh = _make_gh()
        result = await extract_linked_issues(gh, "o/r", None, "main")
        assert result == []

    @pytest.mark.asyncio
    async def test_branch_patterns(self) -> None:
        """Various branch name patterns that should extract issue numbers."""
        gh = _make_gh(get_issue=AsyncMock(return_value=("T", "open", "")))

        for branch in ["fix/123", "issue-456", "bug-789", "feat/100"]:
            gh.get_issue.reset_mock()
            result = await extract_linked_issues(gh, "o/r", None, branch)
            assert len(result) == 1, f"Failed for branch: {branch}"
