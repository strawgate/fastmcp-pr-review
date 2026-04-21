"""Tests for project context and linked issue extraction."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from fastmcp_pr_review.context import extract_linked_issues


@pytest.mark.asyncio
async def test_extract_linked_issues_logs_fetch_failures(caplog: pytest.LogCaptureFixture) -> None:
    gh = MagicMock()
    gh._github.rest.issues.async_get = AsyncMock()

    with caplog.at_level(logging.WARNING):
        issues = await extract_linked_issues(gh, "badrepo", "Fixes #12", "feat/branch")

    assert issues == []
    assert "Failed to fetch linked issue #12 for badrepo" in caplog.text
    gh._github.rest.issues.async_get.assert_not_awaited()
