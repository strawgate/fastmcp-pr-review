"""Project context gathering — shared by all review pipelines.

Reads well-known project docs (README, AGENTS.md, etc.) and extracts
linked issue references from PR body/branch name. These provide the
LLM with project understanding before it reviews any code.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastmcp_pr_review.github_client import GitHubPRClient

logger = logging.getLogger(__name__)

# Well-known files to read for project context, in priority order.
PROJECT_DOC_PATHS = [
    "README.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "CODE_STYLE.md",
    ".coderabbit.yaml",
    ".github/copilot-instructions.md",
]

_MAX_DOC_CHARS = 2000  # per file
_MAX_PROJECT_CONTEXT = 8000  # total cap

_ISSUE_REF_PATTERN = re.compile(
    r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)?\s*#(\d+)",
    re.IGNORECASE,
)


async def gather_project_context(
    gh: GitHubPRClient,
    repo: str,
    ref: str | None = None,
) -> str:
    """Read well-known project docs to understand the codebase.

    Tries each path in PROJECT_DOC_PATHS. Files that don't exist are
    silently skipped. Each file is truncated to _MAX_DOC_CHARS.
    """
    results = await asyncio.gather(
        *(gh.get_file_contents(repo, path, ref) for path in PROJECT_DOC_PATHS),
        return_exceptions=True,
    )

    docs: list[str] = []
    total = 0

    for path, result in zip(PROJECT_DOC_PATHS, results, strict=True):
        if isinstance(result, Exception) or result is None:
            continue
        content = str(result)

        if len(content) > _MAX_DOC_CHARS:
            content = content[:_MAX_DOC_CHARS] + "\n... (truncated)"

        if total + len(content) > _MAX_PROJECT_CONTEXT:
            break

        docs.append(f"### {path}\n{content}")
        total += len(content)

    if docs:
        logger.info("Read %d project docs (%d chars)", len(docs), total)

    return "\n\n".join(docs)


async def extract_linked_issues(
    gh: GitHubPRClient,
    repo: str,
    pr_body: str | None,
    branch_name: str,
) -> list[str]:
    """Extract and fetch linked GitHub issues from PR body and branch.

    Parses #123, fixes #456, closes #789 from the PR body.
    Returns formatted issue summaries.
    """
    refs: set[int] = set()

    if pr_body:
        for match in _ISSUE_REF_PATTERN.finditer(pr_body):
            refs.add(int(match.group(1)))

    # Parse from branch name (e.g. "fix/123", "issue-456")
    for m in re.finditer(r"(?:issue|fix|bug|feat)[/-](\d+)", branch_name, re.IGNORECASE):
        refs.add(int(m.group(1)))

    if not refs:
        return []

    async def fetch_issue(num: int) -> str | None:
        try:
            result = await gh.get_issue(repo, num)
            if result is None:
                return None
            title, state, body = result
            return f"**#{num}: {title}** ({state})\n{body}"
        except Exception:
            return None

    results = await asyncio.gather(*(fetch_issue(n) for n in sorted(refs)))
    issues = [r for r in results if r is not None]

    if issues:
        logger.info("Found %d linked issues: %s", len(issues), sorted(refs))

    return issues
