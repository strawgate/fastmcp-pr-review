"""Quick runner to test PR reviews with token usage logging."""

import asyncio
import logging
import os
import sys
import time

# Enable debug logging for token usage and pipeline
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-5s %(name)s: %(message)s",
)
# Target just our loggers for DEBUG
for name in [
    "fastmcp_pr_review.v1_simple",
    "fastmcp_pr_review.v2_per_file",
    "fastmcp_pr_review.v3_production",
    "fastmcp.client.sampling.handlers.google_genai",
    "fastmcp.server.sampling.run",
]:
    logging.getLogger(name).setLevel(logging.DEBUG)


async def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    from fastmcp import Context

    from fastmcp_pr_review.server import create_server

    repo = sys.argv[1] if len(sys.argv) > 1 else "strawgate/memagent"
    pr_number = int(sys.argv[2]) if len(sys.argv) > 2 else 1821
    version = sys.argv[3] if len(sys.argv) > 3 else "v1"

    print(f"\n{'='*60}")
    print(f"Reviewing {repo}#{pr_number} with {version}")
    print(f"{'='*60}\n")

    server = create_server(
        github_token=os.environ.get("GITHUB_TOKEN") or "",
        gemini_model="gemini-2.5-flash",
    )

    # Get the internal functions directly
    gh_client = None
    for attr in dir(server):
        obj = getattr(server, attr, None)
        if hasattr(obj, "__wrapped__"):
            break

    # Access the gh client from the server's closure
    from fastmcp_pr_review.github_client import GitHubPRClient
    from fastmcp_pr_review.context import gather_project_context, extract_linked_issues

    token = os.environ.get("GITHUB_TOKEN") or ""
    gh = GitHubPRClient(token)

    # Gather context
    print("Gathering PR context...")
    timeline = await gh.get_timeline(repo, pr_number)
    pr = timeline.pr

    import asyncio as aio
    project_ctx, issues = await aio.gather(
        gather_project_context(gh, repo, pr.head_sha),
        extract_linked_issues(gh, repo, pr.body, pr.head_ref),
    )
    print(f"  Files: {len(timeline.files)}")
    print(f"  Project context: {len(project_ctx)} chars")
    print(f"  Linked issues: {len(issues)}")

    # Create a mock-ish context that delegates to the sampling handler
    from unittest.mock import MagicMock, AsyncMock
    from fastmcp.client.sampling.handlers.google_genai import GoogleGenaiSamplingHandler

    handler = GoogleGenaiSamplingHandler(default_model="gemini-2.5-flash")

    # We need a real Context. Use the server's internal sampling.
    # The simplest way: call through the server's tool functions.
    # But those need a real MCP context. Instead, let's use the
    # sampling implementation directly.
    from fastmcp.server.sampling.run import sample_impl, SamplingResult

    # Build a fake context that routes to the handler
    ctx = MagicMock()
    ctx.fastmcp = MagicMock()
    ctx.fastmcp.sampling_handler = handler
    ctx.fastmcp.sampling_handler_behavior = "always"
    ctx.request_context = MagicMock()
    ctx.request_id = None

    # Mock session to report no client capabilities
    ctx.session = MagicMock()
    ctx.session.check_client_capability = MagicMock(return_value=False)

    # Patch ctx.sample to call sample_impl
    async def patched_sample(**kwargs):
        return await sample_impl(ctx, **kwargs)

    ctx.sample = patched_sample

    t0 = time.monotonic()

    if version == "v1":
        from fastmcp_pr_review.v1_simple import simple_review
        result = await simple_review(
            gh, ctx, repo, pr_number,
            project_context=project_ctx,
            linked_issues=issues,
        )
    elif version == "v2":
        from fastmcp_pr_review.v2_per_file import per_file_review
        result = await per_file_review(
            gh, ctx, repo, pr_number,
            project_context=project_ctx,
            linked_issues=issues,
        )
    elif version == "v3":
        from fastmcp_pr_review.v3_production import production_review
        result = await production_review(
            gh, ctx, repo, pr_number,
            project_context=project_ctx,
            linked_issues=issues,
        )
    else:
        print(f"Unknown version: {version}")
        return

    elapsed = time.monotonic() - t0
    u = handler.usage

    print(f"\n{'='*60}")
    print(f"RESULT ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"Verdict: {result.verdict}")
    print(f"Risk: {result.risk_score}/10 | Health: {result.health_score}/100")
    print(f"Files reviewed: {result.files_reviewed} | Skipped: {result.files_skipped}")
    print(f"Comments: {len(result.comments)}")
    if result.stats:
        print(f"  By severity: {result.stats.by_severity}")
        print(f"  By category: {result.stats.by_category}")

    print(f"\nToken usage ({u['calls']} Gemini calls):")
    print(f"  Prompt:   {u['prompt']:>8,}")
    print(f"  Output:   {u['output']:>8,}")
    print(f"  Thoughts: {u['thoughts']:>8,}")
    print(f"  Cached:   {u['cached']:>8,} ({u['cached']*100//max(u['prompt'],1)}% of prompt)")
    print(f"  Total:    {u['total']:>8,}")

    print(f"\nSummary:\n{result.summary}")
    if result.comments:
        print(f"\n{'─'*60}")
        for c in result.comments:
            print(f"\n[{c.severity.value}] {c.path}:{c.line} — {c.title}")
            print(f"  {c.body[:200]}")
            if c.suggested_code:
                print(f"  Suggestion: {c.suggested_code[:100]}")


if __name__ == "__main__":
    asyncio.run(main())
