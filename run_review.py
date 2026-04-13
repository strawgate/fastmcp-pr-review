"""Run PR reviews through the real MCP server with full OTEL tracing."""

import asyncio
import json
import logging
import sys
import time

import logfire

logfire.configure(inspect_arguments=False)
logfire.instrument_mcp()
logfire.instrument_google_genai()

# Enable debug logging for token usage and pipeline
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-5s %(name)s: %(message)s",
)
for name in [
    "fastmcp_pr_review.v1_simple",
    "fastmcp_pr_review.v2_per_file",
    "fastmcp_pr_review.v3_production",
    "fastmcp.client.sampling.handlers.google_genai",
    "fastmcp.server.sampling.run",
]:
    logging.getLogger(name).setLevel(logging.DEBUG)

# Map CLI version names to MCP tool names
TOOL_MAP = {
    "v1": "review_pr_simple",
    "v2": "review_pr",
    "v3": "review_pr_deep",
}


async def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    from fastmcp import Client

    from fastmcp_pr_review.server import create_server

    repo = sys.argv[1] if len(sys.argv) > 1 else "strawgate/memagent"
    pr_number = int(sys.argv[2]) if len(sys.argv) > 2 else 1821
    version = sys.argv[3] if len(sys.argv) > 3 else "v1"

    tool_name = TOOL_MAP.get(version)
    if not tool_name:
        print(f"Unknown version: {version}. Use v1, v2, or v3.")
        return

    print(f"\n{'='*60}")
    print(f"Reviewing {repo}#{pr_number} with {version} ({tool_name})")
    print(f"{'='*60}\n")

    # Create the server and connect a client to it in-process
    server = create_server()

    async with Client(server) as client:
        t0 = time.monotonic()

        # Call the tool through MCP — full protocol, real spans
        args = {"repo": repo, "pr_number": pr_number}
        if version == "v3":
            args["intensity"] = "balanced"

        call_result = await client.call_tool(tool_name, args)
        elapsed = time.monotonic() - t0

        # Extract the JSON result from the CallToolResult
        text = call_result.content[0].text if call_result.content else "{}"
        result = json.loads(text)

    print(f"\n{'='*60}")
    print(f"RESULT ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"Verdict: {result.get('verdict', '?')}")
    print(f"Risk: {result.get('risk_score', '?')}/10 | "
          f"Health: {result.get('health_score', '?')}/100")
    print(f"Files reviewed: {result.get('files_reviewed', '?')} | "
          f"Skipped: {result.get('files_skipped', '?')}")

    comments = result.get("comments", [])
    print(f"Comments: {len(comments)}")

    stats = result.get("stats", {})
    if stats:
        print(f"  By severity: {stats.get('by_severity', {})}")
        print(f"  By category: {stats.get('by_category', {})}")

    print(f"\nSummary:\n{result.get('summary', '(none)')}")

    if comments:
        print(f"\n{'─'*60}")
        for c in comments:
            print(f"\n[{c['severity']}] {c['path']}:{c.get('line', '?')} "
                  f"— {c['title']}")
            print(f"  {c['body'][:200]}")
            if c.get("suggested_code"):
                print(f"  Suggestion: {c['suggested_code'][:100]}")


if __name__ == "__main__":
    asyncio.run(main())
