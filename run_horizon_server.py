"""Horizon import entrypoint for the FastMCP PR review server."""

from pathlib import Path

from dotenv import load_dotenv

from fastmcp_pr_review.server import create_server

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
mcp = create_server()


if __name__ == "__main__":
    mcp.run()
