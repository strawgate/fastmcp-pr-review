"""Horizon import entrypoint for the FastMCP PR review server."""

from fastmcp_pr_review.server import _load_env_file, create_server

_load_env_file()
mcp = create_server()


if __name__ == "__main__":
    mcp.run()
