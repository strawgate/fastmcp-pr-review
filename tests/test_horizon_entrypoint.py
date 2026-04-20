"""Tests for the Horizon import entrypoint."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_horizon_entrypoint_exposes_mcp() -> None:
    from fastmcp_pr_review import server as server_module

    entrypoint = Path(__file__).resolve().parents[1] / "run_horizon_server.py"
    spec = importlib.util.spec_from_file_location("test_run_horizon_server", entrypoint)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    mock_server = MagicMock()

    with (
        patch.object(server_module, "_load_env_file") as load_env_file,
        patch.object(server_module, "create_server", return_value=mock_server) as create_server,
    ):
        spec.loader.exec_module(module)

    load_env_file.assert_called_once_with()
    create_server.assert_called_once_with()
    assert module.mcp is mock_server
