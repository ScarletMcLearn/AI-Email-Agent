"""Workspace-safe pytest fixtures."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def workspace_tmp_path() -> Path:
    """Create a test directory without pytest's restrictive Windows ACL logic."""
    path = Path.cwd() / "tmp" / "test-runs" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path
