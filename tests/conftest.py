"""Shared test fixtures."""

import pytest

from beni.utils import config as cfg


@pytest.fixture(autouse=True)
def _restore_working_dir():
    """Tests that activate a working dir must not leak it into later tests."""
    previous = cfg.get_workdir().root
    yield
    cfg.set_working_dir(previous, ensure=False)
