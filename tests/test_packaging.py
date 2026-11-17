"""Issue 12: pyproject.toml is the single source of version and dependencies."""

from pathlib import Path

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib  # type: ignore

import beni

PYPROJECT = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))


def test_version_is_0_2_0_everywhere():
    assert PYPROJECT["project"]["version"] == "0.2.0"
    assert beni.__version__ == "0.2.0"  # a checkout wins over any stale installed copy
    assert not Path("setup.cfg").exists()


def test_quantisation_deps_are_train_only():
    core = " ".join(PYPROJECT["project"]["dependencies"])
    train = " ".join(PYPROJECT["project"]["optional-dependencies"]["train"])
    for dep in ("torchao", "bitsandbytes"):
        assert dep not in core and dep in train
