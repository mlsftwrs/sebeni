"""Sebeni — self-aware morphotactic generation for extremely low-resource languages."""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

def _resolve_version() -> str:
    """The version of the code that is running.

    A source checkout (pyproject.toml next to this package) wins over any
    installed ``sebeni`` distribution, which may be a different, stale copy.
    pyproject.toml is the single source of the version (#12).
    """
    from pathlib import Path

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if pyproject.is_file():
        try:
            try:
                import tomllib  # Python 3.11+
            except ImportError:  # pragma: no cover
                import tomli as tomllib  # type: ignore
            project = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("project", {})
            if project.get("name") == "sebeni" and project.get("version"):
                return str(project["version"])
        except Exception:  # pragma: no cover
            pass
    try:
        return version("sebeni")
    except PackageNotFoundError:  # pragma: no cover
        return "0+unknown"


__version__ = _resolve_version()

import sys
if not hasattr(sys, "get_int_max_str_digits"):
    sys.get_int_max_str_digits = lambda: 4300
if not hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits = lambda digits: None

import os
import logging
import warnings

# Suppress known harmless upstream deprecation, glog, and environment warnings
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TORCH_LOGS", "-all")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated as an API.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*TRL currently supports vLLM versions.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Enum subclass and is now natively supported by torch.compile.*")
warnings.filterwarnings("ignore", message=".*Calling register_constant.*on Enum subclasses is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=r".*trl.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r".*vllm.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r".*torch.*")

for _mod in ("torch.utils._pytree", "torch._logging", "trl"):
    logging.getLogger(_mod).setLevel(logging.ERROR)

__all__ = ["__version__"]
