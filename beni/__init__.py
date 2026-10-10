"""Sebeni — self-aware morphotactic generation for extremely low-resource languages."""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__ = version("sebeni")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

import sys
if not hasattr(sys, "get_int_max_str_digits"):
    sys.get_int_max_str_digits = lambda: 4300
if not hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits = lambda digits: None

import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated as an API.*", category=UserWarning)

__all__ = ["__version__"]
