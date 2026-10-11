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
