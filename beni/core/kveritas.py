"""K-Veritas integration: one monitored run per command, and the ``KV`` line protocol.

K-Veritas records only what runs as its monitored subprocess
(``kveritas run -- <cmd>``): it hashes that process's whole output and samples
its hardware. Sebeni therefore re-launches a ``train`` / ``exp`` / ``eval``
command once under ``kveritas run`` (see :func:`kveritas_command`); the child
sees :data:`UNDER_KVERITAS_ENV` and does the work, printing ``KV`` lines:

- ``KV <name>=<number> step=<n>``: per-step metrics (rewards, loss, eval);
- ``KV phase <name>``: distill / train / eval boundaries (hardware snapshots);
- ``KV seed <value>``;
- ``KV claim <name>=<number> phase=eval``: the headline Φ / MER / MCS / UWEC.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

UNDER_KVERITAS_ENV = "SEBENI_UNDER_KVERITAS"
# The caller's working directory, restored in the child (kveritas runs it from the session dir).
CALLER_CWD_ENV = "SEBENI_KV_CALLER_CWD"
CLAIM_METRICS = ("phi", "mer", "mcs", "uwec")


def under_kveritas() -> bool:
    """True inside the process that ``kveritas run`` monitors."""
    return os.environ.get(UNDER_KVERITAS_ENV) == "1"


def active(mc: Any = None) -> bool:
    """Print KV lines when monitored, or when ``experiment.kveritas`` is on."""
    if under_kveritas():
        return True
    experiment = getattr(mc, "experiment", None)
    return bool(getattr(experiment, "kveritas", False))


def _number(value: Any) -> Optional[str]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return f"{number:.6g}"


def _name(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "_.-") else "_" for ch in str(name))


def emit_metric(name: str, value: Any, step: Optional[int] = None) -> None:
    """``KV <name>=<value> step=<step>``; non-numeric or non-finite values are skipped."""
    number = _number(value)
    if number is None:
        return
    suffix = f" step={int(step)}" if step is not None else ""
    print(f"KV {_name(name)}={number}{suffix}", flush=True)


def emit_phase(name: str, mc: Any = None) -> None:
    if active(mc):
        print(f"KV phase {_name(name)}", flush=True)


def emit_seed(seed: Any, mc: Any = None) -> None:
    if active(mc) and seed is not None:
        print(f"KV seed {seed}", flush=True)


def emit_claims(report: Dict[str, Any], mc: Any = None, phase: str = "eval") -> None:
    """``KV claim`` lines for the scope's Φ / MER / MCS / UWEC and each language's."""
    if not active(mc):
        return
    rows = [("", report)] + [(f"_{lang}", row or {}) for lang, row in (report.get("by_language") or {}).items()]
    for suffix, row in rows:
        for key in CLAIM_METRICS:
            number = _number(row.get(key))
            if number is not None:
                print(f"KV claim {_name(key + suffix)}={number} phase={phase}", flush=True)


def evidence_files(paths: Iterable[Any]) -> List[str]:
    """Existing files to hash with ``--files`` (absolute, de-duplicated)."""
    out: List[str] = []
    for item in paths:
        if not item:
            continue
        for value in item if isinstance(item, (list, tuple)) else [item]:
            path = Path(str(value)).expanduser()
            if path.is_file():
                resolved = str(path.resolve())
                if resolved not in out:
                    out.append(resolved)
    return out


def kveritas_command(
    binary: str,
    argv: List[str],
    *,
    seal: Optional[Path] = None,
    files: Optional[List[str]] = None,
    python: Optional[str] = None,
) -> List[str]:
    """``kveritas run [--files …] [--seal=…] -- python -m beni.cli.main <argv>``."""
    cmd = [binary, "run"]
    if files:
        cmd += ["--files", ",".join(files)]
    if seal is not None:
        cmd.append(f"--seal={Path(seal).resolve()}")
    cmd += ["--", python or sys.executable, "-m", "beni.cli.main", *argv]
    return cmd


def child_env(caller_cwd: Optional[str] = None) -> Dict[str, str]:
    """Environment for the monitored child: the guard, the caller's cwd, and this ``beni`` on PYTHONPATH."""
    import beni

    env = dict(os.environ)
    env[UNDER_KVERITAS_ENV] = "1"
    env[CALLER_CWD_ENV] = caller_cwd or os.getcwd()
    # The child starts in the session directory, so import the same beni as this process.
    package_root = str(Path(beni.__file__).resolve().parents[1])
    paths = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    if package_root not in paths:
        env["PYTHONPATH"] = os.pathsep.join([package_root, *paths])
    return env


def restore_caller_cwd() -> None:
    """In the monitored child, go back to the directory the user ran the command from."""
    caller = os.environ.get(CALLER_CWD_ENV)
    if under_kveritas() and caller and Path(caller).is_dir():
        os.chdir(caller)
