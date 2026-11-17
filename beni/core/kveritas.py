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
    """A KV name: ``[A-Za-z][A-Za-z0-9_]*`` (other characters become ``_``)."""
    cleaned = "".join(ch if (ch.isascii() and (ch.isalnum() or ch == "_")) else "_" for ch in str(name))
    return cleaned if cleaned[:1].isalpha() else f"m_{cleaned}"


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


def code_manifest(path: Path) -> Path:
    """Write sha256 of every file in the running ``beni`` package to ``path``.

    Passed to ``kveritas run --files`` so the sealed report covers the code
    that ran (the session root is the working dir, not the source tree).
    """
    import hashlib
    import json

    import beni

    root = Path(beni.__file__).resolve().parent
    files = {}
    for item in sorted(root.rglob("*")):
        if item.is_file() and "__pycache__" not in item.parts and item.suffix in {".py", ".yaml", ".yml", ".json", ".txt", ".gram", ".dict"}:
            files[str(item.relative_to(root.parent))] = hashlib.sha256(item.read_bytes()).hexdigest()
    aggregate = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in files.items()).encode("utf-8")).hexdigest()
    # beni.__version__ prefers a source checkout over a stale installed copy (#12).
    payload = {"package_root": str(root), "version": getattr(beni, "__version__", None), "sha256": aggregate, "files": files}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def move_aside_sealed_session(workdir: Path) -> Optional[Path]:
    """A sealed session cannot take more runs; move it aside so a new one starts."""
    import json
    import time

    session = Path(workdir) / ".kveritas"
    try:
        sealed = bool(json.loads((session / "session.json").read_text(encoding="utf-8")).get("sealed"))
    except (OSError, ValueError, AttributeError):
        return None
    if not sealed:
        return None
    target = session.with_name(f".kveritas.sealed-{time.strftime('%Y%m%d-%H%M%S')}")
    session.rename(target)
    return target


def run_forwarding_signals(cmd: List[str], cwd: str, env: Dict[str, str]) -> int:
    """Run ``cmd`` and forward SIGINT / SIGTERM to it, so a stop reaches the monitored run."""
    import signal
    import subprocess

    proc = subprocess.Popen(cmd, cwd=cwd, env=env)

    def _forward(signum, _frame):
        try:
            proc.send_signal(signum)
        except ProcessLookupError:
            pass

    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.signal(sig, _forward)
        except ValueError:  # not in the main thread
            pass
    try:
        return proc.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


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
