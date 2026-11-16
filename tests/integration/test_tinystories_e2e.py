"""End-to-end pipeline checks on roneneldan/TinyStories-1M (CPU).

These tests download the model from the Hugging Face Hub and run real
training steps, so they are marked ``slow`` and only run when
``SEBENI_RUN_SLOW=1`` is set::

    SEBENI_RUN_SLOW=1 pytest -m slow tests/integration

They check pipeline mechanics (distill -> train -> eval, and ``exp``), not
model quality: TinyStories is an English story model.

Device: the harness config uses the GPU when CUDA is available and falls back
to CPU otherwise. Set ``SEBENI_FORCE_CPU=1`` to force the CPU path on a GPU
machine (the config is copied with every ``use_cpu`` set to true).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs" / "test_tinystories.yaml"
TIMEOUT = int(os.environ.get("SEBENI_SLOW_TIMEOUT", "1800"))

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("SEBENI_RUN_SLOW") != "1",
        reason="set SEBENI_RUN_SLOW=1 to run TinyStories end-to-end tests",
    ),
]


def _sebeni_cmd() -> list:
    """Run the CLI from this checkout, not from an installed ``sebeni`` copy."""
    return [sys.executable, "-c", "from beni.cli.main import app; app()"]


def _config(workdir: Path) -> Path:
    if os.environ.get("SEBENI_FORCE_CPU") != "1":
        return CONFIG
    workdir.mkdir(parents=True, exist_ok=True)
    forced = workdir / "test_tinystories.cpu.yaml"
    forced.write_text(
        CONFIG.read_text(encoding="utf-8").replace("use_cpu: false", "use_cpu: true"),
        encoding="utf-8",
    )
    return forced


def _run(*args: str, workdir: Path) -> subprocess.CompletedProcess:
    cmd = [*_sebeni_cmd(), *args, "-c", str(_config(workdir)), "-w", str(workdir)]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH", "")) if p
            ),
            "TOKENIZERS_PARALLELISM": "false",
        },
    )
    assert proc.returncode == 0, (
        f"{' '.join(cmd)} exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-4000:]}"
    )
    return proc


def _assert_eval_report(workdir: Path) -> dict:
    report_path = workdir / "exp" / "eval.json"
    assert report_path.is_file(), f"missing {report_path}"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "bam" in report.get("by_language", {}), report.keys()
    for key in ("mer", "mcs", "uwec"):
        value = report.get(key)
        assert isinstance(value, (int, float)), f"{key}={value!r} is not numeric"
    return report


@pytest.mark.parametrize("algorithm", ["sft", "grpo"])
def test_distill_train_eval(tmp_path: Path, algorithm: str) -> None:
    workdir = tmp_path / f"tiny-{algorithm}"
    _run("distill", workdir=workdir)
    assert (workdir / "data" / "resources" / "frozen.json").is_file()
    _run("train", "--algorithm", algorithm, workdir=workdir)
    assert any((workdir / "models").iterdir()), "train wrote nothing to models/"
    _run("eval", workdir=workdir)
    _assert_eval_report(workdir)


@pytest.mark.parametrize("algorithm", ["dpo", "apo"])
def test_exp(tmp_path: Path, algorithm: str) -> None:
    workdir = tmp_path / f"tiny-exp-{algorithm}"
    _run("exp", "--algorithm", algorithm, workdir=workdir)
    _assert_eval_report(workdir)
