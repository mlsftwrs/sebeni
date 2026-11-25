"""Issue 19: the workbench runners never write configs/matrix/base_mali13.yaml."""

import hashlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

BASE = Path("configs/matrix/base_mali13.yaml")
# The workbench runners are local (gitignored) helper scripts; test them only where present.
needs_workbench = pytest.mark.skipif(
    not Path("scripts/run_workbench_cli.py").is_file(), reason="local workbench scripts not present"
)


def _sha() -> str:
    return hashlib.sha256(BASE.read_bytes()).hexdigest()


@needs_workbench
def test_workbench_py_writes_a_tuned_copy(tmp_path, monkeypatch):
    from scripts import run_workbench_cli

    before = _sha()
    monkeypatch.setattr(sys, "argv", ["run_workbench_cli.py", "--steps", "1", "--output-dir", str(tmp_path)])
    with patch("scripts.run_workbench_cli.subprocess.run", return_value=MagicMock(returncode=0)) as run:
        run_workbench_cli.main()
    assert _sha() == before
    cmd = run.call_args.args[0]
    tuned = Path(cmd[cmd.index("--base-config") + 1])
    assert tuned.parent == tmp_path
    data = yaml.safe_load(tuned.read_text(encoding="utf-8"))
    assert data["trainer"]["bf16"] is True and data["model"]["load_in_4bit"] is True


@needs_workbench
def test_workbench_sh_does_not_write_the_base():
    text = Path("scripts/run_workbench_cli.sh").read_text(encoding="utf-8")
    assert "--base-config" in text
    assert "open('${CONFIG_FILE}', 'w')" not in text


def test_run_matrix_rejects_missing_base_config(monkeypatch, tmp_path, capsys):
    from scripts import run_matrix

    monkeypatch.setattr(sys, "argv", ["run_matrix.py", "--base-config", str(tmp_path / "nope.yaml"), "--dry-run"])
    with pytest.raises(SystemExit) as exc:
        run_matrix.main()
    assert exc.value.code == 2
    assert "--base-config not found" in capsys.readouterr().err


def test_run_matrix_uses_the_given_base_config(monkeypatch, tmp_path):
    from scripts import run_matrix

    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    base["project_name"] = "marker"
    base.setdefault("trainer", {})["weight_decay"] = 0.0123
    custom = tmp_path / "base.yaml"
    custom.write_text(yaml.safe_dump(base), encoding="utf-8")
    out = tmp_path / "out"
    model = run_matrix.load_model_catalog(Path("configs/matrix/models.json"))[0]
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_matrix.py", "--base-config", str(custom), "--dry-run", "--output-dir", str(out),
         "--models", model["slug"], "--algorithms", "sft"],
    )
    run_matrix.main()
    written = list(out.rglob("config.yaml"))
    assert written, "dry run wrote no cell config"
    assert yaml.safe_load(written[0].read_text(encoding="utf-8"))["trainer"]["weight_decay"] == 0.0123
