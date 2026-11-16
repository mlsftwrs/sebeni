"""Issue 20: the whole command runs once under `kveritas run`; no stub run; KV protocol."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from beni.cli.main import app, maybe_kveritas_seal
from beni.core import kveritas as kv

runner = CliRunner()


@pytest.fixture
def kv_config(tmp_path: Path):
    wd = tmp_path / "run"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"algorithm: sft\nworking_dir: {wd}\ndata:\n  languages: [bam]\n"
        "experiment:\n  kveritas: true\n  kveritas_seal: true\n  kveritas_disclosure: names\n",
        encoding="utf-8",
    )
    return config, wd


@pytest.fixture(autouse=True)
def _not_monitored(monkeypatch):
    monkeypatch.delenv(kv.UNDER_KVERITAS_ENV, raising=False)


def _patched(code: int = 0):
    return (
        patch("shutil.which", return_value="/usr/bin/kveritas"),
        patch("subprocess.call", return_value=code),
        patch("beni.cli.main.maybe_kveritas_init"),
    )


@pytest.mark.parametrize(
    "argv, work",
    [
        (["exp", "--algorithm", "sft"], "beni.core.pipeline.run_experiment"),
        (["train", "--algorithm", "sft"], "beni.core.pipeline.run_arm"),
        (["eval"], "beni.core.pipeline.load_policy"),
    ],
)
def test_command_reexecs_once_under_kveritas(kv_config, argv, work):
    config, wd = kv_config
    which, call, init = _patched()
    with which, call as call_mock, init as init_mock, patch(work) as work_mock:
        result = runner.invoke(app, [*argv, "-c", str(config), "-w", str(wd)])
    assert result.exit_code == 0, result.output
    work_mock.assert_not_called()  # the parent only launches the monitored child
    call_mock.assert_called_once()
    cmd = call_mock.call_args.args[0]
    assert cmd[:2] == ["/usr/bin/kveritas", "run"]
    sep = cmd.index("--")
    assert cmd[sep + 2 : sep + 4] == ["-m", "beni.cli.main"]
    assert f"--seal={(wd / 'exp' / 'report.pdf').resolve()}" in cmd[:sep]
    files = cmd[cmd.index("--files") + 1].split(",")
    assert str(config.resolve()) in files
    env = call_mock.call_args.kwargs["env"]
    assert env[kv.UNDER_KVERITAS_ENV] == "1" and env[kv.CALLER_CWD_ENV]
    assert call_mock.call_args.kwargs["cwd"] == str(wd)
    init_mock.assert_called_once_with(wd, disclosure="names", local=False)
    assert not any("sh" == part or "echo" in part for part in cmd)


def test_failed_init_aborts_before_running(kv_config):
    config, wd = kv_config
    with patch("shutil.which", return_value="/usr/bin/kveritas"), patch("subprocess.call") as call_mock, patch(
        "beni.cli.main.maybe_kveritas_init", return_value=False
    ), patch("beni.core.pipeline.run_experiment") as work:
        result = runner.invoke(app, ["exp", "-c", str(config), "-w", str(wd)])
    assert result.exit_code == 1
    assert "kveritas_local" in result.output
    call_mock.assert_not_called()
    work.assert_not_called()


def test_local_init_passes_flag(tmp_path):
    from beni.cli.main import maybe_kveritas_init

    with patch("shutil.which", return_value="/usr/bin/kveritas"), patch("subprocess.run") as run_mock:
        run_mock.return_value = MagicMock(returncode=0)
        assert maybe_kveritas_init(tmp_path, disclosure="open", local=True) is True
    assert run_mock.call_args.args[0] == ["/usr/bin/kveritas", "init", "--disclosure", "open", "--local"]


def test_child_does_not_reexec(kv_config, monkeypatch):
    config, wd = kv_config
    monkeypatch.setenv(kv.UNDER_KVERITAS_ENV, "1")
    which, call, init = _patched()
    with which, call as call_mock, init, patch("beni.core.pipeline.run_experiment", return_value={}) as work:
        result = runner.invoke(app, ["exp", "-c", str(config), "-w", str(wd), "--algorithm", "sft"])
    assert result.exit_code == 0, result.output
    call_mock.assert_not_called()
    work.assert_called_once()


def test_monitored_exit_code_propagates(kv_config):
    config, wd = kv_config
    which, call, init = _patched(code=3)
    with which, call, init:
        result = runner.invoke(app, ["exp", "-c", str(config), "-w", str(wd)])
    assert result.exit_code == 3


def test_missing_binary_runs_unmonitored(kv_config):
    config, wd = kv_config
    with patch("shutil.which", return_value=None), patch("subprocess.call") as call_mock, patch(
        "beni.core.pipeline.run_experiment", return_value={}
    ) as work:
        result = runner.invoke(app, ["exp", "-c", str(config), "-w", str(wd)])
    assert result.exit_code == 0, result.output
    call_mock.assert_not_called()
    work.assert_called_once()
    assert "kveritas not on PATH" in result.output


def test_train_no_seal_flag_overrides_config(kv_config):
    config, wd = kv_config
    which, call, init = _patched()
    with which, call as call_mock, init, patch("beni.core.pipeline.run_arm"):
        runner.invoke(app, ["train", "-c", str(config), "-w", str(wd), "--no-kveritas-seal"])
    assert not any(part.startswith("--seal") for part in call_mock.call_args.args[0])


def test_seal_never_fabricates_a_run(tmp_path):
    (tmp_path / ".kveritas").mkdir()
    (tmp_path / ".kveritas" / "session.json").write_text(json.dumps({"runs": []}), encoding="utf-8")
    with patch("shutil.which", return_value="/usr/bin/kveritas"), patch("subprocess.run") as run_mock:
        maybe_kveritas_seal(tmp_path / "report.pdf", working_dir=tmp_path)
    run_mock.assert_not_called()


def test_restore_caller_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv(kv.UNDER_KVERITAS_ENV, "1")
    monkeypatch.setenv(kv.CALLER_CWD_ENV, str(tmp_path))
    monkeypatch.chdir("/")
    kv.restore_caller_cwd()
    assert Path.cwd() == tmp_path


def test_child_env_puts_this_beni_first():
    import beni

    env = kv.child_env("/somewhere")
    assert env["PYTHONPATH"].split(":")[0] == str(Path(beni.__file__).resolve().parents[1])


# --- line protocol -----------------------------------------------------------


def test_kv_lines(capsys, monkeypatch):
    monkeypatch.setenv(kv.UNDER_KVERITAS_ENV, "1")
    kv.emit_metric("rewards/chosen", 0.25, step=3)
    kv.emit_metric("nan", float("nan"), step=3)
    kv.emit_phase("train")
    kv.emit_seed(42)
    kv.emit_claims({"phi": 0.5, "mer": None, "mcs": 0.25, "by_language": {"bam": {"phi": 0.5, "uwec": 0.7}}})
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "KV rewards_chosen=0.25 step=3",
        "KV phase train",
        "KV seed 42",
        "KV claim phi=0.5 phase=eval",
        "KV claim mcs=0.25 phase=eval",
        "KV claim phi_bam=0.5 phase=eval",
        "KV claim uwec_bam=0.7 phase=eval",
    ]


def test_phases_and_claims_silent_when_inactive(capsys):
    mc = MagicMock()
    mc.experiment.kveritas = False
    kv.emit_phase("train", mc)
    kv.emit_claims({"phi": 0.5}, mc)
    assert capsys.readouterr().out == ""


# --- run_matrix: one monitored subprocess per cell ------------------------------


def test_run_matrix_cell_runs_exp_subprocess(tmp_path):
    from scripts.run_matrix import run_experiment_cell

    run_dir = tmp_path / "cell"

    def fake_run(cmd, env=None, check=False):
        (run_dir / "exp").mkdir(parents=True, exist_ok=True)
        (run_dir / "exp" / "eval.json").write_text(json.dumps({"phi": 0.4, "mer": 0.9}), encoding="utf-8")
        (run_dir / "exp" / "report.pdf").write_bytes(b"%PDF")
        return MagicMock(returncode=0)

    cfg = {"model": {"model_name": "m"}, "algorithm": "dpo", "experiment": {"kveritas": True}}
    with patch("scripts.run_matrix.subprocess.run", side_effect=fake_run) as run_mock:
        row = run_experiment_cell(cfg, run_dir)
    cmd = run_mock.call_args.args[0]
    assert cmd[1:4] == ["-m", "beni.cli.main", "exp"] and cmd[-2:] == ["--algorithm", "dpo"]
    assert row["status"] == "COMPLETED" and row["phi"] == 0.4
    assert (run_dir / "kveritas_report.pdf").read_bytes() == b"%PDF"


def test_in_training_eval_probe_records_nothing(tmp_path):
    from beni.core import Sentence, Token
    from beni.core.pipeline import run_eval
    from beni.core.srl.config import MasterConfig

    mc = MasterConfig()
    mc.working_dir = str(tmp_path / "run")
    mc.apply_workdir()
    mc.data.languages = ["bam"]
    mc.experiment.kveritas = True
    dabax = MagicMock()
    dabax.loader.side_effect = lambda t: [Sentence(text=t, lang="bam", tokens=[Token(surface=w, stage=0) for w in t.split()])]
    with patch("beni.core.morphotactic.dabax.get_dabax", return_value=dabax), patch(
        "beni.core.morphotactic.dabax.resolve_baseline_files", return_value=(None, None, "ck")
    ):
        report = run_eval(mc, records=[{"text": "a b", "lang": "bam"}], record=False)
    assert "by_language" in report
    assert not (tmp_path / "run" / "exp" / "eval.json").exists()
