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
    """(which, monitored-run launcher, init) patches; the launcher mock gets (cmd, cwd=, env=)."""
    return (
        patch("shutil.which", return_value="/usr/bin/kveritas"),
        patch("beni.core.kveritas.run_forwarding_signals", return_value=code),
        patch("beni.cli.main.maybe_kveritas_init"),
    )


@pytest.mark.parametrize(
    "argv, work, report",
    [
        (["exp", "--algorithm", "sft"], "beni.core.pipeline.run_experiment", "report.pdf"),
        (["train", "--algorithm", "sft"], "beni.core.pipeline.run_arm", "report_train.pdf"),
        (["eval"], "beni.core.pipeline.load_policy", "report_eval.pdf"),
    ],
)
def test_command_reexecs_once_under_kveritas(kv_config, argv, work, report):
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
    # The child repeats exactly this invocation (not the host process's sys.argv).
    assert cmd[sep + 4 :] == [*argv, "-c", str(config), "-w", str(wd)]
    assert f"--seal={(wd / 'exp' / report).resolve()}" in cmd[:sep]
    files = cmd[cmd.index("--files") + 1].split(",")
    assert str(config.resolve()) in files
    code = wd / "exp" / "sebeni_code.json"
    assert str(code.resolve()) in files
    assert any(name.endswith("beni/core/kveritas.py") for name in json.loads(code.read_text())["files"])
    env = call_mock.call_args.kwargs["env"]
    assert env[kv.UNDER_KVERITAS_ENV] == "1" and env[kv.CALLER_CWD_ENV]
    assert call_mock.call_args.kwargs["cwd"] == str(wd)
    init_mock.assert_called_once_with(wd, disclosure="names", local=False)
    assert not any("sh" == part or "echo" in part for part in cmd)


def test_failed_init_aborts_before_running(kv_config):
    config, wd = kv_config
    with patch("shutil.which", return_value="/usr/bin/kveritas"), patch(
        "beni.core.kveritas.run_forwarding_signals"
    ) as call_mock, patch(
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
    with patch("shutil.which", return_value=None), patch("beni.core.kveritas.run_forwarding_signals") as call_mock, patch(
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


def test_local_mode_seals_with_configured_key(tmp_path):
    wd = tmp_path / "run"
    key = tmp_path / "k.pem"
    key.write_text("x", encoding="utf-8")
    config = tmp_path / "c.yaml"
    config.write_text(
        f"working_dir: {wd}\nexperiment:\n  kveritas: true\n  kveritas_seal: true\n"
        f"  kveritas_local: true\n  kveritas_local_key: {key}\n",
        encoding="utf-8",
    )
    which, call, init = _patched()
    with which, call as call_mock, init, patch("subprocess.call", return_value=0) as seal_mock:
        result = runner.invoke(app, ["exp", "-c", str(config), "-w", str(wd)])
    assert result.exit_code == 0, result.output
    run_cmd, seal_cmd = call_mock.call_args.args[0], seal_mock.call_args.args[0]
    assert not any(part.startswith("--seal") for part in run_cmd)
    assert seal_cmd == ["/usr/bin/kveritas", "seal", "--local-key", str(key), "-o", str((wd / "exp" / "report.pdf").resolve())]


def test_local_mode_without_key_refuses(tmp_path):
    wd = tmp_path / "run"
    config = tmp_path / "c.yaml"
    config.write_text(
        f"working_dir: {wd}\nexperiment:\n  kveritas: true\n  kveritas_seal: true\n  kveritas_local: true\n",
        encoding="utf-8",
    )
    which, call, init = _patched()
    with which, call as call_mock, init:
        result = runner.invoke(app, ["exp", "-c", str(config), "-w", str(wd)])
    assert result.exit_code == 1 and "openssl genrsa" in result.output
    call_mock.assert_not_called()


@pytest.mark.parametrize("command", ["exp", "eval", "train"])
def test_no_kveritas_opts_out_of_sealing_too(kv_config, command):
    config, wd = kv_config
    which, call, init = _patched()
    work = {"exp": "beni.core.pipeline.run_experiment", "eval": "beni.core.pipeline.load_policy", "train": "beni.core.pipeline.run_arm"}[command]
    with which, call as call_mock, init, patch(work), patch("beni.core.pipeline.run_eval", return_value={}):
        result = runner.invoke(app, [command, "-c", str(config), "-w", str(wd), "--no-kveritas"])
    assert result.exit_code == 0, result.output
    call_mock.assert_not_called()


def test_sealed_session_is_moved_aside(tmp_path):
    session = tmp_path / ".kveritas"
    session.mkdir()
    (session / "session.json").write_text(json.dumps({"sealed": True}), encoding="utf-8")
    moved = kv.move_aside_sealed_session(tmp_path)
    assert moved is not None and moved.name.startswith(".kveritas.sealed-") and not session.exists()
    assert kv.move_aside_sealed_session(tmp_path) is None


def test_kv_names_follow_the_grammar(capsys):
    kv.emit_metric("eval.loss-x", 1.0, step=1)
    kv.emit_metric("1st", 2.0, step=1)
    assert capsys.readouterr().out.splitlines() == ["KV eval_loss_x=1 step=1", "KV m_1st=2 step=1"]


def test_signals_are_forwarded(tmp_path):
    import signal
    import sys
    import threading

    script = tmp_path / "child.py"
    marker = tmp_path / "got"
    script.write_text(
        "import signal, sys, time, pathlib\n"
        f"signal.signal(signal.SIGTERM, lambda *a: (pathlib.Path({str(marker)!r}).write_text('x'), sys.exit(7)))\n"
        "print('ready', flush=True)\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    import os

    timer = threading.Timer(1.5, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.start()
    code = kv.run_forwarding_signals([sys.executable, str(script)], cwd=str(tmp_path), env=dict(os.environ))
    timer.cancel()
    assert code == 7 and marker.exists()


def test_seal_only_still_streams_step_metrics():
    from beni.core.srl.config import MasterConfig

    mc = MasterConfig()
    mc.experiment.kveritas = False
    assert kv.active(mc) is False
    import os

    os.environ[kv.UNDER_KVERITAS_ENV] = "1"
    try:
        assert kv.active(mc) is True  # monitored child (e.g. seal-only) attaches the KV callback
    finally:
        del os.environ[kv.UNDER_KVERITAS_ENV]


def test_run_matrix_cell_ignores_stale_results(tmp_path):
    from scripts.run_matrix import run_experiment_cell

    run_dir = tmp_path / "cell"
    (run_dir / "exp").mkdir(parents=True)
    (run_dir / "exp" / "eval.json").write_text(json.dumps({"phi": 0.9}), encoding="utf-8")
    (run_dir / "exp" / "report.pdf").write_bytes(b"old")
    cfg = {"model": {"model_name": "m"}, "algorithm": "sft", "experiment": {}}
    with patch("scripts.run_matrix.subprocess.run", return_value=MagicMock(returncode=1)):
        row = run_experiment_cell(cfg, run_dir)
    assert row["status"] == "FAILED: exit 1" and row["phi"] is None
    assert not (run_dir / "kveritas_report.pdf").exists()


def test_build_config_local_key(tmp_path):
    import yaml

    from scripts.run_matrix import build_config

    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"model": {}, "trainer": {}, "distillation": {}, "experiment": {}}), encoding="utf-8")
    cfg = build_config(
        base, {"id": "m", "slug": "m"}, "sft", "MULTI13", 1, "x", "global", False, "algorithmic",
        tmp_path, kveritas_local_key=str(tmp_path / "k.pem"),
    )
    assert cfg["experiment"]["kveritas_local"] is True
    assert cfg["experiment"]["kveritas_local_key"] == str((tmp_path / "k.pem").resolve())
