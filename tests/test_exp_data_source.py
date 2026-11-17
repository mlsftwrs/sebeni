"""Issue 11: sebeni exp honours an explicit data.source; presets keep the packaged jsonl."""

from unittest.mock import patch

from typer.testing import CliRunner

from beni.cli.main import app

runner = CliRunner()


def _run(args):
    seen = {}

    def fake(mc):
        seen["source"] = mc.data.source
        seen["dataset"] = mc.experiment.dataset
        return {}

    with patch("beni.core.pipeline.run_experiment", side_effect=fake):
        result = runner.invoke(app, ["exp", *args])
    assert result.exit_code == 0, result.output
    return seen


def test_config_data_source_is_used(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text": "a", "lang": "bam"}\n', encoding="utf-8")
    config = tmp_path / "c.yaml"
    config.write_text(f"data:\n  source: {data}\n  languages: [bam]\n", encoding="utf-8")
    seen = _run(["-c", str(config), "-w", str(tmp_path / "run")])
    assert seen["source"] == str(data)


def test_config_without_source_uses_packaged(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text("data:\n  languages: [bam]\n", encoding="utf-8")
    seen = _run(["-c", str(config), "-w", str(tmp_path / "run")])
    assert seen["source"] is None and seen["dataset"] == "packaged"


def test_preset_ignores_source(tmp_path):
    seen = _run(["--preset", "single", "-w", str(tmp_path / "run")])
    assert seen["source"] is None and seen["dataset"] == "packaged"


def test_run_experiment_reports_train_file(tmp_path, capsys):
    from beni.core.pipeline import run_experiment
    from beni.core.srl.config import MasterConfig

    data = tmp_path / "train.jsonl"
    data.write_text('{"text": "a b", "lang": "bam"}\n{"text": "c d", "lang": "bam"}\n', encoding="utf-8")
    mc = MasterConfig()
    mc.working_dir = str(tmp_path / "run")
    mc.apply_workdir()
    mc.data.languages = ["bam"]
    mc.data.source = str(data)
    with patch("beni.core.pipeline.run_distill"), patch("beni.core.pipeline.run_arm") as arm, patch(
        "beni.core.pipeline.run_eval", return_value={}
    ):
        run_experiment(mc)
    assert len(arm.call_args.args[1]) == 2
    assert f"[exp] train data: {data} (2 rows; bam=2)" in capsys.readouterr().out
