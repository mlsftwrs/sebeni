"""Issue 5: the Hub push gate refuses without an eval snapshot next to the pushed weights."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from beni.core.safety.governor import SafetyGovernor
from beni.core.srl.config import MasterConfig
from beni.core.srl.plugin import AlignmentPlugin


def _plugin(weights_dir: Path) -> AlignmentPlugin:
    plugin = AlignmentPlugin(MasterConfig())
    plugin.model = MagicMock()
    plugin.tokenizer = MagicMock()
    plugin.weights_dir = str(weights_dir)
    return plugin


def _snapshot(directory: Path, **fields) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / SafetyGovernor.SNAPSHOT_NAME).write_text(json.dumps(fields), encoding="utf-8")


def test_push_without_snapshot_refuses_and_writes_nothing(tmp_path):
    plugin = _plugin(tmp_path)
    with pytest.raises(PermissionError, match="sebeni eval"):
        plugin.push_to_hub(repo_id="org/m")
    assert not (tmp_path / "README.md").exists()
    assert not (tmp_path / SafetyGovernor.SNAPSHOT_NAME).exists()
    plugin.model.push_to_hub.assert_not_called()


def test_training_snapshot_without_metrics_refuses(tmp_path):
    (tmp_path / "README.md").write_text("# card", encoding="utf-8")
    _snapshot(tmp_path, phi=0.5, mer=None, mcs=None)
    plugin = _plugin(tmp_path)
    with pytest.raises(PermissionError):
        plugin.push_to_hub(repo_id="org/m")
    assert "missing_eval_metrics" in plugin.governor.spec.fired


def test_push_after_eval_snapshot_succeeds(tmp_path):
    (tmp_path / "README.md").write_text("# card", encoding="utf-8")
    _snapshot(tmp_path, phi=0.5, mer=0.4, mcs=0.6)
    plugin = _plugin(tmp_path)
    plugin.push_to_hub(repo_id="org/m", token="t", private=True)
    plugin.model.push_to_hub.assert_called_once_with(repo_id="org/m", token="t", private=True)
    plugin.tokenizer.push_to_hub.assert_called_once()


def test_write_card_renders_from_snapshot(tmp_path):
    _snapshot(tmp_path, phi=0.5, mer=0.4, mcs=0.6)
    plugin = _plugin(tmp_path)
    plugin.push_to_hub(repo_id="org/m", write_card=True)
    assert (tmp_path / "README.md").is_file()
    plugin.model.push_to_hub.assert_called_once()


def test_write_card_without_snapshot_refuses(tmp_path):
    plugin = _plugin(tmp_path)
    with pytest.raises(PermissionError, match="--write-card only renders"):
        plugin.push_to_hub(repo_id="org/m", write_card=True)
    assert not (tmp_path / "README.md").exists()


def test_push_after_train_reports_refusal(tmp_path, capsys):
    plugin = _plugin(tmp_path)
    block = MagicMock(push_to_hub=True, hub_model_id="org/m", hub_token=None, hub_private_repo=True, output_dir=str(tmp_path))
    plugin.push_after_train(block)  # must not raise
    assert "Skipped push_to_hub after training" in capsys.readouterr().out


# --- run_eval writes the snapshot where push looks ---------------------------


@pytest.fixture
def eval_env(tmp_path):
    from beni.core import Sentence, Token

    mc = MasterConfig()
    mc.working_dir = str(tmp_path / "run")
    mc.apply_workdir()
    mc.data.languages = ["bam"]
    dabax = MagicMock()
    dabax.loader.side_effect = lambda text: [Sentence(text=text, lang="bam", tokens=[Token(surface=w, stage=0) for w in text.split()])]
    with patch("beni.core.morphotactic.dabax.get_dabax", return_value=dabax), patch(
        "beni.core.morphotactic.dabax.resolve_baseline_files", return_value=(None, None, "ck")
    ):
        yield mc


def test_run_eval_snapshot_goes_next_to_weights_and_keeps_training(eval_env, tmp_path):
    from beni.core.pipeline import run_eval

    weights = tmp_path / "run" / "models"
    _snapshot(weights, phi=0.1, blocked_steps=2, target_gate={"n_kept": 3})
    run_eval(eval_env, records=[{"text": "a b", "lang": "bam"}], generate=lambda t, _l: t, weights=str(weights))
    snap = json.loads((weights / SafetyGovernor.SNAPSHOT_NAME).read_text(encoding="utf-8"))
    assert snap["mer"] == 0.0 and snap["weights"] == str(weights)
    assert snap["training"]["blocked_steps"] == 2
    (weights / "README.md").write_text("# card", encoding="utf-8")
    assert SafetyGovernor().allow_hub_push(weights) is True


def test_base_model_eval_snapshot_goes_to_exp(eval_env, tmp_path):
    from beni.core.pipeline import run_eval

    run_eval(eval_env, records=[{"text": "a b", "lang": "bam"}], generate=lambda t, _l: t, weights=None)
    assert (tmp_path / "run" / "exp" / SafetyGovernor.SNAPSHOT_NAME).is_file()
    assert not (tmp_path / "run" / "models" / SafetyGovernor.SNAPSHOT_NAME).exists()
