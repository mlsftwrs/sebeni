"""Issue 5: the Hub push gate refuses without an eval snapshot next to the pushed weights."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from beni.core.safety.governor import SafetyGovernor, weights_fingerprint
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


def _evaluated(directory: Path, weights: bytes = b"w1") -> None:
    """Weights plus an eval snapshot that scored exactly those bytes."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_model.safetensors").write_bytes(weights)
    _snapshot(directory, phi=0.5, mer=0.4, mcs=0.6, weights_sha256=weights_fingerprint(directory))


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


def test_push_after_eval_snapshot_uploads_the_checked_folder(tmp_path):
    (tmp_path / "README.md").write_text("# card", encoding="utf-8")
    _evaluated(tmp_path)
    plugin = _plugin(tmp_path)
    with patch("huggingface_hub.HfApi") as api_cls:
        plugin.push_to_hub(repo_id="org/m", token="t", private=True)
    api = api_cls.return_value
    api.create_repo.assert_called_once_with(repo_id="org/m", private=True, exist_ok=True)
    assert api.upload_folder.call_args.kwargs["folder_path"] == str(tmp_path)
    plugin.model.push_to_hub.assert_not_called()


def test_stale_snapshot_for_other_weights_refuses(tmp_path):
    """Weights changed after the eval (e.g. retrained into the same dir): refuse."""
    (tmp_path / "README.md").write_text("# card", encoding="utf-8")
    _evaluated(tmp_path, weights=b"w1")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"w2")
    plugin = _plugin(tmp_path)
    with patch("huggingface_hub.HfApi") as api_cls, pytest.raises(PermissionError):
        plugin.push_to_hub(repo_id="org/m")
    api_cls.return_value.upload_folder.assert_not_called()
    assert "snapshot_weights_mismatch" in plugin.governor.spec.fired


def test_snapshot_without_weights_hash_refuses(tmp_path):
    (tmp_path / "README.md").write_text("# card", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"w1")
    _snapshot(tmp_path, phi=0.5, mer=0.4, mcs=0.6)
    with pytest.raises(PermissionError):
        _plugin(tmp_path).push_to_hub(repo_id="org/m")


def test_save_model_points_push_at_the_new_weights(tmp_path):
    plugin = _plugin(tmp_path / "old")
    with patch.object(AlignmentPlugin, "write_card_and_snapshot"):
        plugin.save_model(tmp_path / "new")
    assert plugin.weights_dir == str(tmp_path / "new")


def test_write_card_renders_from_snapshot(tmp_path):
    _evaluated(tmp_path)
    plugin = _plugin(tmp_path)
    with patch("huggingface_hub.HfApi") as api_cls:
        plugin.push_to_hub(repo_id="org/m", write_card=True)
    assert (tmp_path / "README.md").is_file()
    api_cls.return_value.upload_folder.assert_called_once()


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
    (weights / "adapter_model.safetensors").write_bytes(b"w1")
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


def test_trl_configs_never_push_by_themselves():
    """HF Trainer must not upload checkpoints itself (it would bypass the gate)."""
    import inspect

    from beni.core.srl.dpo import dpo
    from beni.core.srl.grpo import grpo
    from beni.core.srl.sft import sft

    for module, var in ((sft, "sft_dict"), (dpo, "dpo_dict"), (grpo, "grpo_dict")):
        assert f'{var}["push_to_hub"] = False' in inspect.getsource(module)


@patch("beni.core.srl.plugin.write_model_card")
@patch("beni.core.srl.grpo.grpo.GRPOTrainer")
def test_grpo_trl_args_have_push_disabled(mock_trainer_cls, _card, tmp_path):
    from beni.core.srl.grpo.grpo import SebeniGrpo

    trainer = MagicMock()
    trainer.optimizer = None
    mock_trainer_cls.return_value = trainer
    plugin = SebeniGrpo(MasterConfig())
    plugin.model = MagicMock()
    plugin.tokenizer = MagicMock()
    plugin.config.trainer.use_cpu = True
    plugin.config.trainer.output_dir = str(tmp_path)
    plugin.config.trainer.push_to_hub = True
    plugin.config.trainer.hub_model_id = "org/m"
    plugin.config.trainer.report_to = "none"
    plugin.config.data.eval_ratio = 0.0
    with patch.object(SebeniGrpo, "push_after_train") as after:
        plugin.train([{"text": "a", "lang": "bam", "reference": {"tokens": []}}] * 4, project_name="t")
    assert mock_trainer_cls.call_args.kwargs["args"].push_to_hub is False
    after.assert_called_once()
