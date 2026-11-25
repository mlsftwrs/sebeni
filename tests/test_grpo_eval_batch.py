"""Issue 22: GRPO eval batch must divide evenly into num_generations groups."""

from unittest.mock import MagicMock, patch

import pytest

from beni.core.srl.grpo.grpo import SebeniGrpo, _guard_eval_batch


@pytest.mark.parametrize(
    "start, num_gen, world, expected",
    [
        (1, 4, "1", 4),
        (1, 2, "1", 2),
        (4, 4, "1", 4),
        (3, 4, "1", 4),
        (1, 4, "2", 2),
        (6, 4, "1", 8),
    ],
)
def test_guard_rounds_eval_batch_up(monkeypatch, start, num_gen, world, expected):
    monkeypatch.setenv("WORLD_SIZE", world)
    grpo_dict = {"per_device_eval_batch_size": start, "num_generations": num_gen}
    _guard_eval_batch(grpo_dict)
    assert grpo_dict["per_device_eval_batch_size"] == expected


def test_guard_prefers_num_generations_eval(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    grpo_dict = {"per_device_eval_batch_size": 1, "num_generations": 4, "num_generations_eval": 3}
    _guard_eval_batch(grpo_dict)
    assert grpo_dict["per_device_eval_batch_size"] == 3


@patch("beni.core.srl.plugin.write_model_card")
@patch("beni.core.srl.grpo.grpo.GRPOTrainer")
def test_real_grpo_config_builds_with_eval_split(mock_trainer_cls, _card, tmp_path, monkeypatch):
    """The installed GRPOConfig no longer raises when an eval split exists."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    trainer = MagicMock()
    trainer.optimizer = None
    trainer.train.return_value = {}
    mock_trainer_cls.return_value = trainer

    plugin = SebeniGrpo(model_name="HuggingFaceTB/SmolLM2-135M")
    plugin.model = MagicMock()
    plugin.tokenizer = MagicMock()
    plugin.config.trainer.use_cpu = True
    plugin.config.trainer.output_dir = str(tmp_path)
    plugin.config.trainer.num_generations = 4
    plugin.config.trainer.per_device_train_batch_size = 4
    plugin.config.trainer.gradient_accumulation_steps = 1
    plugin.config.trainer.per_device_eval_batch_size = 1
    plugin.config.trainer.push_to_hub = False
    plugin.config.trainer.report_to = "none"
    plugin.config.data.eval_ratio = 0.25

    rows = [{"text": f"sentence {i}", "lang": "bam", "reference": {"tokens": []}} for i in range(8)]
    plugin.train(rows, project_name="t")

    kwargs = mock_trainer_cls.call_args.kwargs
    assert kwargs.get("eval_dataset") is not None
    assert kwargs["args"].per_device_eval_batch_size == 4


def test_guard_uses_trl_default_when_num_generations_unset(monkeypatch):
    from beni.core.srl.grpo.grpo import _trl_default_num_generations

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    grpo_dict = {"per_device_eval_batch_size": 1}
    _guard_eval_batch(grpo_dict)
    assert grpo_dict["per_device_eval_batch_size"] == _trl_default_num_generations()
