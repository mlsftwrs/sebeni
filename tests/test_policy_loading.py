"""Issue 1: eval / push / generate load the trained policy and decode with the training prompt."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import jinja2
import pytest
from typer.testing import CliRunner

from beni.cli.main import app
from beni.core.srl.config import MasterConfig
from beni.core.srl.plugin import CHATML_CHAT_TEMPLATE, AlignmentPlugin
from beni.core.srl.prompting import (
    build_policy_example,
    build_policy_prompt,
    end_of_turn_ids,
    strip_end_of_turn,
)

runner = CliRunner()


class ChatTokenizer:
    """Offline stand-in for a tokenizer with the ChatML fallback template."""

    chat_template = CHATML_CHAT_TEMPLATE
    eos_token = "<eos>"
    eos_token_id = 0
    unk_token_id = None
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return jinja2.Template(self.chat_template).render(
            messages=messages, add_generation_prompt=add_generation_prompt
        )

    def get_vocab(self):
        return {"<eos>": 0, "<|im_end|>": 7}


# --- prompt builder -------------------------------------------------------


def test_sft_example_starts_with_eval_prompt():
    tok = ChatTokenizer()
    prompt = build_policy_prompt("muso ka taa", tok, ["bam"])
    example = build_policy_example("muso ka taa", '{"tokens": []}', tok, ["bam"])
    assert prompt.endswith("<|im_start|>assistant\n")
    assert example.startswith(prompt)
    assert example[len(prompt):].startswith('{"tokens": []}<|im_end|>')


def test_prompt_without_template_matches_example():
    prompt = build_policy_prompt("muso", None, ["bam"])
    assert prompt.endswith("User: muso\nAssistant: ")
    assert build_policy_example("muso", "{}", None, ["bam"]) == prompt + "{}"


def test_strip_end_of_turn_and_stop_ids():
    tok = ChatTokenizer()
    assert strip_end_of_turn('{"a": 1}<|im_end|>\n<|im_start|>user', tok) == '{"a": 1}'
    assert end_of_turn_ids(tok) == [0, 7]


def test_grpo_dataset_uses_same_messages():
    from beni.core.srl.prompting import policy_messages
    from beni.data.datasets import SebeniDataLoader

    mc = MasterConfig()
    mc.data.languages = ["bam"]
    ds = SebeniDataLoader(mc.data).ds([{"text": "muso", "lang": "bam", "reference": {"tokens": []}}])
    assert ds[0]["prompt"] == policy_messages("muso", ["bam"])


def test_analyze_decodes_greedily_with_training_prompt():
    mc = MasterConfig()
    mc.data.languages = ["bam"]
    mc.experiment.eval_max_new_tokens = 33
    plugin = AlignmentPlugin(mc)
    plugin.tokenizer = ChatTokenizer()
    plugin.model = MagicMock()
    with patch.object(AlignmentPlugin, "generate", return_value='{"x": 1}<|im_end|>junk') as gen:
        out = plugin.analyze("muso ka taa")
    assert out == '{"x": 1}'
    args, kwargs = gen.call_args
    assert args[0] == build_policy_prompt("muso ka taa", plugin.tokenizer, ["bam"])
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_new_tokens"] == 33
    assert kwargs["eos_token_id"] == [0, 7]
    assert kwargs["add_special_tokens"] is False


def test_analyze_uses_training_languages_from_weights(tmp_path):
    from beni.core.srl.plugin import write_policy_meta

    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    write_policy_meta(tmp_path, ["bam", "mku"])
    mc = MasterConfig()
    mc.data.languages = ["bam"]  # e.g. sebeni eval --lang bam on a two-language policy
    plugin = AlignmentPlugin(mc)

    def fake_load(**kwargs):
        plugin.model = MagicMock()
        plugin.tokenizer = ChatTokenizer()

    with patch.object(AlignmentPlugin, "load_models", side_effect=fake_load), patch(
        "peft.PeftModel.from_pretrained", return_value=MagicMock()
    ):
        plugin.load_aligned_weights(tmp_path)
    assert plugin.prompt_languages == ["bam", "mku"]
    with patch.object(AlignmentPlugin, "generate", return_value="{}") as gen:
        plugin.analyze("muso")
    assert gen.call_args.args[0] == build_policy_prompt("muso", plugin.tokenizer, ["bam", "mku"])


def test_save_model_writes_policy_meta(tmp_path):
    mc = MasterConfig()
    mc.data.languages = ["bam", "mku"]
    plugin = AlignmentPlugin(mc)
    with patch.object(AlignmentPlugin, "write_card_and_snapshot"):
        plugin.save_model(tmp_path)
    meta = json.loads((tmp_path / "sebeni_policy.json").read_text(encoding="utf-8"))
    assert meta["prompt_languages"] == ["bam", "mku"]


def test_scope_alias_expands_in_system_prompt():
    from beni.core.language import resolve_scope
    from beni.core.srl.prompting import system_prompt

    assert system_prompt(["multi13"]) == system_prompt(resolve_scope("multi13").languages)


def test_sft_example_prefix_survives_template_injection():
    class ThinkTokenizer(ChatTokenizer):
        bos_token = "<s>"
        chat_template = (
            "<s>{% for m in messages %}{% if loop.last and m['role'] == 'assistant' %}"
            "<|im_start|>assistant\n<think></think>{{ m['content'] }}<|im_end|>\n"
            "{% else %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )

    tok = ThinkTokenizer()
    prompt = build_policy_prompt("muso", tok, ["bam"])
    example = build_policy_example("muso", "{}", tok, ["bam"])
    assert example == prompt[len("<s>"):] + "{}<|im_end|>\n"
    assert "<think>" not in example


def test_generate_greedy_passes_no_sampling_args():
    import torch

    plugin = AlignmentPlugin(MasterConfig())
    tok = MagicMock()
    tok.return_value.to.return_value = {"input_ids": torch.zeros((1, 3), dtype=torch.long)}
    tok.pad_token_id = 0
    tok.decode.return_value = "out"
    plugin.tokenizer = tok
    plugin.model = MagicMock()
    plugin.model.generate.return_value = torch.zeros((1, 5), dtype=torch.long)
    assert plugin.generate("p", max_new_tokens=4, temperature=0.0) == "out"
    kwargs = plugin.model.generate.call_args.kwargs
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs and "top_p" not in kwargs
    assert kwargs["max_new_tokens"] == 4


# --- load_aligned_weights ---------------------------------------------------


def test_load_adapter_attaches_to_clean_base(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    plugin = AlignmentPlugin(MasterConfig())
    base = MagicMock()
    peft_model = MagicMock()

    def fake_load(**kwargs):
        plugin.model = base
        plugin.tokenizer = MagicMock()

    with patch.object(AlignmentPlugin, "load_models", side_effect=fake_load) as load, patch(
        "peft.PeftModel.from_pretrained", return_value=peft_model
    ) as from_pretrained:
        plugin.load_aligned_weights(tmp_path)
    load.assert_called_once_with(apply_peft=False, tokenizer_name=str(tmp_path))
    from_pretrained.assert_called_once_with(base, str(tmp_path), is_trainable=False)
    assert plugin.model is peft_model
    assert plugin.loaded_weights == str(tmp_path)


def test_load_full_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    plugin = AlignmentPlugin(MasterConfig())

    def fake_load(**kwargs):
        plugin.model = MagicMock()
        plugin.tokenizer = MagicMock()

    with patch.object(AlignmentPlugin, "load_models", side_effect=fake_load) as load:
        plugin.load_aligned_weights(tmp_path)
    load.assert_called_once_with(model_name=str(tmp_path), apply_peft=False, tokenizer_name=None)


def test_load_empty_dir_raises(tmp_path):
    with pytest.raises(RuntimeError, match="nothing to load"):
        AlignmentPlugin(MasterConfig()).load_aligned_weights(tmp_path)


def test_load_unreachable_repo_raises(tmp_path):
    with patch("huggingface_hub.snapshot_download", side_effect=OSError("offline")):
        with pytest.raises(RuntimeError, match="neither a local directory"):
            AlignmentPlugin(MasterConfig()).load_aligned_weights("org/missing-model")


# --- CLI ----------------------------------------------------------------------


@pytest.fixture
def workdir(tmp_path: Path):
    wd = tmp_path / "run"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"algorithm: sft\nworking_dir: {wd}\ndata:\n  languages: [bam]\n", encoding="utf-8"
    )
    return config, wd


def _fake_trainer(loaded=None):
    trainer = MagicMock()
    trainer.loaded_weights = loaded

    def load_aligned(path):
        trainer.loaded_weights = str(path)

    trainer.load_aligned_weights.side_effect = load_aligned
    trainer.analyze.return_value = '{"lang": "bam", "tokens": []}'
    return trainer


def _with_adapter(wd: Path) -> Path:
    models = wd / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "adapter_config.json").write_text(json.dumps({}), encoding="utf-8")
    return models


def test_eval_loads_trained_weights(workdir):
    config, wd = workdir
    models = _with_adapter(wd)
    trainer = _fake_trainer()
    with patch("beni.core.srl.unified.SRLTrainer", return_value=trainer), patch(
        "beni.core.pipeline.run_eval", return_value={"scope": "bam", "phi": 0.5}
    ) as run_eval:
        result = runner.invoke(app, ["eval", "-c", str(config), "-w", str(wd)])
    assert result.exit_code == 0, result.output
    trainer.load_aligned_weights.assert_called_once_with(str(models))
    trainer.load_models.assert_not_called()
    assert run_eval.call_args.kwargs["weights"] == str(models)
    decode = run_eval.call_args.kwargs["generate"]
    decode("muso", "bam")
    trainer.analyze.assert_called_once_with("muso")


def test_eval_without_weights_warns_and_uses_base(workdir):
    config, wd = workdir
    trainer = _fake_trainer()
    with patch("beni.core.srl.unified.SRLTrainer", return_value=trainer), patch(
        "beni.core.pipeline.run_eval", return_value={}
    ) as run_eval:
        result = runner.invoke(app, ["eval", "-c", str(config), "-w", str(wd)])
    assert result.exit_code == 0, result.output
    trainer.load_models.assert_called_once()
    trainer.load_aligned_weights.assert_not_called()
    assert "no trained weights" in result.output
    assert run_eval.call_args.kwargs["weights"] is None


def test_eval_explicit_bad_weights_fails(workdir):
    config, wd = workdir
    trainer = _fake_trainer()
    trainer.load_aligned_weights.side_effect = RuntimeError("boom")
    with patch("beni.core.srl.unified.SRLTrainer", return_value=trainer), patch(
        "beni.core.pipeline.run_eval"
    ) as run_eval:
        result = runner.invoke(app, ["eval", "-c", str(config), "-w", str(wd), "--weights", "/nope"])
    assert result.exit_code == 1
    run_eval.assert_not_called()


def test_eval_file_flag_sets_eval_source(workdir, tmp_path):
    config, wd = workdir
    held = tmp_path / "held.jsonl"
    held.write_text('{"text": "a", "lang": "bam"}\n', encoding="utf-8")
    with patch("beni.core.srl.unified.SRLTrainer", return_value=_fake_trainer()), patch(
        "beni.core.pipeline.run_eval", return_value={}
    ) as run_eval:
        result = runner.invoke(app, ["eval", "-c", str(config), "-w", str(wd), "--eval-file", str(held)])
    assert result.exit_code == 0, result.output
    assert run_eval.call_args.args[0].experiment.eval_source == str(held)


def test_push_loads_trained_weights(workdir):
    config, wd = workdir
    models = _with_adapter(wd)
    trainer = _fake_trainer()
    with patch("beni.core.srl.unified.SRLTrainer", return_value=trainer):
        result = runner.invoke(app, ["push", "-c", str(config), "-w", str(wd), "--repo-id", "org/m"])
    assert result.exit_code == 0, result.output
    trainer.load_aligned_weights.assert_called_once_with(str(models))
    trainer.push_to_hub.assert_called_once()


def test_push_refuses_without_trained_weights(workdir):
    config, wd = workdir
    trainer = _fake_trainer()
    with patch("beni.core.srl.unified.SRLTrainer", return_value=trainer):
        result = runner.invoke(app, ["push", "-c", str(config), "-w", str(wd), "--repo-id", "org/m"])
    assert result.exit_code == 1
    trainer.push_to_hub.assert_not_called()
    trainer.load_models.assert_not_called()


def test_generate_loads_trained_weights_and_analyzes(workdir):
    config, wd = workdir
    models = _with_adapter(wd)
    trainer = _fake_trainer()
    with patch("beni.core.srl.unified.SRLTrainer", return_value=trainer):
        result = runner.invoke(
            app, ["generate", "-c", str(config), "-w", str(wd), "--prompt", "muso", "--max-new-tokens", "9"]
        )
    assert result.exit_code == 0, result.output
    trainer.load_aligned_weights.assert_called_once_with(str(models))
    trainer.analyze.assert_called_once_with("muso", max_new_tokens=9)


def test_push_refusal_exits_cleanly_and_passes_write_card(workdir):
    config, wd = workdir
    _with_adapter(wd)
    trainer = _fake_trainer()
    trainer.weights_dir = str(wd / "models")
    trainer.push_to_hub.side_effect = PermissionError("run sebeni eval first")
    with patch("beni.core.srl.unified.SRLTrainer", return_value=trainer):
        result = runner.invoke(
            app, ["push", "-c", str(config), "-w", str(wd), "--repo-id", "org/m", "--write-card"]
        )
    assert result.exit_code == 1
    assert "sebeni eval" in result.output
    kwargs = trainer.push_to_hub.call_args.kwargs
    assert kwargs["write_card"] is True and kwargs["output_dir"] == str(wd / "models")


def test_push_uses_active_arm_hub_config(tmp_path):
    wd = tmp_path / "run"
    config = tmp_path / "sft.yaml"
    config.write_text(
        f"algorithm: sft\nworking_dir: {wd}\ndata:\n  languages: [bam]\n"
        "sft:\n  hub_model_id: org/sft-model\n  hub_token: sft-tok\n  hub_private_repo: false\n",
        encoding="utf-8",
    )
    _with_adapter(wd)
    trainer = _fake_trainer()
    with patch("beni.core.srl.unified.SRLTrainer", return_value=trainer):
        result = runner.invoke(app, ["push", "-c", str(config), "-w", str(wd)])
    assert result.exit_code == 0, result.output
    kwargs = trainer.push_to_hub.call_args.kwargs
    assert kwargs["repo_id"] == "org/sft-model"
    assert kwargs["token"] == "sft-tok"
    assert kwargs["private"] is False
