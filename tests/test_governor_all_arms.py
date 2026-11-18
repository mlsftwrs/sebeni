"""Issue 6: the SafetyGovernor gets real signals on every arm (SFT / DPO / APO / GRPO)."""

from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace

import pytest
import torch

from beni.core.srl.apo.apo import SebeniApo
from beni.core.srl.config import MasterConfig
from beni.core.srl.dpo.dpo import SebeniDpo
from beni.core.srl.grpo.grpo import SebeniGrpo
from beni.core.srl.sft.sft import SebeniSft


class CharTokenizer:
    """ids are code points < 256."""

    eos_token = None
    chat_template = None

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


class TinyLM(torch.nn.Module):
    """Logits are sharper while the 'adapter' is on, so π_θ ≠ π_ref."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(256, 8)
        self.head = torch.nn.Linear(8, 256)
        self.adapter_on = True

    def forward(self, input_ids, attention_mask=None):
        logits = self.head(self.emb(input_ids))
        return SimpleNamespace(logits=logits * (2.0 if self.adapter_on else 1.0))

    @contextlib.contextmanager
    def disable_adapter(self):
        self.adapter_on = False
        try:
            yield
        finally:
            self.adapter_on = True


def _target(lang: str = "bam") -> str:
    return json.dumps({"text": "a", "lang": lang, "tokens": [{"surface": "a", "stage": 0, "analyses": []}]})


def _ids(text: str):
    return [ord(c) for c in text]


def _sft_inputs(prompt: str, target: str):
    ids = _ids(prompt + target)
    labels = [-100] * len(prompt) + _ids(target)
    return {
        "input_ids": torch.tensor([ids]),
        "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        "labels": torch.tensor([labels]),
    }


def _wire(plugin_cls, compute_loss, **trainer_attrs):
    mc = MasterConfig()
    mc.data.languages = ["bam"]
    plugin = plugin_cls(mc)
    model = TinyLM()
    model.train()
    plugin.model = model
    plugin.tokenizer = CharTokenizer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    trainer = SimpleNamespace(
        model=model,
        optimizer=optimizer,
        compute_loss=compute_loss,
        state=SimpleNamespace(log_history=[], global_step=0),
        **trainer_attrs,
    )
    plugin._wire_pre_update_hooks(trainer)
    return plugin, trainer, model, optimizer


def _lm_loss(model, inputs, return_outputs=False, **kwargs):
    outputs = model(inputs["input_ids"], inputs.get("attention_mask"))
    loss = outputs.logits.float().mean()
    return (loss, outputs) if return_outputs else loss


@pytest.mark.parametrize("cls, source", [(SebeniSft, "targets"), (SebeniDpo, "targets"), (SebeniApo, "targets"), (SebeniGrpo, "samples")])
def test_signal_source_per_arm(cls, source):
    assert cls.signal_source == source


def test_sft_feeds_format_language_and_logps():
    plugin, trainer, model, optimizer = _wire(SebeniSft, _lm_loss)
    loss = trainer.compute_loss(model, _sft_inputs("PROMPT ", _target("bam")))
    assert torch.is_tensor(loss) and loss.dim() == 0
    rm = plugin.reward_manager
    assert rm.format_scores == [1.0]
    assert rm.lang_ok == [True]
    assert plugin._last_logps["model_logps"] is not None
    assert plugin._last_logps["ref_logps"] is not None
    assert isinstance(plugin._last_logps["kl"], float)

    loss.backward()
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.step()
    assert any(not torch.equal(o, n) for o, n in zip(before, model.parameters()))
    assert "uncertainty_downweight" in plugin.governor.spec.fired
    assert rm.format_scores == []  # cleared after the step


def test_sft_wrong_language_target_blocks_the_step():
    plugin, trainer, model, optimizer = _wire(SebeniSft, _lm_loss)
    loss = trainer.compute_loss(model, _sft_inputs("PROMPT ", _target("dtm")))
    loss.backward()
    before = [p.detach().clone() for p in model.parameters()]
    assert optimizer.step() is None
    assert all(torch.equal(o, n) for o, n in zip(before, model.parameters()))
    assert "language_mismatch_batch" in plugin.governor.spec.fired


def test_sft_logps_survive_trl_popping_labels():
    """TRL's SFT compute_loss can pop "labels" from the inputs dict."""

    def popping_loss(model, inputs, return_outputs=False, **kwargs):
        inputs.pop("labels")
        return _lm_loss(model, inputs, return_outputs=return_outputs)

    plugin, trainer, model, _ = _wire(SebeniSft, popping_loss)
    trainer.compute_loss(model, _sft_inputs("PROMPT ", _target("bam")))
    assert plugin._last_logps["model_logps"] is not None


def test_sft_logps_when_trl_skips_logits():
    def no_logits_loss(model, inputs, return_outputs=False, **kwargs):
        loss, outputs = _lm_loss(model, inputs, return_outputs=True)
        outputs.logits = None
        return (loss, outputs) if return_outputs else loss

    plugin, trainer, model, _ = _wire(SebeniSft, no_logits_loss)
    trainer.compute_loss(model, _sft_inputs("PROMPT ", _target("bam")))
    assert plugin._last_logps["model_logps"] is not None


def test_sft_return_outputs_is_preserved():
    plugin, trainer, model, _ = _wire(SebeniSft, _lm_loss)
    loss, outputs = trainer.compute_loss(model, _sft_inputs("P ", _target()), return_outputs=True)
    assert hasattr(outputs, "logits")


@pytest.mark.parametrize("cls", [SebeniDpo, SebeniApo])
def test_dpo_feeds_chosen_targets_and_log_ratio(cls):
    metrics = {"train": {"rewards/chosen": [0.2], "rewards/rejected": [-0.1]}}
    plugin, trainer, model, optimizer = _wire(cls, _lm_loss, _metrics=metrics, beta=0.1)
    prompt, chosen, rejected = "P ", _target("bam"), "not json"
    rows = [_ids(prompt + chosen), _ids(prompt + rejected)]
    width = max(len(r) for r in rows)
    input_ids = torch.tensor([r + [32] * (width - len(r)) for r in rows])
    completion_mask = torch.tensor(
        [[0] * len(prompt) + [1] * (len(r) - len(prompt)) + [0] * (width - len(r)) for r in rows]
    )
    trainer.compute_loss(model, {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "completion_mask": completion_mask})
    rm = plugin.reward_manager
    assert rm.format_scores == [1.0]  # chosen half only
    assert rm.lang_ok == [True]
    n_c, n_r = len(chosen), len(rejected)  # one token per character
    expected = (0.2 / (0.1 * n_c) + 0.1 / (0.1 * n_r)) / 2
    assert plugin._last_logps["log_ratio"] == pytest.approx(expected)
    assert plugin._last_logps["kl"] == pytest.approx(expected)


def test_dpo_kl_bound_blocks_the_step():
    metrics = {"train": {"rewards/chosen": [5.0], "rewards/rejected": [-5.0]}}
    plugin, trainer, model, optimizer = _wire(SebeniDpo, _lm_loss, _metrics=metrics, beta=0.1)
    plugin.governor.spec.max_kl = 0.1  # per-token |log-ratio| here is 5 / (0.1 * len) ≈ 0.6
    ids = torch.tensor([_ids("P " + _target()), _ids("P " + _target())])
    mask = torch.tensor([[0, 0] + [1] * (ids.shape[1] - 2)] * 2)
    loss = trainer.compute_loss(model, {"input_ids": ids, "attention_mask": torch.ones_like(ids), "completion_mask": mask})
    loss.backward()
    assert optimizer.step() is None
    assert "kl_bound" in plugin.governor.spec.fired


def test_gate_targets_drops_refused_rows():
    plugin = SebeniSft(MasterConfig())
    rows = [
        {"target": _target("bam"), "lang": "bam"},
        {"target": "not json", "lang": "bam"},
        {"target": _target("dtm"), "lang": "bam"},
        {"target": _target("bam,dtm"), "lang": "bam"},
    ]
    kept = plugin.gate_targets(rows, completion_key="target")
    assert kept == rows[:1]
    report = plugin.target_gate_report
    assert report == {
        "n_kept": 1,
        "n_dropped": 3,
        "format_invalid_target": 1,
        "language_mismatch_target": 1,
        "mixed_language_target": 1,
    }
    assert {"format_invalid_target", "language_mismatch_target", "mixed_language_target"} <= set(
        plugin.governor.spec.fired
    )


def test_gate_targets_reads_chat_chosen():
    plugin = SebeniDpo(MasterConfig())
    rows = [{"chosen": [{"role": "assistant", "content": _target("dtm")}], "language": "bam"}]
    assert plugin.gate_targets(rows, completion_key="chosen", lang_key="language") == []


def test_gate_targets_off_when_safety_disabled():
    mc = MasterConfig()
    mc.safety.enabled = False
    plugin = SebeniSft(mc)
    rows = [{"target": "not json", "lang": "bam"}]
    assert plugin.gate_targets(rows, completion_key="target") == rows


# --- issue 24: over-length targets are dropped, not truncated -----------------


class LenTokenizer:
    """One token per character."""

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": list(range(len(text)))}


def test_drop_overlength_rows():
    plugin = SebeniSft(MasterConfig())
    plugin.tokenizer = LenTokenizer()
    rows = [{"p": "x" * 5, "a": "y" * 4}, {"p": "x" * 5, "a": "y" * 10}]
    kept = plugin.drop_overlength(rows, lambda r: r["p"], lambda r: [r["a"]], max_length=10)
    assert kept == rows[:1]
    assert plugin.target_gate_report["overlength_target"] == 1
    assert "overlength_target" in plugin.governor.spec.fired


def test_drop_overlength_uses_longest_answer():
    plugin = SebeniDpo(MasterConfig())
    plugin.tokenizer = LenTokenizer()
    rows = [{"p": "x" * 4, "c": "y" * 3, "r": "z" * 9}]
    with pytest.raises(RuntimeError, match="longer than max_length"):
        plugin.drop_overlength(rows, lambda r: r["p"], lambda r: [r["c"], r["r"]], max_length=10)


def test_drop_overlength_noop_without_limit():
    plugin = SebeniSft(MasterConfig())
    plugin.tokenizer = LenTokenizer()
    rows = [{"p": "x" * 50, "a": ""}]
    assert plugin.drop_overlength(rows, lambda r: r["p"], lambda r: [r["a"]], max_length=None) == rows


class KeepLM(TinyLM):
    """Accepts logits_to_keep like HF causal LMs."""

    def forward(self, input_ids, attention_mask=None, logits_to_keep=0):
        out = super().forward(input_ids, attention_mask)
        if logits_to_keep:
            out.logits = out.logits[:, -logits_to_keep:, :]
        return out


def test_target_logps_with_logits_to_keep_matches_full():
    plugin = SebeniSft(MasterConfig())
    inputs = _sft_inputs("PROMPT ", _target())
    forward = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
    labels = inputs["labels"]
    with torch.no_grad():
        kept = plugin._target_logps(KeepLM(), forward, labels)
        full_logits = TinyLM()(inputs["input_ids"]).logits
        shifted = labels[:, 1:]
        full = torch.log_softmax(full_logits[:, :-1].float(), -1).gather(-1, shifted.clamp(min=0).unsqueeze(-1)).squeeze(-1)[shifted != -100]
    assert torch.allclose(kept, full, atol=1e-5)
    assert kept.numel() == len(_target())


def test_last_logps_cleared_after_each_step():
    plugin, trainer, model, optimizer = _wire(SebeniSft, _lm_loss)
    loss = trainer.compute_loss(model, _sft_inputs("PROMPT ", _target("bam")))
    loss.backward()
    optimizer.step()
    assert plugin._last_logps["model_logps"] is None and plugin._last_logps["kl"] is None


def test_consecutive_blocked_steps_warn(capsys):
    plugin, trainer, model, optimizer = _wire(SebeniSft, _lm_loss)
    for _ in range(10):
        loss = trainer.compute_loss(model, _sft_inputs("P ", _target("dtm")))
        loss.backward()
        optimizer.step()
    assert plugin.blocked_steps == 10
    assert "10 consecutive optimizer steps refused" in capsys.readouterr().out


def test_dpo_length_counts_rendered_conversation():
    class Tok:
        chat_template = "x"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return "".join(f"<{m['role']}>{m['content']}</s>" for m in messages)

    plugin = SebeniDpo(MasterConfig())
    plugin.tokenizer = Tok()
    row = {
        "prompt": [{"role": "user", "content": "hi"}],
        "chosen": [{"role": "assistant", "content": "{}"}],
    }
    assert plugin._rendered_conversation(row, "chosen") == "<user>hi</s><assistant>{}</s>"


def _ref():
    return json.loads(_target("bam"))


@pytest.mark.parametrize("cls, trainer_path, config_path", [
    (SebeniSft, "trl.SFTTrainer", "trl.SFTConfig"),
    (SebeniDpo, "trl.DPOTrainer", "trl.DPOConfig"),
])
def test_arm_train_applies_dataset_gates(cls, trainer_path, config_path, tmp_path):
    from unittest.mock import MagicMock, patch

    mc = MasterConfig()
    mc.working_dir = str(tmp_path)
    mc.apply_workdir()
    mc.data.languages = ["bam"]
    mc.data.eval_ratio = 0.0
    plugin = cls(mc)
    plugin.model = MagicMock()
    plugin.tokenizer = None  # plain-text prompt layout; the TRL trainer is mocked
    rows = [{"text": f"a {i}", "lang": "bam", "reference": _ref()} for i in range(3)]
    trainer = MagicMock()
    trainer.optimizer = None
    with patch(trainer_path, return_value=trainer), patch(config_path), patch.object(
        cls, "gate_targets", side_effect=lambda r, **k: r
    ) as gate, patch.object(cls, "drop_overlength", side_effect=lambda r, **k: r) as drop, patch.object(
        cls, "save_model"
    ), patch("beni.data.datasets.build_dabax_reference", return_value=_ref()):
        plugin.train(rows, project_name="t")
    gate.assert_called_once()
    drop.assert_called_once()


def test_grad_accum_aggregates_log_ratios_across_microbatches():
    metrics = {"train": {"rewards/chosen": [0.2], "rewards/rejected": [-0.2]}}
    plugin, trainer, model, optimizer = _wire(SebeniDpo, _lm_loss, _metrics=metrics, beta=0.1)
    ids = torch.tensor([_ids("P " + _target()), _ids("P " + _target())])
    mask = torch.tensor([[0, 0] + [1] * (ids.shape[1] - 2)] * 2)
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "completion_mask": mask}

    l1 = trainer.compute_loss(model, batch)
    first_kl = plugin._last_logps["kl"]
    metrics["train"]["rewards/chosen"].append(1.8)
    metrics["train"]["rewards/rejected"].append(-1.8)
    l2 = trainer.compute_loss(model, batch)
    second_kl = plugin._last_logps["kl"]
    assert second_kl > first_kl

    (l1 + l2).backward()
    seen = {}
    plugin.register_hook(lambda _m, _r, meta: seen.update(meta))
    optimizer.step()
    assert seen["kl"] == pytest.approx((first_kl + second_kl) / 2.0)
    assert plugin._step_logps == []


def test_kveritas_callback_on_evaluate_uses_sft_raw_text_and_skips_chatml():
    from unittest.mock import patch

    from beni.core.srl.grpo.callbacks import KVeritasCallback

    mc = MasterConfig()
    mc.data.languages = ["bam"]
    eval_ds = [
        {"prompt": "<|im_start|>system\n...\n<|im_start|>user\nmuso\n", "completion": "{}", "raw_text": "muso", "lang": "bam"},
        {"prompt": "<|im_start|>system\n...\n<|im_start|>user\njunk\n", "completion": "{}"},
    ]
    cb = KVeritasCallback(config=mc, eval_dataset=eval_ds)
    state = SimpleNamespace(global_step=1)
    with patch("beni.core.pipeline.run_eval", return_value={"phi": 0.5}) as mock_eval:
        cb.on_evaluate(args=None, state=state, control=None, metrics={})
    mock_eval.assert_called_once_with(mc, [{"text": "muso", "lang": "bam"}], record=False)
